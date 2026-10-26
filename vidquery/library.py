"""Remove a video from one library, and restore it, without deleting anything.

A library is one ``DATABASE_PATH`` together with its ``UPLOAD_DIR`` and
``GENERATED_DIR``.  Removing a video from it:

1. exports the video's rows (video, segments, entity appearances) into a new
   entry of the library's own trash, ``<database dir>/library_trash/<database stem>/``;
2. moves the files this library owns into that entry: the media file only when
   it lies inside ``UPLOAD_DIR`` and no other video row uses it, and
   ``GENERATED_DIR/<video_id>``.  Media outside ``UPLOAD_DIR`` (a shared source
   registered in place, such as a dataset clip) is never moved or modified;
3. deletes the rows in one SQLite transaction.

If step 2 or 3 fails, the files already moved are moved back and the rows stay,
so the library is unchanged.  :func:`restore_removed_video` reverses a completed
removal.  The optional Neo4j mirror is updated after the SQLite commit; if that
fails, its stale nodes are harmless because graph candidates are always filtered
through SQLite, which stays authoritative.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .config import Settings
from .domain import ProcessingState, utc_now
from .pipeline import ProcessingConflictError
from .storage import SQLiteRepository, VideoNotFoundError

LOGGER = logging.getLogger(__name__)

REMOVAL_SCHEMA = "vidquery-removal-v1"
MANIFEST_NAME = "manifest.json"
ROWS_NAME = "rows.json"
VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REMOVABLE_STATES = frozenset(
    {ProcessingState.UPLOADED, ProcessingState.READY, ProcessingState.FAILED}
)
RESTORABLE_STATUSES = frozenset({"removed", "removing"})


class LibraryError(RuntimeError):
    """Base class; the message is safe to show to users."""


class InvalidVideoIdError(LibraryError, ValueError):
    pass


class RemovalConflictError(LibraryError):
    """The video is busy (processing, or one of its files is in use); nothing changed."""


class RemovalFailedError(LibraryError):
    def __init__(self, message: str, *, rolled_back: bool, entry: Path | None) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back
        self.entry = entry


class RestoreError(LibraryError):
    pass


def validate_video_id(video_id: str) -> str:
    if not VIDEO_ID_PATTERN.fullmatch(video_id) or ".." in video_id:
        raise InvalidVideoIdError("Invalid video id")
    return video_id


def library_trash_dir(settings: Settings) -> Path:
    """The trash of the library whose database is ``settings.database_path``."""

    database = Path(settings.database_path).resolve()
    return database.parent / "library_trash" / database.stem


def restore_command_for(settings: Settings, entry: Path) -> str:
    return (
        f'python -m vidquery restore-video "{entry}" '
        f'--database "{Path(settings.database_path).resolve()}" '
        f'--upload-dir "{Path(settings.upload_dir).resolve()}" '
        f'--generated-dir "{Path(settings.generated_dir).resolve()}"'
    )


_LIBRARY_LOCKS: dict[Path, Lock] = {}
_REGISTRY_LOCK = Lock()


def _library_lock(database_path: Path) -> Lock:
    with _REGISTRY_LOCK:
        return _LIBRARY_LOCKS.setdefault(Path(database_path).resolve(), Lock())


def _within(path: Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rename(source: Path, destination: Path) -> None:
    """Move within one volume.  No copy fallback: a move either happens whole or not at all."""

    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, destination)


def _undo_moves(moves: list[tuple[Path, Path]]) -> list[dict[str, str]]:
    """Move ``(source, destination)`` pairs back, newest first; returns those that stayed."""

    stranded = []
    for source, destination in reversed(moves):
        try:
            _rename(destination, source)
        except OSError:
            LOGGER.exception("Could not move %s back to %s", destination, source)
            stranded.append({"original": str(source), "now_at": str(destination)})
    return stranded


@dataclass(frozen=True, slots=True)
class RemovalResult:
    video_id: str
    display_name: str
    entry: Path
    rows: dict[str, int]
    files: list[dict[str, str]]
    neo4j: str
    warnings: list[str]
    restore_command: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "display_name": self.display_name,
            "removed": True,
            "trash_entry": str(self.entry),
            "rows_deleted": self.rows,
            "files": self.files,
            "neo4j": self.neo4j,
            "warnings": self.warnings,
            "restore_command": self.restore_command,
        }


class LibraryRemovalService:
    def __init__(
        self,
        settings: Settings,
        repository: SQLiteRepository,
        *,
        processor: Any | None = None,
        frame_index: Any | None = None,
        neo4j_factory: Callable[[], Any] | None = None,
        lock_timeout: float = 30.0,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.processor = processor
        self.frame_index = frame_index
        self.neo4j_factory = neo4j_factory
        self.lock_timeout = lock_timeout
        self.trash_dir = library_trash_dir(settings)

    def remove(self, video_id: str) -> RemovalResult:
        """Remove one video; raises ``VideoNotFoundError`` if this library lacks it."""

        validate_video_id(video_id)
        lock = _library_lock(self.repository.path)
        if not lock.acquire(timeout=self.lock_timeout):
            raise RemovalConflictError(
                "Another removal in this library is still running; try again."
            )
        try:
            with self._reservation(video_id):
                return self._remove(video_id)
        except ProcessingConflictError as exc:
            raise RemovalConflictError(
                "This video is processing; remove it once it is READY or FAILED."
            ) from exc
        finally:
            lock.release()

    def _reservation(self, video_id: str) -> AbstractContextManager[Any]:
        # Processors without reservations (test stubs) only get the library lock.
        reserve = getattr(self.processor, "reserved", None)
        return reserve(video_id) if reserve is not None else nullcontext()

    def _remove(self, video_id: str) -> RemovalResult:
        record = self.repository.get_video(video_id)
        if record.state not in REMOVABLE_STATES:
            raise RemovalConflictError(
                f"{record.display_name} is still processing ({record.state.value}); "
                "remove it once it is READY or FAILED."
            )
        rows = self.repository.export_video_rows(video_id)
        counts = {table: len(items) for table, items in rows.items()}
        planned, files = self._plan_files(record.video_id, Path(record.stored_path))

        entry = self.trash_dir / f"{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}-{video_id}"
        entry.mkdir(parents=True)
        manifest: dict[str, Any] = {
            "schema_version": REMOVAL_SCHEMA,
            "status": "removing",
            "video_id": video_id,
            "display_name": record.display_name,
            "content_sha256": record.content_sha256,
            "database": str(Path(self.repository.path).resolve()),
            "upload_dir": str(Path(self.settings.upload_dir).resolve()),
            "generated_dir": str(Path(self.settings.generated_dir).resolve()),
            "removed_at": utc_now().isoformat(),
            "restore_command": restore_command_for(self.settings, entry),
            "rows": counts,
            "files": files,
            "moved": [
                {"kind": kind, "original": str(source), "trashed": relative}
                for kind, source, relative in planned
            ],
        }
        _write_json(entry / ROWS_NAME, rows)
        _write_json(entry / MANIFEST_NAME, manifest)

        moves: list[tuple[Path, Path]] = []
        try:
            for _kind, source, relative in planned:
                destination = entry / relative
                _rename(source, destination)
                moves.append((source, destination))
            deleted = self.repository.delete_video(video_id, expected_counts=counts)
        except Exception as exc:
            stranded = _undo_moves(moves)
            rolled_back = not stranded
            self._forget(video_id)
            stranded_paths = {item["original"] for item in stranded}
            manifest["status"] = "rolled_back" if rolled_back else "rollback_incomplete"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["files"] = [
                {
                    **item,
                    "action": "stranded_in_trash"
                    if str(Path(item["path"]).resolve()) in stranded_paths
                    else "unchanged",
                }
                if item["action"] == "moved"
                else item
                for item in files
            ]
            manifest["stranded"] = stranded
            _write_json(entry / MANIFEST_NAME, manifest)
            if not rolled_back:
                raise RemovalFailedError(
                    "Removal failed and some files could not be moved back; "
                    f"see {entry / MANIFEST_NAME}.",
                    rolled_back=False,
                    entry=entry,
                ) from exc
            if isinstance(exc, PermissionError):
                raise RemovalConflictError(
                    "A file of this video is in use (for example, open in a player). "
                    "Nothing was removed; close it and try again."
                ) from exc
            raise RemovalFailedError(
                "Removal failed and was rolled back; the library is unchanged.",
                rolled_back=True,
                entry=entry,
            ) from exc

        self._forget(video_id)
        warnings: list[str] = []
        neo4j = self._remove_from_mirror(video_id, warnings)
        manifest.update(status="removed", rows=deleted, neo4j=neo4j, warnings=warnings)
        try:
            _write_json(entry / MANIFEST_NAME, manifest)
        except OSError:
            LOGGER.exception("Removed %s but could not update %s", video_id, entry)
            warnings.append(f"Removed, but the trash manifest in {entry} still says 'removing'.")
        return RemovalResult(
            video_id=video_id,
            display_name=record.display_name,
            entry=entry,
            rows=deleted,
            files=files,
            neo4j=neo4j,
            warnings=warnings,
            restore_command=manifest["restore_command"],
        )

    def _plan_files(
        self, video_id: str, media: Path
    ) -> tuple[list[tuple[str, Path, str]], list[dict[str, str]]]:
        planned: list[tuple[str, Path, str]] = []
        files: list[dict[str, str]] = []

        def keep(kind: str, path: Path, action: str, reason: str = "") -> None:
            files.append({"kind": kind, "path": str(path), "action": action, "reason": reason})

        if media.is_symlink():
            keep("media", media, "left_in_place", "symbolic link")
        elif not media.exists():
            keep("media", media, "missing", "the stored media file was already missing")
        elif not media.is_file() or not _within(media, self.settings.upload_dir):
            keep("media", media, "left_in_place", "outside this library's upload directory")
        elif any(
            other.video_id != video_id and Path(other.stored_path).resolve() == media.resolve()
            for other in self.repository.list_videos()
        ):
            keep("media", media, "left_in_place", "also used by another video in this library")
        else:
            planned.append(("media", media.resolve(), f"media/{media.name}"))
            keep("media", media, "moved")

        generated_root = Path(self.settings.generated_dir).resolve()
        generated = Path(self.settings.generated_dir) / video_id
        if generated.is_symlink() or (
            generated.exists() and generated.resolve().parent != generated_root
        ):
            keep("generated", generated, "left_in_place", "not a directory of this library")
        elif not generated.is_dir():
            keep("generated", generated, "absent")
        else:
            planned.append(("generated", generated.resolve(), f"generated/{video_id}"))
            keep("generated", generated, "moved")
        return planned, files

    def _forget(self, video_id: str) -> None:
        forget = getattr(self.frame_index, "forget", None)
        if forget is not None:
            forget(video_id)

    def _remove_from_mirror(self, video_id: str, warnings: list[str]) -> str:
        if not self.settings.enable_neo4j or self.neo4j_factory is None:
            return "disabled"
        try:
            indexer = self.neo4j_factory()
            try:
                indexer.remove_video(video_id)
            finally:
                indexer.close()
        except Exception:
            LOGGER.exception("Neo4j mirror removal failed for %s", video_id)
            warnings.append(
                "The Neo4j mirror still holds this video's nodes. Results are unaffected "
                "because graph candidates are filtered through SQLite."
            )
            return "stale"
        return "removed"


def list_removed(settings: Settings) -> list[dict[str, Any]]:
    """Every entry of this library's trash, oldest first."""

    trash = library_trash_dir(settings)
    entries = []
    for manifest_path in sorted(trash.glob(f"*/{MANIFEST_NAME}")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {"status": "unreadable"}
        entries.append(
            {
                "entry": str(manifest_path.parent),
                "status": manifest.get("status"),
                "video_id": manifest.get("video_id"),
                "display_name": manifest.get("display_name"),
                "removed_at": manifest.get("removed_at"),
                "restored_at": manifest.get("restored_at"),
            }
        )
    return entries


def restore_removed_video(
    settings: Settings,
    repository: SQLiteRepository,
    entry: Path,
    *,
    neo4j_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Put a removed video's files and rows back into the library it came from."""

    entry = Path(entry).resolve()
    if entry.parent != library_trash_dir(settings):
        raise RestoreError(f"{entry} is not an entry of this library's trash")
    try:
        manifest = json.loads((entry / MANIFEST_NAME).read_text(encoding="utf-8"))
        rows = json.loads((entry / ROWS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RestoreError(f"{entry} has no readable removal record") from exc
    if manifest.get("schema_version") != REMOVAL_SCHEMA:
        raise RestoreError("Unknown removal record format")
    if manifest.get("database") != str(Path(repository.path).resolve()):
        raise RestoreError(
            f"This entry was removed from the library at {manifest.get('database')}; "
            "restore it with that library's --database, --upload-dir and --generated-dir"
        )
    if manifest.get("status") not in RESTORABLE_STATUSES:
        raise RestoreError(f"Nothing to restore: this entry is {manifest.get('status')}")
    video_id = validate_video_id(str(manifest["video_id"]))
    if any(row.get("video_id") != video_id for items in rows.values() for row in items):
        raise RestoreError("The removal record holds rows of another video")

    try:
        existing = repository.get_video(video_id)
    except VideoNotFoundError:
        existing = None
    if existing is not None and existing.content_sha256 != manifest["content_sha256"]:
        raise RestoreError(f"A different video with id {video_id} is in the library")
    if existing is None:
        duplicate = repository.find_by_hash(manifest["content_sha256"])
        if duplicate is not None:
            raise RestoreError(
                f"This video is in the library again as {duplicate.video_id}; "
                "remove that copy first"
            )

    roots = {"media": settings.upload_dir, "generated": settings.generated_dir}
    pending: list[tuple[Path, Path]] = []
    for item in manifest.get("moved", []):
        original = Path(item["original"])
        trashed = (entry / item["trashed"]).resolve()
        if item.get("kind") not in roots or not _within(original, roots[item["kind"]]):
            raise RestoreError(f"{original} is outside this library's directories")
        if not trashed.is_relative_to(entry):
            raise RestoreError(f"{item['trashed']} points outside the trash entry")
        if not trashed.exists():
            if original.exists() and manifest["status"] == "removing":
                continue  # interrupted before this file was moved
            raise RestoreError(f"{trashed} is missing from the trash entry")
        if original.exists():
            raise RestoreError(f"{original} already exists; nothing was restored")
        pending.append((trashed, original))

    moves: list[tuple[Path, Path]] = []
    try:
        for trashed, original in pending:
            _rename(trashed, original)
            moves.append((trashed, original))
        inserted = repository.insert_video_rows(rows) if existing is None else {}
    except Exception as exc:
        if _undo_moves(moves):
            raise RestoreError(
                f"Restore failed and some files are now outside {entry}; see the log"
            ) from exc
        raise RestoreError(f"Restore failed and was rolled back: {exc}") from exc

    neo4j = "disabled"
    if settings.enable_neo4j and neo4j_factory is not None:
        try:
            indexer = neo4j_factory()
            try:
                indexer.ingest(
                    repository.get_video(video_id),
                    repository.list_segments([video_id]),
                    repository.list_entity_appearances([video_id]),
                )
            finally:
                indexer.close()
            neo4j = "mirrored"
        except Exception:
            LOGGER.exception("Neo4j mirror restore failed for %s", video_id)
            neo4j = "stale"
    manifest.update(status="restored", restored_at=utc_now().isoformat(), neo4j_restore=neo4j)
    _write_json(entry / MANIFEST_NAME, manifest)
    return {
        "video_id": video_id,
        "display_name": manifest.get("display_name"),
        "rows_inserted": inserted,
        "files_restored": [str(original) for _trashed, original in moves],
        "neo4j": neo4j,
    }
