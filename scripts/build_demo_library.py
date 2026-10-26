"""Build a self-contained video library by copying videos out of existing libraries.

Each ``--video DB GENERATED_DIR VIDEO_ID`` names a source library (its SQLite
database and generated-media directory) and one video in it. The video's
database rows (video, segments, entity appearances), its media file, and its
generated directory are *copied* into ``--output-dir``:

    <output-dir>/library.sqlite3
    <output-dir>/uploads/<video_id><suffix>
    <output-dir>/generated/<video_id>/
    <output-dir>/library_manifest.json

Sources are opened read-only and never modified. The copied media must match
the video's recorded ``content_sha256`` and each generated directory must
match its source file count and size; otherwise the build stops. The new
database points only at the copies inside ``--output-dir``: ``stored_path``
and each segment's ``processing_metadata.thumbnail_path`` are rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidquery.storage import SQLiteRepository

TABLES = ("videos", "segments", "entity_appearances")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_stats(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def relocate_thumbnail(payload_json: str, generated: Path, generated_copy: Path) -> str:
    """Point a segment's thumbnail at the copy; refuse thumbnails outside ``generated``."""
    payload = json.loads(payload_json)
    metadata = payload.get("processing_metadata") or {}
    raw = metadata.get("thumbnail_path")
    if not raw:
        return payload_json
    try:
        relative = Path(raw).resolve().relative_to(generated.resolve())
    except ValueError:
        raise SystemExit(
            f"thumbnail {raw} is outside {generated}; the copy would depend on it") from None
    metadata["thumbnail_path"] = str((generated_copy / relative).resolve())
    return json.dumps(payload)


def copy_video(source_db: Path, source_generated: Path, video_id: str, output: Path,
               target: sqlite3.Connection) -> dict[str, Any]:
    source = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    video = source.execute("select * from videos where video_id = ?", (video_id,)).fetchone()
    if video is None:
        raise SystemExit(f"{video_id} is not in {source_db}")
    media = Path(video["stored_path"])
    media_copy = output / "uploads" / f"{video_id}{media.suffix.lower() or '.mp4'}"
    shutil.copy2(media, media_copy)
    if _sha256(media_copy) != video["content_sha256"]:
        raise SystemExit(f"{media} does not match the recorded content_sha256 of {video_id}")
    generated = source_generated / video_id
    generated_copy = output / "generated" / video_id
    if generated.is_dir():
        shutil.copytree(generated, generated_copy)
        if _tree_stats(generated) != _tree_stats(generated_copy):
            raise SystemExit(f"generated copy of {video_id} is incomplete")
    counts = {}
    for table in TABLES:
        rows = [dict(row) for row in source.execute(
            f"select * from {table} where video_id = ?", (video_id,))]
        if table == "videos":
            rows[0]["stored_path"] = str(media_copy.resolve())
        if table == "segments":
            for row in rows:
                row["payload_json"] = relocate_thumbnail(
                    row["payload_json"], generated, generated_copy)
        for row in rows:
            columns = ", ".join(row)
            placeholders = ", ".join(f":{name}" for name in row)
            target.execute(f"insert into {table} ({columns}) values ({placeholders})", row)
        counts[table] = len(rows)
    source.close()
    return {
        "video_id": video_id,
        "display_name": video["display_name"],
        "source_database": source_db.as_posix(),
        "source_media": media.as_posix(),
        "source_generated": generated.as_posix() if generated.is_dir() else None,
        "media_copy": media_copy.as_posix(),
        "content_sha256": video["content_sha256"],
        "generated_files_and_bytes": (
            list(_tree_stats(generated_copy)) if generated.is_dir() else None),
        "rows": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", nargs=3, action="append", required=True,
                        metavar=("DB", "GENERATED_DIR", "VIDEO_ID"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--database-name", default="library.sqlite3")
    args = parser.parse_args()
    database = args.output_dir / args.database_name
    if database.exists() or (args.output_dir / "uploads").exists():
        raise SystemExit(f"{args.output_dir} already holds a library; choose a new directory")
    (args.output_dir / "uploads").mkdir(parents=True)
    (args.output_dir / "generated").mkdir()
    SQLiteRepository(database)  # creates the canonical schema
    target = sqlite3.connect(database)
    target.execute("PRAGMA foreign_keys = ON")
    entries = []
    with target:
        for source_db, source_generated, video_id in args.video:
            entries.append(copy_video(Path(source_db), Path(source_generated), video_id,
                                      args.output_dir, target))
    integrity = target.execute("pragma integrity_check").fetchone()[0]
    fk_errors = target.execute("pragma foreign_key_check").fetchall()
    target.execute("pragma wal_checkpoint(TRUNCATE)")
    target.close()
    if integrity != "ok" or fk_errors:
        raise SystemExit(f"new library failed checks: integrity={integrity}, fk={fk_errors}")
    manifest = {
        "schema_version": "vidquery-library-manifest-v1",
        "database": database.as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
        "videos": entries,
        "integrity_check": integrity,
    }
    (args.output_dir / "library_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({entry["display_name"]: entry["rows"] for entry in entries}, indent=1))


if __name__ == "__main__":
    main()
