from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vidquery import library as library_module
from vidquery.api import create_app
from vidquery.domain import EntityAppearance, ProcessingState
from vidquery.library import (
    LibraryRemovalService,
    RemovalConflictError,
    RemovalFailedError,
    RestoreError,
    library_trash_dir,
    list_removed,
    restore_removed_video,
)
from vidquery.media import VideoIngestionService
from vidquery.pipeline import ProcessingConflictError, VideoProcessingService
from vidquery.storage import SQLiteRepository, VideoNotFoundError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): _sha256(item) for item in path.rglob("*") if item.is_file()
    }


class Library:
    """One library (database, uploads, generated) with processed videos."""

    def __init__(self, settings_factory, root: Path, **overrides):
        self.settings = settings_factory(
            data_dir=root / "data",
            upload_dir=root / "uploads",
            generated_dir=root / "generated",
            database_path=root / "library.sqlite3",
            **overrides,
        )
        self.repository = SQLiteRepository(self.settings.database_path)
        self.processor = VideoProcessingService(self.settings, self.repository)
        self.ingestion = VideoIngestionService(self.settings, self.repository)

    def upload(self, video: Path, name: str = "clip.mp4") -> str:
        with video.open("rb") as stream:
            record, _ = self.ingestion.ingest_stream(stream, name, "video/mp4")
        return self._process(record.video_id)

    def register_in_place(self, video: Path) -> str:
        record, _ = self.ingestion.register_existing(video)
        return self._process(record.video_id)

    def _process(self, video_id: str) -> str:
        assert self.processor.process(video_id).state is ProcessingState.READY
        self.repository.replace_entity_appearances(
            video_id,
            [
                EntityAppearance(
                    appearance_id=f"{video_id}-a1",
                    video_id=video_id,
                    track_id="track-1",
                    class_label="person",
                    embedding=[0.1, 0.2],
                    appearance_embedding_model="test",
                    appearance_embedding_version="1",
                    observation_count=2,
                )
            ],
        )
        return video_id

    def service(self, **kwargs) -> LibraryRemovalService:
        kwargs.setdefault("processor", self.processor)
        return LibraryRemovalService(self.settings, self.repository, **kwargs)

    def client(self) -> TestClient:
        return TestClient(create_app(self.settings, self.repository, self.processor))


@pytest.fixture
def make_video(tmp_path):
    """Distinct MP4s: the library deduplicates identical content."""

    import cv2
    import numpy as np

    def factory(name: str, shade: int) -> Path:
        path = tmp_path / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (160, 120))
        if not writer.isOpened():
            pytest.skip("OpenCV MP4 writer is unavailable")
        for index in range(20):
            frame = np.full((120, 160, 3), shade, dtype=np.uint8)
            cv2.rectangle(frame, (10 + index, 20), (60 + index, 100), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        return path

    return factory


@pytest.fixture
def lib(settings_factory, tmp_path):
    return Library(settings_factory, tmp_path / "library-a")


def test_api_removes_rows_and_moves_library_files_to_trash(lib, make_video):
    keep = lib.upload(make_video("keep.mp4", 40), "keep.mp4")
    video_id = lib.upload(make_video("remove.mp4", 90), "remove.mp4")
    media = Path(lib.repository.get_video(video_id).stored_path)
    generated = lib.settings.generated_dir / video_id
    media_sha, generated_tree = _sha256(media), _tree(generated)
    exported = lib.repository.export_video_rows(video_id)
    assert exported["segments"] and exported["entity_appearances"]

    client = lib.client()
    response = client.delete(f"/api/videos/{video_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["removed"] is True and body["display_name"] == "remove.mp4"
    assert body["rows_deleted"] == {
        "videos": 1,
        "segments": len(exported["segments"]),
        "entity_appearances": 1,
    }
    assert {item["kind"]: item["action"] for item in body["files"]} == {
        "media": "moved",
        "generated": "moved",
    }
    entry = Path(body["trash_entry"])
    assert entry.parent == library_trash_dir(lib.settings)
    assert entry.is_relative_to(lib.settings.database_path.parent.resolve())
    assert not media.exists() and not generated.exists()
    assert _sha256(entry / "media" / media.name) == media_sha
    assert _tree(entry / "generated" / video_id) == generated_tree
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "removed"
    assert json.loads((entry / "rows.json").read_text(encoding="utf-8")) == exported
    assert "restore-video" in body["restore_command"]

    with pytest.raises(VideoNotFoundError):
        lib.repository.export_video_rows(video_id)
    assert lib.repository.list_segments([video_id]) == []
    assert lib.repository.list_entity_appearances([video_id]) == []
    assert [video["video_id"] for video in client.get("/api/videos").json()] == [keep]
    assert client.get(f"/api/videos/{video_id}/stream").status_code == 404
    assert lib.repository.list_segments([keep])  # the other video is untouched

    repeated = client.delete(f"/api/videos/{video_id}")
    assert repeated.status_code == 404
    assert repeated.json()["detail"] == "Video was not found in this library"


def test_api_rejects_unknown_and_malformed_ids_without_side_effects(lib, make_video):
    lib.upload(make_video("a.mp4", 40))
    client = lib.client()
    assert client.delete("/api/videos/missing-video").status_code == 404
    for bad in ("x..y", ".hidden", "-x", "a" * 129, "white space"):
        response = client.delete(f"/api/videos/{bad}")
        assert response.status_code == 422, bad
        assert response.json()["detail"] == "Invalid video id"
    assert not library_trash_dir(lib.settings).exists()
    assert len(client.get("/api/videos").json()) == 1


def test_shared_source_registered_in_place_is_never_moved(lib, make_video):
    source = make_video("dataset-clip.mp4", 120)
    source_sha = _sha256(source)
    video_id = lib.register_in_place(source)
    assert Path(lib.repository.get_video(video_id).stored_path) == source.resolve()

    result = lib.service().remove(video_id)

    media = next(item for item in result.files if item["kind"] == "media")
    assert media["action"] == "left_in_place"
    assert media["reason"] == "outside this library's upload directory"
    assert source.is_file() and _sha256(source) == source_sha
    assert not (result.entry / "media").exists()
    assert not (lib.settings.generated_dir / video_id).exists()  # library-owned, moved
    restored = restore_removed_video(lib.settings, lib.repository, result.entry)
    assert restored["rows_inserted"]["videos"] == 1
    assert _sha256(source) == source_sha


def test_media_path_escaping_upload_dir_is_left_in_place(lib, make_video, tmp_path):
    video_id = lib.upload(make_video("a.mp4", 40))
    outside = tmp_path / "outside.mp4"
    shutil.copyfile(lib.repository.get_video(video_id).stored_path, outside)
    escaping = lib.settings.upload_dir / ".." / ".." / "outside.mp4"
    assert escaping.resolve() == outside.resolve()
    with lib.repository._connect() as connection:  # simulate a crafted stored_path
        connection.execute(
            "UPDATE videos SET stored_path = ? WHERE video_id = ?", (str(escaping), video_id)
        )

    result = lib.service().remove(video_id)

    media = next(item for item in result.files if item["kind"] == "media")
    assert media["action"] == "left_in_place"
    assert outside.is_file()


def test_missing_files_do_not_block_removal(lib, make_video):
    video_id = lib.upload(make_video("a.mp4", 40))
    os.remove(lib.repository.get_video(video_id).stored_path)
    shutil.move(lib.settings.generated_dir / video_id, lib.settings.data_dir / "elsewhere")

    result = lib.service().remove(video_id)

    assert {item["kind"]: item["action"] for item in result.files} == {
        "media": "missing",
        "generated": "absent",
    }
    with pytest.raises(VideoNotFoundError):
        lib.repository.get_video(video_id)


def test_processing_video_cannot_be_removed_and_reservation_blocks_processing(lib, make_video):
    video_id = lib.upload(make_video("a.mp4", 40))
    client = lib.client()

    with lib.processor.reserved(video_id):  # the pipeline is processing this video
        response = client.delete(f"/api/videos/{video_id}")
        assert response.status_code == 409
        assert "processing" in response.json()["detail"]
        with pytest.raises(ProcessingConflictError):
            lib.processor.process(video_id)

    lib.repository.update_state(video_id, ProcessingState.VISUAL_PROCESSING)
    assert client.delete(f"/api/videos/{video_id}").status_code == 409
    assert lib.repository.get_video(video_id).state is ProcessingState.VISUAL_PROCESSING
    assert not library_trash_dir(lib.settings).exists()

    lib.repository.update_state(video_id, ProcessingState.FAILED)
    assert client.delete(f"/api/videos/{video_id}").status_code == 200


def test_database_failure_rolls_back_moved_files(lib, make_video, monkeypatch):
    video_id = lib.upload(make_video("a.mp4", 40))
    media = Path(lib.repository.get_video(video_id).stored_path)
    generated = lib.settings.generated_dir / video_id
    before_rows = lib.repository.export_video_rows(video_id)
    before_tree = _tree(generated)

    def fail(*args, **kwargs):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(lib.repository, "delete_video", fail)
    response = lib.client().delete(f"/api/videos/{video_id}")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Removal failed and was rolled back; the library is unchanged."
    )
    assert media.is_file() and _tree(generated) == before_tree
    assert lib.repository.export_video_rows(video_id) == before_rows
    [entry] = list_removed(lib.settings)
    assert entry["status"] == "rolled_back"
    manifest = json.loads((Path(entry["entry"]) / "manifest.json").read_text(encoding="utf-8"))
    assert {item["action"] for item in manifest["files"]} == {"unchanged"}
    assert manifest["stranded"] == []
    with pytest.raises(RestoreError, match="Nothing to restore"):
        restore_removed_video(lib.settings, lib.repository, Path(entry["entry"]))


def test_file_in_use_rolls_back_and_reports_conflict(lib, make_video, monkeypatch):
    video_id = lib.upload(make_video("a.mp4", 40))
    media = Path(lib.repository.get_video(video_id).stored_path)
    real_rename = library_module._rename
    calls = []

    def rename(source, destination):
        calls.append(source)
        if len(calls) == 2:  # the generated directory is "open in another program"
            raise PermissionError(32, "The process cannot access the file")
        real_rename(source, destination)

    monkeypatch.setattr(library_module, "_rename", rename)
    response = lib.client().delete(f"/api/videos/{video_id}")

    assert response.status_code == 409
    assert "in use" in response.json()["detail"]
    assert media.is_file()  # moved, then moved back
    assert (lib.settings.generated_dir / video_id).is_dir()
    assert lib.repository.get_video(video_id).video_id == video_id


def test_incomplete_rollback_is_reported(lib, make_video, monkeypatch):
    video_id = lib.upload(make_video("a.mp4", 40))
    real_rename = library_module._rename
    calls = []

    def rename(source, destination):
        calls.append(source)
        if len(calls) >= 2:  # second move and the move back both fail
            raise OSError("device unavailable")
        real_rename(source, destination)

    monkeypatch.setattr(library_module, "_rename", rename)
    with pytest.raises(RemovalFailedError) as caught:
        lib.service().remove(video_id)

    assert caught.value.rolled_back is False and caught.value.entry is not None
    manifest = json.loads((caught.value.entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rollback_incomplete"
    actions = {item["kind"]: item["action"] for item in manifest["files"]}
    assert actions == {"media": "stranded_in_trash", "generated": "unchanged"}
    [stranded] = manifest["stranded"]
    assert Path(stranded["now_at"]).is_file() and not Path(stranded["original"]).exists()
    assert lib.repository.get_video(video_id)  # rows were never deleted


def test_restore_puts_rows_and_files_back_exactly(lib, make_video):
    video_id = lib.upload(make_video("a.mp4", 40))
    media = Path(lib.repository.get_video(video_id).stored_path)
    generated = lib.settings.generated_dir / video_id
    rows = lib.repository.export_video_rows(video_id)
    media_sha, tree = _sha256(media), _tree(generated)
    result = lib.service().remove(video_id)

    restored = restore_removed_video(lib.settings, lib.repository, result.entry)

    assert restored["rows_inserted"] == {"videos": 1, "segments": len(rows["segments"]),
                                         "entity_appearances": 1}
    assert lib.repository.export_video_rows(video_id) == rows
    assert _sha256(media) == media_sha and _tree(generated) == tree
    assert list_removed(lib.settings)[0]["status"] == "restored"
    with pytest.raises(RestoreError, match="Nothing to restore"):
        restore_removed_video(lib.settings, lib.repository, result.entry)
    client = lib.client()
    assert [video["video_id"] for video in client.get("/api/videos").json()] == [video_id]
    assert client.get(f"/api/videos/{video_id}/thumbnail?timestamp=0").status_code == 200


def test_restore_refuses_conflicts_and_changes_nothing(lib, make_video):
    source = make_video("a.mp4", 40)
    video_id = lib.upload(source)
    result = lib.service().remove(video_id)
    media_name = next(item for item in result.files if item["kind"] == "media")["path"]

    again = lib.upload(source)  # the same content uploaded again under a new id
    with pytest.raises(RestoreError, match="in the library again"):
        restore_removed_video(lib.settings, lib.repository, result.entry)
    assert (result.entry / "media").is_dir() and not Path(media_name).exists()
    lib.service().remove(again)

    tampered = json.loads((result.entry / "manifest.json").read_text(encoding="utf-8"))
    tampered["moved"][0]["trashed"] = "../../../escape.mp4"
    (result.entry / "manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RestoreError, match="outside the trash entry"):
        restore_removed_video(lib.settings, lib.repository, result.entry)


def test_tampered_rows_cannot_inject_sql(lib):
    with pytest.raises(ValueError, match="no column"):
        lib.repository.insert_video_rows(
            {"videos": [{"video_id": "x", "display_name) VALUES (1); DROP TABLE videos; --": 1}]}
        )
    assert lib.repository.list_videos() == []


def test_libraries_are_isolated(settings_factory, tmp_path, make_video):
    library_a = Library(settings_factory, tmp_path / "library-a")
    library_b = Library(settings_factory, tmp_path / "library-b")
    source = make_video("shared.mp4", 40)
    in_a, in_b = library_a.upload(source), library_b.upload(source)
    b_rows = library_b.repository.export_video_rows(in_b)
    b_media = Path(library_b.repository.get_video(in_b).stored_path)
    b_tree = _tree(library_b.settings.generated_dir / in_b)

    # Removing B's video id through A's API is a miss: libraries share nothing.
    assert library_a.client().delete(f"/api/videos/{in_b}").status_code == 404
    result = library_a.service().remove(in_a)

    assert result.entry.is_relative_to((tmp_path / "library-a").resolve())
    assert library_b.repository.export_video_rows(in_b) == b_rows
    assert b_media.is_file() and _tree(library_b.settings.generated_dir / in_b) == b_tree
    assert source.is_file()
    assert not library_trash_dir(library_b.settings).exists()
    with pytest.raises(RestoreError, match="not an entry of this library's trash"):
        restore_removed_video(library_b.settings, library_b.repository, result.entry)


def test_concurrent_removals_of_one_video_succeed_once(lib, make_video):
    video_id = lib.upload(make_video("a.mp4", 40))
    other = lib.upload(make_video("b.mp4", 90))
    service = lib.service()
    outcomes: list[str] = []
    barrier = threading.Barrier(5)

    def remove(target):
        barrier.wait()
        try:
            service.remove(target)
            outcomes.append(f"removed {target}")
        except VideoNotFoundError:
            outcomes.append("not found")
        except RemovalConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=remove, args=(video_id,)) for _ in range(4)]
    threads.append(threading.Thread(target=remove, args=(other,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(f"removed {video_id}") == 1
    assert outcomes.count(f"removed {other}") == 1
    assert outcomes.count("not found") + outcomes.count("conflict") == 3
    assert lib.repository.list_videos() == []
    assert len([item for item in list_removed(lib.settings) if item["status"] == "removed"]) == 2


def test_interrupted_removal_can_be_restored(lib, make_video, monkeypatch):
    video_id = lib.upload(make_video("a.mp4", 40))
    media = Path(lib.repository.get_video(video_id).stored_path)
    real_rename = library_module._rename
    calls = []

    def crash(source, destination):
        calls.append(source)
        if len(calls) == 2:
            raise SystemExit("server killed between the two moves")
        real_rename(source, destination)

    monkeypatch.setattr(library_module, "_rename", crash)
    with pytest.raises(SystemExit):
        lib.service().remove(video_id)
    monkeypatch.setattr(library_module, "_rename", real_rename)
    [entry] = list_removed(lib.settings)
    assert entry["status"] == "removing" and not media.exists()

    restored = restore_removed_video(lib.settings, lib.repository, Path(entry["entry"]))

    assert restored["rows_inserted"] == {}  # the rows were never deleted
    assert media.is_file() and (lib.settings.generated_dir / video_id).is_dir()


class RecordingIndexer:
    def __init__(self, fail: bool = False):
        self.fail, self.closed = fail, False
        self.removed: list[str] = []

    def remove_video(self, video_id):
        if self.fail:
            raise ConnectionError("neo4j down")
        self.removed.append(video_id)

    def close(self):
        self.closed = True


def test_neo4j_mirror_is_updated_and_its_failure_does_not_block(lib, make_video):
    from dataclasses import replace

    lib.settings = replace(lib.settings, enable_neo4j=True)
    first, second = lib.upload(make_video("a.mp4", 40)), lib.upload(make_video("b.mp4", 90))
    indexer = RecordingIndexer()
    result = lib.service(neo4j_factory=lambda: indexer).remove(first)
    assert result.neo4j == "removed" and indexer.removed == [first] and indexer.closed

    down = RecordingIndexer(fail=True)
    result = lib.service(neo4j_factory=lambda: down).remove(second)
    assert result.neo4j == "stale" and result.warnings
    with pytest.raises(VideoNotFoundError):
        lib.repository.get_video(second)


def test_neo4j_remove_video_deletes_only_that_video():
    from vidquery.neo4j import CanonicalNeo4jIndexer

    queries: list[tuple[str, dict | None]] = []

    class Recorder(CanonicalNeo4jIndexer):
        def __init__(self) -> None:  # no driver: queries are only recorded
            pass

        def run_query(self, query: str, parameters: dict | None = None) -> list[dict]:
            queries.append((query, parameters))
            return []

    Recorder().remove_video("video-1")
    assert all(params in (None, {"video_id": "video-1"}) for _, params in queries)
    assert any("DETACH DELETE s, v" in query for query, _ in queries)


def test_frame_embedding_cache_is_dropped_on_removal(lib, make_video):
    video_id = lib.upload(make_video("a.mp4", 40))
    forgotten = []
    index = type("Index", (), {"forget": lambda self, vid: forgotten.append(vid)})()
    lib.service(frame_index=index).remove(video_id)
    assert forgotten == [video_id]


def test_frontend_offers_confirmed_removal(lib):
    client = lib.client()
    page = client.get("/").text
    assert 'id="remove-dialog"' in page and 'value="cancel"' in page and 'id="toast"' in page
    script = client.get("/assets/app.js").text
    assert 'method: "DELETE"' in script
    assert '$("#remove-dialog").returnValue === "confirm"' in script
