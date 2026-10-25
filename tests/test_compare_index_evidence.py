from __future__ import annotations

from test_search_and_storage import evidence_segment, video_record

from scripts.compare_index_evidence import compare
from vidquery.storage import SQLiteRepository


def _index(path, video_id, transcript):
    repository = SQLiteRepository(path)
    repository.create_video(video_record(video_id))
    segment = evidence_segment(video_id).model_copy(update={"transcript": transcript})
    repository.replace_segments(video_id, [segment])
    return path


def test_identical_evidence_matches_across_random_video_ids(tmp_path):
    left = _index(tmp_path / "a.sqlite3", "video-1", "hello")
    right = _index(tmp_path / "b.sqlite3", "video-2", "hello")
    report = compare(left, right)
    assert report["segments_compared"] == 1
    assert report["identical"] is True


def test_transcript_change_is_counted(tmp_path):
    left = _index(tmp_path / "a.sqlite3", "video-1", "hello")
    right = _index(tmp_path / "b.sqlite3", "video-1", "hullo")
    report = compare(left, right)
    assert report["identical"] is False
    assert report["segments_differing_by_field"]["transcript"] == 1
    assert report["segments_differing_by_field"]["detections"] == 0
