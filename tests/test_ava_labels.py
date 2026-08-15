from pathlib import Path

from vidquery.ava_adapter import _canonical_action, import_fused_ava
from vidquery.ava_labels import AVA_V22_ACTIONS
from vidquery.config import Settings
from vidquery.domain import ProcessingState, VideoRecord, utc_now
from vidquery.storage import SQLiteRepository


def test_official_ava_v22_action_ids_are_not_shifted():
    assert len(AVA_V22_ACTIONS) == 80
    assert AVA_V22_ACTIONS[11] == "sit"
    assert AVA_V22_ACTIONS[12] == "stand"
    assert AVA_V22_ACTIONS[17] == "carry/hold"
    assert AVA_V22_ACTIONS[43] == "point to"
    assert AVA_V22_ACTIONS[63] == "write"
    assert AVA_V22_ACTIONS[74] == "listen to"
    assert AVA_V22_ACTIONS[79] == "talk to"
    assert AVA_V22_ACTIONS[80] == "watch person"


def test_fused_action_tokens_use_official_v22_names():
    assert _canonical_action("action_17") == "carry/hold"
    assert _canonical_action("action_63") == "write"
    assert _canonical_action("action_79") == "talk to"
    assert _canonical_action("unexpected") == "unexpected"


def test_import_prefers_official_ids_over_stale_fused_names(tmp_path):
    now = utc_now()
    video = VideoRecord(
        video_id="video",
        display_name="video.mp4",
        original_filename="video.mp4",
        stored_path=str(tmp_path / "video.mp4"),
        content_sha256="a" * 64,
        duration=10,
        width=100,
        height=100,
        fps=25,
        has_audio=False,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
    )
    repository = SQLiteRepository(tmp_path / "db.sqlite3")
    repository.create_video(video)
    fused = tmp_path / "sample.fused.json"
    fused.write_text(
        """{
          "video_id": "sample", "split": "train", "frames": [{
            "timestamp": 1, "nodes": [{
              "node_id": 0, "class_name": "person", "confidence": 1,
              "bbox": [0, 0, 50, 50],
              "matched_annotation": {"action_ids": [12, 17],
                                     "action_labels": ["sit", "write"]}
            }], "edges": [], "audio_segments": []
          }]
        }""",
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        generated_dir=tmp_path / "generated",
        database_path=tmp_path / "db.sqlite3",
        whisper_cache_dir=tmp_path / "whisper",
    )

    import_fused_ava(
        video=video, fused_path=Path(fused), settings=settings, repository=repository
    )

    assert repository.list_segments([video.video_id])[0].actions == [
        "carry/hold",
        "stand",
    ]
