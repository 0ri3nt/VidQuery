from __future__ import annotations

import json
from datetime import UTC, datetime

from vidquery.demo_manifest import apply_demo_manifest
from vidquery.domain import CanonicalSegment, ProcessingState, RelationshipSource, VideoRecord
from vidquery.storage import SQLiteRepository


def test_demo_manifest_adds_idempotent_explicit_manual_evidence(tmp_path):
    repository = SQLiteRepository(tmp_path / "demo.sqlite3")
    now = datetime.now(UTC)
    video = VideoRecord(
        video_id="demo-video",
        display_name="demo.mp4",
        original_filename="demo.mp4",
        stored_path="demo.mp4",
        content_sha256="d" * 64,
        duration=5,
        width=1280,
        height=720,
        fps=25,
        has_audio=True,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
    )
    repository.create_video(video)
    repository.replace_segments(
        video.video_id,
        [
            CanonicalSegment(
                segment_id="demo-video:0:5000",
                video_id=video.video_id,
                start_time=0,
                end_time=5,
                frame_ids=["frame-0"],
            )
        ],
    )
    manifest = {
        "schema_version": "1.0",
        "duration_seconds": 5,
        "entities": [
            {"id": "presenter", "class": "person", "normalized_bbox": [0.1, 0.1, 0.4, 0.9]},
            {"id": "laptop", "class": "laptop", "normalized_bbox": [0.2, 0.7, 0.5, 0.9]},
        ],
        "relationships": [{"subject": "presenter", "predicate": "near", "object": "laptop"}],
        "segments": [
            {
                "start_time": 0,
                "end_time": 5,
                "speaker": "SPEAKER_00",
                "transcript": "We discuss architecture.",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    first = apply_demo_manifest(repository, video_id=video.video_id, manifest_path=manifest_path)[0]
    second = apply_demo_manifest(repository, video_id=video.video_id, manifest_path=manifest_path)[
        0
    ]

    assert second.transcript == "We discuss architecture."
    assert second.speakers == ["SPEAKER_00"]
    assert second.entities == ["laptop", "person"]
    assert len(second.detections) == len(first.detections) == 2
    assert len(second.relationships) == len(first.relationships) == 1
    relationship = second.relationships[0]
    assert relationship.source_method == RelationshipSource.MANUAL_ANNOTATION
    labels = {item.detection_id: item.class_label for item in second.detections}
    assert labels[relationship.source_id] == "person"
    assert labels[relationship.target_id] == "laptop"
    assert second.processing_metadata["manual_annotation_overlay"] is True
