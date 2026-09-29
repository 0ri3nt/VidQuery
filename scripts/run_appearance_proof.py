"""Build and query the controlled red/black seated-person appearance fixture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from PIL import Image

from vidquery.api import create_app
from vidquery.appearance import TrackAppearanceExtractor, configured_appearance_encoder
from vidquery.config import PROJECT_ROOT, get_settings
from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    FrameRecord,
    ProcessingState,
    VideoRecord,
)
from vidquery.geometry import centroid, normalize_bbox
from vidquery.models import YoloVisualDetector
from vidquery.storage import SQLiteRepository, VideoNotFoundError
from vidquery.tracking import assign_detection_tracks

VIDEO_ID = "appearance-proof-v1"
FIXTURE_DIR = PROJECT_ROOT / "evaluation" / "fixtures" / "appearance"


class _NoopProcessor:
    def process(self, video_id: str) -> None:
        del video_id


def _frame_records() -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    for index, (filename, timestamp) in enumerate(
        (("red_shirt_seated.png", 0.0), ("black_shirt_seated.png", 5.0))
    ):
        path = FIXTURE_DIR / filename
        with Image.open(path) as image:
            width, height = image.size
        frames.append(
            FrameRecord(
                frame_id=f"{VIDEO_ID}_{int(timestamp * 1000):012d}",
                video_id=VIDEO_ID,
                frame_index=index,
                timestamp=timestamp,
                path=str(path.resolve()),
                width=width,
                height=height,
            )
        )
    return frames


def _chair_annotation(frame: FrameRecord, ordinal: int) -> Detection:
    # The chair is unambiguously visible but occluded enough that YOLOv8n does
    # not return it. This explicit controlled-fixture annotation is kept
    # separate from the real YOLO person detection and real CLIP crop evidence.
    box = BoundingBox(x1=540, y1=430, x2=995, y2=1005)
    normalized = normalize_bbox(box, frame.width, frame.height)
    return Detection(
        detection_id=f"fixture-chair-{ordinal}",
        frame_id=frame.frame_id,
        video_id=VIDEO_ID,
        timestamp=frame.timestamp,
        class_id=56,
        class_label="chair",
        confidence=1.0,
        pixel_bbox=box,
        normalized_bbox=normalized,
        centroid=centroid(box),
        centroid_normalized=centroid(normalized),
        track_id=f"fixture-chair-track-{ordinal}",
    )


def main() -> None:
    settings = replace(
        get_settings(),
        enable_appearance_features=True,
        appearance_device="cpu",
        enable_yolo=True,
        allow_model_downloads=False,
        semantic_retrieval_mode="hashing",
        enable_query_planner=False,
        enable_rag_generation=False,
        enable_neo4j=False,
    )
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    video_path = FIXTURE_DIR / "red_black_seated_fixture.mp4"
    now = datetime.now(UTC)
    video = VideoRecord(
        video_id=VIDEO_ID,
        display_name=video_path.name,
        original_filename=video_path.name,
        stored_path=str(video_path.resolve()),
        content_sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),
        duration=10.0,
        width=768,
        height=512,
        fps=5.0,
        has_audio=False,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
        current_stage="appearance proof indexed",
    )
    try:
        repository.get_video(VIDEO_ID)
    except VideoNotFoundError:
        repository.create_video(video)

    frames = _frame_records()
    people = assign_detection_tracks(YoloVisualDetector(settings).detect(frames))
    if len(people) != 2 or {item.class_label for item in people} != {"person"}:
        raise RuntimeError("controlled proof expected exactly one YOLO person per scene")
    encoder = configured_appearance_encoder(settings)
    if encoder is None:
        raise RuntimeError("appearance encoder is not configured")
    extractor = TrackAppearanceExtractor(
        encoder,
        attribute_min_confidence=settings.attribute_min_confidence,
        max_observations=settings.appearance_max_observations,
    )
    appearances, attached_people = extractor.extract(
        video_id=VIDEO_ID,
        frames=frames,
        detections=people,
        existing=repository.list_entity_appearances([VIDEO_ID]),
    )
    by_timestamp = {item.timestamp: item for item in attached_people}
    segments: list[CanonicalSegment] = []
    for index, frame in enumerate(frames):
        person = by_timestamp[frame.timestamp]
        chair = _chair_annotation(frame, index)
        color = "red" if index == 0 else "black"
        segments.append(
            CanonicalSegment(
                segment_id=(
                    f"{VIDEO_ID}:{int(frame.timestamp * 1000)}:"
                    f"{int((frame.timestamp + 5) * 1000)}"
                ),
                video_id=VIDEO_ID,
                start_time=frame.timestamp,
                end_time=frame.timestamp + 5,
                frame_ids=[frame.frame_id],
                detections=[person, chair],
                entities=["person", "chair"],
                actions=["sit"],
                processing_metadata={
                    "controlled_fixture": True,
                    "expected_upper_clothing_color": color,
                    "person_source": "yolov8n",
                    "chair_source": "controlled_fixture_ground_truth",
                    "action_sources": {"sit": "controlled_fixture_ground_truth"},
                    "action_confidences": {"sit": 1.0},
                    "appearance_source": "frozen_open_clip_crop",
                },
            )
        )
    repository.replace_entity_appearances(VIDEO_ID, appearances)
    repository.replace_segments(VIDEO_ID, segments)

    app = create_app(settings, repository, _NoopProcessor())  # type: ignore[arg-type]
    client = TestClient(app)
    queries = [
        "find the red-shirt person sitting on the chair",
        "find the black-shirt person sitting on the chair",
        "find a person sitting on the chair",
    ]
    responses = []
    for query in queries:
        response = client.post(
            "/api/search",
            json={"query": query, "video_ids": [VIDEO_ID], "limit": 10},
        )
        response.raise_for_status()
        payload = response.json()
        responses.append(
            {
                "query": query,
                "parsed_query": payload["parsed_query"],
                "results": [
                    {
                        "rank": rank,
                        "segment_id": item["segment_id"],
                        "start_time": item["start_time"],
                        "end_time": item["end_time"],
                        "score": item["score"],
                        "match_reason": item["match_reason"],
                        "appearance_matches": item["appearance_matches"],
                    }
                    for rank, item in enumerate(payload["results"], start=1)
                ],
            }
        )
    output = {
        "proof_version": "appearance-proof-v1",
        "fixture_video": str(video_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model": {
            "name": encoder.model_name,
            "version": encoder.model_version,
            "weights_cached": True,
        },
        "detector": "YOLOv8n person crops",
        "controlled_annotations": ["chair", "sit"],
        "appearance_record_count": len(appearances),
        "queries": responses,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
