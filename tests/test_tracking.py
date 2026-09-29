from __future__ import annotations

from vidquery.domain import (
    BoundingBox,
    Detection,
    FrameRecord,
    Relationship,
    RelationshipQuery,
    RelationshipSource,
)
from vidquery.relationships import matching_relationships
from vidquery.segments import build_canonical_segments
from vidquery.tracking import assign_detection_tracks, smooth_relationship_intervals


def _detection(
    detection_id: str,
    class_label: str,
    timestamp: float,
    box: tuple[float, float, float, float],
) -> Detection:
    normalized = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    return Detection(
        detection_id=detection_id,
        frame_id=f"frame-{timestamp}",
        video_id="video-1",
        timestamp=timestamp,
        class_id=0 if class_label == "person" else 1,
        class_label=class_label,
        confidence=0.9,
        pixel_bbox=BoundingBox(
            x1=box[0] * 100,
            y1=box[1] * 100,
            x2=box[2] * 100,
            y2=box[3] * 100,
        ),
        normalized_bbox=normalized,
        centroid=((box[0] + box[2]) * 50, (box[1] + box[3]) * 50),
        centroid_normalized=(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
        ),
    )


def _relationship(
    source: Detection,
    target: Detection,
    predicate: str,
    confidence: float,
) -> Relationship:
    return Relationship(
        relationship_id=f"{source.detection_id}-{predicate}-{target.detection_id}",
        source_id=source.detection_id,
        target_id=target.detection_id,
        predicate=predicate,
        confidence=confidence,
        source_method=RelationshipSource.RELATIONSHIP_GNN,
        timestamp=source.timestamp,
        model_version="relation-test-v1",
        start_time=source.timestamp,
        end_time=source.timestamp + 1,
    )


def test_class_aware_tracking_is_deterministic_and_respects_gaps():
    detections = [
        _detection("person-0", "person", 0, (0.1, 0.1, 0.3, 0.5)),
        _detection("chair-0", "chair", 0, (0.1, 0.1, 0.3, 0.5)),
        _detection("person-1", "person", 1, (0.12, 0.1, 0.32, 0.5)),
        _detection("person-4", "person", 4, (0.12, 0.1, 0.32, 0.5)),
    ]
    first = assign_detection_tracks(detections, max_gap_seconds=1.5)
    second = assign_detection_tracks(detections, max_gap_seconds=1.5)
    tracks = {item.detection_id: item.track_id for item in first}

    assert tracks == {item.detection_id: item.track_id for item in second}
    assert tracks["person-0"] == tracks["person-1"]
    assert tracks["person-0"] != tracks["chair-0"]
    assert tracks["person-1"] != tracks["person-4"]


def test_relationship_smoothing_aggregates_confidence_and_preserves_direction():
    detections = assign_detection_tracks(
        [
            _detection("person-0", "person", 0, (0.1, 0.1, 0.3, 0.5)),
            _detection("chair-0", "chair", 0, (0.5, 0.1, 0.8, 0.6)),
            _detection("person-1", "person", 1, (0.11, 0.1, 0.31, 0.5)),
            _detection("chair-1", "chair", 1, (0.51, 0.1, 0.81, 0.6)),
        ]
    )
    by_id = {item.detection_id: item for item in detections}
    relationships = [
        _relationship(by_id["person-0"], by_id["chair-0"], "left_of", 0.6),
        _relationship(by_id["person-1"], by_id["chair-1"], "left_of", 0.8),
        _relationship(by_id["chair-0"], by_id["person-0"], "left_of", 0.7),
    ]

    smoothed = smooth_relationship_intervals(
        relationships, detections, sample_interval=1, max_gap_seconds=1.5
    )

    assert len(smoothed) == 2
    forward = next(item for item in smoothed if item.source_id.startswith("person"))
    reverse = next(item for item in smoothed if item.source_id.startswith("chair"))
    assert forward.observation_count == 2
    assert forward.temporal_smoothed is True
    assert forward.confidence == 0.7
    assert (forward.start_time, forward.end_time) == (0, 2)
    assert reverse.observation_count == 1


def test_smoothed_tuple_is_localized_into_each_overlapping_segment():
    raw_detections = [
        _detection("person-4", "person", 4, (0.1, 0.1, 0.3, 0.5)),
        _detection("chair-4", "chair", 4, (0.35, 0.1, 0.6, 0.5)),
        _detection("person-5", "person", 5, (0.11, 0.1, 0.31, 0.5)),
        _detection("chair-5", "chair", 5, (0.36, 0.1, 0.61, 0.5)),
    ]
    detections = assign_detection_tracks(raw_detections)
    by_id = {item.detection_id: item for item in detections}
    smoothed = smooth_relationship_intervals(
        [
            _relationship(by_id["person-4"], by_id["chair-4"], "near", 0.7),
            _relationship(by_id["person-5"], by_id["chair-5"], "near", 0.9),
        ],
        detections,
        sample_interval=1,
    )
    frames = [
        FrameRecord(
            frame_id=f"frame-{timestamp}",
            video_id="video-1",
            frame_index=index,
            timestamp=timestamp,
            path=f"frame-{timestamp}.jpg",
            width=100,
            height=100,
        )
        for index, timestamp in enumerate((4.0, 5.0))
    ]

    segments = build_canonical_segments(
        video_id="video-1",
        duration=10,
        segment_duration=5,
        frames=frames,
        detections=detections,
        relationships=smoothed,
        transcripts=[],
    )
    query = RelationshipQuery(
        subject="person", predicate="near", object="chair", directionality="symmetric"
    )

    assert len(segments[0].relationships) == 1
    assert len(segments[1].relationships) == 1
    assert matching_relationships(segments[0], query)
    assert matching_relationships(segments[1], query)
    assert segments[0].relationships[0].source_id == "person-4"
    assert segments[1].relationships[0].source_id == "person-5"
    assert segments[0].relationships[0].observation_count == 2
