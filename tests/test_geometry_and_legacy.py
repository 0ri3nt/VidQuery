from __future__ import annotations

import pytest

from extraction.visual import compute_iou, match_detections_to_annotations
from modelling.gnn_dataset import build_sparse_graphs
from modelling.scene_graph_builder import SceneGraphBuilder
from vidquery.domain import BoundingBox, Detection, FrameRecord, RelationshipSource
from vidquery.geometry import bbox_iou, centroid, interval_overlap, normalize_bbox
from vidquery.segments import build_canonical_segments, build_spatial_relationships


def detection(identifier: str, frame_id: str, label: str, box: tuple[float, float, float, float]):
    normalized = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    pixel = BoundingBox(
        x1=box[0] * 100,
        y1=box[1] * 100,
        x2=box[2] * 100,
        y2=box[3] * 100,
    )
    return Detection(
        detection_id=identifier,
        frame_id=frame_id,
        video_id="video-1",
        timestamp=0.5,
        class_id=0,
        class_label=label,
        confidence=0.9,
        pixel_bbox=pixel,
        normalized_bbox=normalized,
        centroid=centroid(pixel),
        centroid_normalized=centroid(normalized),
    )


def test_bounding_box_rejects_reversed_coordinates():
    with pytest.raises(ValueError):
        BoundingBox(x1=5, y1=0, x2=2, y2=1)


def test_normalize_bbox_clamps_to_image_bounds():
    result = normalize_bbox(BoundingBox(x1=-2, y1=10, x2=120, y2=80), 100, 100)
    assert result.as_list() == [0.0, 0.1, 1.0, 0.8]


def test_centroid_is_box_midpoint():
    assert centroid(BoundingBox(x1=2, y1=4, x2=6, y2=10)) == (4.0, 7.0)


def test_iou_handles_overlap_and_disjoint_boxes():
    first = BoundingBox(x1=0, y1=0, x2=2, y2=2)
    second = BoundingBox(x1=1, y1=1, x2=3, y2=3)
    assert bbox_iou(first, second) == pytest.approx(1 / 7)
    assert bbox_iou(first, BoundingBox(x1=3, y1=3, x2=4, y2=4)) == 0


def test_interval_overlap_uses_half_open_duration():
    assert interval_overlap(0, 2, 1, 3) == 1
    assert interval_overlap(0, 1, 1, 2) == 0
    with pytest.raises(ValueError):
        interval_overlap(2, 1, 0, 1)


def test_legacy_iou_matches_canonical_value():
    assert compute_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_ava_annotation_matching_attaches_action_labels():
    detections = [{"bbox": [0, 0, 100, 100]}]
    annotations = [
        {"entity_id": 7, "bbox": [0, 0, 1, 1], "action_id": 17, "action_label": "write"}
    ]
    matched = match_detections_to_annotations(detections, annotations, 100, 100)
    assert matched[0]["matched_annotation"]["entity_id"] == 7
    assert matched[0]["matched_annotation"]["action_labels"] == ["write"]


def test_scene_builder_iou_and_near_relation():
    builder = SceneGraphBuilder(near_threshold=30)
    left = {"node_id": 1, "bbox": [0, 0, 20, 20], "center": [10, 10]}
    right = {"node_id": 2, "bbox": [15, 0, 35, 20], "center": [25, 10]}
    predicates = {item["type"] for item in builder.get_spatial_relationships(left, right)}
    assert "overlaps_with" in predicates
    assert "near" in predicates


def test_spatial_relations_have_geometry_provenance():
    items = [
        detection("p", "f", "person", (0.1, 0.1, 0.4, 0.8)),
        detection("l", "f", "laptop", (0.38, 0.4, 0.65, 0.7)),
    ]
    relationships = build_spatial_relationships(items, near_threshold=0.5)
    assert any(item.predicate == "near" for item in relationships)
    assert all(
        item.source_method is RelationshipSource.BOUNDING_BOX_GEOMETRY
        for item in relationships
    )


def test_frame_assignment_to_canonical_windows(tmp_path):
    frames = [
        FrameRecord(
            frame_id="f0",
            video_id="video-1",
            frame_index=0,
            timestamp=0.2,
            path=str(tmp_path / "f0.jpg"),
            width=100,
            height=100,
        ),
        FrameRecord(
            frame_id="f1",
            video_id="video-1",
            frame_index=1,
            timestamp=1.2,
            path=str(tmp_path / "f1.jpg"),
            width=100,
            height=100,
        ),
    ]
    segments = build_canonical_segments(
        video_id="video-1",
        duration=2,
        segment_duration=1,
        frames=frames,
        detections=[],
        relationships=[],
        transcripts=[],
    )
    assert segments[0].frame_ids == ["f0"]
    assert segments[1].frame_ids == ["f1"]


def test_gnn_feature_construction_includes_audio_features():
    graphs = build_sparse_graphs(
        [
            {
                "video_id": "v",
                "timestamp": 1,
                "nodes": [
                    {"bbox": [0, 0, 10, 20], "center": [5, 10], "confidence": 0.8}
                ],
                "audio_segments": [
                    {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hello"}
                ],
            }
        ]
    )
    graph = graphs[0]
    assert tuple(graph.x.shape) == (1, 10)
    assert tuple(graph.edge_index.shape) == (2, 0)
