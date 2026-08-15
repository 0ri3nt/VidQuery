from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from vidquery.vidor_improved import (
    HARD_NEGATIVE_EDGE,
    IMPROVED_PAIR_FEATURE_NAMES,
    RANDOM_NEGATIVE_EDGE,
    VIDOR_IMPROVED_CHECKPOINT_SCHEMA,
    VISUAL_FEATURE_DIM,
    ImprovedRelationModel,
    ImprovedVidORRelationPredictor,
    build_improved_relation_graphs,
    configured_relation_checkpoint_status,
    controlled_candidate_pairs,
    improved_relation_checkpoint_status,
    union_box,
)
from vidquery.vidor_relation import (
    VIDOR_OBJECTS,
    VIDOR_PREDICATES,
    RelationModel,
    VidORRelationPredictor,
    build_relation_graphs,
    relation_gnn_status,
    strict_video_split,
)


def _write_annotation_zip(path):
    payload = {
        "version": "VERSION 1.0",
        "video_id": "synthetic-video",
        "video_hash": "hash",
        "video_path": "0000/synthetic-video.mp4",
        "frame_count": 2,
        "fps": 2.0,
        "width": 100,
        "height": 100,
        "subject/objects": [
            {"tid": 0, "category": "adult"},
            {"tid": 1, "category": "chair"},
        ],
        "trajectories": [
            [
                {
                    "tid": 0,
                    "bbox": {"xmin": 10, "ymin": 10, "xmax": 40, "ymax": 80},
                    "generated": 0,
                },
                {
                    "tid": 1,
                    "bbox": {"xmin": 45, "ymin": 40, "xmax": 80, "ymax": 90},
                    "generated": 0,
                },
            ],
            [
                {
                    "tid": 0,
                    "bbox": {"xmin": 12, "ymin": 10, "xmax": 42, "ymax": 80},
                    "generated": 0,
                },
                {
                    "tid": 1,
                    "bbox": {"xmin": 45, "ymin": 40, "xmax": 80, "ymax": 90},
                    "generated": 0,
                },
            ],
        ],
        "relation_instances": [
            {
                "subject_tid": 0,
                "object_tid": 1,
                "predicate": "next_to",
                "begin_fid": 0,
                "end_fid": 2,
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("training/0000/synthetic-video.json", json.dumps(payload))


def test_vidor_versioned_maps_are_complete():
    assert len(VIDOR_OBJECTS) == 80
    assert len(set(VIDOR_OBJECTS)) == 80
    assert len(VIDOR_PREDICATES) == 50
    assert len(set(VIDOR_PREDICATES)) == 50
    assert "speak_to" in VIDOR_PREDICATES
    assert "hold" in VIDOR_PREDICATES


def test_vidor_split_is_video_level_and_deterministic():
    video_ids = [f"video-{index}" for index in range(30)]
    first = strict_video_split(video_ids, seed=2026)
    second = strict_video_split(list(reversed(video_ids)), seed=2026)

    assert first == second
    assert len(first["train"]) == 24
    assert len(first["validation"]) == 3
    assert len(first["test"]) == 3
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["validation"]) & set(first["test"]))


def test_vidor_graph_has_positive_and_background_candidate_edges(tmp_path):
    annotation_zip = tmp_path / "annotations.zip"
    _write_annotation_zip(annotation_zip)
    graphs, statistics = build_relation_graphs(
        annotation_zip,
        [{"member": "training/0000/synthetic-video.json"}],
        Counter({"next_to": 1}),
        max_frames_per_video=1,
        nearest_k=2,
    )

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.x.shape == (2, 8)
    assert graph.edge_attr.shape == (2, 8)
    assert graph.edge_y.shape == (2, 50)
    assert int((graph.edge_y.sum(dim=1) > 0).sum()) == 1
    assert int((graph.edge_y.sum(dim=1) == 0).sum()) == 1
    assert statistics["candidate_edge_recall"] == pytest.approx(1.0)


def test_vidor_relation_gnn_forward_and_batching(tmp_path):
    from torch_geometric.loader import DataLoader

    annotation_zip = tmp_path / "annotations.zip"
    _write_annotation_zip(annotation_zip)
    graphs, _ = build_relation_graphs(
        annotation_zip,
        [{"member": "training/0000/synthetic-video.json"}],
        Counter({"next_to": 1}),
        max_frames_per_video=1,
        nearest_k=2,
    )
    batch = next(iter(DataLoader([graphs[0], graphs[0]], batch_size=2)))
    model: Any = RelationModel(use_graph=True)
    logits = model(batch.x, batch.class_ids, batch.edge_index, batch.edge_attr)

    assert logits.shape == (4, 50)
    assert batch.edge_y.shape == (4, 50)


class _DeterministicVisualEncoder:
    feature_dim = VISUAL_FEATURE_DIM
    encoder_name = "test-deterministic-visual-encoder"

    def encode(self, image, node_boxes, pair_boxes):
        import torch

        assert image.size == (100, 100)
        node = torch.stack(
            [
                torch.full((VISUAL_FEATURE_DIM,), float(index + 1))
                for index in range(len(node_boxes))
            ]
        )
        pair = torch.stack(
            [
                torch.full((VISUAL_FEATURE_DIM,), float(index + 11))
                for index in range(len(pair_boxes))
            ]
        )
        return node, pair


def test_vidor_union_box_and_cached_pair_visual_features(tmp_path):
    from PIL import Image

    assert union_box((0.1, 0.2, 0.4, 0.7), (0.5, 0.1, 0.8, 0.9)) == (
        0.1,
        0.1,
        0.8,
        0.9,
    )
    annotation_zip = tmp_path / "annotations.zip"
    _write_annotation_zip(annotation_zip)
    frame = tmp_path / "frames" / "0000" / "synthetic-video" / "frame_000001.jpg"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), (20, 40, 60)).save(frame)
    graphs, statistics = build_improved_relation_graphs(
        annotation_zip,
        [{"member": "training/0000/synthetic-video.json"}],
        Counter({"next_to": 1}),
        frame_root=tmp_path / "frames",
        visual_encoder=_DeterministicVisualEncoder(),
        max_frames_per_video=1,
    )

    graph = graphs[0]
    assert graph.node_visual.shape == (2, VISUAL_FEATURE_DIM)
    assert graph.union_visual.shape == (2, VISUAL_FEATURE_DIM)
    assert float(graph.node_visual[0, 0]) == 1.0
    assert float(graph.node_visual[1, 0]) == 2.0
    assert float(graph.union_visual[0, 0]) == 11.0
    assert statistics["candidate_edge_recall"] == pytest.approx(1.0)
    assert statistics["visual_cache"]["computed_once_and_cached"] is True


def test_vidor_controlled_hard_and_random_negative_sampling():
    features = [
        [0.05, 0.05, 0.25, 0.8, 0.15, 0.27, 1.0, 1.0],
        [0.22, 0.3, 0.40, 0.65, 0.063, 0.51, 1.0, 0.0],
        [0.42, 0.3, 0.60, 0.65, 0.063, 0.51, 1.0, 0.0],
        [0.80, 0.05, 0.98, 0.25, 0.036, 0.9, 1.0, 0.0],
    ]
    pairs, edge_types, stats = controlled_candidate_pairs(
        features,
        ["adult", "cup", "laptop", "chair"],
        {(0, 1)},
        hard_negative_ratio=2.0,
        random_negative_ratio=1.0,
        seed=7,
    )

    assert pairs[0] == (0, 1)
    assert HARD_NEGATIVE_EDGE in edge_types
    assert RANDOM_NEGATIVE_EDGE in edge_types
    assert stats["positive"] == 1
    assert stats["hard_nearby_negative"] == 2
    assert stats["random_background_negative"] == 1
    assert controlled_candidate_pairs(
        features,
        ["adult", "cup", "laptop", "chair"],
        {(0, 1)},
        seed=7,
    ) == controlled_candidate_pairs(
        features,
        ["adult", "cup", "laptop", "chair"],
        {(0, 1)},
        seed=7,
    )


def test_vidor_improved_mlp_and_gnn_receive_identical_pair_features(tmp_path):
    import torch

    annotation_zip = tmp_path / "annotations.zip"
    _write_annotation_zip(annotation_zip)
    from PIL import Image

    frame = tmp_path / "frames" / "0000" / "synthetic-video" / "frame_000001.jpg"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "black").save(frame)
    graphs, _ = build_improved_relation_graphs(
        annotation_zip,
        [{"member": "training/0000/synthetic-video.json"}],
        Counter({"next_to": 1}),
        frame_root=tmp_path / "frames",
        visual_encoder=_DeterministicVisualEncoder(),
        max_frames_per_video=1,
    )
    graph = graphs[0]
    mlp: Any = ImprovedRelationModel(use_graph=False)
    gnn: Any = ImprovedRelationModel(use_graph=True)
    gnn.class_embedding.load_state_dict(mlp.class_embedding.state_dict())
    arguments = (
        graph.x,
        graph.class_ids,
        graph.node_visual,
        graph.node_motion,
        graph.edge_index,
        graph.edge_attr,
        graph.union_visual,
    )

    assert torch.equal(mlp.pair_inputs(*arguments), gnn.pair_inputs(*arguments))
    assert mlp(*arguments).shape == (2, 50)
    assert gnn(*arguments).shape == (2, 50)


def test_vidor_improved_checkpoint_schema_validation(tmp_path):
    import torch

    checkpoint = tmp_path / "model.pt"
    metadata = tmp_path / "model.metadata.json"
    model: Any = ImprovedRelationModel(use_graph=True)
    payload = {
        "schema_version": VIDOR_IMPROVED_CHECKPOINT_SCHEMA,
        "model_version": "vidor-test-v2",
        "state_dict": model.state_dict(),
        "thresholds": torch.full((50,), 0.5),
        "object_classes": list(VIDOR_OBJECTS),
        "predicates": list(VIDOR_PREDICATES),
        "node_feature_names": [
            "x1",
            "y1",
            "x2",
            "y2",
            "area",
            "aspect_ratio",
            "annotation_confidence",
            "is_person",
        ],
        "motion_feature_names": [
            "center_delta_x",
            "center_delta_y",
            "log_area_ratio_to_previous",
            "previous_frame_visible",
        ],
        "pair_input_feature_names": list(IMPROVED_PAIR_FEATURE_NAMES),
        "visual_feature_dim": VISUAL_FEATURE_DIM,
        "use_graph": True,
    }
    torch.save(payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata.write_text(
        json.dumps(
            {
                "schema_version": VIDOR_IMPROVED_CHECKPOINT_SCHEMA,
                "model_version": "vidor-test-v2",
                "checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert (
        improved_relation_checkpoint_status(checkpoint, metadata, "vidor-test-v2")
        == "available_validated"
    )
    payload["visual_feature_dim"] = 256
    torch.save(payload, checkpoint)
    metadata.write_text(
        json.dumps(
            {
                "schema_version": VIDOR_IMPROVED_CHECKPOINT_SCHEMA,
                "model_version": "vidor-test-v2",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert improved_relation_checkpoint_status(checkpoint, metadata) == (
        "unavailable_feature_schema_mismatch"
    )


def test_configured_vidor_status_selects_pair_visual_schema(tmp_path):
    checkpoint = tmp_path / "missing-v2.pt"
    metadata = tmp_path / "missing-v2.metadata.json"

    assert configured_relation_checkpoint_status(
        checkpoint, metadata, "vidor-relation-pair-visual-gat-v2"
    ) == "unavailable_missing_checkpoint"


def test_configured_trained_pair_visual_checkpoint_is_available():
    checkpoint = Path(
        "data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt"
    )
    metadata = checkpoint.with_suffix(".metadata.json")
    if not checkpoint.is_file() or not metadata.is_file():
        pytest.skip("trained VidOR pair-visual GNN checkpoint is not present")

    assert configured_relation_checkpoint_status(
        checkpoint, metadata, "vidor-relation-pair-visual-gat-v2"
    ) == "available_validated"


def test_trained_vidor_pair_visual_predictor_requires_pixels_and_is_deterministic(
    tmp_path,
):
    from PIL import Image

    from vidquery.domain import BoundingBox, Detection

    checkpoint = Path(
        "data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt"
    )
    metadata = checkpoint.with_suffix(".metadata.json")
    if not checkpoint.is_file() or not metadata.is_file():
        pytest.skip("trained VidOR pair-visual GNN checkpoint is not present")
    detections = [
        Detection(
            detection_id="person-1",
            frame_id="frame-1",
            video_id="video-1",
            timestamp=1.0,
            class_id=0,
            class_label="person",
            confidence=0.9,
            pixel_bbox=BoundingBox(x1=10, y1=10, x2=40, y2=90),
            normalized_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.9),
            centroid=(25.0, 50.0),
            centroid_normalized=(0.25, 0.5),
            track_id="person-track",
        ),
        Detection(
            detection_id="chair-1",
            frame_id="frame-1",
            video_id="video-1",
            timestamp=1.0,
            class_id=56,
            class_label="chair",
            confidence=0.8,
            pixel_bbox=BoundingBox(x1=45, y1=35, x2=80, y2=90),
            normalized_bbox=BoundingBox(x1=0.45, y1=0.35, x2=0.8, y2=0.9),
            centroid=(62.5, 62.5),
            centroid_normalized=(0.625, 0.625),
            track_id="chair-track",
        ),
    ]
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), "navy").save(frame)
    predictor = ImprovedVidORRelationPredictor(
        checkpoint,
        metadata,
        device="cpu",
        expected_model_version="vidor-relation-pair-visual-gat-v2",
        visual_encoder=_DeterministicVisualEncoder(),
    )

    first = predictor.predict_detections(detections, frame_path=frame)
    second = predictor.predict_detections(detections, frame_path=frame)

    assert first == second
    assert all(row["model_version"] == "vidor-relation-pair-visual-gat-v2" for row in first)
    assert all(
        row["visual_provenance"] == "frozen_resnet18_subject_object_union_roi" for row in first
    )
    with pytest.raises(FileNotFoundError, match="requires a readable frame"):
        predictor.predict_detections(detections, frame_path=tmp_path / "missing.jpg")


def test_trained_vidor_checkpoint_loads_and_is_deterministic():
    from vidquery.domain import BoundingBox, Detection

    checkpoint = Path("data/app/models/vidor_relation/vidor-relation-gat-v1.pt")
    metadata = checkpoint.with_suffix(".metadata.json")
    if not checkpoint.is_file() or not metadata.is_file():
        pytest.skip("trained VidOR relationship checkpoint is not present")
    assert (
        relation_gnn_status(checkpoint, metadata, "vidor-relation-gat-v1") == "available_validated"
    )
    detections = [
        Detection(
            detection_id="person-1",
            frame_id="frame-1",
            video_id="video-1",
            timestamp=1.0,
            class_id=0,
            class_label="person",
            confidence=0.9,
            pixel_bbox=BoundingBox(x1=10, y1=10, x2=40, y2=90),
            normalized_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.9),
            centroid=(25.0, 50.0),
            centroid_normalized=(0.25, 0.5),
        ),
        Detection(
            detection_id="chair-1",
            frame_id="frame-1",
            video_id="video-1",
            timestamp=1.0,
            class_id=56,
            class_label="chair",
            confidence=0.8,
            pixel_bbox=BoundingBox(x1=45, y1=35, x2=80, y2=90),
            normalized_bbox=BoundingBox(x1=0.45, y1=0.35, x2=0.8, y2=0.9),
            centroid=(62.5, 62.5),
            centroid_normalized=(0.625, 0.625),
        ),
    ]
    predictor = VidORRelationPredictor(
        checkpoint,
        metadata,
        device="cpu",
        expected_model_version="vidor-relation-gat-v1",
    )

    first = predictor.predict_detections(detections)
    second = predictor.predict_detections(detections)

    assert first == second
    assert all(row["model_version"] == "vidor-relation-gat-v1" for row in first)
