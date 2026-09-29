from __future__ import annotations

import hashlib
import json

import pytest

from vidquery.action_model import ACTION_LABELS
from vidquery.gnn_action_model import (
    GNN_MODEL_SCHEMA_VERSION,
    NODE_FEATURE_NAMES,
    GraphSAGEActionPredictor,
    GraphSAGEPersonActionModel,
    build_frame_graph,
    gnn_action_model_status,
)


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_graph_builder_creates_spatial_message_passing_edges_and_person_targets():
    graph = build_frame_graph(
        video_id="v",
        timestamp=1,
        nodes=[
            {
                "node_id": 7,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [10, 10, 50, 90],
                "center": [30, 50],
                "matched_annotation": {"action_ids": [12, 79]},
            },
            {
                "node_id": 8,
                "class_name": "laptop",
                "confidence": 0.8,
                "bbox": [45, 35, 75, 65],
                "center": [60, 50],
            },
        ],
        audio_segments=[],
        width=100,
        height=100,
        require_targets=True,
    )

    assert graph is not None
    assert graph.edge_pairs == ((0, 1), (1, 0))
    assert graph.person_node_ids == (7,)
    assert graph.targets[0][1] == 1.0
    assert graph.targets[0][4] == 1.0


def test_graphsage_checkpoint_loads_and_runs_inference(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "graphsage.pt"
    metadata_path = tmp_path / "graphsage.metadata.json"
    model = GraphSAGEPersonActionModel(len(NODE_FEATURE_NAMES), len(ACTION_LABELS))
    torch.save(
        {
            "schema_version": GNN_MODEL_SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "feature_mean": torch.zeros(len(NODE_FEATURE_NAMES)),
            "feature_std": torch.ones(len(NODE_FEATURE_NAMES)),
            "labels": list(ACTION_LABELS),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "thresholds": torch.zeros(len(ACTION_LABELS)),
        },
        checkpoint,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": GNN_MODEL_SCHEMA_VERSION,
                "node_feature_names": list(NODE_FEATURE_NAMES),
                "labels": list(ACTION_LABELS),
                "validation": {"status": "validated"},
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    assert gnn_action_model_status(checkpoint, metadata_path) == "available_validated"
    predictor = GraphSAGEActionPredictor(checkpoint, metadata_path)
    predictions = predictor.predict_graph(
        video_id="v",
        timestamp=0,
        nodes=[
            {
                "node_id": 3,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [5, 5, 50, 95],
                "center": [27.5, 50],
            },
            {
                "node_id": 4,
                "class_name": "chair",
                "confidence": 0.8,
                "bbox": [45, 30, 85, 95],
                "center": [65, 62.5],
            },
        ],
        audio_segments=[],
        width=100,
        height=100,
    )

    assert set(predictions) == {3}
    assert {item["label"] for item in predictions[3]} == set(ACTION_LABELS)
