import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from vidquery.ava_ablation import (
    STATIC_SCHEMA,
    StaticActionModel,
    rare_class_graph_sampling_weights,
    validate_static_checkpoint,
)
from vidquery.ava_gnn80 import AVA80_LABELS, AVA80GNNPredictor, ava80_gnn_status
from vidquery.ava_improved import (
    TEMPORAL_CLIP_ENCODER,
    attach_frozen_temporal_clip_features,
    attach_motion_features,
    coverage_balanced_video_split,
)


def _graph(video_id: str, label: int, timestamp: int = 900):
    target = torch.zeros((1, 80))
    target[0, label - 1] = 1
    graph = Data(
        x=torch.tensor([[0.1, 0.1, 0.3, 0.5]], dtype=torch.float32),
        y=target,
        person_mask=torch.tensor([True]),
        loss_mask=torch.tensor([True]),
    )
    graph.video_id = video_id
    graph.timestamp = float(timestamp)
    graph.node_ids = ["ava:person-1"]
    return graph


def test_balanced_split_is_video_disjoint_and_deterministic():
    graphs = [_graph(f"video-{index:02d}", index % 4 + 1) for index in range(12)]
    first = coverage_balanced_video_split(graphs, seed=7)
    second = coverage_balanced_video_split(graphs, seed=7)
    assert first == second
    splits = [set(first[name]) for name in ("train", "validation", "test")]
    assert not splits[0] & splits[1]
    assert not splits[0] & splits[2]
    assert not splits[1] & splits[2]


def test_motion_features_use_stable_annotation_track_without_identity_recognition():
    graphs = [_graph("video", 1, 900), _graph("video", 1, 901), _graph("video", 1, 902)]
    graphs[1].x[0, :4] += torch.tensor([0.1, 0.0, 0.1, 0.0])
    result = attach_motion_features(graphs)
    assert result[0].track_ids == result[1].track_ids == result[2].track_ids
    assert result[1].motion_x[0, 0] == 1
    assert result[1].motion_x[0, 1] == 1
    assert result[1].motion_x[0, 2] > 0
    assert result[1].motion_x[0, 9] == 2


def test_frozen_temporal_clip_embedding_is_cached_and_loaded(tmp_path: Path):
    graphs = [_graph("video", 1, timestamp) for timestamp in (900, 901, 902)]
    for index, graph in enumerate(graphs):
        frame = torch.zeros((1, 512), dtype=torch.float32)
        frame[0, index] = 1.0
        graph.frame_visual_x = frame

    first = attach_frozen_temporal_clip_features(graphs, cache_dir=tmp_path)
    cache = tmp_path / TEMPORAL_CLIP_ENCODER / "video.pt"
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    assert cache.is_file()
    assert first[1].temporal_visual_x.shape == (1, 512)

    payload["embeddings"] = torch.full((3, 512), 0.125)
    torch.save(payload, cache)
    second = attach_frozen_temporal_clip_features(graphs, cache_dir=tmp_path)
    assert torch.all(second[1].temporal_visual_x == 0.125)


def test_rare_class_sampler_weights_rare_positive_graphs_more():
    graphs = [_graph(f"common-{index}", 1) for index in range(9)]
    graphs.append(_graph("rare", 2))

    weights, metadata = rare_class_graph_sampling_weights(graphs, torch)

    assert weights[-1] > weights[0]
    assert metadata["strategy"] == "inverse_sqrt_person_support_graph_sampler"
    assert metadata["supported_training_classes"] == 2


def test_improved_static_checkpoint_is_validated_and_predictable(tmp_path: Path):
    graph = _graph("video", 1)
    graph.class_ids = torch.tensor([0])
    graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    graph.edge_attr = torch.empty((0, 8), dtype=torch.float32)
    graph.handcrafted_x = graph.x.clone()
    model = StaticActionModel(4, use_edges=True, mlp_only=False)
    checkpoint = tmp_path / "improved.pt"
    torch.save(
        {
            "schema_version": STATIC_SCHEMA,
            "model_version": "improved-test",
            "state_dict": model.state_dict(),
            "feature_mean": torch.zeros(4),
            "feature_std": torch.ones(4),
            "thresholds": torch.ones(80),
            "feature_attribute": "handcrafted_x",
            "feature_dimension": 4,
            "use_edges": True,
            "mlp_only": False,
            "labels": list(AVA80_LABELS),
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = tmp_path / "improved.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": STATIC_SCHEMA,
                "model_version": "improved-test",
                "validation_metrics": {},
                "checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    assert ava80_gnn_status(checkpoint, metadata, "improved-test") == "available_validated"
    assert validate_static_checkpoint(
        checkpoint,
        expected_feature_attribute="handcrafted_x",
        expected_feature_dimension=4,
    )["valid"] is True
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        validate_static_checkpoint(checkpoint, expected_feature_dimension=5)
    predictor = AVA80GNNPredictor(
        checkpoint, metadata, expected_model_version="improved-test"
    )
    assert list(predictor.predict_graph(graph)) == ["ava:person-1"]
