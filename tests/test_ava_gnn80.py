from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from vidquery.api import create_app
from vidquery.ava_gnn80 import (
    AVA80_GNN_SCHEMA,
    AVA80_LABELS,
    AVA80GAT,
    COCO_CLASS_NAMES,
    CONTINUOUS_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    AVA80GNNPredictor,
    ava80_gnn_status,
    build_pyg_timestamp_graph,
    masked_multilabel_loss,
    multi_hot_action_target,
    strict_video_split,
)
from vidquery.ava_labels import load_ava_v22_label_map
from vidquery.domain import (
    BoundingBox,
    Detection,
    ProcessingState,
    RelationshipSource,
    VideoRecord,
)
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.segments import ActionEvidence, build_canonical_segments
from vidquery.storage import SQLiteRepository


def _nodes():
    return [
        {
            "node_id": "person_1",
            "class_name": "person",
            "confidence": 0.9,
            "bbox": [5, 5, 55, 95],
            "annotated": True,
            "action_ids": [12, 79],
        },
        {
            "node_id": "chair_1",
            "class_name": "chair",
            "confidence": 0.8,
            "bbox": [45, 35, 90, 95],
            "annotated": False,
            "action_ids": [],
        },
    ]


def _graph():
    return build_pyg_timestamp_graph(
        video_id="video",
        timestamp=902,
        nodes=_nodes(),
        audio_segments=[
            {"start": 901, "end": 903, "speaker": "SPEAKER_00", "text": "hello"}
        ],
        width=100,
        height=100,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_official_ava80_label_map_loads():
    labels = load_ava_v22_label_map()
    assert len(labels) == 80
    assert [label.label_id for label in labels] == list(range(1, 81))
    assert labels[0].official_name == "bend/bow (at the waist)"
    assert labels[16].official_name == "carry/hold (an object)"
    assert labels[79].official_name == "watch (a person)"
    assert {label.label_type for label in labels} == {
        "PERSON_MOVEMENT",
        "OBJECT_MANIPULATION",
        "PERSON_INTERACTION",
    }


def test_ava80_multi_hot_target_has_exact_action_positions():
    target = multi_hot_action_target([1, 12, 80])
    assert len(target) == 80
    assert sum(target) == 3
    assert target[0] == target[11] == target[79] == 1
    with pytest.raises(ValueError, match="invalid AVA"):
        multi_hot_action_target([81])


def test_strict_video_split_has_no_video_leakage():
    split = strict_video_split(["a", "b", "c", "d", "e"], seed=7)
    train, validation, test = map(set, split.values())
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == {"a", "b", "c", "d", "e"}
    assert split == strict_video_split(["e", "d", "c", "b", "a"], seed=7)


def test_graph_construction_batching_and_object_node_masking():
    torch = pytest.importorskip("torch")
    loader_module = pytest.importorskip("torch_geometric.loader")
    graph = _graph()

    assert graph.x.shape == (2, len(CONTINUOUS_FEATURE_NAMES))
    assert graph.edge_attr.shape[1] == len(EDGE_FEATURE_NAMES)
    assert graph.y.shape == (2, 80)
    assert graph.person_mask.tolist() == [True, False]
    assert graph.loss_mask.tolist() == [True, False]
    assert graph.y[0, 11] == 1 and graph.y[0, 78] == 1
    assert not graph.y[1].any()

    batch = next(iter(loader_module.DataLoader([graph, graph], batch_size=2)))
    assert batch.num_graphs == 2
    assert batch.x.shape[0] == 4
    assert int(batch.loss_mask.sum()) == 2

    logits = torch.zeros((2, 80), requires_grad=True)
    targets = graph.y.clone()
    targets[1] = 1
    loss = masked_multilabel_loss(
        logits, targets, graph.loss_mask, torch.nn.BCEWithLogitsLoss()
    )
    expected = torch.nn.BCEWithLogitsLoss()(logits[:1], targets[:1])
    assert torch.allclose(loss, expected)


def test_gat_forward_checkpoint_thresholds_and_deterministic_inference(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    torch.manual_seed(123)
    model = AVA80GAT(
        len(CONTINUOUS_FEATURE_NAMES), len(EDGE_FEATURE_NAMES), len(COCO_CLASS_NAMES)
    )
    model.eval()
    graph = _graph()
    logits = model(graph.x, graph.class_ids, graph.edge_index, graph.edge_attr)
    assert logits.shape == (2, 80)

    checkpoint = tmp_path / "ava80_gat.pt"
    metadata_path = tmp_path / "ava80_gat.metadata.json"
    thresholds = torch.full((80,), 0.5)
    torch.save(
        {
            "schema_version": AVA80_GNN_SCHEMA,
            "model_version": "test-v1",
            "state_dict": model.state_dict(),
            "feature_mean": torch.zeros(len(CONTINUOUS_FEATURE_NAMES)),
            "feature_std": torch.ones(len(CONTINUOUS_FEATURE_NAMES)),
            "labels": list(AVA80_LABELS),
            "label_ids": list(range(1, 81)),
            "thresholds": thresholds,
            "continuous_feature_names": list(CONTINUOUS_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "class_names": list(COCO_CLASS_NAMES),
        },
        checkpoint,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": AVA80_GNN_SCHEMA,
                "model_version": "test-v1",
                "thresholds": {label: 0.5 for label in AVA80_LABELS},
                "validation": {"status": "validated"},
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    assert ava80_gnn_status(checkpoint, metadata_path, "test-v1") == "available_validated"
    predictor = AVA80GNNPredictor(
        checkpoint, metadata_path, expected_model_version="test-v1"
    )
    first = predictor.predict_graph(graph)
    second = predictor.predict_graph(graph)
    assert first == second
    assert set(first) == {"person_1"}


def _detection(identifier, label):
    return Detection(
        detection_id=identifier,
        frame_id="frame-1",
        video_id="video",
        timestamp=1,
        class_id=0,
        class_label=label,
        confidence=0.9,
        pixel_bbox=BoundingBox(x1=0, y1=0, x2=50, y2=90),
        normalized_bbox=BoundingBox(x1=0, y1=0, x2=0.5, y2=0.9),
        centroid=(25, 45),
        centroid_normalized=(0.25, 0.45),
    )


def test_gnn_action_target_resolution_reaches_canonical_neo4j_and_api(
    settings_factory, tmp_path
):
    person = _detection("person-1", "person")
    chair = _detection("chair-1", "chair")
    segments = build_canonical_segments(
        video_id="video",
        duration=5,
        segment_duration=5,
        frames=[],
        detections=[person, chair],
        relationships=[],
        transcripts=[],
        actions=[
            ActionEvidence(
                timestamp=1,
                label="point to",
                source_method=RelationshipSource.GNN_ACTION_MODEL,
                confidence=0.81,
                source_entity_id="person-1",
                target_entity_id="chair-1",
                resolved_predicate="point_to",
            )
        ],
    )
    segment = segments[0]
    assert segment.actions == ["point to"]
    assert segment.relationships[0].source_method is (
        RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER
    )
    assert segment.processing_metadata["gnn_action_instances"][0]["provenance"] == (
        "gnn_action_plus_target_resolver"
    )
    neo4j_row = CanonicalNeo4jIndexer._segment_row(segment)
    assert neo4j_row["relationships"][0]["source_method"] == (
        "gnn_action_plus_target_resolver"
    )

    now = datetime.now(UTC)
    video = VideoRecord(
        video_id="video",
        display_name="unseen.mp4",
        original_filename="unseen.mp4",
        stored_path=str(tmp_path / "unseen.mp4"),
        content_sha256="a" * 64,
        duration=5,
        width=100,
        height=100,
        fps=25,
        has_audio=False,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
    )
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    repository.create_video(video)
    repository.replace_segments("video", [segment])
    payload = TestClient(create_app(settings, repository)).post(
        "/api/search", json={"query": "point to", "video_ids": ["video"]}
    ).json()
    assert payload["results"][0]["actions"] == ["point to"]
    assert payload["results"][0]["action_evidence"][0]["provenance"] == (
        "gnn_action_plus_target_resolver"
    )
    assert "gnn_action_model" in payload["results"][0]["match_reason"]
