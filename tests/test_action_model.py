import hashlib
import json

import pytest

from vidquery.action_model import (
    ACTION_LABELS,
    FEATURE_NAMES,
    MODEL_SCHEMA_VERSION,
    PersonActionMLP,
    PersonActionPredictor,
    PersonActionSample,
    action_model_status,
    deterministic_video_split,
    extract_person_feature_rows,
)


def test_action_feature_schema_is_stable_and_person_scoped():
    rows = extract_person_feature_rows(
        [
            {
                "node_id": 1,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [10, 20, 50, 80],
                "center": [30, 50],
            },
            {"node_id": 2, "class_name": "chair", "bbox": [40, 40, 70, 90]},
        ],
        [{"start": 0, "end": 2, "speaker": "SPEAKER_00", "text": "hello"}],
        width=100,
        height=100,
    )

    assert len(rows) == 1
    assert rows[0][0] == 1
    assert len(rows[0][1]) == len(FEATURE_NAMES)
    assert rows[0][1][FEATURE_NAMES.index("context_has_chair")] == 1.0


def test_action_split_is_deterministic_and_video_disjoint():
    samples = [
        PersonActionSample(str(index), 0, 0, (0.0,), (0.0,))
        for index in range(5)
    ]

    split = deterministic_video_split(samples)

    assert split == {
        "train": ["0", "1", "2"],
        "validation": ["3"],
        "test": ["4"],
    }
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_validated_checkpoint_loads_and_runs_inference(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "model.pt"
    metadata_path = tmp_path / "model.metadata.json"
    model = PersonActionMLP(len(FEATURE_NAMES), len(ACTION_LABELS))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "feature_mean": torch.zeros(len(FEATURE_NAMES)),
            "feature_std": torch.ones(len(FEATURE_NAMES)),
            "labels": list(ACTION_LABELS),
            "feature_names": list(FEATURE_NAMES),
            "thresholds": torch.full((len(ACTION_LABELS),), 0.5),
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "labels": list(ACTION_LABELS),
                "feature_names": list(FEATURE_NAMES),
                "checkpoint_sha256": digest,
                "validation": {"status": "validated"},
            }
        ),
        encoding="utf-8",
    )

    assert action_model_status(checkpoint, metadata_path) == "available_validated"
    predictions = PersonActionPredictor(checkpoint, metadata_path).predict(
        [0.0] * len(FEATURE_NAMES)
    )
    assert [item["label"] for item in predictions] == list(ACTION_LABELS)


def test_missing_action_checkpoint_is_never_activated(tmp_path):
    assert (
        action_model_status(tmp_path / "missing.pt", tmp_path / "missing.json")
        == "unavailable_no_valid_checkpoint"
    )
