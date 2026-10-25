from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_ava_shards import _graph

from scripts.build_ava_scaling_split import build_split
from vidquery.ava_compare import _resampling_weights, compare_predictions, weighted_metrics
from vidquery.ava_expansion import build_evaluation_shards
from vidquery.ava_gnn80 import classification_metrics, decode_ava_predictions
from vidquery.ava_shards import ShardGraphDataset, write_shard
from vidquery.eval_registry import (
    EvaluationLeakageError,
    assert_not_selected_on,
    eval_only_video_ids,
)

CALIBRATION = sorted(eval_only_video_ids(groups=["ava_holdout_v1_calibration"]))
V1_TEST = sorted(eval_only_video_ids(groups=["ava_holdout_v1_test"]))
V2_TEST = sorted(eval_only_video_ids(groups=["ava_holdout_v2_test"]))


def _random_predictions(seed: int, persons: int = 60):
    rng = np.random.default_rng(seed)
    truth = rng.random((persons, 80)) < 0.06
    scores = np.clip(truth * 0.4 + rng.random((persons, 80)) * 0.7, 0, 1).astype(np.float32)
    thresholds = np.full(80, 0.5, dtype=np.float32)
    return truth, scores, thresholds


def test_weighted_metrics_equal_classification_metrics_on_repeated_rows():
    truth, scores, thresholds = _random_predictions(1)
    predicted = decode_ava_predictions(scores, thresholds)
    weights = np.random.default_rng(2).integers(0, 4, len(truth))
    repeated = np.repeat(np.arange(len(truth)), weights)
    exact = classification_metrics(truth[repeated].astype(np.float32), scores[repeated], thresholds)
    weighted = weighted_metrics(truth, predicted, scores, weights)
    for key in ("macro_f1_supported_classes", "micro_f1", "mean_average_precision"):
        assert weighted[key] == pytest.approx(exact[key], abs=1e-5)


def test_stratified_resampling_keeps_each_video_size():
    groups = np.asarray(["a@1", "a@1", "a@2", "b@1", "b@2", "b@3"])
    strata = np.asarray(["a", "a", "a", "b", "b", "b"])
    weights = _resampling_weights(groups, strata, np.random.default_rng(0))
    # 2 keyframes drawn in video a, 3 in video b; each keyframe weight hits all its persons.
    unique_draws_a = weights[strata == "a"]
    assert weights[0] == weights[1]
    assert sum(weights[[0, 2]]) == 2 and sum(weights[3:]) == 3
    assert unique_draws_a.min() >= 0


def test_compare_predictions_requires_identical_test_persons_and_reports_pairs():
    truth, scores, thresholds = _random_predictions(3)
    base = {
        "targets": truth,
        "probabilities": scores,
        "predicted": decode_ava_predictions(scores, thresholds),
        "thresholds": thresholds,
        "detector_only": np.ones(len(truth), dtype=bool),
        "videos": np.asarray(["v1"] * 30 + ["v2"] * 30),
        "keyframes": np.asarray([f"v{1 + index // 30}@{index // 3}" for index in range(60)]),
    }
    better_scores = np.clip(scores + truth * 0.3, 0, 1).astype(np.float32)
    better = {**base, "probabilities": better_scores,
              "predicted": decode_ava_predictions(better_scores, thresholds)}
    report = compare_predictions({"a": base, "b": better}, pairs=[("a", "b")], samples=50)
    entry = report["paired_bootstrap"]["keyframe_within_video"]["b - a"]
    assert entry["mean_average_precision"]["difference"] > 0
    assert set(report["per_video"]) == {"v1", "v2"}
    shuffled = {**base, "keyframes": base["keyframes"][::-1]}
    with pytest.raises(ValueError, match="same test persons"):
        compare_predictions({"a": base, "c": shuffled}, pairs=[], samples=5)


def test_scaling_split_freezes_validation_and_test_and_refuses_leakage():
    base = {"seed": 2026, "train": ["t1"], "validation": CALIBRATION, "test": V1_TEST}
    shards = {"t1", "t2", "t3", *CALIBRATION, *V1_TEST}
    split = build_split(base, shards, extra_test=V2_TEST, eval_shards=set(V2_TEST))
    assert split["validation"] == CALIBRATION
    assert split["test"] == [*V1_TEST, *V2_TEST]
    assert split["train"] == ["t1", "t2", "t3"]
    with pytest.raises(EvaluationLeakageError, match="without a verified shard"):
        build_split(base, shards, extra_test=V2_TEST, eval_shards=set())
    with pytest.raises(EvaluationLeakageError, match="must not have training shards"):
        build_split(base, shards | {V2_TEST[0]}, extra_test=V2_TEST, eval_shards=set(V2_TEST))
    leaking = {**base, "validation": [*CALIBRATION, V2_TEST[0]]}
    with pytest.raises(EvaluationLeakageError):
        build_split(leaking, shards | {V2_TEST[0]}, extra_test=[], eval_shards=set())


def test_selection_guard_allows_calibration_but_not_test_only_videos():
    assert_not_selected_on(CALIBRATION)
    with pytest.raises(EvaluationLeakageError, match="test-only"):
        assert_not_selected_on([V1_TEST[0]])


def test_evaluation_shards_refuse_unregistered_videos_and_the_training_directory(tmp_path):
    with pytest.raises(EvaluationLeakageError, match="not registered"):
        build_evaluation_shards(
            ["someTrainVideo"], shard_dir=tmp_path / "eval", training_shard_dir=tmp_path / "train"
        )
    with pytest.raises(EvaluationLeakageError, match="training shard directory"):
        build_evaluation_shards(
            V2_TEST, shard_dir=tmp_path / "train", training_shard_dir=tmp_path / "train"
        )


def test_dataset_reads_extra_shard_directories(tmp_path):
    train_dir, eval_dir = Path(tmp_path / "train"), Path(tmp_path / "eval")
    write_shard([_graph("a", 1.0)], train_dir, video_id="a", provenance={})
    write_shard([_graph("b", 1.0), _graph("b", 2.0)], eval_dir, video_id="b", provenance={})
    dataset = ShardGraphDataset(train_dir, ["a", "b"], extra_shard_dirs=[eval_dir])
    assert len(dataset) == 3
    assert dataset.video_dirs == {"a": train_dir, "b": eval_dir}
    assert {graph.video_id for graph in dataset.iter_graphs()} == {"a", "b"}
