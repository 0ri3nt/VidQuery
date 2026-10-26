from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.compare_moment_runs import compare
from scripts.run_with_profile import main as run_with_profile
from scripts.run_with_profile import resolve_profile, validate_checkpoints
from vidquery.pipeline import checkpoint_digest

PROFILES = {
    "common": {"WHISPER_TEMPERATURE_FALLBACK": "false"},
    "models": {"a": {"GNN_MODEL_VERSION": "x"}, "b": {"RELATION_GNN_MODEL_VERSION": "y"}},
    "profiles": {"ab": ["a", "b"], "broken": ["a", "missing"]},
}


def test_profiles_merge_common_and_model_groups():
    assert resolve_profile(PROFILES, "ab") == {
        "WHISPER_TEMPERATURE_FALLBACK": "false",
        "GNN_MODEL_VERSION": "x",
        "RELATION_GNN_MODEL_VERSION": "y",
    }
    with pytest.raises(SystemExit, match="unknown profile"):
        resolve_profile(PROFILES, "nope")
    with pytest.raises(SystemExit, match="unknown model group"):
        resolve_profile(PROFILES, "broken")


def test_unusable_checkpoints_stop_the_run(tmp_path):
    overrides = {
        "GNN_CHECKPOINT": str(tmp_path / "missing.pt"),
        "GNN_METADATA": str(tmp_path / "missing.json"),
        "GNN_MODEL_VERSION": "x",
    }
    with pytest.raises(SystemExit, match="not usable"):
        validate_checkpoints(overrides)


def test_command_runs_with_overrides_and_records_them(tmp_path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"common": {"VQ_TEST_FLAG": "on"}, "models": {},
                                    "profiles": {"plain": []}}), encoding="utf-8")
    out, record = tmp_path / "out.txt", tmp_path / "record.json"
    code = run_with_profile([
        "--profiles", str(profiles), "--profile", "plain", "--set", f"VQ_OUT={out}",
        "--record", str(record), "--", sys.executable, "-c",
        "import os, pathlib; "
        "pathlib.Path(os.environ['VQ_OUT']).write_text(os.environ['VQ_TEST_FLAG'])",
    ])
    assert code == 0
    assert out.read_text() == "on"
    assert json.loads(record.read_text())["overrides"]["VQ_TEST_FLAG"] == "on"


def test_checkpoint_digest_is_cached_and_missing_is_none(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    first = checkpoint_digest(path)
    assert first == checkpoint_digest(path)
    assert len(first or "") == 64
    assert checkpoint_digest(tmp_path / "absent.pt") is None


def _result(p1_values, rejected, split="test"):
    rows = []
    for index, value in enumerate(p1_values):
        rows.append({"id": f"p{index}", "kind": "positive", "negative": False,
                     "max_confidence": 0.5 + value / 10,
                     "metrics": {"precision_at_1": value, "r1_iou0.5": value,
                                 "r1_iou0.7": 0.0, "r5_iou0.5": value}})
    for index, value in enumerate(rejected):
        rows.append({"id": f"n{index}", "kind": "in_domain_negative", "negative": True,
                     "max_confidence": 0.1, "metrics": {"negative_rejected": value}})
    summary = {"precision_at_1": sum(p1_values) / len(p1_values), "r1_iou0.5": 0, "r1_iou0.7": 0,
               "r5_iou0.5": 0, "mean_iou": 0, "positive_answered_rate": 1, "coverage": 1,
               "aurc": 0, "rejection_precision": 0, "rejection_recall": 0, "rejection_f1": 0,
               "answerability_auroc": 0,
               "negative_rejection_rate_by_kind": {"in_domain_negative": 0.5}}
    return {"summary": {"k": summary}, "rows": {"k": rows}, "plans": [{"id": "p0"}],
            "run_metadata": {"split": split}}


MANIFEST = {"splits": {"test": {"queries": [
    {"id": "p0", "kind": "positive", "video": "v1", "answerable": True},
    {"id": "p1", "kind": "positive", "video": "v2", "answerable": True},
    {"id": "n0", "kind": "in_domain_negative", "video": "v1", "answerable": False},
    {"id": "n1", "kind": "in_domain_negative", "video": "v2", "answerable": False},
]}}}


def test_compare_moment_runs_pairs_by_position_and_reports_per_video():
    results = {"a": _result([0.0, 1.0], [False, False]), "b": _result([1.0, 1.0], [True, False])}
    report = compare(results, MANIFEST, key="k", pairs=[("a", "b")], samples=50, seed=1)
    pair = report["paired_video_bootstrap"]["b - a"]
    assert pair["precision_at_1"]["difference"] == 0.5
    assert pair["precision_at_1"]["queries_better"] == 1
    assert pair["rejected_in_domain_negative"]["difference"] == 0.5
    assert report["per_video_precision_at_1"]["v1"] == {"positives": 1, "a": 0.0, "b": 1.0}
    assert pair["videos_p1_better"] == 1
    assert report["metrics"]["a"]["auroc_vs_in_domain_negative"] == 1.0
    reordered = _result([0.0, 1.0], [False, False])
    reordered["rows"]["k"].reverse()
    with pytest.raises(ValueError, match="in order"):
        compare({"a": results["a"], "c": reordered}, MANIFEST, key="k", pairs=[], samples=5,
                seed=1)


def test_profiles_file_in_repository_names_existing_model_groups():
    payload = json.loads(Path("evaluation/configs/model_profiles_2026-10-26.json").read_text())
    for name in payload["profiles"]:
        overrides = resolve_profile(payload, name)
        assert overrides["WHISPER_TEMPERATURE_FALLBACK"] == "false"
        assert {"GNN_MODEL_VERSION", "RELATION_GNN_MODEL_VERSION"} <= set(overrides)
