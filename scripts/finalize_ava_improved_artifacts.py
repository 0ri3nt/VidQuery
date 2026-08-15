"""Write self-describing, versioned manifests beside every AVA80 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("data/app/models/ava80_improved/ablations"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/app/models/ava80_improved/split_manifest.json"),
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=Path("evaluation/results/ava-expansion-final-coverage.json"),
    )
    parser.add_argument(
        "--graph-metadata",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / args.artifact_dir
    split_path = root / args.split_manifest
    coverage_path = root / args.coverage_report
    graph_metadata_path = root / args.graph_metadata

    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    graph_metadata = json.loads(graph_metadata_path.read_text(encoding="utf-8"))
    data_revision_inputs = {
        split_path.as_posix(): _sha256(split_path),
        coverage_path.as_posix(): _sha256(coverage_path),
        graph_metadata_path.as_posix(): _sha256(graph_metadata_path),
    }
    data_revision = hashlib.sha256(
        json.dumps(data_revision_inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    git_revision = _git_revision(root)

    written: list[str] = []
    for checkpoint_path in sorted(artifact_dir.glob("*.pt")):
        metadata_path = checkpoint_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = checkpoint.get("labels")
        if not isinstance(labels, list) or len(labels) != 80:
            raise ValueError(f"{checkpoint_path} does not contain the AVA80 label map")

        normalization: dict[str, Any]
        if "feature_mean" in checkpoint and "feature_std" in checkpoint:
            normalization = {
                "feature_mean": checkpoint["feature_mean"],
                "feature_std": checkpoint["feature_std"],
            }
        else:
            normalization = {
                "provided_by_static_checkpoint": checkpoint.get("static_checkpoint"),
                "static_checkpoint_sha256": checkpoint.get("static_checkpoint_sha256"),
            }

        bundle = {
            "schema_version": "ava80-checkpoint-bundle-v1",
            "model_version": checkpoint.get("model_version"),
            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "metadata": metadata_path.relative_to(root).as_posix(),
            "metadata_sha256": _sha256(metadata_path),
            "label_map": labels,
            "prediction_thresholds": checkpoint.get("thresholds"),
            "feature_schema": checkpoint.get(
                "feature_schema", graph_metadata.get("feature_schema")
            ),
            "feature_attribute": checkpoint.get("feature_attribute"),
            "normalization": normalization,
            "backbone": graph_metadata.get("feature_schema", {}).get("backbone"),
            "model_config": {
                "architecture": metadata.get("architecture"),
                "loss_mode": metadata.get("loss_mode"),
                "graph_edges_consumed": metadata.get("graph_edges_consumed"),
                "seed": metadata.get("seed"),
                "hardware": metadata.get("hardware"),
            },
            "split_manifest": split_manifest,
            "class_coverage": coverage,
            "metrics": {
                "validation": metadata.get("validation_metrics"),
                "test": metadata.get("test_metrics"),
                "static_test": metadata.get("static_test_metrics"),
                "deterministic_smoothing_test": metadata.get(
                    "deterministic_smoothing_test_metrics"
                ),
            },
            "training_history": metadata.get("history"),
            "training_duration_seconds": metadata.get("training_duration_seconds"),
            "semantic_scope": metadata.get("semantic_scope"),
            "git_revision": git_revision,
            "data_revision": data_revision,
            "data_revision_inputs": data_revision_inputs,
        }
        bundle_path = checkpoint_path.with_suffix(".bundle.json")
        bundle_path.write_text(
            json.dumps(_jsonable(bundle), indent=2) + "\n", encoding="utf-8"
        )
        written.append(bundle_path.relative_to(root).as_posix())

    print(json.dumps({"written": written, "data_revision": data_revision}, indent=2))


if __name__ == "__main__":
    main()
