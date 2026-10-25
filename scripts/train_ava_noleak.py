"""Retrain the AVA action GNN without ground-truth-revealing features.

The active checkpoint (`ava80-gat-v5-scale30-roi-frame`) consumes
``detector_confidence`` and ``annotation_only``: ground-truth boxes that YOLO
missed enter training with flag 1 and confidence 0, which never happens at
inference.  This script:

1. builds ``visual_noleak_x`` (``visual_x`` minus those two columns) on the
   cached graphs of the active model's exact 24/3/3 video split;
2. trains GATv2 variants (learning-rate sweep) and the existing MLP baseline
   with test metrics withheld;
3. selects one variant by the rule declared below, on validation only;
4. evaluates the active checkpoint and the selected models on test under two
   conditions: all supervised persons (ground-truth-assisted boxes) and
   detector-only persons (boxes YOLO actually produced);
5. writes per-class metrics, prediction frequency, and density diagnostics.

Selection rule (fixed before any test evaluation): highest validation
supported-class macro F1 on detector-only persons, tie-break validation mAP.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vidquery.ava_ablation import StaticActionModel, train_static_ablation
from vidquery.eval_registry import assert_not_trained_on
from vidquery.ava_gnn80 import (
    AVA80_LABELS,
    AVA80_NOLEAK_ATTRIBUTE,
    AVA80_NOLEAK_DROPPED,
    AVA80_NOLEAK_SCHEMA,
    classification_metrics,
    decode_ava_predictions,
    drop_leaking_columns,
    leaking_column_indices,
)

ACTIVE = Path("data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.pt")


def _load_graphs(cache: Path, split: dict[str, Any]) -> list[Any]:
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    wanted = {video for name in ("train", "validation", "test") for video in split[name]}
    graphs = [graph for graph in payload["graphs"] if str(graph.video_id) in wanted]
    annotation_column = leaking_column_indices()[1]
    for graph in graphs:
        graph.annotation_only_flag = graph.visual_x[:, annotation_column] > 0.5
        setattr(graph, AVA80_NOLEAK_ATTRIBUTE, drop_leaking_columns(graph.visual_x))
    return graphs


def _model_from_checkpoint(path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    factory: Any = StaticActionModel(
        int(payload["feature_dimension"]),
        use_edges=bool(payload.get("use_edges")),
        mlp_only=bool(payload.get("mlp_only")),
    )
    model = factory.to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _predict(path: Path, graphs: list[Any], device: torch.device) -> dict[str, Any]:
    model, payload = _model_from_checkpoint(path, device)
    attribute = str(payload["feature_attribute"])
    mean = payload["feature_mean"].to(device)
    std = payload["feature_std"].to(device)
    thresholds = payload["thresholds"].numpy()
    probabilities, targets, detector_only = [], [], []
    segment_predicted: dict[tuple[str, int], set[int]] = defaultdict(set)
    segment_truth: dict[tuple[str, int], set[int]] = defaultdict(set)
    started = time.perf_counter()
    with torch.inference_mode():
        for graph in graphs:
            features = getattr(graph, attribute).to(device)
            logits = model(
                (features - mean) / std,
                graph.class_ids.to(device),
                graph.edge_index.to(device),
                graph.edge_attr.to(device),
            )
            probability = torch.sigmoid(logits).cpu()
            mask = graph.loss_mask.bool()
            probabilities.append(probability[mask])
            targets.append(graph.y[mask])
            detector_only.append(~graph.annotation_only_flag[mask])
            # Deployment density: decode every detector person, union per 5 s segment.
            persons = graph.person_mask.bool() & ~graph.annotation_only_flag
            key = (str(graph.video_id), int(float(graph.timestamp) // 5))
            if persons.any():
                decoded = decode_ava_predictions(probability[persons].numpy(), thresholds)
                for row in decoded:
                    segment_predicted[key].update(np.flatnonzero(row).tolist())
            for row in graph.y[mask].numpy():
                segment_truth[key].update(np.flatnonzero(row).tolist())
            segment_predicted.setdefault(key, set())
    elapsed = time.perf_counter() - started
    return {
        "probabilities": torch.cat(probabilities).numpy(),
        "targets": torch.cat(targets).numpy(),
        "detector_only": torch.cat(detector_only).numpy(),
        "thresholds": thresholds,
        "segment_predicted": segment_predicted,
        "segment_truth": segment_truth,
        "seconds": elapsed,
        "graphs": len(graphs),
    }


def _metrics(prediction: dict[str, Any], *, detector_only: bool) -> dict[str, Any]:
    keep = prediction["detector_only"] if detector_only else np.ones_like(
        prediction["detector_only"], dtype=bool
    )
    targets = prediction["targets"][keep]
    probabilities = prediction["probabilities"][keep]
    metrics = classification_metrics(targets, probabilities, prediction["thresholds"])
    decoded = decode_ava_predictions(probabilities, prediction["thresholds"])
    frequency = decoded.sum(axis=0)
    for row, count in zip(metrics["per_class"], frequency, strict=True):
        row["prediction_count"] = int(count)
    metrics["persons"] = int(keep.sum())
    metrics["mean_true_labels_per_person"] = round(float(targets.sum(axis=1).mean()), 4)
    metrics["mean_predicted_labels_per_person"] = round(float(decoded.sum(axis=1).mean()), 4)
    metrics["persons_with_any_prediction"] = round(float(decoded.any(axis=1).mean()), 4)
    return metrics


def _density(prediction: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(prediction["segment_predicted"])
    predicted = [len(prediction["segment_predicted"][key]) for key in keys]
    truth = [len(prediction["segment_truth"].get(key, set())) for key in keys]
    label_share = np.zeros(80)
    truth_share = np.zeros(80)
    for key in keys:
        for index in prediction["segment_predicted"][key]:
            label_share[index] += 1
        for index in prediction["segment_truth"].get(key, set()):
            truth_share[index] += 1
    top = np.argsort(-label_share)[:8]
    return {
        "segments": len(keys),
        "segments_with_any_prediction": round(
            float(np.mean([value > 0 for value in predicted])), 4
        ),
        "mean_predicted_labels_per_segment": round(float(np.mean(predicted)), 4),
        "mean_true_labels_per_segment": round(float(np.mean(truth)), 4),
        "most_predicted": [
            {
                "label": AVA80_LABELS[index],
                "predicted_segment_share": round(float(label_share[index] / len(keys)), 4),
                "true_segment_share": round(float(truth_share[index] / len(keys)), 4),
            }
            for index in top
        ],
    }


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "macro_f1",
        "macro_f1_supported_classes",
        "micro_f1",
        "mean_average_precision",
        "supported_classes",
        "persons",
        "mean_true_labels_per_person",
        "mean_predicted_labels_per_person",
        "persons_with_any_prediction",
    )
    return {key: metrics[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument(
        "--split", type=Path, default=Path("data/app/models/ava80_scaling/30/split_manifest.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/app/models/ava80_noleak"))
    parser.add_argument(
        "--report", type=Path, default=Path("evaluation/results/ava80-noleak-comparison.json")
    )
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    for path in (args.output_dir, args.report):
        if path.exists():
            raise SystemExit(f"{path} exists; choose a new versioned path")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    assert_not_trained_on(split["train"], context="AVA leakage-free training split")
    load_started = time.perf_counter()
    graphs = _load_graphs(args.graph_cache, split)
    load_seconds = time.perf_counter() - load_started
    by_split = {
        name: [graph for graph in graphs if str(graph.video_id) in set(split[name])]
        for name in ("train", "validation", "test")
    }
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    feature_schema = {
        "base_attribute": "visual_x",
        "backbone": "torchvision-resnet18-imagenet1k-v1",
        "dropped_feature_names": list(AVA80_NOLEAK_DROPPED),
        "dropped_feature_columns": leaking_column_indices(),
        "dimensions": int(getattr(by_split["train"][0], AVA80_NOLEAK_ATTRIBUTE).shape[1]),
    }
    extra = {
        "dropped_feature_names": list(AVA80_NOLEAK_DROPPED),
        "source_graph_cache": args.graph_cache.as_posix(),
        "source_split_manifest": args.split.as_posix(),
    }
    variants: list[dict[str, Any]] = []

    def train(name: str, learning_rate: float, mlp_only: bool) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metadata = train_static_ablation(
            graphs=graphs,
            split=split,
            feature_attribute=AVA80_NOLEAK_ATTRIBUTE,
            use_edges=not mlp_only,
            mlp_only=mlp_only,
            loss_mode="bce_positive_weights",
            output_dir=args.output_dir,
            model_version=name,
            feature_schema=feature_schema,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            device_name=str(device),
            evaluate_test=False,
            learning_rate=learning_rate,
            schema_version=AVA80_NOLEAK_SCHEMA,
            extra_payload=extra,
        )
        checkpoint = Path(metadata["checkpoint"])
        validation = _predict(checkpoint, by_split["validation"], device)
        variants.append(
            {
                "model_version": name,
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "learning_rate": learning_rate,
                "architecture": "MLP" if mlp_only else "two-layer GATv2",
                "best_epoch": metadata["checkpoint_selection"]["best_epoch"],
                "epochs_run": len(metadata["history"]),
                "training_seconds": metadata["training_duration_seconds"],
                "peak_gpu_memory_mb": (
                    round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                    if device.type == "cuda"
                    else None
                ),
                "validation_all_persons": _summary(_metrics(validation, detector_only=False)),
                "validation_detector_only": _summary(_metrics(validation, detector_only=True)),
            }
        )

    for learning_rate in args.learning_rates:
        train(f"ava80-gat-v7-noleak-lr{learning_rate:g}", learning_rate, mlp_only=False)
    gat_variants = [item for item in variants if item["architecture"] != "MLP"]

    def score(item: dict[str, Any]) -> tuple[float, float]:
        values = item["validation_detector_only"]
        return float(values["macro_f1_supported_classes"]), float(values["mean_average_precision"])

    selected = max(gat_variants, key=score)
    train(f"ava80-mlp-v7-noleak-lr{selected['learning_rate']:g}", selected["learning_rate"], True)

    # Test evaluation happens only now, after the selection rule was applied.
    test_report: dict[str, Any] = {}
    for label, path in (
        ("active_v5_with_leaking_features", ACTIVE),
        ("selected_noleak_gat", Path(selected["checkpoint"])),
        ("noleak_mlp_baseline", Path(variants[-1]["checkpoint"])),
    ):
        prediction = _predict(path, by_split["test"], device)
        all_persons = _metrics(prediction, detector_only=False)
        detector = _metrics(prediction, detector_only=True)
        test_report[label] = {
            "checkpoint": path.as_posix(),
            "all_persons": _summary(all_persons),
            "detector_only": _summary(detector),
            "per_class_detector_only": detector["per_class"],
            "segment_density_detector_persons": _density(prediction),
            "inference_seconds": round(prediction["seconds"], 3),
            "graphs": prediction["graphs"],
        }
    report = {
        "schema_version": "ava80-noleak-comparison-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "highest validation supported-class macro F1 on detector-only persons; "
            "tie-break validation mAP; test withheld until selection"
        ),
        "split_manifest": args.split.as_posix(),
        "graph_counts": {name: len(values) for name, values in by_split.items()},
        "annotation_only_share": {
            name: round(
                float(
                    torch.cat([g.annotation_only_flag[g.loss_mask.bool()] for g in values])
                    .float()
                    .mean()
                ),
                4,
            )
            for name, values in by_split.items()
        },
        "graph_cache_load_seconds": round(load_seconds, 1),
        "device": str(device),
        "variants": variants,
        "selected": selected["model_version"],
        "test": test_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "test"}, indent=2)[:4000])
    for label, values in test_report.items():
        print(label, "all", values["all_persons"], "\n   detector", values["detector_only"])


if __name__ == "__main__":
    main()
