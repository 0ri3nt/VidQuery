from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vidquery.ava_ablation import train_static_ablation, train_temporal_ablation
from vidquery.ava_improved import MOTION_FEATURE_NAMES, RESNET18_BACKBONE


def _score(metadata: dict[str, Any], split_name: str = "validation_metrics") -> tuple[float, float]:
    metrics = metadata[split_name]
    return (
        float(metrics["macro_f1_supported_classes"]),
        float(metrics["mean_average_precision"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AVA80 visual/motion/temporal ablations")
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/app/models/ava80_improved/split_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/app/models/ava80_improved/ablations"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    payload = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    graphs = list(payload["graphs"])
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    dimensions = {
        name: int(getattr(graphs[0], name).shape[1])
        for name in ("handcrafted_x", "roi_x", "visual_x", "visual_motion_x")
    }
    base_schema = {
        "dataset": payload["schema_version"],
        "backbone": RESNET18_BACKBONE,
        "dimensions": dimensions,
        "motion_features": list(MOTION_FEATURE_NAMES),
    }
    specifications = [
        ("ava80-mlp-v2-balanced", "handcrafted_x", False, True, "bce_positive_weights"),
        ("ava80-gat-v2-no-edges", "handcrafted_x", False, False, "bce_positive_weights"),
        ("ava80-gat-v2-balanced", "handcrafted_x", True, False, "bce_positive_weights"),
        ("ava80-gat-v3-roi", "roi_x", True, False, "bce_positive_weights"),
        ("ava80-gat-v3-visual", "visual_x", True, False, "bce_positive_weights"),
        (
            "ava80-gat-v3-visual-motion-bce",
            "visual_motion_x",
            True,
            False,
            "bce_positive_weights",
        ),
        ("ava80-gat-v3-visual-motion-focal", "visual_motion_x", True, False, "focal"),
        (
            "ava80-gat-v3-visual-motion-balanced",
            "visual_motion_x",
            True,
            False,
            "class_balanced",
        ),
    ]
    results: dict[str, Any] = {}
    for index, (version, feature, edges, mlp, loss) in enumerate(specifications, start=1):
        print(f"[ablation {index}/{len(specifications)}] {version}", flush=True)
        results[version] = train_static_ablation(
            graphs=graphs,
            split=split,
            feature_attribute=feature,
            use_edges=edges,
            mlp_only=mlp,
            loss_mode=loss,
            output_dir=args.output_dir,
            model_version=version,
            feature_schema={**base_schema, "active_feature_attribute": feature},
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
        )
    motion_candidates = {
        name: metadata
        for name, metadata in results.items()
        if "visual-motion" in name
    }
    selected_motion_name = max(motion_candidates, key=lambda name: _score(motion_candidates[name]))
    temporal = train_temporal_ablation(
        graphs=graphs,
        split=split,
        static_checkpoint=args.output_dir / f"{selected_motion_name}.pt",
        output_dir=args.output_dir,
        model_version="ava80-temporal-gat-v1",
        loss_mode=str(motion_candidates[selected_motion_name]["loss_mode"]),
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        device_name=args.device,
    )
    results["ava80-temporal-gat-v1"] = temporal
    candidates = {
        name: metadata
        for name, metadata in results.items()
        if "validation_metrics" in metadata
    }
    best_name = max(candidates, key=lambda name: _score(candidates[name]))
    old_metrics_path = Path("data/app/models/ava80/held_out_metrics.json")
    data_metrics_path = Path("data/app/models/ava80_v2_data/held_out_metrics.json")
    report = {
        "schema_version": "ava80-improved-ablation-report-v1",
        "selection_policy": (
            "highest validation supported-class macro F1, then validation mAP; "
            "test metrics are not used for selection"
        ),
        "selected_model": best_name,
        "selected_validation_score": _score(candidates[best_name]),
        "split_manifest": str(args.split_manifest.as_posix()),
        "graph_cache": str(args.graph_cache.as_posix()),
        "feature_schema": base_schema,
        "old_five_video_metrics": (
            json.loads(old_metrics_path.read_text(encoding="utf-8"))
            if old_metrics_path.is_file() else None
        ),
        "data_only_17_video_metrics": (
            json.loads(data_metrics_path.read_text(encoding="utf-8"))
            if data_metrics_path.is_file() else None
        ),
        "ablations": {
            name: {
                "validation_metrics": metadata["validation_metrics"],
                "test_metrics": metadata["test_metrics"],
                "loss_mode": metadata["loss_mode"],
                "checkpoint": metadata["checkpoint"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                **({
                    "static_test_metrics": metadata["static_test_metrics"],
                    "deterministic_smoothing_test_metrics": metadata[
                        "deterministic_smoothing_test_metrics"
                    ],
                } if "static_test_metrics" in metadata else {}),
            }
            for name, metadata in results.items()
        },
    }
    report_path = args.output_dir / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_model": best_name,
        "selected_validation_score": report["selected_validation_score"],
        "selected_test_metrics": candidates[best_name]["test_metrics"],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
