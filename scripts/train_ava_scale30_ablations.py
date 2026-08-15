from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vidquery.ava_ablation import train_static_ablation, train_temporal_ablation
from vidquery.ava_improved import MOTION_FEATURE_NAMES, RESNET18_BACKBONE


def _score(metadata: dict[str, Any]) -> tuple[float, float]:
    metrics = metadata["validation_metrics"]
    return (
        float(metrics["macro_f1_supported_classes"]),
        float(metrics["mean_average_precision"]),
    )


def _load_or_train(
    *,
    metadata_path: Path,
    graphs: list[Any],
    split: dict[str, Any],
    feature_attribute: str,
    loss_mode: str,
    output_dir: Path,
    model_version: str,
    feature_schema: dict[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return train_static_ablation(
        graphs=graphs,
        split=split,
        feature_attribute=feature_attribute,
        use_edges=True,
        mlp_only=False,
        loss_mode=loss_mode,
        output_dir=output_dir,
        model_version=model_version,
        feature_schema={
            **feature_schema,
            "active_feature_attribute": feature_attribute,
        },
        seed=seed,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        device_name=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled visual, motion, loss, and temporal AVA30 ablations"
    )
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/app/models/ava80_scaling/30/split_manifest.json"),
    )
    parser.add_argument(
        "--roi-metadata",
        type=Path,
        default=Path(
            "data/app/models/ava80_scaling/30/"
            "ava80-gat-v4-scale30-roi.metadata.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/app/models/ava80_scaling/30/ablations"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    payload = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    graphs = list(payload["graphs"])
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    selected_video_ids = set(split["train"] + split["validation"] + split["test"])
    graphs = [graph for graph in graphs if str(graph.video_id) in selected_video_ids]
    if {str(graph.video_id) for graph in graphs} != selected_video_ids:
        raise ValueError("one or more scale-30 split videos are absent from the graph cache")

    dimensions = {
        name: int(getattr(graphs[0], name).shape[1])
        for name in ("handcrafted_x", "roi_x", "visual_x", "visual_motion_x")
    }
    feature_schema = {
        "dataset": payload["schema_version"],
        "backbone": RESNET18_BACKBONE,
        "dimensions": dimensions,
        "motion_features": list(MOTION_FEATURE_NAMES),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {
        "ava80-gat-v4-scale30-roi": json.loads(
            args.roi_metadata.read_text(encoding="utf-8")
        )
    }
    specifications = (
        ("ava80-gat-v5-scale30-handcrafted", "handcrafted_x", "bce_positive_weights"),
        ("ava80-gat-v5-scale30-roi-frame", "visual_x", "bce_positive_weights"),
        (
            "ava80-gat-v5-scale30-roi-frame-motion-bce",
            "visual_motion_x",
            "bce_positive_weights",
        ),
        ("ava80-gat-v5-scale30-roi-frame-motion-focal", "visual_motion_x", "focal"),
        (
            "ava80-gat-v5-scale30-roi-frame-motion-balanced",
            "visual_motion_x",
            "class_balanced",
        ),
    )
    for index, (version, feature, loss) in enumerate(specifications, start=1):
        print(f"[scale30 ablation {index}/{len(specifications)}] {version}", flush=True)
        results[version] = _load_or_train(
            metadata_path=args.output_dir / f"{version}.metadata.json",
            graphs=graphs,
            split=split,
            feature_attribute=feature,
            loss_mode=loss,
            output_dir=args.output_dir,
            model_version=version,
            feature_schema=feature_schema,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
        )

    motion_names = [name for name in results if "frame-motion" in name]
    static_for_temporal = max(motion_names, key=lambda name: _score(results[name]))
    temporal_version = "ava80-temporal-gat-v2-scale30"
    temporal_metadata = args.output_dir / f"{temporal_version}.metadata.json"
    if temporal_metadata.is_file():
        temporal = json.loads(temporal_metadata.read_text(encoding="utf-8"))
    else:
        temporal = train_temporal_ablation(
            graphs=graphs,
            split=split,
            static_checkpoint=args.output_dir / f"{static_for_temporal}.pt",
            output_dir=args.output_dir,
            model_version=temporal_version,
            loss_mode=str(results[static_for_temporal]["loss_mode"]),
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
        )

    static_winner = max(results, key=lambda name: _score(results[name]))
    all_candidates = {**results, temporal_version: temporal}
    validation_winner = max(all_candidates, key=lambda name: _score(all_candidates[name]))
    report = {
        "schema_version": "ava80-scale30-ablation-report-v1",
        "selection_policy": (
            "highest validation supported-class macro F1, then validation mAP; "
            "test metrics are reported only after selection"
        ),
        "split_manifest": args.split_manifest.as_posix(),
        "graph_cache": args.graph_cache.as_posix(),
        "feature_schema": feature_schema,
        "static_deployment_winner": static_winner,
        "overall_validation_winner": validation_winner,
        "temporal_static_source": static_for_temporal,
        "ablations": {
            name: {
                "feature_attribute": metadata.get("feature_attribute"),
                "loss_mode": metadata["loss_mode"],
                "validation_metrics": metadata["validation_metrics"],
                "test_metrics": metadata["test_metrics"],
                "training_duration_seconds": metadata["training_duration_seconds"],
                "checkpoint": metadata["checkpoint"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                **(
                    {
                        "static_test_metrics": metadata["static_test_metrics"],
                        "deterministic_smoothing_test_metrics": metadata[
                            "deterministic_smoothing_test_metrics"
                        ],
                    }
                    if "static_test_metrics" in metadata
                    else {}
                ),
            }
            for name, metadata in all_candidates.items()
        },
    }
    report_path = args.output_dir / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": report_path.as_posix(),
                "static_deployment_winner": static_winner,
                "static_validation_score": _score(results[static_winner]),
                "overall_validation_winner": validation_winner,
                "overall_validation_score": _score(all_candidates[validation_winner]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
