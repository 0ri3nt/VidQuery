from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from vidquery.ava_ablation import train_static_ablation
from vidquery.ava_improved import RESNET18_BACKBONE
from vidquery.ava_scaling import (
    build_fixed_holdout_scaling_splits,
    graph_corpus_statistics,
    ordered_video_ids,
)


def _score(metadata: dict[str, Any]) -> tuple[float, float]:
    metrics = metadata["validation_metrics"]
    return (
        float(metrics["macro_f1_supported_classes"]),
        float(metrics["mean_average_precision"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed-holdout AVA GATv2 data scaling runs")
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument(
        "--baseline-split",
        type=Path,
        default=Path("data/app/models/ava80_improved/split_manifest.json"),
    )
    parser.add_argument(
        "--baseline-metadata",
        type=Path,
        default=Path(
            "data/app/models/ava80_improved/ablations/ava80-gat-v3-roi.metadata.json"
        ),
    )
    parser.add_argument(
        "--expansion-state",
        type=Path,
        default=Path("data/app/features/ava80/expansion_state.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/app/models/ava80_scaling")
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[17, 30, 50])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    payload = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    graphs = list(payload["graphs"])
    baseline_split = json.loads(args.baseline_split.read_text(encoding="utf-8"))
    expansion_state = (
        json.loads(args.expansion_state.read_text(encoding="utf-8"))
        if args.expansion_state.is_file()
        else {}
    )
    expansion_order = [
        str(video_id)
        for batch in expansion_state.get("batches", [])
        for video_id in batch.get("requested_video_ids", [])
    ]
    manifests = build_fixed_holdout_scaling_splits(
        graphs,
        baseline_split,
        args.sizes,
        expansion_order=expansion_order,
    )
    if 17 not in manifests:
        raise ValueError("the historical 17-video baseline split is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical = json.loads(args.baseline_metadata.read_text(encoding="utf-8"))
    runs: dict[str, Any] = {}
    for corpus_size, split in manifests.items():
        run_dir = args.output_dir / str(corpus_size)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "ava80-fixed-holdout-scaling-split-v1",
            "seed": args.seed,
            "corpus_size": corpus_size,
            "selection_rule": (
                "frozen historical validation/test videos; class-aware expansion videos "
                "are added only to training"
            ),
            **split,
        }
        (run_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        statistics = graph_corpus_statistics(graphs, split)
        if corpus_size == 17:
            metadata = historical
            reused = True
        else:
            version = f"ava80-gat-v4-scale{corpus_size}-roi"
            metadata_path = run_dir / f"{version}.metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                reused = True
            else:
                metadata = train_static_ablation(
                    graphs=graphs,
                    split=split,
                    feature_attribute="roi_x",
                    use_edges=True,
                    mlp_only=False,
                    loss_mode="bce_positive_weights",
                    output_dir=run_dir,
                    model_version=version,
                    feature_schema={
                        "dataset": payload["schema_version"],
                        "backbone": RESNET18_BACKBONE,
                        "active_feature_attribute": "roi_x",
                        "feature_dimension": int(graphs[0].roi_x.shape[1]),
                    },
                    seed=args.seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    device_name=args.device,
                )
                reused = False
        runs[str(corpus_size)] = {
            "corpus_statistics": statistics,
            "model_version": metadata["model_version"],
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "validation_metrics": metadata["validation_metrics"],
            "test_metrics": metadata["test_metrics"],
            "training_duration_seconds": metadata["training_duration_seconds"],
            "inference_latency_mean_milliseconds_per_graph": metadata[
                "inference_latency_mean_milliseconds_per_graph"
            ],
            "reused_existing_artifact": reused,
        }

    selected_size = max(runs, key=lambda value: _score(runs[value]))
    report = {
        "schema_version": "ava80-data-scaling-report-v1",
        "selection_policy": (
            "highest validation supported-class macro F1, then validation mAP; "
            "held-out test metrics are never used for selection"
        ),
        "available_video_count": len(ordered_video_ids(graphs)),
        "requested_corpus_sizes": args.sizes,
        "completed_corpus_sizes": sorted(int(value) for value in runs),
        "unavailable_corpus_sizes": sorted(set(args.sizes).difference(map(int, runs))),
        "selected_corpus_size": int(selected_size),
        "selected_model": runs[selected_size]["model_version"],
        "runs": runs,
    }
    report_path = args.output_dir / "scaling_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": report_path.as_posix(), **report}, indent=2))


if __name__ == "__main__":
    main()
