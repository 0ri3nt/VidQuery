from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from vidquery.ava_ablation import (
    evaluate_static_checkpoint,
    train_static_ablation,
    validate_static_checkpoint,
)
from vidquery.ava_improved import (
    RESNET18_BACKBONE,
    TEMPORAL_CLIP_ENCODER,
    attach_frozen_temporal_clip_features,
    materialize_feature_variants,
    save_improved_graph_cache,
)
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


def _summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": metadata["model_version"],
        "checkpoint": metadata["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "feature_attribute": metadata["feature_attribute"],
        "sampling": metadata.get("sampling"),
        "validation_metrics": metadata["validation_metrics"],
        "test_metrics": metadata.get("test_metrics"),
        "test_evaluation_status": metadata.get("test_evaluation_status"),
        "training_duration_seconds": metadata["training_duration_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final controlled ROI/frame versus temporal AVA experiment"
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument(
        "--baseline-split",
        type=Path,
        default=Path("data/app/models/ava80_scaling/30/split_manifest.json"),
    )
    parser.add_argument(
        "--expansion-state",
        type=Path,
        default=Path("data/app/features/ava80/expansion_state.json"),
    )
    parser.add_argument(
        "--old-metadata",
        type=Path,
        default=Path(
            "data/app/models/ava80_scaling/30/ablations/"
            "ava80-gat-v5-scale30-roi-frame.metadata.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/app/models/ava80_final")
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maximum-videos", type=int, default=None)
    parser.add_argument("--skip-graph-cache", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    source_graphs = list(payload["graphs"])
    dataset_schema = str(payload["schema_version"])
    available_video_ids = ordered_video_ids(source_graphs)
    if args.maximum_videos is not None:
        if args.maximum_videos < 30:
            raise ValueError("maximum videos cannot be below the 30-video baseline")
        expansion_state = json.loads(args.expansion_state.read_text(encoding="utf-8"))
        expansion_order = [
            str(video_id)
            for batch in expansion_state.get("batches", [])
            for video_id in batch.get("requested_video_ids", [])
        ]
        ordered = []
        for video_id in [*expansion_order, *available_video_ids]:
            if video_id in available_video_ids and video_id not in ordered:
                ordered.append(video_id)
        keep = set(ordered[: args.maximum_videos])
        source_graphs = [
            graph for graph in source_graphs if str(graph.video_id) in keep
        ]
        available_video_ids = ordered_video_ids(source_graphs)
    if len(available_video_ids) < 30:
        raise ValueError("final AVA experiment requires at least the existing 30 videos")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal = attach_frozen_temporal_clip_features(
        source_graphs,
        cache_dir=args.output_dir / "temporal_embedding_cache",
    )
    del source_graphs, payload
    gc.collect()
    graphs = materialize_feature_variants(temporal)
    del temporal
    gc.collect()
    graph_cache = args.output_dir / "ava80_roi_frame_temporal_graphs.pt"
    graph_metadata = (
        {"cache_skipped": True, "reason": "in-memory training avoids a redundant copy"}
        if args.skip_graph_cache
        else save_improved_graph_cache(graphs, graph_cache)
    )

    baseline_split = json.loads(args.baseline_split.read_text(encoding="utf-8"))
    expansion_state = json.loads(args.expansion_state.read_text(encoding="utf-8"))
    expansion_order = [
        str(video_id)
        for batch in expansion_state.get("batches", [])
        for video_id in batch.get("requested_video_ids", [])
    ]
    corpus_size = len(available_video_ids)
    manifests = build_fixed_holdout_scaling_splits(
        graphs,
        baseline_split,
        [corpus_size],
        expansion_order=expansion_order,
    )
    if corpus_size not in manifests:
        raise ValueError("failed to create the fixed-holdout final AVA split")
    split = manifests[corpus_size]
    split_manifest = {
        "schema_version": "ava80-final-fixed-holdout-split-v1",
        "seed": args.seed,
        "corpus_size": corpus_size,
        "selection_rule": (
            "the scale-30 validation/test videos stay frozen; expansion videos are "
            "added to training only"
        ),
        **split,
    }
    split_path = args.output_dir / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    statistics = graph_corpus_statistics(graphs, split)
    dimensions = {
        "visual_x": int(graphs[0].visual_x.shape[1]),
        "visual_temporal_x": int(graphs[0].visual_temporal_x.shape[1]),
    }
    shared_schema = {
        "dataset": dataset_schema,
        "backbone": RESNET18_BACKBONE,
        "temporal_clip_encoder": TEMPORAL_CLIP_ENCODER,
        "dimensions": dimensions,
    }
    specifications = (
        (
            f"ava80-gat-v6-scale{corpus_size}-roi-frame-rare",
            "visual_x",
        ),
        (
            f"ava80-gat-v6-scale{corpus_size}-roi-frame-temporal-rare",
            "visual_temporal_x",
        ),
    )
    candidates: dict[str, dict[str, Any]] = {}
    for version, feature_attribute in specifications:
        metadata = train_static_ablation(
            graphs=graphs,
            split=split,
            feature_attribute=feature_attribute,
            use_edges=True,
            mlp_only=False,
            loss_mode="bce_positive_weights",
            output_dir=args.output_dir,
            model_version=version,
            feature_schema={
                **shared_schema,
                "active_feature_attribute": feature_attribute,
                "feature_dimension": dimensions[feature_attribute],
            },
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device_name=args.device,
            rare_class_sampling=True,
            evaluate_test=False,
        )
        validate_static_checkpoint(
            Path(metadata["checkpoint"]),
            expected_feature_attribute=feature_attribute,
            expected_feature_dimension=dimensions[feature_attribute],
        )
        candidates[version] = metadata

    selected_version = max(candidates, key=lambda name: _score(candidates[name]))
    selected = candidates[selected_version]
    held_out = evaluate_static_checkpoint(
        checkpoint_path=Path(selected["checkpoint"]),
        graphs=graphs,
        split=split,
        split_name="test",
        batch_size=args.batch_size,
        device_name=args.device,
    )
    selected["test_metrics"] = held_out["metrics"]
    selected["test_evaluation_status"] = "evaluated_after_final_validation_selection"
    selected["held_out_evaluation"] = held_out
    selected_metadata_path = args.output_dir / f"{selected_version}.metadata.json"
    selected_metadata_path.write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )

    old = json.loads(args.old_metadata.read_text(encoding="utf-8"))
    deployment_winner = (
        selected if _score(selected) > _score(old) else old
    )
    report = {
        "schema_version": "ava80-final-temporal-comparison-v1",
        "selection_policy": (
            "primary validation supported-class macro F1, validation mAP tie-break; "
            "held-out test labels were evaluated only for the selected checkpoint"
        ),
        "dataset": {
            "videos": corpus_size,
            "timestamp_graphs": len(graphs),
            "nodes": sum(int(graph.num_nodes) for graph in graphs),
            "supervised_person_nodes": sum(
                int(graph.loss_mask.sum()) for graph in graphs
            ),
            "split_statistics": statistics,
            "graph_cache": None if args.skip_graph_cache else graph_cache.as_posix(),
            "graph_cache_metadata": graph_metadata,
            "split_manifest": split_path.as_posix(),
        },
        "old_30_video_roi_frame": _summary(old),
        "larger_data_candidates": {
            name: _summary(metadata) for name, metadata in candidates.items()
        },
        "selected_experiment_model": selected_version,
        "selected_experiment_checkpoint": selected["checkpoint"],
        "selected_experiment_validation_score": {
            "macro_f1_supported_classes": _score(selected)[0],
            "mean_average_precision": _score(selected)[1],
        },
        "selected_experiment_held_out_metrics": held_out["metrics"],
        "deployment_selected_model": deployment_winner["model_version"],
        "deployment_selected_checkpoint": deployment_winner["checkpoint"],
        "deployment_selected_checkpoint_sha256": deployment_winner[
            "checkpoint_sha256"
        ],
        "deployment_selection_reason": (
            "retained the existing scale-30 checkpoint because neither scale-37 "
            "candidate exceeded its validation supported-class macro F1"
            if deployment_winner is old
            else "activated the winning scale-37 candidate by validation metrics"
        ),
        "commands": {
            "expansion": (
                "python -m vidquery expand-ava-dataset --target-videos 75 "
                "--batch-size 1 --workers 3 --target-support 25 --min-free-gb 4"
            ),
            "training": (
                "python scripts/train_ava_final_temporal.py --epochs "
                f"{args.epochs} --patience {args.patience} --batch-size "
                f"{args.batch_size} --device {args.device}"
            ),
        },
        "limitations": [
            (
                "The temporal descriptor is fixed short-clip pooling over cached "
                "ImageNet-pretrained ResNet18 frame embeddings because validated AVA "
                "source media is intentionally deleted after feature extraction."
            ),
            "The GNN remains person-centric and does not predict explicit action targets.",
        ],
    }
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": report_path.as_posix(),
                "videos": corpus_size,
                "selected_experiment_model": selected_version,
                "selected_experiment_checkpoint": selected["checkpoint"],
                "deployment_selected_model": deployment_winner["model_version"],
                "deployment_selected_checkpoint": deployment_winner["checkpoint"],
                "validation": selected["validation_metrics"],
                "test": held_out["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
