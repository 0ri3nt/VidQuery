"""Train the leakage-free AVA action model by streaming slim shards.

Only a few decoded shards are held in memory at a time
(``ShardGraphDataset(cache_shards=...)``), so the full 299-video AVA set fits
this laptop's RAM.  Splits come from a JSON manifest with train/validation/test
video lists; evaluation-only videos are refused for training, test-only
registry videos are refused for validation, and test metrics are withheld
unless ``--evaluate-test`` is given.  ``--test-shard-dir`` adds directories of
evaluation-only shards that are read for the test split only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vidquery.ava_ablation import train_static_ablation
from vidquery.ava_gnn80 import (
    AVA80_NOLEAK_ATTRIBUTE,
    AVA80_NOLEAK_DROPPED,
    AVA80_NOLEAK_SCHEMA,
    leaking_column_indices,
)
from vidquery.ava_shards import SHARD_SCHEMA, ShardBlockSampler, ShardGraphDataset
from vidquery.eval_registry import assert_not_selected_on, assert_not_trained_on


def _peak_memory_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    peak = float(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0) or 0)
    return round(peak / 2**20, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=Path("data/app/features/ava80_shards"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--test-shard-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    checkpoint = args.output_dir / f"{args.model_version}.pt"
    if checkpoint.exists():
        raise SystemExit(f"{checkpoint} exists; choose a new model version")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    assert_not_trained_on(split["train"], context="shard training split")
    assert_not_selected_on(split["validation"], context="shard validation split")
    datasets = {
        name: ShardGraphDataset(
            args.shard_dir,
            split[name],
            feature_attribute=AVA80_NOLEAK_ATTRIBUTE,
            cache_shards=args.cache_shards,
            extra_shard_dirs=args.test_shard_dir if name == "test" else (),
        )
        for name in ("train", "validation", "test")
    }
    samplers = {
        "train": ShardBlockSampler(datasets["train"], seed=args.seed, shuffle=True),
        "validation": ShardBlockSampler(datasets["validation"], seed=args.seed, shuffle=False),
        "test": ShardBlockSampler(datasets["test"], seed=args.seed, shuffle=False),
    }
    started = time.perf_counter()
    metadata = train_static_ablation(
        graphs=[],
        split=split,
        feature_attribute=AVA80_NOLEAK_ATTRIBUTE,
        use_edges=not args.mlp_only,
        mlp_only=args.mlp_only,
        loss_mode="bce_positive_weights",
        output_dir=args.output_dir,
        model_version=args.model_version,
        feature_schema={
            "base_attribute": "visual_x",
            "backbone": "torchvision-resnet18-imagenet1k-v1",
            "dropped_feature_names": list(AVA80_NOLEAK_DROPPED),
            "dropped_feature_columns": leaking_column_indices(),
            "source_shard_schema": SHARD_SCHEMA,
        },
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device_name=args.device,
        evaluate_test=args.evaluate_test,
        learning_rate=args.learning_rate,
        schema_version=AVA80_NOLEAK_SCHEMA,
        extra_payload={
            "dropped_feature_names": list(AVA80_NOLEAK_DROPPED),
            "source_shard_dir": args.shard_dir.as_posix(),
            "source_split_manifest": args.split.as_posix(),
            "test_shard_dirs": [path.as_posix() for path in args.test_shard_dir],
        },
        datasets=datasets,
        samplers=samplers,
    )
    summary = {
        "model_version": metadata["model_version"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "best_epoch": metadata["checkpoint_selection"]["best_epoch"],
        "epochs_run": len(metadata["history"]),
        "graph_counts": metadata["graph_counts"],
        "validation": {
            key: metadata["validation_metrics"][key]
            for key in ("macro_f1_supported_classes", "micro_f1", "mean_average_precision")
        },
        "test": (
            {
                key: metadata["test_metrics"][key]
                for key in ("macro_f1_supported_classes", "micro_f1", "mean_average_precision")
            }
            if metadata.get("test_metrics")
            else "withheld"
        ),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "peak_process_memory_mb": _peak_memory_mb(),
        "shard_loads": {name: dataset.loads for name, dataset in datasets.items()},
    }
    (args.output_dir / f"{args.model_version}.run.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
