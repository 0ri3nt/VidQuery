from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidquery.vidor_relation import build_and_train_vidor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build explicit VidOR relation graphs, baselines, and GATv2 model"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/source/vidor/training-annotation.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/app/models/vidor_relation"),
    )
    parser.add_argument("--target-videos", type=int, default=500)
    parser.add_argument("--max-frames-per-video", type=int, default=6)
    parser.add_argument("--nearest-k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = build_and_train_vidor(
        annotation_zip=args.annotations,
        output_dir=args.output_dir,
        target_videos=args.target_videos,
        max_frames_per_video=args.max_frames_per_video,
        nearest_k=args.nearest_k,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device_name=args.device,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": (args.output_dir / "held_out_metrics.json").as_posix(),
                "dataset": report["dataset"],
                "validation_comparison": report["validation_comparison"],
                "test": {
                    "geometry_rules": report["baselines"]["geometry_rules"],
                    "class_pair_prior": report["baselines"]["class_pair_prior"],
                    "pairwise_mlp": report["pairwise_mlp"]["test_metrics"],
                    "relationship_gnn": report["relationship_gnn"]["test_metrics"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
