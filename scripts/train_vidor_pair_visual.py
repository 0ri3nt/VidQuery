from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vidquery.vidor_improved import build_and_train_improved_vidor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen ResNet subject/object/union features and train matched "
            "VidOR pairwise MLP and GATv2 models"
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/source/vidor/training-annotation.zip"),
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        required=True,
        help="Extracted official frames arranged as GROUP/VIDEO/frame_XXXXXX.jpg",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("data/source/vidor/selected-frames/split_manifest.json"),
        help="Preserves the baseline video's original train/validation/test assignment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/app/models/vidor_relation_pair_visual"),
    )
    parser.add_argument("--max-frames-per-video", type=int, default=6)
    parser.add_argument("--hard-negative-ratio", type=float, default=2.0)
    parser.add_argument("--random-negative-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    source_manifest_path = Path(manifest.get("source_split_manifest", args.baseline_manifest))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    predicate_support: Counter[str] = Counter()
    for row in source_manifest["selected_videos"]:
        predicate_support.update(row["support"])
    split = {
        name: [str(value) for value in manifest[name]] for name in ("train", "validation", "test")
    }
    report = build_and_train_improved_vidor(
        annotation_zip=args.annotations,
        frame_root=args.frame_root,
        output_dir=args.output_dir,
        selected=manifest["selected_videos"],
        split=split,
        predicate_support=predicate_support,
        max_frames_per_video=args.max_frames_per_video,
        hard_negative_ratio=args.hard_negative_ratio,
        random_negative_ratio=args.random_negative_ratio,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "output": (args.output_dir / "held_out_metrics_pair_visual.json").as_posix(),
                "dataset": report["dataset"],
                "validation_winner": report["validation_winner"],
                "pairwise_mlp": {
                    "validation": report["pairwise_mlp"]["validation_metrics"],
                    "test": report["pairwise_mlp"]["test_metrics"],
                },
                "relationship_gnn": {
                    "validation": report["relationship_gnn"]["validation_metrics"],
                    "test": report["relationship_gnn"]["test_metrics"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
