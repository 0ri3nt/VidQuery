from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidquery.vidor_improved import build_and_train_improved_vidor
from vidquery.vidor_relation import frame_selection_support


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
    parser.add_argument(
        "--version-tag",
        default="v2",
        help="Model version suffix; use a new tag (e.g. v3-scale119) for any retrain",
    )
    parser.add_argument(
        "--compare-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Also evaluate these checkpoints on the same test graphs (repeatable)",
    )
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=None,
        help="Reuse a pair-visual graph cache built for exactly this selection",
    )
    parser.add_argument("--init-gnn-checkpoint", type=Path, default=None)
    parser.add_argument("--init-mlp-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--withhold-test",
        action="store_true",
        help="Do not score test; select on validation now, compare on test afterwards",
    )
    args = parser.parse_args()
    for kind in ("mlp", "gat"):
        target = args.output_dir / f"vidor-relation-pair-visual-{kind}-{args.version_tag}.pt"
        if target.exists():
            raise SystemExit(f"{target} exists; choose a new --version-tag or --output-dir")

    manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    source_manifest_path = Path(manifest.get("source_split_manifest", args.baseline_manifest))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    # Frozen counts (extended selections) keep earlier videos' frames unchanged.
    predicate_support = frame_selection_support(source_manifest)
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
        version_tag=args.version_tag,
        graph_cache=args.graph_cache,
        init_checkpoints={
            "gnn": args.init_gnn_checkpoint,
            "mlp": args.init_mlp_checkpoint,
        },
        evaluate_test=not args.withhold_test,
    )
    comparisons = {}
    if args.compare_checkpoint and args.withhold_test:
        raise SystemExit("--compare-checkpoint scores test; drop it when withholding test")
    if args.compare_checkpoint:
        import torch

        from vidquery.vidor_improved import evaluate_relation_checkpoint, scalar_metrics
        from vidquery.vidor_relation import _graph_split

        graphs = torch.load(
            Path(report["dataset"]["cache"]),
            map_location="cpu",
            weights_only=False,
        )["graphs"]
        test_graphs = _graph_split(graphs, split)["test"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for path in args.compare_checkpoint:
            comparisons[path.as_posix()] = scalar_metrics(
                evaluate_relation_checkpoint(path, test_graphs, device)
            )
        (args.output_dir / "comparison_on_new_test.json").write_text(
            json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
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
                "compared_checkpoints_on_same_test": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
