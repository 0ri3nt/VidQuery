from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path("data/app/cache/torch").resolve()))

import torch  # noqa: E402

from vidquery.ava_improved import (  # noqa: E402
    attach_motion_features,
    attach_pretrained_visual_features,
    coverage_balanced_video_split,
    materialize_feature_variants,
    save_improved_graph_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen ResNet18 ROI/frame embeddings and AVA motion features"
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=Path(
            "data/app/models/ava80_v2_data/feature_cache/ava80_graph_dataset.pt"
        ),
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path("data/app/models/ava80_improved/visual_embedding_cache"),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/app/models/ava80_improved/split_manifest.json"),
    )
    parser.add_argument("--frame-root", type=Path, default=Path("data/ava/extracted_frames"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    payload = torch.load(args.base_cache, map_location="cpu", weights_only=False)
    graphs = list(payload["graphs"])
    print(f"Loaded {len(graphs)} timestamp graphs", flush=True)
    tracked = attach_motion_features(graphs)
    del graphs
    gc.collect()
    visual = attach_pretrained_visual_features(
        tracked,
        frame_root=args.frame_root,
        cache_dir=args.embedding_cache_dir,
        device_name=args.device,
    )
    del tracked
    gc.collect()
    improved = materialize_feature_variants(visual)
    del visual
    gc.collect()
    split = coverage_balanced_video_split(improved, seed=args.seed)
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    metadata = save_improved_graph_cache(improved, args.output_cache)
    print(json.dumps({
        "cache": metadata,
        "split": {
            "train_videos": len(split["train"]),
            "validation_videos": len(split["validation"]),
            "test_videos": len(split["test"]),
            "classes_with_positive_examples": split["classes_with_positive_examples"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
