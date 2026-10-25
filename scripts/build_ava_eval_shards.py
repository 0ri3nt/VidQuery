"""Build test-only AVA shards for evaluation-registry videos in their own directory.

Defaults to the registry's untouched ``ava_holdout_v2_test`` videos. The
shards use the same feature path as training shards but live outside the
training shard directory, so only ``--test-shard-dir`` can read them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidquery.ava_expansion import build_evaluation_shards
from vidquery.config import get_settings
from vidquery.eval_registry import eval_only_video_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument(
        "--shard-dir", type=Path, default=Path("data/app/features/ava80_eval_shards")
    )
    parser.add_argument(
        "--training-shard-dir", type=Path, default=Path("data/app/features/ava80_shards")
    )
    parser.add_argument("--temporary-root", type=Path, default=Path("data/temp/ava_eval_shards"))
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    videos = args.videos or sorted(eval_only_video_ids(groups=["ava_holdout_v2_test"]))
    state = build_evaluation_shards(
        videos,
        shard_dir=args.shard_dir,
        training_shard_dir=args.training_shard_dir,
        temporary_root=args.temporary_root,
        yolo_model=get_settings().yolo_model,
        device=args.device,
        workers=args.workers,
        min_free_bytes=int(args.min_free_gb * 1024**3),
    )
    print(json.dumps({key: state.get(key) for key in (
        "requested_video_ids", "completed_video_ids", "storage_blocked", "seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
