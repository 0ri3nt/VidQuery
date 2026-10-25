"""Write a versioned AVA scaling split with frozen validation and test videos.

Validation and test are copied from ``--base-split`` (the 30-video scaling
split: 3 calibration videos, 3 held-out test videos). Every other verified
training shard goes to train. ``--extra-test`` appends evaluation-only test
videos whose shards live in ``--eval-shard-dir``. The builder refuses
registry videos in train, test-only registry videos in validation, any split
overlap, and test videos without a shard.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidquery.ava_shards import completed_shard_video_ids
from vidquery.eval_registry import (
    EvaluationLeakageError,
    assert_not_selected_on,
    assert_not_trained_on,
)


def build_split(
    base_split: dict[str, Any],
    training_shards: set[str],
    *,
    extra_test: list[str],
    eval_shards: set[str],
) -> dict[str, Any]:
    validation = [str(video) for video in base_split["validation"]]
    test = [str(video) for video in base_split["test"]]
    fixed = set(validation) | set(test)
    train = sorted(training_shards - fixed - set(extra_test))
    missing = sorted(
        {video for video in [*validation, *test] if video not in training_shards}
        | {video for video in extra_test if video not in eval_shards}
    )
    if missing:
        raise EvaluationLeakageError(f"split videos without a verified shard: {missing}")
    if set(extra_test) & training_shards:
        raise EvaluationLeakageError("evaluation-only test videos must not have training shards")
    assert_not_trained_on(train, context="AVA scaling train split")
    assert_not_selected_on(validation, context="AVA scaling validation split")
    all_test = [*test, *(video for video in extra_test if video not in test)]
    sets = [set(train), set(validation), set(all_test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise EvaluationLeakageError("AVA scaling split overlaps")
    return {
        "schema_version": "ava80-fixed-holdout-scaling-split-v2",
        "seed": int(base_split.get("seed", 2026)),
        "corpus_size": len(train) + len(validation) + len(all_test),
        "selection_rule": (
            "validation and test frozen from the base split; every other verified "
            "training shard is train; evaluation-only test videos appended"
        ),
        "train": train,
        "validation": validation,
        "test": all_test,
        "test_groups": {"base_test": test, "evaluation_only_test": list(extra_test)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-split",
        type=Path,
        default=Path("data/app/models/ava80_scaling/30/split_manifest.json"),
    )
    parser.add_argument("--shard-dir", type=Path, default=Path("data/app/features/ava80_shards"))
    parser.add_argument(
        "--eval-shard-dir", type=Path, default=Path("data/app/features/ava80_eval_shards")
    )
    parser.add_argument("--extra-test", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    eval_shards = (
        completed_shard_video_ids(args.eval_shard_dir) if args.eval_shard_dir.is_dir() else set()
    )
    split = build_split(
        json.loads(args.base_split.read_text(encoding="utf-8")),
        completed_shard_video_ids(args.shard_dir),
        extra_test=list(args.extra_test),
        eval_shards=eval_shards,
    )
    split["base_split"] = args.base_split.as_posix()
    split["shard_dir"] = args.shard_dir.as_posix()
    split["eval_shard_dir"] = args.eval_shard_dir.as_posix()
    split["created_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(split[name]) for name in ("train", "validation", "test")}))


if __name__ == "__main__":
    main()
