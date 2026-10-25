"""Score frozen AVA checkpoints on the identical test persons and compare them.

Test is touched only here, and only after ``--decision`` exists: a file
written beforehand that records the validation-only model choices. Shards
are read from ``--shard-dir`` or, for evaluation-only videos, from
``--eval-shard-dir``. Output: point metrics (all persons and detector-only),
per-video metrics, and paired bootstrap CIs (see ``vidquery.ava_compare``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from vidquery.ava_compare import compare_predictions, predict_checkpoint
from vidquery.ava_shards import load_shard, shard_paths, validate_shard


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=Path("data/app/features/ava80_shards"))
    parser.add_argument(
        "--eval-shard-dir", type=Path, default=Path("data/app/features/ava80_eval_shards")
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--pair", action="append", default=[], help="A:B reports B - A")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    if not args.decision.is_file():
        raise SystemExit("write the validation-only decision file before scoring test")
    import torch

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    split = json.loads(args.split.read_text(encoding="utf-8"))
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    graphs = []
    sources = {}
    for video in split["test"]:
        directory = next(
            path for path in (args.shard_dir, args.eval_shard_dir)
            if shard_paths(path, video)[0].is_file()
        )
        sources[video] = {"dir": directory.as_posix(), **validate_shard(directory, video)}
        graphs.extend(load_shard(directory, video))
    started = time.perf_counter()
    predictions = {
        name: predict_checkpoint(Path(path), graphs, device) for name, path in checkpoints.items()
    }
    inference_seconds = round(time.perf_counter() - started, 1)
    pairs = [tuple(item.split(":", 1)) for item in args.pair]
    report = {
        "schema_version": "ava80-checkpoint-comparison-v1",
        "split": args.split.as_posix(),
        "split_sha256": _sha256(args.split),
        "decision_file": args.decision.as_posix(),
        "decision_sha256": _sha256(args.decision),
        "test_videos": split["test"],
        "test_groups": split.get("test_groups"),
        "test_shards": {
            video: {key: value for key, value in info.items() if key in ("dir", "sha256", "graphs")}
            for video, info in sources.items()
        },
        "checkpoints": {
            name: {
                "path": Path(path).as_posix(),
                "sha256": _sha256(Path(path)),
                "model_version": predictions[name]["model_version"],
                "feature_attribute": predictions[name]["feature_attribute"],
            }
            for name, path in checkpoints.items()
        },
        "inference_seconds": inference_seconds,
        "primary_metric": "macro_f1_supported_classes on all supervised persons",
        "results": {
            subset: compare_predictions(
                predictions, pairs=pairs, samples=args.samples, seed=args.seed, subset=subset
            )
            for subset in ("all_persons", "detector_only")
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({subset: value["point"] for subset, value in report["results"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
