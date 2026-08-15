from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidquery.ava_coverage import write_coverage_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select AVA videos that close missing/rare action-label deficits"
    )
    parser.add_argument(
        "--annotation",
        action="append",
        type=Path,
        default=[],
        help="AVA action annotation CSV; repeat for train and validation",
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/ava/videos"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/ava-expansion-selection.json"),
    )
    parser.add_argument("--rare-max-support", type=int, default=5)
    parser.add_argument("--target-support", type=int, default=10)
    parser.add_argument("--max-additional-videos", type=int, default=30)
    args = parser.parse_args()
    annotations = args.annotation or [
        Path("data/ava/annotations/ava_train_v2.2.csv"),
        Path("data/ava/annotations/ava_val_v2.2.csv"),
    ]
    report = write_coverage_selection(
        annotation_paths=annotations,
        video_dir=args.video_dir,
        output_path=args.output,
        rare_max_support=args.rare_max_support,
        target_support=args.target_support,
        max_additional_videos=args.max_additional_videos,
    )
    weak_rows = [row for row in report["actions"] if row["status"] != "covered"]
    print(json.dumps({
        "missing_actions": [
            [row["action_id"], row["canonical_name"], row["current_support"]]
            for row in weak_rows if row["status"] == "missing"
        ],
        "extremely_rare_actions": [
            [row["action_id"], row["canonical_name"], row["current_support"]]
            for row in weak_rows if row["status"] == "extremely_rare"
        ],
        "selected_video_ids": report["selected_video_ids"],
        "projected_actions_present": report["projected_actions_present"],
        "priority_actions_below_target": report[
            "priority_actions_below_target_after_selection"
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
