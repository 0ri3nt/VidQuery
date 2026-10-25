"""Extend the 500-video VidOR selection with training videos only.

Writes a new versioned manifest: validation and test copied unchanged, the
next-ranked videos added to train, and frame-selection counts frozen at the
base selection's values. It then checks that every base video selects
exactly the frames it selected before.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from vidquery.vidor_relation import (
    _annotation_summaries,
    _selected_frames,
    extend_relation_selection,
    frame_selection_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/source/vidor/training-annotation.zip")
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("data/app/models/vidor_relation/split_manifest.json"),
    )
    parser.add_argument("--target-videos", type=int, default=2000)
    parser.add_argument("--max-frames-per-video", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    summaries, global_support = _annotation_summaries(args.annotations)
    extended = extend_relation_selection(summaries, global_support, base, args.target_videos)
    old_support, new_support = frame_selection_support(base), frame_selection_support(extended)
    changed = []
    with zipfile.ZipFile(args.annotations) as archive:
        for row in base["selected_videos"]:
            relations = json.loads(archive.read(row["member"]))["relation_instances"]
            before = _selected_frames(relations, old_support, args.max_frames_per_video)
            after = _selected_frames(relations, new_support, args.max_frames_per_video)
            if before != after:
                changed.append(str(row["video_id"]))
    if changed:
        raise SystemExit(f"frame selection changed for base videos: {changed[:10]}")
    extended["base_manifest"] = args.base_manifest.as_posix()
    extended["created_at"] = datetime.now(UTC).isoformat()
    extended["frame_invariant"] = {
        "base_videos_checked": len(base["selected_videos"]),
        "base_videos_with_changed_frames": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(extended, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": len(extended["selected_videos"]),
        "train": len(extended["train"]),
        "validation": len(extended["validation"]),
        "test": len(extended["test"]),
        "frame_invariant": extended["frame_invariant"],
    }))


if __name__ == "__main__":
    main()
