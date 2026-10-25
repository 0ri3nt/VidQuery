"""Compare the stored evidence of two SQLite indexes of the same videos.

Segments are matched by (source filename stem, segment start), because video
IDs are random per registration. Per field, the report counts segments whose
evidence differs. IDs that embed the random video ID (frame IDs, OCR frame
lists) are normalised first, so only real evidence changes count.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    names = {
        video_id: Path(filename).stem
        for video_id, filename in connection.execute(
            "select video_id, original_filename from videos"
        )
    }
    segments = {}
    for video_id, start, payload in connection.execute(
        "select video_id, start_time, payload_json from segments"
    ):
        segments[(names[video_id], round(float(start), 3))] = json.loads(payload)
    connection.close()
    return segments


def _rounded(value: Any, digits: int) -> float:
    return round(float(value), digits) if value is not None else -1.0


def _views(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "detections": [
            (item["class_label"], round(item["timestamp"], 3), round(item["confidence"], 4))
            for item in segment.get("detections", [])
        ],
        "transcript": segment.get("transcript"),
        "actions": sorted(map(str, segment.get("actions", []))),
        "relationships": sorted(
            (item["source_method"], item["predicate"],
             _rounded(item.get("start_time", item.get("timestamp")), 2),
             _rounded(item.get("confidence"), 4))
            for item in segment.get("relationships", [])
        ),
        "ocr_text": sorted(item.get("text", "") for item in segment.get("ocr_evidence", [])),
        "speakers": [
            item.get("speaker_label") if isinstance(item, dict) else item
            for item in segment.get("speakers", [])
        ],
        "frame_count": len(segment.get("frame_ids", [])),
    }


def compare(left: Path, right: Path) -> dict[str, Any]:
    a, b = _load(left), _load(right)
    shared = sorted(set(a) & set(b))
    differing: Counter[str] = Counter()
    for key in shared:
        views_a, views_b = _views(a[key]), _views(b[key])
        for field, value in views_a.items():
            differing[field] += value != views_b[field]
    return {
        "left": left.as_posix(),
        "right": right.as_posix(),
        "segments_compared": len(shared),
        "segments_only_left": len(set(a) - set(b)),
        "segments_only_right": len(set(b) - set(a)),
        "segments_differing_by_field": {field: differing[field] for field in sorted(differing)},
        "identical": not differing.total() and len(a) == len(b) == len(shared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = compare(args.left, args.right)
    text = json.dumps(report, indent=1)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"{args.output} exists; choose a new file name")
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
