from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ava_labels import AVA_V22_ACTIONS, load_ava_v22_label_map


@dataclass(frozen=True, slots=True)
class VideoCoverage:
    video_id: str
    action_counts: Counter[int]
    annotation_split: str


def load_video_coverage(annotation_paths: Sequence[Path]) -> dict[str, VideoCoverage]:
    """Count AVA action rows per video without loading the full CSVs into memory."""
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    split_by_video: dict[str, str] = {}
    for annotation_path in annotation_paths:
        split_name = "validation" if "val" in annotation_path.name else "train"
        with Path(annotation_path).open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) < 7:
                    continue
                video_id = row[0]
                counts[video_id][int(row[6])] += 1
                split_by_video[video_id] = split_name
    return {
        video_id: VideoCoverage(video_id, action_counts, split_by_video[video_id])
        for video_id, action_counts in counts.items()
    }


def _combined_support(
    coverage: dict[str, VideoCoverage], video_ids: Iterable[str]
) -> Counter[int]:
    combined: Counter[int] = Counter()
    for video_id in video_ids:
        item = coverage.get(video_id)
        if item is not None:
            combined.update(item.action_counts)
    return combined


def select_coverage_videos(
    coverage: dict[str, VideoCoverage],
    current_video_ids: Iterable[str],
    *,
    excluded_video_ids: Iterable[str] = (),
    rare_max_support: int = 5,
    target_support: int = 10,
    max_additional_videos: int = 30,
) -> dict[str, Any]:
    """Greedily cover weak labels, then remove redundant selections.

    The objective is deliberately narrow: improve labels that are absent or have at
    most ``rare_max_support`` rows in the current local media. Each round chooses the
    video that closes the largest fraction of the remaining per-label deficits.
    Deterministic tie breaking makes the manifest reproducible.
    """
    current_ids = sorted(set(current_video_ids).intersection(coverage))
    current_support = _combined_support(coverage, current_ids)
    priority_ids = tuple(
        action_id
        for action_id in range(1, 81)
        if current_support[action_id] <= rare_max_support
    )
    remaining = set(coverage).difference(current_ids).difference(excluded_video_ids)
    selected: list[str] = []
    projected = current_support.copy()

    def deficits(support: Counter[int]) -> dict[int, int]:
        return {
            action_id: max(0, target_support - support[action_id])
            for action_id in priority_ids
        }

    while len(selected) < max_additional_videos:
        missing = deficits(projected)
        if not any(missing.values()) or not remaining:
            break

        def score(
            video_id: str, deficits_by_action: dict[int, int] = missing
        ) -> tuple[float, int, int, str]:
            counts = coverage[video_id].action_counts
            fractional_gain = sum(
                min(deficits_by_action[action_id], counts[action_id]) / target_support
                for action_id in priority_ids
                if deficits_by_action[action_id]
            )
            labels_helped = sum(
                bool(deficits_by_action[action_id] and counts[action_id])
                for action_id in priority_ids
            )
            raw_gain = sum(
                min(deficits_by_action[action_id], counts[action_id])
                for action_id in priority_ids
            )
            return fractional_gain, labels_helped, raw_gain, video_id

        best = max(remaining, key=score)
        if score(best)[0] <= 0:
            break
        selected.append(best)
        projected.update(coverage[best].action_counts)
        remaining.remove(best)

    # Greedy selection can retain an early video that becomes redundant later.
    # Remove it when every priority label still meets the target without it.
    changed = True
    while changed:
        changed = False
        for video_id in tuple(selected):
            candidate = _combined_support(
                coverage, [*current_ids, *(item for item in selected if item != video_id)]
            )
            if all(candidate[action_id] >= target_support for action_id in priority_ids):
                selected.remove(video_id)
                projected = candidate
                changed = True
                break

    projected = _combined_support(coverage, [*current_ids, *selected])
    official_names = {
        item.label_id: item.official_name for item in load_ava_v22_label_map()
    }
    action_rows = []
    for action_id in range(1, 81):
        containing = sorted(
            (
                item
                for item in coverage.values()
                if item.video_id not in current_ids and item.action_counts[action_id]
            ),
            key=lambda item: (-item.action_counts[action_id], item.video_id),
        )
        action_rows.append(
            {
                "action_id": action_id,
                "canonical_name": AVA_V22_ACTIONS[action_id],
                "official_name": official_names[action_id],
                "current_support": current_support[action_id],
                "status": (
                    "missing"
                    if current_support[action_id] == 0
                    else "extremely_rare"
                    if current_support[action_id] <= rare_max_support
                    else "covered"
                ),
                "candidate_videos": [
                    {
                        "video_id": item.video_id,
                        "additional_support": item.action_counts[action_id],
                        "annotation_split": item.annotation_split,
                    }
                    for item in containing
                ],
                "selected_videos": [
                    video_id for video_id in selected if coverage[video_id].action_counts[action_id]
                ],
                "expected_support_after_download": projected[action_id],
            }
        )

    uncovered_priority = [
        action_id for action_id in priority_ids if projected[action_id] < target_support
    ]
    return {
        "schema_version": "ava-v2.2-coverage-selection-v1",
        "selection_method": "deterministic_greedy_deficit_cover_with_redundancy_pruning",
        "rare_max_support": rare_max_support,
        "target_support": target_support,
        "max_additional_videos": max_additional_videos,
        "current_video_ids": current_ids,
        "current_video_count": len(current_ids),
        "current_actions_present": sum(
            current_support[action_id] > 0 for action_id in range(1, 81)
        ),
        "missing_action_ids": [
            action_id for action_id in range(1, 81) if current_support[action_id] == 0
        ],
        "extremely_rare_action_ids": [
            action_id
            for action_id in range(1, 81)
            if 0 < current_support[action_id] <= rare_max_support
        ],
        "priority_action_ids": list(priority_ids),
        "selected_video_ids": selected,
        "selected_video_count": len(selected),
        "projected_total_video_count": len(current_ids) + len(selected),
        "projected_actions_present": sum(
            projected[action_id] > 0 for action_id in range(1, 81)
        ),
        "priority_actions_below_target_after_selection": uncovered_priority,
        "downloads": [
            {
                "video_id": video_id,
                "filename": f"{video_id}.mp4",
                "annotation_split": coverage[video_id].annotation_split,
                "s3_url": f"https://s3.amazonaws.com/ava-dataset/trainval/{video_id}.mp4",
            }
            for video_id in selected
        ],
        "actions": action_rows,
    }


def write_coverage_selection(
    *,
    annotation_paths: Sequence[Path],
    video_dir: Path,
    output_path: Path,
    rare_max_support: int = 5,
    target_support: int = 10,
    max_additional_videos: int = 30,
) -> dict[str, Any]:
    coverage = load_video_coverage(annotation_paths)
    current_video_ids = {
        path.stem
        for path in Path(video_dir).iterdir()
        if path.suffix.lower() in {".mp4", ".mkv", ".webm"}
    }
    report = select_coverage_videos(
        coverage,
        current_video_ids,
        rare_max_support=rare_max_support,
        target_support=target_support,
        max_additional_videos=max_additional_videos,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
