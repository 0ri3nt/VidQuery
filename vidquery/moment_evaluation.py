"""Temporal moment retrieval and answerability evaluation (Charades-STA, NA-VMR).

Metric definitions (all per query, then averaged):

* ``tIoU``: temporal intersection over union of a predicted interval and the
  ground-truth moment.
* ``R@1@m`` / ``R@5@m``: the official Charades-STA metric. The top-1 (any of
  the top-5) predicted interval has tIoU >= m, m in {0.3, 0.5, 0.7}. Ranking
  only; abstention is ignored, as in the original protocol.
* ``mIoU``: mean tIoU of the top-1 interval (0 if nothing was retrieved).
* ``P@1``: the query was *answered* (not abstained) and its top-1 interval has
  tIoU >= 0.5. For full Review-2 this is R@1@0.5 counted only on answered
  queries; for systems without abstention it equals R@1@0.5.
* NA-VMR: a negative query is *rejected* only when the system abstains. Any
  answer, weak or not, is a failure to reject. Reported per negative kind.

Predicted intervals (``span``):

* ``native``: what each system reports (Review-1: the 5 s segment; Review-2:
  the localization interval).
* ``segment``: always the 5 s segment of the result.
* ``merge``: the top-1 segment extended over adjacent retrieved segments whose
  score is at least ``merge_ratio`` times the top score.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from .challenge_evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    ChallengeMethods,
    RankedMoment,
    _bootstrap_interval,
    _selective_metrics,
    _without_abstention,
    source_video_id,
)
from .query_planner import QueryPlanningService
from .search import LocalHybridSearchEngine
from .storage import SQLiteRepository

IOU_THRESHOLDS = (0.3, 0.5, 0.7)
SPANS = ("native", "segment", "merge")


def temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def predicted_intervals(
    ranked: list[RankedMoment], *, span: str, merge_ratio: float = 0.8
) -> list[tuple[float, float]]:
    if span not in SPANS:
        raise ValueError(f"span must be one of {SPANS}")
    if span == "native":
        return [(item.interval_start, item.interval_end) for item in ranked]
    intervals = [(item.segment_start, item.segment_end) for item in ranked]
    if span == "segment" or not ranked:
        return intervals
    top = ranked[0]
    floor = merge_ratio * top.score
    eligible = sorted(
        (item.segment_start, item.segment_end)
        for item in ranked
        if item.score >= floor
    )
    start, end = top.segment_start, top.segment_end
    changed = True
    while changed:
        changed = False
        for left, right in eligible:
            if abs(left - end) < 1e-6:
                end, changed = right, True
            elif abs(right - start) < 1e-6:
                start, changed = left, True
    return [(start, end), *intervals[1:]]


def _query_row(
    item: dict[str, Any], run: Any, *, span: str, merge_ratio: float
) -> dict[str, Any]:
    ranked = list(run.ranked)
    abstained = run.interpretation == "insufficient_evidence" or not ranked
    confidence = ranked[0].confidence if ranked else 0.0
    row: dict[str, Any] = {
        "id": item["id"],
        "kind": item["kind"],
        "negative": not item["answerable"],
        "abstained": abstained,
        "confidence": float(confidence) if confidence is not None else None,
        "top_verdict": ranked[0].verdict if ranked else None,
        "result_count": len(ranked),
    }
    if not item["answerable"]:
        row["metrics"] = {
            "abstained": abstained,
            "confidence": row["confidence"],
            "negative_rejected": abstained,
        }
        return row
    truth = (float(item["start"]), float(item["end"]))
    intervals = predicted_intervals(ranked, span=span, merge_ratio=merge_ratio)
    ious = [temporal_iou(interval, truth) for interval in intervals]
    top_iou = ious[0] if ious else 0.0
    segment_overlap = [
        temporal_iou((moment.segment_start, moment.segment_end), truth) > 0 for moment in ranked
    ]
    if not ranked:
        stage = "no_result"
    elif top_iou >= 0.5:
        stage = "correct"
    elif segment_overlap and segment_overlap[0]:
        stage = "localization"  # right region, interval too short/long
    elif any(segment_overlap[1:]):
        stage = "ranking"  # a correct segment was retrieved but not ranked first
    else:
        stage = "retrieval"  # nothing retrieved overlaps the moment
    metrics: dict[str, Any] = {
        "abstained": abstained,
        "confidence": row["confidence"],
        "top_iou": round(top_iou, 4),
        "precision_at_1": float(top_iou >= 0.5 and not abstained),
        "ranking_precision_at_1": float(top_iou >= 0.5),
        "stage": stage,
    }
    for threshold in IOU_THRESHOLDS:
        metrics[f"r1_iou{threshold}"] = float(top_iou >= threshold)
        metrics[f"r5_iou{threshold}"] = float(any(value >= threshold for value in ious[:5]))
    row["metrics"] = metrics
    row["predicted"] = [list(map(lambda value: round(value, 2), pair)) for pair in intervals[:5]]
    row["truth"] = list(truth)
    return row


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if not row["negative"]]
    negatives = [row for row in rows if row["negative"]]
    summary: dict[str, Any] = {
        "positive_query_count": len(positives),
        "negative_query_count": len(negatives),
        "negative_counts_by_kind": dict(Counter(row["kind"] for row in negatives)),
    }
    metrics = [row["metrics"] for row in positives]
    keys = ["precision_at_1", "ranking_precision_at_1"] + [
        f"r{k}_iou{threshold}" for k in (1, 5) for threshold in IOU_THRESHOLDS
    ]
    for key in keys:
        summary[key] = _mean(m[key] for m in metrics)
    summary["mean_iou"] = _mean(m["top_iou"] for m in metrics)
    summary["positive_answered_rate"] = _mean(float(not m["abstained"]) for m in metrics)
    summary["failure_stages"] = dict(Counter(m["stage"] for m in metrics))
    summary["negative_rejection_rate_by_kind"] = {
        kind: _mean(float(row["metrics"]["negative_rejected"]) for row in negatives
                    if row["kind"] == kind)
        for kind in sorted({row["kind"] for row in negatives})
    }
    summary["ci95"] = {
        key: _bootstrap_interval([m[key] for m in metrics])
        for key in ("precision_at_1", "r1_iou0.5", "r1_iou0.7", "r5_iou0.5")
    }
    if negatives:
        selective_rows = [
            {"negative": row["negative"], "metrics": row["metrics"]} for row in rows
        ]
        selective = _selective_metrics(selective_rows)
        summary.update({key: value for key, value in selective.items() if key != "risk_coverage"})
    return summary


def content_blind_baselines(
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    durations: dict[str, float],
    *,
    segment_duration: float = 5.0,
) -> dict[str, Any]:
    """Random 5 s segment (expected value) and the best fixed fractional window.

    The fixed window is chosen on validation and applied unchanged to test,
    so it measures how much the dataset's moment-position bias alone yields.
    """

    def random_segment(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        output: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            duration = float(durations[row["video"]])
            count = max(1, int(duration // segment_duration) + (duration % segment_duration > 0))
            truth = (row["start"], row["end"])
            values = [
                temporal_iou((index * segment_duration,
                              min(duration, (index + 1) * segment_duration)), truth)
                for index in range(count)
            ]
            for threshold in IOU_THRESHOLDS:
                output[f"r1_iou{threshold}"].append(
                    sum(value >= threshold for value in values) / len(values)
                )
            output["mean_iou"].append(sum(values) / len(values))
        return {key: _mean(values) for key, values in output.items()}

    def fixed(rows: list[dict[str, Any]], start: float, end: float) -> dict[str, float | None]:
        output: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            duration = float(durations[row["video"]])
            value = temporal_iou((start * duration, end * duration), (row["start"], row["end"]))
            for threshold in IOU_THRESHOLDS:
                output[f"r1_iou{threshold}"].append(float(value >= threshold))
            output["mean_iou"].append(value)
        return {key: _mean(values) for key, values in output.items()}

    grid = [round(0.05 * step, 2) for step in range(21)]
    best = max(
        ((start, end) for start in grid for end in grid if end > start),
        key=lambda window: (fixed(validation, *window)["r1_iou0.5"] or 0.0, -window[0]),
    )
    return {
        "random_segment": {"validation": random_segment(validation), "test": random_segment(test)},
        "fixed_window_from_validation": {
            "window_fraction": list(best),
            "validation": fixed(validation, *best),
            "test": fixed(test, *best),
        },
    }


def evaluate_moment_benchmark(
    repository: SQLiteRepository,
    queries: list[dict[str, Any]],
    *,
    planner: QueryPlanningService,
    engine: LocalHybridSearchEngine,
    limit: int = 5,
    spans: Iterable[str] = SPANS,
    merge_ratio: float = 0.8,
    resolution_margin: float = 0.25,
) -> dict[str, Any]:
    by_source: dict[str, str] = {}
    for video in repository.list_videos():
        by_source[source_video_id(video)] = video.video_id
    missing = sorted({item["video"] for item in queries} - set(by_source))
    if missing:
        raise ValueError(f"{len(missing)} benchmark videos are not indexed, e.g. {missing[:5]}")
    methods = ChallengeMethods(
        repository, planner, engine=engine, resolution_margin=resolution_margin
    )
    spans = list(spans)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plans: list[dict[str, Any]] = []
    for item in queries:
        planning = planner.plan(item["query"])
        plans.append(
            {
                "id": item["id"],
                "intent": planning.plan.intent,
                "actions": planning.plan.actions,
                "entities": planning.plan.visual_entities,
                "spoken_terms": planning.plan.spoken_terms,
            }
        )
        video_ids = [by_source[item["video"]]]
        runs = {
            "review1": methods.review1_segment_start(item["query"], planning, video_ids, limit),
        }
        full = methods.review2_full(item["query"], planning, video_ids, limit)
        runs["review2_full"] = full
        runs["review2_full_no_abstention"] = _without_abstention(full)
        for method, run in runs.items():
            for span in spans:
                rows[f"{method}|{span}"].append(
                    _query_row(item, run, span=span, merge_ratio=merge_ratio)
                )
    return {
        "summary": {key: summarize_rows(values) for key, values in rows.items()},
        "plans": plans,
        "rows": dict(rows),
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
    }

