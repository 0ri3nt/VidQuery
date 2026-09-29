"""Review-2 challenge benchmark: ambiguity, timestamp precision, abstention.

The Review-1 benchmark (``evaluation/queries_independent.json``) labels
relevance at the five-second segment grid, so it cannot measure sub-segment
timestamp error and is already near-saturated (P@1 0.96).  This module runs a
*different* manifest whose ground truth is a point-in-time ``peak_time`` plus a
tight interval, and whose queries are deliberately ambiguous, timestamp-
specific, or negative.  Three systems are compared on identical inputs:

* ``review1_segment_start``  - Review-1 behaviour: hybrid ranking, segment
  start returned as the timestamp, alternatives merged by max score.
* ``review2_localized``      - Review-1 ranking plus the coarse-to-fine
  localizer (isolates the timestamp gain).
* ``review2_full``           - hypotheses + reliability ranking + localization
  + duplicate collapse + abstention.

Every number is machine-generated from the manifest; nothing is hand-typed.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from .domain import QueryPlan, SearchRequest, SearchResult
from .hypotheses import fuse_hypotheses
from .query_planner import QueryPlanningOutcome, QueryPlanningService
from .search import LocalHybridSearchEngine
from .storage import SQLiteRepository

SCHEMA_VERSION = "challenge-1.0"
REQUIRED_QUERY_FIELDS = ("id", "query", "category", "source_video_id")
MINIMUM_COUNTS = {"ambiguous": 15, "timestamp": 15, "negative": 5}


class ChallengeManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RankedMoment:
    video_id: str
    segment_start: float
    segment_end: float
    time_point: float
    interval_start: float
    interval_end: float
    score: float
    verdict: str | None
    confidence: float | None
    precision: str | None


@dataclass(slots=True)
class MethodRun:
    ranked: list[RankedMoment]
    selected_intent: str | None
    selected_hypothesis_id: str | None
    offered_hypothesis_ids: list[str] = field(default_factory=list)
    offered_intents: list[str] = field(default_factory=list)
    interpretation: str = "single"
    latency_ms: float = 0.0


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    """Raise on structural errors; return non-fatal warnings."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ChallengeManifestError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ChallengeManifestError("manifest must contain a non-empty 'queries' list")
    seen: set[str] = set()
    for item in queries:
        for name in REQUIRED_QUERY_FIELDS:
            if not item.get(name):
                raise ChallengeManifestError(f"query is missing required field {name!r}: {item}")
        if item["id"] in seen:
            raise ChallengeManifestError(f"duplicate query id {item['id']!r}")
        seen.add(item["id"])
        if item.get("status") == "example":
            raise ChallengeManifestError(
                f"query {item['id']!r} is still the template example; replace it with a "
                "real, manually labelled query"
            )
        negative = bool(item.get("negative", False))
        relevant = item.get("relevant", [])
        if negative and relevant:
            raise ChallengeManifestError(f"negative query {item['id']!r} must not list relevant")
        if not negative and not relevant:
            raise ChallengeManifestError(f"positive query {item['id']!r} needs relevant moments")
        for moment in relevant:
            for name in ("source_video_id", "peak_time", "start_time", "end_time"):
                if moment.get(name) is None:
                    raise ChallengeManifestError(
                        f"relevant moment of {item['id']!r} is missing {name!r}"
                    )
            if not moment["start_time"] <= moment["peak_time"] <= moment["end_time"]:
                raise ChallengeManifestError(
                    f"relevant moment of {item['id']!r} has peak outside its interval"
                )
    warnings: list[str] = []
    counts = Counter(str(item["category"]) for item in queries)
    for category, minimum in MINIMUM_COUNTS.items():
        if counts.get(category, 0) < minimum:
            warnings.append(
                f"category {category!r} has {counts.get(category, 0)} queries; "
                f"the Review-2 protocol asks for at least {minimum}"
            )
    if len(queries) < 40:
        warnings.append(f"manifest has {len(queries)} queries; the protocol asks for 40-60")
    return warnings


def _source_ids(repository: SQLiteRepository) -> dict[str, str]:
    return {
        video.video_id: Path(video.original_filename).stem for video in repository.list_videos()
    }


def _moment_from_result(
    result: SearchResult, *, use_localization: bool, with_verdict: bool = True
) -> RankedMoment:
    """Project a result onto the timestamp contract of the method under test.

    ``with_verdict=False`` reproduces Review-1, which had no abstention layer
    and therefore answered every query with full confidence.
    """

    verdict = result.evidence.verdict if (with_verdict and result.evidence) else None
    confidence = result.evidence.confidence if (with_verdict and result.evidence) else None
    if use_localization and result.localization is not None:
        localization = result.localization
        return RankedMoment(
            video_id=result.video_id,
            segment_start=result.start_time,
            segment_end=result.end_time,
            time_point=localization.peak_time,
            interval_start=localization.start_time,
            interval_end=localization.end_time,
            score=result.score,
            verdict=verdict,
            confidence=confidence,
            precision=localization.precision,
        )
    return RankedMoment(
        video_id=result.video_id,
        segment_start=result.start_time,
        segment_end=result.end_time,
        time_point=result.start_time,
        interval_start=result.start_time,
        interval_end=result.end_time,
        score=result.score,
        verdict=verdict,
        confidence=confidence,
        precision="segment",
    )


class ChallengeMethods:
    """The three systems under comparison, sharing one index and one planner."""

    def __init__(
        self,
        repository: SQLiteRepository,
        planner: QueryPlanningService,
        *,
        engine: LocalHybridSearchEngine | None = None,
        resolution_margin: float = 0.25,
    ) -> None:
        self.repository = repository
        self.planner = planner
        self.engine = engine or LocalHybridSearchEngine(repository)
        self.resolution_margin = resolution_margin

    def _retrieve(self, plan: QueryPlan, request: SearchRequest) -> list[SearchResult]:
        return self.engine.search(request, plan_override=plan).results

    def review1_segment_start(
        self, query: str, planning: QueryPlanningOutcome, video_ids: list[str], limit: int
    ) -> MethodRun:
        request = SearchRequest(
            query=query, video_ids=video_ids, limit=limit, ranking="hybrid", localize=False
        )
        combined = {
            item.segment_id: item for item in self._retrieve(planning.plan, request)
        }
        for alternative in planning.alternatives:
            for item in self._retrieve(alternative, request):
                previous = combined.get(item.segment_id)
                if previous is None or item.score > previous.score:
                    combined[item.segment_id] = item
        ranked = sorted(
            combined.values(), key=lambda item: (-item.score, item.start_time, item.video_id)
        )[:limit]
        return MethodRun(
            ranked=[
                _moment_from_result(item, use_localization=False, with_verdict=False)
                for item in ranked
            ],
            selected_intent=planning.plan.intent,
            selected_hypothesis_id=None,
            interpretation="single",
        )

    def review2_localized(
        self, query: str, planning: QueryPlanningOutcome, video_ids: list[str], limit: int
    ) -> MethodRun:
        request = SearchRequest(
            query=query, video_ids=video_ids, limit=limit, ranking="hybrid", localize=True
        )
        combined = {
            item.segment_id: item for item in self._retrieve(planning.plan, request)
        }
        for alternative in planning.alternatives:
            for item in self._retrieve(alternative, request):
                previous = combined.get(item.segment_id)
                if previous is None or item.score > previous.score:
                    combined[item.segment_id] = item
        ranked = sorted(
            combined.values(), key=lambda item: (-item.score, item.start_time, item.video_id)
        )[:limit]
        # Localization-only ablation: Review-1 ranking and no abstention layer.
        return MethodRun(
            ranked=[
                _moment_from_result(item, use_localization=True, with_verdict=False)
                for item in ranked
            ],
            selected_intent=planning.plan.intent,
            selected_hypothesis_id=None,
            interpretation="single",
        )

    def review2_full(
        self, query: str, planning: QueryPlanningOutcome, video_ids: list[str], limit: int
    ) -> MethodRun:
        request = SearchRequest(
            query=query,
            video_ids=video_ids,
            limit=limit,
            ranking="reliability",
            localize=True,
            collapse_duplicate_evidence=True,
        )
        hypotheses = list(planning.hypotheses)
        if not hypotheses:
            response = self.engine.search(request, plan_override=planning.plan)
            return MethodRun(
                ranked=[
                    _moment_from_result(item, use_localization=True) for item in response.results
                ],
                selected_intent=planning.plan.intent,
                selected_hypothesis_id=None,
                interpretation=response.interpretation,
            )
        retrieved = {
            item.hypothesis_id: self._retrieve(item.plan, request) for item in hypotheses
        }
        fused = fuse_hypotheses(
            hypotheses, retrieved, limit=limit, resolution_margin=self.resolution_margin
        )
        selected = next(
            (
                item
                for item in fused.hypotheses
                if item.hypothesis_id == fused.selected_hypothesis_id
            ),
            None,
        )
        offered = (
            [item for item in fused.hypotheses if item.result_count > 0]
            if fused.interpretation == "clarification_suggested"
            else ([selected] if selected else [])
        )
        return MethodRun(
            ranked=[_moment_from_result(item, use_localization=True) for item in fused.results],
            selected_intent=selected.plan.intent if selected else None,
            selected_hypothesis_id=fused.selected_hypothesis_id,
            offered_hypothesis_ids=[item.hypothesis_id for item in offered],
            offered_intents=[item.plan.intent for item in offered],
            interpretation=fused.interpretation,
        )


def _hit(moment: RankedMoment, relevant: dict[str, Any], tolerance: float) -> bool:
    overlaps = min(moment.interval_end, relevant["end_time"]) > max(
        moment.interval_start, relevant["start_time"]
    )
    close = abs(moment.time_point - float(relevant["peak_time"])) <= tolerance
    return overlaps or close


def _iou(moment: RankedMoment, relevant: dict[str, Any]) -> float:
    intersection = max(
        0.0,
        min(moment.interval_end, relevant["end_time"])
        - max(moment.interval_start, relevant["start_time"]),
    )
    union = max(moment.interval_end, relevant["end_time"]) - min(
        moment.interval_start, relevant["start_time"]
    )
    return intersection / union if union > 0 else 0.0


def _query_metrics(
    item: dict[str, Any], run: MethodRun, source_ids: dict[str, str]
) -> dict[str, Any]:
    tolerance = float(item.get("tolerance_seconds", 2.0))
    negative = bool(item.get("negative", False))
    ranked = [
        moment
        for moment in run.ranked
        if source_ids.get(moment.video_id) == item["source_video_id"]
    ]
    abstained = run.interpretation == "insufficient_evidence" or not ranked
    top_supported = bool(
        ranked and (ranked[0].verdict == "supported" or ranked[0].verdict is None)
    )
    metrics: dict[str, Any] = {
        "interpretation": run.interpretation,
        "abstained": abstained,
    }
    if negative:
        metrics.update(
            {
                "negative_rejected": abstained,
                "confident_wrong": bool(ranked) and top_supported and not abstained,
            }
        )
        return metrics

    relevant = item.get("relevant", [])
    first_rank = 0
    best_relevant: dict[str, Any] | None = None
    for rank, moment in enumerate(ranked, start=1):
        matches = [target for target in relevant if _hit(moment, target, tolerance)]
        if matches:
            first_rank = rank
            best_relevant = min(
                matches, key=lambda target: abs(moment.time_point - float(target["peak_time"]))
            )
            break
    top1_hit = first_rank == 1
    top1 = ranked[0] if ranked else None
    timestamp_error: float | None = None
    within_tolerance = False
    temporal_iou = 0.0
    if top1 is not None and top1_hit and best_relevant is not None:
        timestamp_error = abs(top1.time_point - float(best_relevant["peak_time"]))
        within_tolerance = timestamp_error <= tolerance
        temporal_iou = _iou(top1, best_relevant)
    elif top1 is not None:
        nearest = min(
            (abs(top1.time_point - float(target["peak_time"])) for target in relevant),
            default=None,
        )
        timestamp_error = nearest

    expected_intents = set(item.get("expected_intents", []) or [])
    expected_ids = set(item.get("expected_hypothesis_ids", []) or [])
    intent_strict: bool | None = None
    intent_lenient: bool | None = None
    if expected_intents or expected_ids:
        strict_ok = (run.selected_intent in expected_intents) or (
            run.selected_hypothesis_id in expected_ids
        )
        offered_ok = bool(expected_intents & set(run.offered_intents)) or bool(
            expected_ids & set(run.offered_hypothesis_ids)
        )
        intent_strict = strict_ok and run.interpretation != "clarification_suggested"
        intent_lenient = strict_ok or offered_ok

    metrics.update(
        {
            "precision_at_1": float(top1_hit),
            "hit_at_5": float(0 < first_rank <= 5),
            "mrr": 1.0 / first_rank if first_rank else 0.0,
            "timestamp_error_seconds": (
                round(timestamp_error, 3) if timestamp_error is not None else None
            ),
            "within_tolerance": within_tolerance,
            "temporal_iou": round(temporal_iou, 4),
            "returned_precision": top1.precision if top1 else None,
            "confident_wrong": bool(top1 is not None and not top1_hit and top_supported),
            "false_abstention": abstained and any(
                _hit(moment, target, tolerance) for moment in run.ranked for target in relevant
            ),
            "intent_correct_strict": intent_strict,
            "intent_correct_lenient": intent_lenient,
        }
    )
    return metrics


def _rate(values: list[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    return round(sum(bool(value) for value in known) / len(known), 4) if known else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if not row["negative"]]
    negatives = [row for row in rows if row["negative"]]
    metrics = [row["metrics"] for row in positives]
    errors_on_hits = [
        row["metrics"]["timestamp_error_seconds"]
        for row in positives
        if row["metrics"]["precision_at_1"]
        and row["metrics"]["timestamp_error_seconds"] is not None
    ]
    summary: dict[str, Any] = {
        "positive_query_count": len(positives),
        "negative_query_count": len(negatives),
        "precision_at_1": round(mean(m["precision_at_1"] for m in metrics), 4) if metrics else None,
        "hit_at_5": round(mean(m["hit_at_5"] for m in metrics), 4) if metrics else None,
        "mrr": round(mean(m["mrr"] for m in metrics), 4) if metrics else None,
        "within_tolerance_rate": _rate([m["within_tolerance"] for m in metrics]),
        "temporal_iou": round(mean(m["temporal_iou"] for m in metrics), 4) if metrics else None,
        "mean_abs_timestamp_error_on_hits": (
            round(mean(errors_on_hits), 3) if errors_on_hits else None
        ),
        "median_abs_timestamp_error_on_hits": (
            round(median(errors_on_hits), 3) if errors_on_hits else None
        ),
        "intent_accuracy_strict": _rate([m["intent_correct_strict"] for m in metrics]),
        "intent_accuracy_lenient": _rate([m["intent_correct_lenient"] for m in metrics]),
        "confident_wrong_rate": _rate(
            [m["confident_wrong"] for m in metrics]
            + [row["metrics"]["confident_wrong"] for row in negatives]
        ),
        "false_abstention_rate": _rate([m["false_abstention"] for m in metrics]),
        "negative_rejection_rate": _rate(
            [row["metrics"]["negative_rejected"] for row in negatives]
        ),
        "clarification_rate": _rate(
            [row["metrics"]["interpretation"] == "clarification_suggested" for row in rows]
        ),
        "mean_latency_ms": round(mean(row["latency_ms"] for row in rows), 3) if rows else None,
        "returned_precision_histogram": dict(
            Counter(
                str(m["returned_precision"]) for m in metrics if m["returned_precision"] is not None
            )
        ),
    }
    return summary


def evaluate_challenge(
    repository: SQLiteRepository,
    dataset_path: Path,
    *,
    planner: QueryPlanningService,
    limit: int = 5,
    engine: LocalHybridSearchEngine | None = None,
    resolution_margin: float = 0.25,
) -> dict[str, Any]:
    manifest = load_manifest(dataset_path)
    warnings = validate_manifest(manifest)
    queries: list[dict[str, Any]] = manifest["queries"]
    source_ids = _source_ids(repository)
    canonical_by_source = {source: canonical for canonical, source in source_ids.items()}
    missing = sorted(
        {str(item["source_video_id"]) for item in queries} - set(canonical_by_source)
    )
    if missing:
        raise ChallengeManifestError(
            "manifest references videos that are not indexed: " + ", ".join(missing)
        )

    methods = ChallengeMethods(
        repository, planner, engine=engine, resolution_margin=resolution_margin
    )
    runners: dict[str, Callable[[str, QueryPlanningOutcome, list[str], int], MethodRun]] = {
        "review1_segment_start": methods.review1_segment_start,
        "review2_localized": methods.review2_localized,
        "review2_full": methods.review2_full,
    }
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in queries:
        planning = planner.plan(item["query"])
        video_ids = [canonical_by_source[str(item["source_video_id"])]]
        for method, runner in runners.items():
            started = time.perf_counter()
            run = runner(str(item["query"]), planning, video_ids, limit)
            latency_ms = (time.perf_counter() - started) * 1000
            metrics = _query_metrics(item, run, source_ids)
            rows[method].append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "category": item["category"],
                    "source_video_id": item["source_video_id"],
                    "negative": bool(item.get("negative", False)),
                    "latency_ms": round(latency_ms, 3),
                    "metrics": metrics,
                    "selected_intent": run.selected_intent,
                    "selected_hypothesis_id": run.selected_hypothesis_id,
                    "offered_hypothesis_ids": run.offered_hypothesis_ids,
                    "results": [
                        {
                            "source_video_id": source_ids.get(moment.video_id, moment.video_id),
                            "segment_start": moment.segment_start,
                            "segment_end": moment.segment_end,
                            "time_point": round(moment.time_point, 3),
                            "interval": [
                                round(moment.interval_start, 3),
                                round(moment.interval_end, 3),
                            ],
                            "score": round(moment.score, 4),
                            "verdict": moment.verdict,
                            "confidence": moment.confidence,
                            "precision": moment.precision,
                        }
                        for moment in run.ranked
                    ],
                }
            )

    summary: dict[str, Any] = {}
    for method, method_rows in rows.items():
        summary[method] = {
            "overall": _summarize(method_rows),
            "by_category": {
                category: _summarize([row for row in method_rows if row["category"] == category])
                for category in sorted({row["category"] for row in method_rows})
            },
        }
    comparison = {}
    if "review1_segment_start" in summary and "review2_full" in summary:
        baseline = summary["review1_segment_start"]["overall"]
        improved = summary["review2_full"]["overall"]
        for key in (
            "precision_at_1",
            "within_tolerance_rate",
            "temporal_iou",
            "mean_abs_timestamp_error_on_hits",
            "intent_accuracy_strict",
            "intent_accuracy_lenient",
            "confident_wrong_rate",
            "negative_rejection_rate",
        ):
            if baseline.get(key) is not None and improved.get(key) is not None:
                comparison[key] = {
                    "review1": baseline[key],
                    "review2": improved[key],
                    "delta": round(improved[key] - baseline[key], 4),
                }
    return {
        "schema_version": "challenge-results-1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(Path(dataset_path).as_posix()),
        "dataset_schema_version": manifest["schema_version"],
        "query_count": len(queries),
        "category_counts": dict(Counter(str(item["category"]) for item in queries)),
        "manifest_warnings": warnings,
        "cutoff": limit,
        "planner_status_histogram": dict(
            Counter(planner.plan(item["query"]).status for item in queries)
        ),
        "summary": summary,
        "review1_vs_review2": comparison,
        "queries": dict(rows),
    }


def write_challenge_results(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
