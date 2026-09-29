"""End-to-end diagnostics for action/relationship retrieval failures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Settings, get_settings
from .domain import CanonicalSegment, QueryPlan, SearchRequest
from .hypotheses import generate_hypotheses
from .query_planner import QueryPlanningService
from .reliability import (
    SOURCE_RELIABILITY,
    reliability_prior,
)
from .search import LocalHybridSearchEngine, StructuredQueryParser
from .storage import SQLiteRepository


@dataclass
class FailureCategory:
    code: str
    detail: str


@dataclass
class DiagnoseReport:
    query: str
    video_ids: list[str]
    deterministic_plan: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    planner_status: str | None
    index_action_inventory: dict[str, Any]
    index_relationship_inventory: dict[str, Any]
    raw_action_hits: list[dict[str, Any]]
    raw_relationship_hits: list[dict[str, Any]]
    ranking_modes: dict[str, Any]
    failure_categories: list[FailureCategory] = field(default_factory=list)
    summary: str = ""


def _plan_dict(plan: QueryPlan) -> dict[str, Any]:
    return plan.model_dump()


def _inventory(segments: list[CanonicalSegment]) -> tuple[dict[str, Any], dict[str, Any]]:
    action_counts: dict[str, int] = {}
    action_max: dict[str, float] = {}
    action_sources: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    relation_sources: dict[str, int] = {}
    for segment in segments:
        for action in segment.actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        for item in segment.processing_metadata.get("action_instances", []) or []:
            label = str(item.get("action", ""))
            conf = float(item.get("confidence", 0.0))
            source = str(item.get("source_method", "unknown"))
            action_sources[source] = action_sources.get(source, 0) + 1
            action_max[label] = max(action_max.get(label, 0.0), conf)
        for relationship in segment.relationships:
            predicate = relationship.predicate
            relation_counts[predicate] = relation_counts.get(predicate, 0) + 1
            source = relationship.source_method.value
            relation_sources[source] = relation_sources.get(source, 0) + 1
    return (
        {
            "unique_actions": sorted(action_counts),
            "segment_action_counts": action_counts,
            "max_confidence": action_max,
            "sources": action_sources,
        },
        {
            "unique_predicates": sorted(relation_counts),
            "predicate_counts": relation_counts,
            "sources": relation_sources,
        },
    )


def _raw_action_hits(
    segments: list[CanonicalSegment], actions: list[str]
) -> list[dict[str, Any]]:
    wanted = {item.lower() for item in actions}
    hits: list[dict[str, Any]] = []
    if not wanted:
        return hits
    for segment in segments:
        instances = [
            item
            for item in segment.processing_metadata.get("action_instances", []) or []
            if str(item.get("action", "")).lower() in wanted
        ]
        labels = [action for action in segment.actions if action.lower() in wanted]
        if not labels and not instances:
            continue
        best = max(instances, key=lambda item: float(item.get("confidence", 0.0)), default=None)
        hits.append(
            {
                "segment_id": segment.segment_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "actions": labels,
                "best_instance": best,
                "instance_count": len(instances),
            }
        )
    hits.sort(
        key=lambda item: (
            -(float((item["best_instance"] or {}).get("confidence", 0.0))),
            item["start_time"],
        )
    )
    return hits


def _raw_relationship_hits(
    segments: list[CanonicalSegment], plan: QueryPlan
) -> list[dict[str, Any]]:
    from .relationships import matching_relationships

    hits: list[dict[str, Any]] = []
    queries = list(plan.relationship_tuples)
    if not queries and plan.relationships:
        from .domain import RelationshipQuery
        from .relationships import relationship_directionality

        queries = [
            RelationshipQuery(
                subject="person",
                predicate=predicate,
                object="person",
                directionality=relationship_directionality(predicate),
            )
            for predicate in plan.relationships
        ]
    if not queries:
        return hits
    for segment in segments:
        matched = []
        for relation_query in queries:
            matches = matching_relationships(segment, relation_query)
            if matches:
                best = matches[0]
                matched.append(
                    {
                        "predicate": best.predicate,
                        "source_class": best.source_class,
                        "target_class": best.target_class,
                        "confidence": best.confidence,
                        "source_method": best.source_method.value,
                        "timestamp": best.timestamp,
                    }
                )
        if matched:
            hits.append(
                {
                    "segment_id": segment.segment_id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "matches": matched,
                }
            )
    hits.sort(key=lambda item: (-max(m["confidence"] for m in item["matches"]), item["start_time"]))
    return hits


def _search_mode(
    engine: LocalHybridSearchEngine,
    plan: QueryPlan,
    *,
    query: str,
    video_ids: list[str],
    limit: int,
    ranking: str,
) -> dict[str, Any]:
    request = SearchRequest(
        query=query,
        video_ids=video_ids or None,
        limit=limit,
        ranking=ranking,  # type: ignore[arg-type]
        localize=True,
        collapse_duplicate_evidence=True,
    )
    response = engine.search(request, plan_override=plan)
    return {
        "ranking": ranking,
        "interpretation": response.interpretation,
        "result_count": len(response.results),
        "results": [
            {
                "start_time": item.start_time,
                "end_time": item.end_time,
                "score": item.score,
                "peak_time": None if item.localization is None else item.localization.peak_time,
                "precision": None if item.localization is None else item.localization.precision,
                "verdict": None if item.evidence is None else item.evidence.verdict,
                "confidence": None if item.evidence is None else item.evidence.confidence,
                "explanation": item.match_reason,
                "contributions": (
                    []
                    if item.evidence is None
                    else [contribution.model_dump() for contribution in item.evidence.contributions]
                ),
            }
            for item in response.results
        ],
    }


def _classify(
    plan: QueryPlan,
    action_hits: list[dict[str, Any]],
    relationship_hits: list[dict[str, Any]],
    reliability_results: dict[str, Any],
    hybrid_results: dict[str, Any],
    action_inventory: dict[str, Any],
) -> list[FailureCategory]:
    categories: list[FailureCategory] = []
    if plan.actions:
        missing = [
            action
            for action in plan.actions
            if action not in action_inventory.get("max_confidence", {})
        ]
        if missing:
            categories.append(
                FailureCategory(
                    "A_DETECTION_FAILURE",
                    f"Requested actions never stored in the index: {missing}",
                )
            )
        elif not action_hits:
            categories.append(
                FailureCategory(
                    "B_STORAGE_OR_D_RETRIEVAL_FAILURE",
                    "Actions exist in inventory but no segment action hit was produced",
                )
            )
        elif reliability_results.get("result_count", 0) == 0 and hybrid_results.get(
            "result_count", 0
        ):
            categories.append(
                FailureCategory(
                    "F_RELIABILITY_FAILURE",
                    "Hybrid ranking returns hits but reliability ranking returns none",
                )
            )
        elif reliability_results.get("result_count", 0) == 0:
            categories.append(
                FailureCategory(
                    "E_RANKING_OR_F_RELIABILITY_FAILURE",
                    "Raw action evidence exists but ranked search returned no results",
                )
            )
    if plan.relationship_tuples or plan.relationships:
        if not relationship_hits:
            categories.append(
                FailureCategory(
                    "A_DETECTION_FAILURE",
                    "No stored relationships match the parsed relationship query",
                )
            )
        elif reliability_results.get("result_count", 0) == 0 and hybrid_results.get(
            "result_count", 0
        ):
            categories.append(
                FailureCategory(
                    "F_RELIABILITY_FAILURE",
                    "Relationship evidence ranks under hybrid but not reliability",
                )
            )
    if not plan.actions and not plan.relationship_tuples and not plan.relationships:
        if plan.spoken_terms:
            categories.append(
                FailureCategory(
                    "C_QUERY_UNDERSTANDING_FAILURE",
                    "Query collapsed to spoken/transcript terms instead of action/relationship",
                )
            )
    if not categories and reliability_results.get("result_count", 0):
        categories.append(
            FailureCategory(
                "OK_OR_G_LOCALIZATION_CHECK",
                "Evidence was retrieved; verify peak timestamps against manual ground truth",
            )
        )
    return categories


def diagnose_query(
    query: str,
    *,
    repository: SQLiteRepository | None = None,
    settings: Settings | None = None,
    video_ids: list[str] | None = None,
    limit: int = 5,
    use_planner: bool = True,
) -> DiagnoseReport:
    settings = settings or get_settings()
    repository = repository or SQLiteRepository(settings.database_path)
    engine = LocalHybridSearchEngine(
        repository,
        evidence_supported_threshold=settings.evidence_supported_threshold,
        evidence_weak_threshold=settings.evidence_weak_threshold,
    )
    parser = StructuredQueryParser()
    deterministic = parser.parse(query)
    hypotheses = [
        item.model_dump() for item in generate_hypotheses(query, deterministic)
    ]
    planner_status = None
    plan = deterministic
    if use_planner and settings.enable_query_planner:
        planner = QueryPlanningService(settings)
        outcome = planner.plan(query)
        planner_status = outcome.status
        plan = outcome.plan
        if outcome.hypotheses:
            hypotheses = [item.model_dump() for item in outcome.hypotheses]

    segments = repository.list_segments(video_ids=video_ids or None)
    action_inventory, relationship_inventory = _inventory(segments)
    action_hits = _raw_action_hits(segments, plan.actions)
    relationship_hits = _raw_relationship_hits(segments, plan)
    hybrid = _search_mode(
        engine, plan, query=query, video_ids=video_ids or [], limit=limit, ranking="hybrid"
    )
    reliability = _search_mode(
        engine,
        plan,
        query=query,
        video_ids=video_ids or [],
        limit=limit,
        ranking="reliability",
    )
    categories = _classify(
        plan, action_hits, relationship_hits, reliability, hybrid, action_inventory
    )
    summary_parts = [item.code for item in categories] or ["NO_ISSUE_CLASSIFIED"]
    return DiagnoseReport(
        query=query,
        video_ids=video_ids or [],
        deterministic_plan=_plan_dict(deterministic),
        hypotheses=hypotheses,
        planner_status=planner_status,
        index_action_inventory=action_inventory,
        index_relationship_inventory=relationship_inventory,
        raw_action_hits=action_hits[:20],
        raw_relationship_hits=relationship_hits[:20],
        ranking_modes={
            "hybrid": hybrid,
            "reliability": reliability,
            "source_reliability_priors": {
                key: SOURCE_RELIABILITY[key]
                for key in (
                    "gnn_action_model",
                    "relationship_gnn",
                    "bounding_box_geometry",
                    "whisper_transcript",
                    "yolo_detection",
                )
                if key in SOURCE_RELIABILITY
            },
            "action_prior_example": reliability_prior("gnn_action_model"),
        },
        failure_categories=categories,
        summary=",".join(summary_parts),
    )


def report_to_dict(report: DiagnoseReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["failure_categories"] = [asdict(item) for item in report.failure_categories]
    return payload


def format_report(report: DiagnoseReport) -> str:
    lines = [
        f"Query: {report.query!r}",
        f"Summary: {report.summary}",
        "",
        "Deterministic plan:",
        json.dumps(report.deterministic_plan, indent=2),
        "",
        "Hypotheses:",
        json.dumps(report.hypotheses, indent=2),
        "",
        "Index action inventory (max confidence):",
        json.dumps(report.index_action_inventory.get("max_confidence", {}), indent=2),
        "",
        "Raw action hits:",
        json.dumps(report.raw_action_hits[:10], indent=2),
        "",
        "Raw relationship hits:",
        json.dumps(report.raw_relationship_hits[:10], indent=2),
        "",
        "Hybrid ranking:",
        json.dumps(report.ranking_modes["hybrid"], indent=2),
        "",
        "Reliability ranking:",
        json.dumps(report.ranking_modes["reliability"], indent=2),
        "",
        "Failure categories:",
    ]
    for item in report.failure_categories:
        lines.append(f"- {item.code}: {item.detail}")
    lines.append("")
    lines.append(
        "Source reliability priors: "
        + json.dumps(report.ranking_modes["source_reliability_priors"])
    )
    return "\n".join(lines)
