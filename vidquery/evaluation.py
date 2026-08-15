from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .ava_gnn80 import AVA80_LABELS
from .domain import QueryPlan, SearchRequest
from .search import (
    GraphQueryRunner,
    LocalHybridSearchEngine,
    Neo4jGraphSearchEngine,
    StructuredQueryParser,
    _tokens,
)
from .storage import SQLiteRepository


@dataclass(frozen=True, slots=True)
class RankedSegment:
    video_id: str
    start_time: float
    end_time: float
    score: float


def _source_ids(repository: SQLiteRepository) -> dict[str, str]:
    """Map canonical UUIDs to stable source IDs used by evaluation manifests."""
    return {
        video.video_id: Path(video.original_filename).stem
        for video in repository.list_videos()
    }


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union else 0.0


def _matches(result: RankedSegment, relevant: dict[str, Any], source_ids: dict[str, str]) -> bool:
    if source_ids.get(result.video_id) != relevant["source_video_id"]:
        return False
    tolerance = float(relevant.get("tolerance_seconds", 0.0))
    overlaps = min(result.end_time, relevant["end_time"]) > max(
        result.start_time, relevant["start_time"]
    )
    close = abs(result.start_time - relevant["start_time"]) <= tolerance
    return overlaps or close


def _source_for_query(item: dict[str, Any]) -> str:
    source = item.get("relevant_video_id")
    if source:
        return str(source)
    relevant = item.get("relevant", [])
    if relevant:
        return str(relevant[0]["source_video_id"])
    raise ValueError(f"query {item.get('id', '<unknown>')} has no source-video scope")


def _video_scope(item: dict[str, Any], canonical_by_source: dict[str, str]) -> list[str]:
    source = _source_for_query(item)
    try:
        return [canonical_by_source[source]]
    except KeyError as exc:
        raise ValueError(
            f"evaluation source video {source!r} is not present in the SQLite index"
        ) from exc


def _transcript_lexical_rank(
    repository: SQLiteRepository,
    query: str,
    *,
    video_ids: list[str] | None = None,
    limit: int,
) -> list[RankedSegment]:
    """Narrow transcript-only token-overlap baseline."""
    plan = StructuredQueryParser().parse(query)
    terms = plan.spoken_terms or [token for token in _tokens(query) if len(token) > 2]
    ranked: list[RankedSegment] = []
    if not terms:
        return ranked
    for segment in repository.list_segments(video_ids):
        transcript_tokens = set(_tokens(segment.transcript))
        score = sum(term in transcript_tokens for term in terms) / len(terms)
        if score:
            ranked.append(
                RankedSegment(
                    video_id=segment.video_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    score=score,
                )
            )
    ranked.sort(key=lambda item: (-item.score, item.start_time, item.video_id))
    return ranked[:limit]


class TranscriptSemanticRanker:
    """SentenceTransformer transcript-only baseline with a frozen model.

    Model loading is local-only by default. Set ``ALLOW_MODEL_DOWNLOADS=true``
    explicitly to permit the SentenceTransformers library to fetch the model.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed; install the evaluation extra"
            ) from exc

        allow_downloads = os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=not allow_downloads)
        self.segments = [
            segment for segment in repository.list_segments() if segment.transcript.strip()
        ]
        texts = [segment.transcript for segment in self.segments]
        if texts:
            self.embeddings = self.model.encode(
                texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            dimensions = int(self.model.get_sentence_embedding_dimension() or 0)
            self.embeddings = np.empty((0, dimensions), dtype=np.float32)

    def rank(self, query: str, *, video_ids: list[str], limit: int) -> list[RankedSegment]:
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_vector
        scope = set(video_ids)
        ranked = [
            RankedSegment(
                video_id=segment.video_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                score=max(0.0, min(1.0, float(score))),
            )
            for segment, score in zip(self.segments, scores, strict=True)
            if segment.video_id in scope and score > 0
        ]
        ranked.sort(key=lambda item: (-item.score, item.start_time, item.video_id))
        return ranked[:limit]


def _visual_entity_rank(
    repository: SQLiteRepository,
    query: str,
    *,
    video_ids: list[str],
    limit: int,
) -> list[RankedSegment]:
    entities = StructuredQueryParser().parse(query).visual_entities
    if not entities:
        return []
    ranked: list[RankedSegment] = []
    for segment in repository.list_segments(video_ids):
        present = {item.lower() for item in segment.entities}
        if not all(entity in present for entity in entities):
            continue
        ranked.append(
            RankedSegment(
                video_id=segment.video_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                score=1.0,
            )
        )
    ranked.sort(key=lambda item: (item.start_time, item.video_id))
    return ranked[:limit]


class RelationshipTupleAblationParser(StructuredQueryParser):
    """Reproduce the old entity-plus-predicate co-occurrence behavior."""

    def parse(self, query: str) -> QueryPlan:
        return super().parse(query).model_copy(update={"relationship_tuples": []})


class LearnedActionAblationParser(StructuredQueryParser):
    """Remove only action constraints supplied by the learned checkpoint."""

    def __init__(self, learned_actions: set[str]):
        self.learned_actions = learned_actions

    def parse(self, query: str) -> QueryPlan:
        plan = super().parse(query)
        return plan.model_copy(
            update={
                "actions": [
                    action for action in plan.actions if action not in self.learned_actions
                ]
            }
        )


def _engine_rank(
    engine: LocalHybridSearchEngine | Neo4jGraphSearchEngine,
    query: str,
    *,
    video_ids: list[str],
    limit: int,
) -> list[RankedSegment]:
    response = engine.search(SearchRequest(query=query, video_ids=video_ids, limit=limit))
    return [
        RankedSegment(
            video_id=result.video_id,
            start_time=result.start_time,
            end_time=result.end_time,
            score=result.score,
        )
        for result in response.results
    ]


def _learned_action_rank(
    repository: SQLiteRepository,
    prediction_index: dict[tuple[str, float], dict[str, float]],
    source_ids: dict[str, str],
    query: str,
    *,
    video_ids: list[str],
    limit: int,
) -> list[RankedSegment]:
    plan = StructuredQueryParser().parse(query)
    required = {action for action in plan.actions if action in AVA80_LABELS}
    if not required:
        return []
    source = source_ids[video_ids[0]]
    candidates: set[str] = set()
    confidences: dict[str, float] = {}
    for segment in repository.list_segments(video_ids):
        scores = prediction_index.get((source, segment.start_time), {})
        if not required.issubset(scores):
            continue
        candidates.add(segment.segment_id)
        confidences[segment.segment_id] = sum(scores[action] for action in required) / len(
            required
        )
    engine = LocalHybridSearchEngine(
        repository, parser=LearnedActionAblationParser(required)
    )
    response = engine.search(
        SearchRequest(query=query, video_ids=video_ids, limit=50),
        candidate_segment_ids=candidates,
    )
    ranked = [
        RankedSegment(
            video_id=result.video_id,
            start_time=result.start_time,
            end_time=result.end_time,
            score=min(
                1.0,
                0.8 * result.score + 0.2 * confidences.get(result.segment_id, 0.0),
            ),
        )
        for result in response.results
    ]
    ranked.sort(key=lambda item: (-item.score, item.start_time, item.video_id))
    return ranked[:limit]


def _query_metrics(
    ranked: list[RankedSegment], relevant: list[dict[str, Any]], source_ids: dict[str, str]
) -> dict[str, float]:
    if not relevant:
        return {
            "precision_at_1": 0.0,
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "timestamp_tolerance_rate": 0.0,
            "temporal_iou": 0.0,
            "negative_rejection": float(not ranked),
        }

    matched_relevant: set[int] = set()
    first_rank = 0
    first_iou = 0.0
    tolerance_hit = 0.0
    for rank, result in enumerate(ranked, start=1):
        candidates = [
            (index, target)
            for index, target in enumerate(relevant)
            if index not in matched_relevant and _matches(result, target, source_ids)
        ]
        if not candidates:
            continue
        # A tolerance window can make one result eligible for adjacent labelled
        # windows. Attribute it to the best temporal overlap, not JSON order.
        index, target = max(
            candidates,
            key=lambda candidate: (
                _overlap(
                    result.start_time,
                    result.end_time,
                    candidate[1]["start_time"],
                    candidate[1]["end_time"],
                ),
                -abs(result.start_time - candidate[1]["start_time"]),
            ),
        )
        matched_relevant.add(index)
        if not first_rank:
            first_rank = rank
            first_iou = _overlap(
                result.start_time,
                result.end_time,
                target["start_time"],
                target["end_time"],
            )
            tolerance_hit = float(
                abs(result.start_time - target["start_time"])
                <= float(target.get("tolerance_seconds", 0.0))
            )

    top1_hits = sum(
        any(_matches(result, target, source_ids) for target in relevant)
        for result in ranked[:1]
    )
    top5_hits = sum(
        any(_matches(result, target, source_ids) for target in relevant)
        for result in ranked[:5]
    )
    return {
        "precision_at_1": float(top1_hits),
        "precision_at_5": top5_hits / 5.0,
        "recall_at_5": len(matched_relevant) / len(relevant),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "timestamp_tolerance_rate": tolerance_hit,
        "temporal_iou": first_iou,
        "negative_rejection": 0.0,
    }


METRIC_NAMES = (
    "precision_at_1",
    "precision_at_5",
    "recall_at_5",
    "mrr",
    "timestamp_tolerance_rate",
    "temporal_iou",
)


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if not row["negative"]]
    negatives = [row for row in rows if row["negative"]]

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        positive_selected = [row for row in selected if not row["negative"]]
        negative_selected = [row for row in selected if row["negative"]]
        values: dict[str, Any] = {
            metric: round(
                mean(row["metrics"][metric] for row in positive_selected), 4
            )
            if positive_selected
            else None
            for metric in METRIC_NAMES
        }
        values.update(
            {
                "mean_search_latency_ms": round(
                    mean(row["metrics"]["latency_ms"] for row in selected), 4
                )
                if selected
                else None,
                # Backward-compatible key used by the earlier smoke result schema.
                "latency_ms": round(
                    mean(row["metrics"]["latency_ms"] for row in selected), 4
                )
                if selected
                else None,
                "negative_rejection_rate": round(
                    mean(row["metrics"]["negative_rejection"] for row in negative_selected),
                    4,
                )
                if negative_selected
                else None,
                "positive_query_count": len(positive_selected),
                "negative_query_count": len(negative_selected),
            }
        )
        return values

    categories = {
        category: aggregate([row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    }
    videos = {
        source: aggregate([row for row in rows if row["source_video_id"] == source])
        for source in sorted({row["source_video_id"] for row in rows})
    }
    return {
        "status": "evaluated",
        "overall": aggregate(rows),
        "by_category": categories,
        "by_video": videos,
        "positive_query_count": len(positives),
        "negative_query_count": len(negatives),
    }


def evaluate(
    repository: SQLiteRepository,
    dataset_path: Path,
    *,
    limit: int = 5,
    graph_runner: GraphQueryRunner | None = None,
    learned_action_index: dict[tuple[str, float], dict[str, float]] | None = None,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    queries = dataset.get("queries", [])
    if not queries:
        raise ValueError("evaluation dataset contains no queries")

    source_ids = _source_ids(repository)
    canonical_by_source = {source: canonical for canonical, source in source_ids.items()}
    for item in queries:
        _video_scope(item, canonical_by_source)

    exact_local = LocalHybridSearchEngine(repository)
    ablated_local = LocalHybridSearchEngine(
        repository, parser=RelationshipTupleAblationParser()
    )
    rankers: dict[
        str, Callable[[str, list[str], int], list[RankedSegment]]
    ] = {
        "transcript_lexical": lambda query, video_ids, cutoff: _transcript_lexical_rank(
            repository, query, video_ids=video_ids, limit=cutoff
        ),
        "visual_entity": lambda query, video_ids, cutoff: _visual_entity_rank(
            repository, query, video_ids=video_ids, limit=cutoff
        ),
        "hybrid_without_relationship_tuples": (
            lambda query, video_ids, cutoff: _engine_rank(
                ablated_local, query, video_ids=video_ids, limit=cutoff
            )
        ),
        # Retained so old smoke/regression consumers keep a stable method name.
        "hybrid": lambda query, video_ids, cutoff: _engine_rank(
            exact_local, query, video_ids=video_ids, limit=cutoff
        ),
    }
    method_status: dict[str, dict[str, str]] = {}
    method_filters: dict[str, Callable[[dict[str, Any]], bool]] = {}

    try:
        semantic = TranscriptSemanticRanker(repository, semantic_model_name)
        rankers["transcript_semantic"] = lambda query, video_ids, cutoff: semantic.rank(
            query, video_ids=video_ids, limit=cutoff
        )
    except Exception as exc:  # pragma: no cover - environment-dependent optional model
        method_status["transcript_semantic"] = {
            "status": "not_evaluated",
            "reason": f"semantic model unavailable: {type(exc).__name__}: {exc}",
        }

    if graph_runner is None:
        method_status["exact_relationship_aware_graph"] = {
            "status": "not_evaluated",
            "reason": "Neo4j graph runner was not supplied",
        }
    else:
        graph_engine = Neo4jGraphSearchEngine(repository, graph_runner)
        rankers["exact_relationship_aware_graph"] = (
            lambda query, video_ids, cutoff: _engine_rank(
                graph_engine, query, video_ids=video_ids, limit=cutoff
            )
        )

    if learned_action_index is None:
        method_status["learned_model_enhanced"] = {
            "status": "not_evaluated",
            "reason": "no validated learned-model prediction index was supplied",
        }
    else:
        rankers["learned_model_enhanced"] = (
            lambda query, video_ids, cutoff: _learned_action_rank(
                repository,
                learned_action_index,
                source_ids,
                query,
                video_ids=video_ids,
                limit=cutoff,
            )
        )
        method_filters["learned_model_enhanced"] = lambda item: (
            item.get("expected_action") in AVA80_LABELS
            and item.get("category") in {"ava_action", "multimodal"}
        )

    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method, ranker in rankers.items():
        selected_queries = [
            item
            for item in queries
            if method_filters.get(method, lambda _item: True)(item)
        ]
        for item in selected_queries:
            video_ids = _video_scope(item, canonical_by_source)
            started = time.perf_counter()
            ranked = ranker(item["query"], video_ids, limit)
            latency_ms = (time.perf_counter() - started) * 1000
            relevant = item.get("relevant", [])
            metrics = _query_metrics(ranked, relevant, source_ids)
            source = _source_for_query(item)
            methods[method].append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "query": item["query"],
                    "source_video_id": source,
                    "negative": bool(item.get("negative", not relevant)),
                    "metrics": {**metrics, "latency_ms": round(latency_ms, 3)},
                    "results": [
                        {
                            "source_video_id": source_ids.get(
                                result.video_id, result.video_id
                            ),
                            "start_time": result.start_time,
                            "end_time": result.end_time,
                            "score": round(result.score, 4),
                        }
                        for result in ranked
                    ],
                }
            )

    summaries: dict[str, Any] = {
        method: _summary_for_rows(rows) for method, rows in methods.items()
    }
    if "learned_model_enhanced" in summaries:
        learned_summary = summaries["learned_model_enhanced"]
        learned_summary["status"] = "evaluated_supported_subset"
        learned_summary["eligible_query_count"] = len(
            methods["learned_model_enhanced"]
        )
        learned_summary["corpus_query_count"] = len(queries)
        learned_summary["query_coverage"] = round(
            len(methods["learned_model_enhanced"]) / len(queries), 4
        )
    summaries.update(method_status)

    return {
        "schema_version": "2.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(Path(dataset_path).as_posix()),
        "dataset_schema_version": str(dataset.get("schema_version", "unspecified")),
        "query_count": len(queries),
        "positive_query_count": sum(not item.get("negative", False) for item in queries),
        "negative_query_count": sum(bool(item.get("negative", False)) for item in queries),
        "indexed_video_count": len(repository.list_videos()),
        "indexed_segment_count": len(repository.list_segments()),
        "cutoff": limit,
        "semantic_model": semantic_model_name,
        "summary": summaries,
        "queries": dict(methods),
    }


def write_evaluation(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
