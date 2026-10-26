"""Evaluate VidQuery on the Charades-STA + NA-VMR benchmark (one split per run).

Writes summary metrics, per-query rows, query plans, content-blind baselines,
and run metadata (priors, thresholds, encoder, indexed models, manifest hash).
Tune only on ``validation``; run ``test`` once per frozen configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from vidquery.config import get_settings
from vidquery.moment_evaluation import SPANS, content_blind_baselines, evaluate_moment_benchmark
from vidquery.query_planner import QueryPlanningService
from vidquery.reliability import load_reliability_priors, load_reliability_thresholds
from vidquery.search import configured_search_engine, require_semantic_encoder
from vidquery.storage import SQLiteRepository

INDEXED_MODEL_KEYS = (
    "whisper_decoding",
    "gnn_model_version",
    "gnn_checkpoint_sha256",
    "relationship_gnn_model_version",
    "relationship_gnn_checkpoint_sha256",
)


def _indexed_models(repository: SQLiteRepository) -> dict[str, dict[str, int]]:
    """Segment counts per value of each model-provenance key in the index.

    Indexes built before 26 October 2026 lack ``whisper_decoding`` (they used
    the temperature fallback) and the checkpoint hashes; those show as None.
    """

    counts: dict[str, Counter[str]] = {key: Counter() for key in INDEXED_MODEL_KEYS}
    for segment in repository.list_segments(None):
        for key in INDEXED_MODEL_KEYS:
            counts[key][str(segment.processing_metadata.get(key))] += 1
    return {key: dict(value) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("evaluation/charades_sta_navmr_benchmark.json")
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reliability-priors", default="default")
    parser.add_argument(
        "--spans",
        nargs="+",
        default=list(SPANS),
        help="native, segment, merge, expand:SECONDS, expand_peak:SECONDS",
    )
    parser.add_argument("--merge-ratio", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--kinds", nargs="+", default=None,
                        help="Restrict to query kinds (positive, in_domain_negative, ...)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = manifest["splits"][args.split]
    queries = [
        {**item, "duration": split["durations"].get(item["video"])}
        for item in split["queries"]
        if args.kinds is None or item["kind"] in args.kinds
    ]

    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    priors_path = None if args.reliability_priors == "default" else Path(args.reliability_priors)
    priors = load_reliability_priors(priors_path) if priors_path else None
    thresholds = load_reliability_thresholds(priors_path) if priors_path else None
    engine = configured_search_engine(
        repository, settings, reliability_priors=priors, thresholds=thresholds
    )
    require_semantic_encoder(engine, settings)
    planner = QueryPlanningService(replace(settings, enable_query_planner=False))
    started = datetime.now(UTC)
    result = evaluate_moment_benchmark(
        repository,
        queries,
        planner=planner,
        engine=engine,
        limit=args.limit,
        spans=args.spans,
        merge_ratio=args.merge_ratio,
        resolution_margin=settings.hypothesis_resolution_margin,
    )
    durations = {
        video: value
        for payload in manifest["splits"].values()
        for video, value in payload["durations"].items()
    }
    positives = {
        name: [item for item in payload["queries"] if item["answerable"]]
        for name, payload in manifest["splits"].items()
    }
    result["content_blind_baselines"] = content_blind_baselines(
        positives["validation"], positives["test"], durations,
        segment_duration=settings.segment_duration,
    )
    result["run_metadata"] = {
        "split": args.split,
        "query_count": len(queries),
        "query_kinds": dict(Counter(item["kind"] for item in queries)),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "reliability_priors": priors_path.as_posix() if priors_path else "default_hand_picked",
        "evidence_thresholds": [
            engine.evidence_weak_threshold,
            engine.evidence_supported_threshold,
        ],
        "spans": args.spans,
        "merge_ratio": args.merge_ratio,
        "limit": args.limit,
        "text_encoder": getattr(engine.encoder, "model_name", type(engine.encoder).__name__),
        "phonetic_match": settings.enable_phonetic_match,
        "database": settings.database_path.as_posix(),
        "indexed_gnn_action_model_versions": dict(
            Counter(
                str(segment.processing_metadata.get("gnn_model_version"))
                for segment in repository.list_segments(None)
            )
        ),
        "indexed_models": _indexed_models(repository),
        "search_settings": {
            "query_planner": "rule-based only (LLM planner disabled for evaluation)",
            "soft_constraint_matching": settings.soft_constraint_matching,
            "frame_semantic_retrieval": settings.enable_frame_semantic_retrieval,
            "frame_semantic_weight": settings.frame_semantic_weight,
            "frame_semantic_floor": settings.frame_semantic_floor,
            "frame_semantic_ceiling": settings.frame_semantic_ceiling,
            "neo4j": settings.enable_neo4j,
        },
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for key, summary in result["summary"].items():
        print(
            key,
            {name: summary.get(name) for name in (
                "precision_at_1", "r1_iou0.5", "r1_iou0.7", "r5_iou0.5", "mean_iou",
                "positive_answered_rate", "negative_rejection_rate_by_kind", "answerability_auroc",
                "aurc", "failure_stages")},
        )


if __name__ == "__main__":
    main()
