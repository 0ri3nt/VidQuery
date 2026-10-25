"""Collect calibration observations for learned reliability priors.

Two passes, both on one split (``calibration`` unless stated):

1. Observations (default): every contribution of every returned result, under
   the hand-picked priors, labelled correct/incorrect.  ``--limit 1`` restricts
   this to each query's top answer ("top-1 calibration").
2. Decisions (``--decisions-output``): the exact ``review2_full`` run the
   evaluator scores, under ``--priors`` (hand-picked when omitted).  Each row
   records the query's maximum fused confidence among returned results (the
   quantity the abstention rule thresholds) and whether the top answer is
   correct.  ``scripts/fit_reliability.py --decisions`` fits thresholds on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from vidquery.challenge_evaluation import (
    ChallengeManifestError,
    ChallengeMethods,
    _hit,
    _moment_from_result,
    load_manifest,
    source_video_id,
    validate_manifest,
)
from vidquery.config import get_settings
from vidquery.domain import SearchRequest
from vidquery.query_planner import QueryPlanningService
from vidquery.reliability import load_reliability_priors
from vidquery.search import configured_search_engine, require_semantic_encoder
from vidquery.storage import SQLiteRepository


def _manifest_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_queries(manifest: dict, split: str) -> list[dict]:
    videos = manifest.get("videos") or {}
    return [
        item
        for item in manifest["queries"]
        if (videos.get(str(item["source_video_id"])) or {}).get("split") == split
    ]


def _services(allow_hashing_fallback: bool, priors_path: Path | None = None):
    settings = get_settings()
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    planner = QueryPlanningService(replace(settings, enable_query_planner=False))
    priors = load_reliability_priors(priors_path) if priors_path is not None else None
    # Priors only: the decision pass measures confidences, so it must not
    # apply thresholds from an earlier calibration file.
    engine = configured_search_engine(
        repository,
        settings,
        reliability_priors=priors,
        thresholds=(settings.evidence_supported_threshold, settings.evidence_weak_threshold),
    )
    if not allow_hashing_fallback:
        require_semantic_encoder(engine, settings)
    return settings, repository, planner, engine


def collect_observations(
    manifest_path: Path, *, split: str, limit: int, allow_hashing_fallback: bool = False
) -> dict:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    # Observations are always collected under the hand-picked priors, so the
    # fitted values never depend on a previously learned file.
    _settings, repository, planner, engine = _services(allow_hashing_fallback)
    source_ids = {video.video_id: source_video_id(video) for video in repository.list_videos()}
    observations: list[dict] = []
    queries = _split_queries(manifest, split)

    for item in queries:
        source = str(item["source_video_id"])
        plan = planner.plan(str(item["query"])).plan
        response = engine.search(
            SearchRequest(
                query=str(item["query"]),
                video_ids=[video_id for video_id, value in source_ids.items() if value == source],
                limit=limit,
                ranking="reliability",
                localize=True,
            ),
            plan_override=plan,
        )
        for rank, result in enumerate(response.results, start=1):
            if source_ids.get(result.video_id) != source or result.evidence is None:
                continue
            moment = _moment_from_result(result, use_localization=True)
            correct = (not item.get("negative", False)) and any(
                _hit(moment, relevant, float(item.get("tolerance_seconds", 2.0)))
                for relevant in item.get("relevant", [])
            )
            for contribution in result.evidence.contributions:
                observations.append(
                    {
                        "query_id": item["id"],
                        "rank": rank,
                        "source_method": contribution.source_method,
                        "modality": contribution.modality,
                        "raw_score": contribution.raw_score,
                        "correct": correct,
                    }
                )

    return {
        "schema_version": "reliability-calibration-1.1",
        "split": split,
        "limit": limit,
        "manifest": str(manifest_path),
        "manifest_sha256": _manifest_checksum(manifest_path),
        "created_at": datetime.now(UTC).isoformat(),
        "query_count": len(queries),
        "observation_count": len(observations),
        "observations": observations,
    }


def collect_decisions(
    manifest_path: Path,
    *,
    split: str,
    limit: int,
    priors_path: Path | None,
    allow_hashing_fallback: bool = False,
) -> dict:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    settings, repository, planner, engine = _services(allow_hashing_fallback, priors_path)
    methods = ChallengeMethods(
        repository,
        planner,
        engine=engine,
        resolution_margin=settings.hypothesis_resolution_margin,
    )
    source_ids = {video.video_id: source_video_id(video) for video in repository.list_videos()}
    canonical = {source: video for video, source in source_ids.items()}
    rows: list[dict] = []
    for item in _split_queries(manifest, split):
        source = str(item["source_video_id"])
        planning = planner.plan(str(item["query"]))
        run = methods.review2_full(str(item["query"]), planning, [canonical[source]], limit)
        ranked = [moment for moment in run.ranked if source_ids.get(moment.video_id) == source]
        confidences = [float(moment.confidence or 0.0) for moment in ranked]
        tolerance = float(item.get("tolerance_seconds", 2.0))
        top1_correct = bool(
            ranked
            and not item.get("negative", False)
            and any(_hit(ranked[0], target, tolerance) for target in item.get("relevant", []))
        )
        rows.append(
            {
                "query_id": item["id"],
                "category": item["category"],
                "negative": bool(item.get("negative", False)),
                "result_count": len(ranked),
                "max_confidence": max(confidences) if confidences else None,
                "top1_confidence": confidences[0] if confidences else None,
                "top1_correct": top1_correct,
            }
        )
    return {
        "schema_version": "reliability-decisions-1.0",
        "split": split,
        "limit": limit,
        "priors": priors_path.as_posix() if priors_path is not None else "default_hand_picked",
        "manifest": str(manifest_path),
        "manifest_sha256": _manifest_checksum(manifest_path),
        "created_at": datetime.now(UTC).isoformat(),
        "decisions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/queries_challenge.json"))
    parser.add_argument("--split", choices=("calibration", "test"), default="calibration")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Observation output path")
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=None,
        help="Also write review2_full decision rows for threshold fitting",
    )
    parser.add_argument("--priors", type=Path, default=None, help="Priors for the decision pass")
    parser.add_argument("--decision-limit", type=int, default=5)
    parser.add_argument("--allow-hashing-fallback", action="store_true")
    args = parser.parse_args()
    if args.output is None and args.decisions_output is None:
        raise SystemExit("give --output and/or --decisions-output")
    if args.split == "test":
        print("[WARN] collecting on the test split; never fit priors or thresholds on it")
    for path in (args.output, args.decisions_output):
        if path is not None and path.exists():
            raise SystemExit(f"{path} exists; choose a new versioned file name")
    try:
        if args.output is not None:
            result = collect_observations(
                args.manifest,
                split=args.split,
                limit=args.limit,
                allow_hashing_fallback=args.allow_hashing_fallback,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {result['observation_count']} observations to {args.output}")
        if args.decisions_output is not None:
            decisions = collect_decisions(
                args.manifest,
                split=args.split,
                limit=args.decision_limit,
                priors_path=args.priors,
                allow_hashing_fallback=args.allow_hashing_fallback,
            )
            args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
            args.decisions_output.write_text(
                json.dumps(decisions, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote {len(decisions['decisions'])} decisions to {args.decisions_output}")
    except ChallengeManifestError as exc:
        raise SystemExit(f"Challenge manifest is invalid: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
