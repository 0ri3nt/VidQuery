"""Collect calibration observations for learned reliability priors."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from vidquery.challenge_evaluation import (
    ChallengeManifestError,
    _hit,
    _moment_from_result,
    load_manifest,
    source_video_id,
    validate_manifest,
)
from vidquery.config import get_settings
from vidquery.domain import SearchRequest
from vidquery.query_planner import QueryPlanningService
from vidquery.search import LocalHybridSearchEngine, configured_text_encoder
from vidquery.storage import SQLiteRepository


def _manifest_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_observations(
    manifest_path: Path, *, split: str, limit: int, top_k: int = 1
) -> dict:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    settings = get_settings()
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    planner = QueryPlanningService(replace(settings, enable_query_planner=False))
    engine = LocalHybridSearchEngine(
        repository,
        encoder=configured_text_encoder(settings),
        semantic_min_similarity=settings.semantic_min_similarity,
        appearance_settings=settings,
        evidence_supported_threshold=settings.evidence_supported_threshold,
        evidence_weak_threshold=settings.evidence_weak_threshold,
    )
    source_ids = {video.video_id: source_video_id(video) for video in repository.list_videos()}
    observations: list[dict] = []
    query_count = 0

    for item in manifest["queries"]:
        source = str(item["source_video_id"])
        split_entry = (manifest.get("videos") or {}).get(source, {})
        if split_entry.get("split") != split:
            continue
        query_count += 1
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
        # Only the results a user acts on are scored; lower ranks are mostly
        # unlabelled near-misses and would bias every prior downward.
        for result in response.results[:top_k]:
            if source_ids.get(result.video_id) != source or result.evidence is None:
                continue
            moment = _moment_from_result(result, use_localization=True)
            correct = (not item.get("negative", False)) and any(
                _hit(moment, relevant, float(item.get("tolerance_seconds", 2.0)))
                for relevant in item.get("relevant", [])
            )
            for contribution in result.evidence.contributions:
                if contribution.raw_score <= 0:
                    continue
                observations.append(
                    {
                        "query_id": item["id"],
                        "source_method": contribution.source_method,
                        "modality": contribution.modality,
                        "raw_score": contribution.raw_score,
                        "correct": correct,
                    }
                )

    return {
        "schema_version": "reliability-calibration-1.0",
        "split": split,
        "top_k": top_k,
        "manifest": str(manifest_path),
        "manifest_sha256": _manifest_checksum(manifest_path),
        "created_at": datetime.now(UTC).isoformat(),
        "query_count": query_count,
        "observation_count": len(observations),
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/queries_challenge.json"))
    parser.add_argument("--split", choices=("calibration", "test"), default="calibration")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--top-k", type=int, default=1, help="Score only the top K results per query"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    try:
        result = collect_observations(
            args.manifest, split=args.split, limit=args.limit, top_k=args.top_k
        )
    except ChallengeManifestError as exc:
        raise SystemExit(f"Challenge manifest is invalid: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {result['observation_count']} observations to {args.output}")


if __name__ == "__main__":
    main()
