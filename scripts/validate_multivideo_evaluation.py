from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vidquery.config import get_settings
from vidquery.domain import CanonicalSegment, RelationshipQuery
from vidquery.relationships import (
    matching_relationships,
    normalize_predicate,
    relationship_directionality,
)
from vidquery.search import StructuredQueryParser, _tokens
from vidquery.storage import SQLiteRepository


def _satisfies(segment: CanonicalSegment, query: dict[str, Any]) -> bool:
    expected_entities = {str(item).lower() for item in query["expected_entities"]}
    if not expected_entities.issubset({item.lower() for item in segment.entities}):
        return False
    expected_action = query["expected_action"]
    if expected_action and expected_action.lower() not in {
        item.lower() for item in segment.actions
    }:
        return False
    expected_speaker = query["expected_speaker"]
    if expected_speaker and expected_speaker not in segment.speakers:
        return False
    expected_concept = query["expected_spoken_concept"]
    if expected_concept:
        evidence_tokens = set(
            _tokens(
                " ".join(
                    [segment.transcript, *(item.text for item in segment.ocr_evidence)]
                )
            )
        )
        if not set(_tokens(expected_concept)).issubset(evidence_tokens):
            return False
    expected_relation = query["expected_relation"]
    if expected_relation:
        relation = RelationshipQuery(
            **expected_relation,
            directionality=relationship_directionality(expected_relation["predicate"]),
        )
        if not matching_relationships(segment, relation):
            return False
    return True


def _query_plan_satisfies(segment: CanonicalSegment, query_text: str) -> bool:
    """Check whether explicit canonical evidence makes a negative query positive."""

    plan = StructuredQueryParser().parse(query_text)
    has_constraint = bool(
        plan.visual_entities
        or plan.actions
        or plan.relationships
        or plan.relationship_tuples
        or plan.spoken_terms
        or plan.speaker
    )
    if not has_constraint:
        return False
    entities = {item.lower() for item in segment.entities}
    if not set(plan.visual_entities).issubset(entities):
        return False
    actions = {item.lower() for item in segment.actions}
    if not set(plan.actions).issubset(actions):
        return False
    if plan.speaker and plan.speaker not in segment.speakers:
        return False
    evidence_tokens = set(
        _tokens(
            " ".join(
                [segment.transcript, *(item.text for item in segment.ocr_evidence)]
            )
        )
    )
    if not set(plan.spoken_terms).issubset(evidence_tokens):
        return False
    if any(not matching_relationships(segment, relation) for relation in plan.relationship_tuples):
        return False
    if plan.relationships and not set(plan.relationships).issubset(
        {normalize_predicate(item.predicate) for item in segment.relationships}
    ):
        return False
    return True


def validate(repository: SQLiteRepository, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_by_source = {
        Path(video.original_filename).stem: video.video_id
        for video in repository.list_videos()
    }
    errors: list[str] = []
    for query in manifest["queries"]:
        source = query["relevant_video_id"]
        video_id = canonical_by_source.get(source)
        if video_id is None:
            errors.append(f"{query['id']}: source video {source!r} is not indexed")
            continue
        segments = repository.list_segments([video_id])
        if query["negative"]:
            matches = [
                segment.start_time
                for segment in segments
                if _query_plan_satisfies(segment, str(query["query"]))
            ]
            if matches:
                errors.append(f"{query['id']}: negative query matched at {matches}")
            continue
        for relevant in query["relevant"]:
            segment = next(
                (
                    candidate
                    for candidate in segments
                    if candidate.start_time == relevant["start_time"]
                    and candidate.end_time == relevant["end_time"]
                ),
                None,
            )
            if segment is None:
                errors.append(
                    f"{query['id']}: no canonical segment at "
                    f"{relevant['start_time']}-{relevant['end_time']}"
                )
            elif not _satisfies(segment, query):
                errors.append(
                    f"{query['id']}: labelled segment "
                    f"{relevant['start_time']}-{relevant['end_time']} lacks expected evidence"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the five-video evaluation labels")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/queries_multivideo.json"),
    )
    args = parser.parse_args()
    settings = get_settings()
    errors = validate(SQLiteRepository(settings.database_path), args.manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    query_count = len(json.loads(args.manifest.read_text(encoding="utf-8"))["queries"])
    print(f"Validated {query_count} queries against canonical SQLite evidence")


if __name__ == "__main__":
    main()
