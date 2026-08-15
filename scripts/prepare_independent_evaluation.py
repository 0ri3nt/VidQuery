# Dense one-line query declarations intentionally mirror the generated JSON rows.
# ruff: noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vidquery.config import get_settings
from vidquery.domain import CanonicalSegment, RelationshipQuery
from vidquery.relationships import (
    matching_relationships,
    relationship_directionality,
)
from vidquery.storage import SQLiteRepository


@dataclass(frozen=True, slots=True)
class QuerySpec:
    id: str
    query: str
    category: str
    source_video_id: str
    expected_entities: tuple[str, ...] = ()
    expected_relation: tuple[str, str, str] | None = None
    expected_action: str | None = None
    expected_spoken_concept: str | None = None
    expected_speaker: str | None = None
    starts: tuple[float, ...] = ()
    tolerance: float = 0.0
    negative: bool = False
    label_source: str = "canonical_evidence_review"
    notes: str = ""


SPECS = [
    # Transcript and semantic transcript intents.
    QuerySpec("tx-architecture", "architecture", "transcript", "final_demo", expected_spoken_concept="architecture", starts=(0, 15), label_source="controlled_manifest"),
    QuerySpec("tx-software-architecture", "software architecture", "transcript", "final_demo", expected_spoken_concept="software architecture", starts=(0,), label_source="controlled_manifest"),
    QuerySpec("tx-sqlite-metadata", "SQLite metadata", "transcript", "final_demo", expected_spoken_concept="SQLite metadata", starts=(5,), label_source="controlled_manifest"),
    QuerySpec("tx-deployment-docker", "deployment Docker Neo4j", "transcript", "final_demo", expected_spoken_concept="deployment", starts=(10,), label_source="controlled_manifest"),
    QuerySpec("tx-retrieval-architecture", "retrieval architecture", "transcript", "final_demo", expected_spoken_concept="retrieval architecture", starts=(15,), label_source="controlled_manifest"),
    QuerySpec("tx-search-evidence", "speech objects relationships", "transcript", "final_demo", expected_spoken_concept="speech", starts=(20,), label_source="controlled_manifest"),
    QuerySpec("tx-semantic-deployment", "release the application to production", "transcript", "final_demo", expected_spoken_concept="deployment", starts=(10,), tolerance=5, label_source="controlled_manifest", notes="Deliberate paraphrase for semantic retrieval."),
    QuerySpec("tx-distractor-architecture", "software architecture", "transcript", "distractor_demo", expected_spoken_concept="software architecture", starts=(0,), label_source="controlled_manifest"),
    # Visual entity intents.
    QuerySpec("vis-final-laptop", "laptop", "visual_entity", "final_demo", expected_entities=("laptop",), label_source="controlled_manifest"),
    QuerySpec("vis-final-whiteboard", "whiteboard", "visual_entity", "final_demo", expected_entities=("whiteboard",), label_source="controlled_manifest"),
    QuerySpec("vis-final-table", "table", "visual_entity", "final_demo", expected_entities=("dining table",), label_source="automatic_detection_review"),
    QuerySpec("vis-heldout-phone", "cell phone", "visual_entity", "hbYvDvJrpNk", expected_entities=("cell phone",), label_source="official_ava_heldout"),
    QuerySpec("vis-heldout-cup", "cup", "visual_entity", "hbYvDvJrpNk", expected_entities=("cup",), label_source="official_ava_heldout"),
    QuerySpec("vis-heldout-handbag", "handbag", "visual_entity", "hbYvDvJrpNk", expected_entities=("handbag",), label_source="official_ava_heldout"),
    QuerySpec("vis-heldout-couch", "couch", "visual_entity", "hbYvDvJrpNk", expected_entities=("couch",), label_source="official_ava_heldout"),
    QuerySpec("vis-user-person", "person", "visual_entity", "She How many dates have we been to", expected_entities=("person",), label_source="automatic_detection_review"),
    # Explicit tuple intents.
    QuerySpec("rel-final-person-laptop", "person near laptop", "spatial_relationship", "final_demo", expected_entities=("person", "laptop"), expected_relation=("person", "near", "laptop"), label_source="controlled_manifest"),
    QuerySpec("rel-final-chair-behind-laptop", "chair behind laptop", "spatial_relationship", "final_demo", expected_entities=("chair", "laptop"), expected_relation=("chair", "behind", "laptop"), label_source="learned_relation_review"),
    QuerySpec("rel-final-laptop-above-table", "laptop above table", "spatial_relationship", "final_demo", expected_entities=("laptop", "dining table"), expected_relation=("laptop", "above", "dining table"), label_source="learned_relation_review"),
    QuerySpec("rel-final-person-front-person", "person in front of person", "spatial_relationship", "final_demo", expected_entities=("person",), expected_relation=("person", "in_front_of", "person"), label_source="learned_relation_review"),
    QuerySpec("rel-heldout-person-contains-phone", "person contains cell phone", "spatial_relationship", "hbYvDvJrpNk", expected_entities=("person", "cell phone"), expected_relation=("person", "contains", "cell phone"), label_source="official_ava_geometry"),
    QuerySpec("rel-heldout-phone-inside-person", "cell phone inside person", "spatial_relationship", "hbYvDvJrpNk", expected_entities=("cell phone", "person"), expected_relation=("cell phone", "inside", "person"), label_source="official_ava_geometry"),
    QuerySpec("rel-heldout-person-left-chair", "person left of chair", "spatial_relationship", "hbYvDvJrpNk", expected_entities=("person", "chair"), expected_relation=("person", "left_of", "chair"), label_source="official_ava_geometry"),
    QuerySpec("rel-heldout-plant-contains-vase", "potted plant contains vase", "spatial_relationship", "hbYvDvJrpNk", expected_entities=("potted plant", "vase"), expected_relation=("potted plant", "contains", "vase"), label_source="official_ava_geometry"),
    # Held-out AVA person-action intents.
    QuerySpec("act-hand-shake", "person hand shake", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="hand shake", label_source="official_ava_heldout"),
    QuerySpec("act-answer-phone", "person answer phone", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="answer phone", label_source="official_ava_heldout"),
    QuerySpec("act-fall-down", "person fall down", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="fall down", label_source="official_ava_heldout"),
    QuerySpec("act-take-photo", "person take photo", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="take photo", label_source="official_ava_heldout"),
    QuerySpec("act-write", "person writing", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="write", label_source="official_ava_heldout"),
    QuerySpec("act-kiss", "person kiss", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="kiss", label_source="official_ava_heldout"),
    QuerySpec("act-crawl", "person crawl", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="crawl", label_source="official_ava_heldout"),
    QuerySpec("act-martial-art", "person martial art", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="martial art", label_source="official_ava_heldout"),
    QuerySpec("act-text-phone", "person text on phone", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="text on phone", label_source="official_ava_heldout"),
    QuerySpec("act-work-computer", "person work on computer", "ava_action", "hbYvDvJrpNk", expected_entities=("person",), expected_action="work on computer", label_source="official_ava_heldout"),
    # Speaker intents use manually authored transcript concepts plus actual diarization clusters.
    QuerySpec("spk00-architecture", "SPEAKER_00 software architecture", "speaker", "final_demo", expected_spoken_concept="software architecture", expected_speaker="SPEAKER_00", starts=(0,), label_source="controlled_manifest_plus_pyannote"),
    QuerySpec("spk01-sqlite", "SPEAKER_01 SQLite metadata", "speaker", "final_demo", expected_spoken_concept="SQLite metadata", expected_speaker="SPEAKER_01", starts=(5,), label_source="controlled_manifest_plus_pyannote"),
    QuerySpec("spk00-deployment", "SPEAKER_00 deployment Docker", "speaker", "final_demo", expected_spoken_concept="deployment", expected_speaker="SPEAKER_00", starts=(10,), label_source="controlled_manifest_plus_pyannote"),
    QuerySpec("spk01-whiteboard", "SPEAKER_01 whiteboard retrieval", "speaker", "final_demo", expected_spoken_concept="retrieval", expected_speaker="SPEAKER_01", starts=(15,), label_source="controlled_manifest_plus_pyannote"),
    QuerySpec("spk00-relationships", "SPEAKER_00 objects relationships", "speaker", "final_demo", expected_spoken_concept="relationships", expected_speaker="SPEAKER_00", starts=(20,), label_source="controlled_manifest_plus_pyannote"),
    QuerySpec("spk-distractor", "SPEAKER_00 software architecture", "speaker", "distractor_demo", expected_spoken_concept="software architecture", expected_speaker="SPEAKER_00", starts=(0,), label_source="controlled_manifest_plus_pyannote"),
    # Multimodal conjunctions.
    QuerySpec("mm-deployment-laptop", "deployment person near laptop", "multimodal", "final_demo", expected_entities=("person", "laptop"), expected_relation=("person", "near", "laptop"), expected_spoken_concept="deployment", starts=(10,), label_source="controlled_manifest"),
    QuerySpec("mm-retrieval-whiteboard", "retrieval architecture person near whiteboard", "multimodal", "final_demo", expected_entities=("person", "whiteboard"), expected_relation=("person", "near", "whiteboard"), expected_spoken_concept="retrieval architecture", starts=(15,), label_source="controlled_manifest"),
    QuerySpec("mm-sqlite-laptop", "SQLite metadata laptop", "multimodal", "final_demo", expected_entities=("laptop",), expected_spoken_concept="SQLite metadata", starts=(5,), label_source="controlled_manifest"),
    QuerySpec("mm-speech-person", "speech relationships person", "multimodal", "final_demo", expected_entities=("person",), expected_spoken_concept="speech", starts=(20,), label_source="controlled_manifest"),
    QuerySpec("mm-heldout-writing-table", "person writing table", "multimodal", "hbYvDvJrpNk", expected_entities=("person", "dining table"), expected_action="write", label_source="official_ava_heldout"),
    QuerySpec("mm-heldout-answer-phone", "person answer phone remote", "multimodal", "hbYvDvJrpNk", expected_entities=("person", "remote"), expected_action="answer phone", label_source="official_ava_heldout"),
    QuerySpec("mm-heldout-photo-phone", "person take photo cell phone", "multimodal", "hbYvDvJrpNk", expected_entities=("person", "cell phone"), expected_action="take photo", label_source="official_ava_heldout"),
    QuerySpec("mm-heldout-computer-laptop", "person work on computer laptop", "multimodal", "hbYvDvJrpNk", expected_entities=("person", "laptop"), expected_action="work on computer", label_source="official_ava_heldout"),
    # OCR intents.
    QuerySpec("ocr-kubernetes", "Kubernetes", "ocr", "ocr_demo", expected_spoken_concept="kubernetes", starts=(0,), label_source="controlled_ocr_fixture"),
    QuerySpec("ocr-deployment", "deployment", "ocr", "ocr_demo", expected_spoken_concept="deployment", starts=(0,), label_source="controlled_ocr_fixture"),
    QuerySpec("ocr-knowledge-graph", "knowledge graph", "ocr", "ocr_demo", expected_spoken_concept="knowledge graph", starts=(0, 5), label_source="controlled_ocr_fixture"),
    QuerySpec("ocr-k8s-deployment", "Kubernetes deployment", "ocr", "ocr_demo", expected_spoken_concept="kubernetes", starts=(0,), label_source="controlled_ocr_fixture"),
    # Video-scoped negatives and distractors.
    QuerySpec("neg-final-dog", "dog", "negative", "final_demo", negative=True, label_source="controlled_manifest"),
    QuerySpec("neg-final-car", "person near car", "negative", "final_demo", negative=True, label_source="controlled_manifest"),
    QuerySpec("neg-final-speaker", "SPEAKER_99 architecture", "negative", "final_demo", negative=True, label_source="controlled_manifest"),
    QuerySpec("neg-final-swim", "person swimming", "negative", "final_demo", negative=True, label_source="controlled_manifest"),
    QuerySpec("neg-ocr-laptop", "laptop", "negative", "ocr_demo", negative=True, label_source="controlled_ocr_fixture"),
    QuerySpec("neg-distractor-whiteboard", "whiteboard", "negative", "distractor_demo", negative=True, label_source="controlled_manifest"),
    QuerySpec("neg-heldout-airplane", "airplane", "negative", "hbYvDvJrpNk", negative=True, label_source="official_ava_heldout"),
    QuerySpec("neg-user-elephant", "elephant", "negative", "She How many dates have we been to", negative=True, label_source="automatic_detection_review"),
    QuerySpec("neg-final-postgres", "PostgreSQL", "negative", "final_demo", negative=True, label_source="controlled_manifest"),
    QuerySpec("neg-ocr-speaker", "SPEAKER_00 Kubernetes", "negative", "ocr_demo", negative=True, label_source="controlled_ocr_fixture"),
]


def _relation_query(value: tuple[str, str, str]) -> RelationshipQuery:
    subject, predicate, target = value
    return RelationshipQuery(
        subject=subject,
        predicate=predicate,
        object=target,
        directionality=relationship_directionality(predicate),
    )


def _matches_spec(segment: CanonicalSegment, spec: QuerySpec) -> bool:
    entities = {item.lower() for item in segment.entities}
    if not set(spec.expected_entities).issubset(entities):
        return False
    if spec.expected_action and spec.expected_action not in {
        item.lower() for item in segment.actions
    }:
        return False
    if spec.expected_relation and not matching_relationships(
        segment, _relation_query(spec.expected_relation)
    ):
        return False
    if spec.expected_speaker and spec.expected_speaker not in segment.speakers:
        return False
    if spec.expected_spoken_concept:
        evidence_text = " ".join(
            [segment.transcript, *(item.text for item in segment.ocr_evidence)]
        ).lower()
        if spec.expected_spoken_concept.lower() not in evidence_text:
            return False
    return True


def _query_row(
    spec: QuerySpec,
    segments: list[CanonicalSegment],
) -> dict[str, Any]:
    if spec.negative:
        relevant_segments: list[CanonicalSegment] = []
    elif spec.starts:
        by_start = {segment.start_time: segment for segment in segments}
        missing = [start for start in spec.starts if start not in by_start]
        if missing:
            raise ValueError(f"{spec.id} refers to missing segment starts: {missing}")
        relevant_segments = [by_start[start] for start in spec.starts]
        invalid = [segment.start_time for segment in relevant_segments if not _matches_spec(segment, spec)]
        if invalid:
            raise ValueError(f"{spec.id} expected evidence is absent at: {invalid}")
    else:
        relevant_segments = [segment for segment in segments if _matches_spec(segment, spec)]
        if not relevant_segments:
            raise ValueError(f"{spec.id} has no matching canonical evidence")

    expected_relation = None
    if spec.expected_relation:
        expected_relation = {
            "subject": spec.expected_relation[0],
            "predicate": spec.expected_relation[1],
            "object": spec.expected_relation[2],
        }
    first = relevant_segments[0] if relevant_segments else None
    return {
        "id": spec.id,
        "query": spec.query,
        "category": spec.category,
        "relevant_video_id": spec.source_video_id,
        "ground_truth_start_time": first.start_time if first else None,
        "ground_truth_end_time": first.end_time if first else None,
        "timestamp_tolerance": spec.tolerance,
        "expected_entities": list(spec.expected_entities),
        "expected_relation": expected_relation,
        "expected_action": spec.expected_action,
        "expected_spoken_concept": spec.expected_spoken_concept,
        "expected_speaker": spec.expected_speaker,
        "negative": spec.negative,
        "label_source": spec.label_source,
        "notes": spec.notes,
        "relevant": [
            {
                "source_video_id": spec.source_video_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "tolerance_seconds": spec.tolerance,
            }
            for segment in relevant_segments
        ],
    }


def main() -> None:
    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    source_to_id = {
        Path(video.original_filename).stem: video.video_id
        for video in repository.list_videos()
    }
    sources = sorted({spec.source_video_id for spec in SPECS})
    missing = sorted(set(sources) - set(source_to_id))
    if missing:
        raise ValueError(f"evaluation videos are absent from SQLite: {missing}")

    split_path = Path("data/app/models/ava80_scaling/30/split_manifest.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    training_overlap = sorted(set(sources) & set(split["train"]))
    if training_overlap:
        raise ValueError(f"evaluation leaks AVA action training videos: {training_overlap}")

    segments_by_source = {
        source: repository.list_segments([source_to_id[source]]) for source in sources
    }
    rows = [_query_row(spec, segments_by_source[spec.source_video_id]) for spec in SPECS]
    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1

    output = {
        "schema_version": "3.0-independent",
        "description": (
            "Five-video evaluation with manually selected query intents and "
            "evidence-validated intervals. The only AVA member is from the strict "
            "held-out action-model test split; other fixtures are outside AVA training."
        ),
        "videos": sources,
        "video_count": len(sources),
        "query_count": len(rows),
        "positive_query_count": sum(not row["negative"] for row in rows),
        "negative_query_count": sum(row["negative"] for row in rows),
        "category_counts": categories,
        "independence_audit": {
            "action_model_split_manifest": split_path.as_posix(),
            "ava_training_overlap": training_overlap,
            "ava_validation_overlap": sorted(set(sources) & set(split["validation"])),
            "ava_test_overlap": sorted(set(sources) & set(split["test"])),
            "passed": not training_overlap,
        },
        "queries": rows,
    }
    path = Path("evaluation/queries_independent.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": path.as_posix(),
                "videos": sources,
                "queries": len(rows),
                "categories": categories,
                "independence_audit": output["independence_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
