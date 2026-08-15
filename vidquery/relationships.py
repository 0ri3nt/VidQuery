from __future__ import annotations

import re

from .domain import (
    CanonicalSegment,
    MatchedRelationship,
    Relationship,
    RelationshipQuery,
)

RELATIONSHIP_ALIASES = {
    "speaking with each other": "talk_to",
    "speaking with": "talk_to",
    "shaking hands with": "shake_hand_with",
    "speaking to": "talk_to",
    "speaks to": "talk_to",
    "holding hands with": "hold_hand_of",
    "holding": "hold",
    "holds": "hold",
    "looking at": "watch",
    "pointing to": "point_to",
    "points to": "point_to",
    "in front of": "in_front_of",
    "behind": "behind",
    "beneath": "below",
    "overlapping with": "overlaps_with",
    "interacting with": "interacts_with",
    "writing on": "write_on",
    "talking to": "talk_to",
    "overlaps with": "overlaps_with",
    "next to": "near",
    "close to": "near",
    "beside": "near",
    "near": "near",
    "left of": "left_of",
    "right of": "right_of",
    "above": "above",
    "below": "below",
    "inside": "inside",
    "contains": "contains",
    "overlapping": "overlaps_with",
    "overlaps": "overlaps_with",
    "touching": "touches",
}

_PREDICATE_NORMALIZATION = {
    **RELATIONSHIP_ALIASES,
    "talk to": "talk_to",
    "talks to": "talk_to",
    "speak to": "talk_to",
    "speaks to": "talk_to",
    "speak with": "talk_to",
    "speaking with": "talk_to",
    "next to": "near",
    "next_to": "near",
    "speak_to": "talk_to",
    "touch": "touches",
    "beneath": "below",
    "write on": "write_on",
    "writes on": "write_on",
}

_ENTITY_NORMALIZATION = {
    "people": "person",
    "persons": "person",
    "someone": "person",
    "somebody": "person",
    "man": "person",
    "woman": "person",
    "computer": "laptop",
    "notebook": "laptop",
    "white board": "whiteboard",
    "board": "whiteboard",
    "phone": "cell phone",
    "mobile": "cell phone",
    "table": "dining table",
    "desk": "dining table",
    "television": "tv",
}

SYMMETRIC_PREDICATES = frozenset(
    {
        "near",
        "overlaps_with",
        "touches",
        "hug",
        "kiss",
        "shake_hand_with",
        "hold_hand_of",
    }
)
DIRECTIONAL_PREDICATES = frozenset(
    {"left_of", "right_of", "above", "below", "inside", "contains"}
)


def _words(value: str) -> str:
    return " ".join(re.sub(r"[-_]+", " ", value.strip().lower()).split())


def normalize_predicate(value: str) -> str:
    words = _words(value)
    return _PREDICATE_NORMALIZATION.get(words, words.replace(" ", "_"))


def normalize_entity_class(value: str) -> str:
    words = _words(value)
    return _ENTITY_NORMALIZATION.get(words, words)


def relationship_directionality(predicate: str) -> str:
    return "symmetric" if normalize_predicate(predicate) in SYMMETRIC_PREDICATES else "directed"


def materialize_relationship(
    relationship: Relationship, segment: CanonicalSegment
) -> MatchedRelationship | None:
    detections = {item.detection_id: item for item in segment.detections}
    source = detections.get(relationship.source_id)
    target = detections.get(relationship.target_id)
    if source is None and relationship.source_track_id:
        source = next(
            (
                item
                for item in segment.detections
                if item.track_id == relationship.source_track_id
            ),
            None,
        )
    if target is None and relationship.target_track_id:
        target = next(
            (
                item
                for item in segment.detections
                if item.track_id == relationship.target_track_id
            ),
            None,
        )
    if source is None or target is None:
        return None
    predicate = normalize_predicate(relationship.predicate)
    return MatchedRelationship(
        relationship_id=relationship.relationship_id,
        source_id=source.detection_id,
        source_class=normalize_entity_class(source.class_label),
        predicate=predicate,
        target_id=target.detection_id,
        target_class=normalize_entity_class(target.class_label),
        directionality=relationship_directionality(predicate),
        confidence=relationship.confidence,
        source_method=relationship.source_method,
        timestamp=relationship.timestamp,
        model_version=relationship.model_version,
        start_time=relationship.start_time,
        end_time=relationship.end_time,
        source_track_id=relationship.source_track_id,
        target_track_id=relationship.target_track_id,
        observation_count=relationship.observation_count,
        temporal_smoothed=relationship.temporal_smoothed,
    )


def matching_relationships(
    segment: CanonicalSegment, query: RelationshipQuery
) -> list[MatchedRelationship]:
    matches: list[MatchedRelationship] = []
    for relationship in segment.relationships:
        evidence = materialize_relationship(relationship, segment)
        if evidence is None or evidence.predicate != normalize_predicate(query.predicate):
            continue
        forward = (
            evidence.source_class == normalize_entity_class(query.subject)
            and evidence.target_class == normalize_entity_class(query.object)
        )
        reverse = (
            evidence.directionality == "symmetric"
            and evidence.source_class == normalize_entity_class(query.object)
            and evidence.target_class == normalize_entity_class(query.subject)
        )
        if forward or reverse:
            matches.append(evidence)
    return sorted(matches, key=lambda item: (-item.confidence, item.relationship_id))
