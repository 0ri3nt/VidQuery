from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .appearance import APPEARANCE_COLORS, cosine_similarity
from .ava_labels import AVA_V22_ACTIONS
from .domain import (
    ActionPredictionEvidence,
    AppearanceConstraint,
    AppearanceMatchEvidence,
    EntityAppearance,
    MatchedRelationship,
    QueryPlan,
    RelationshipQuery,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .relationships import (
    RELATIONSHIP_ALIASES,
    matching_relationships,
    normalize_predicate,
    relationship_directionality,
)
from .storage import SQLiteRepository

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
LOGGER = logging.getLogger(__name__)

ENTITY_ALIASES = {
    "people": "person",
    "someone": "person",
    "somebody": "person",
    "man": "person",
    "woman": "person",
    "guy": "person",
    "person": "person",
    "persons": "person",
    "laptop": "laptop",
    "computer": "laptop",
    "notebook": "laptop",
    "whiteboard": "whiteboard",
    "white board": "whiteboard",
    "board": "whiteboard",
    "phone": "cell phone",
    "cell phone": "cell phone",
    "mobile": "cell phone",
    "book": "book",
    "chair": "chair",
    "table": "dining table",
    "desk": "dining table",
    "cup": "cup",
    "cups": "cup",
    "mug": "cup",
    "mugs": "cup",
    "bottle": "bottle",
    "backpack": "backpack",
    "bag": "backpack",
    "keyboard": "keyboard",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "suitcase": "suitcase",
    "couch": "couch",
    "sofa": "couch",
    "bed": "bed",
    "remote": "remote",
    "bowl": "bowl",
    "vase": "vase",
    "clock": "clock",
    "potted plant": "potted plant",
    "tv": "tv",
    "television": "tv",
    "police car": "car",
    "police vehicle": "car",
    "automobile": "car",
    "vehicle": "car",
    "car": "car",
    "cars": "car",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "bus": "bus",
    "buses": "bus",
    "train": "train",
    "truck": "truck",
    "trucks": "truck",
    "boat": "boat",
    "boats": "boat",
    "traffic light": "traffic light",
    "fire hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "parking meter": "parking meter",
    "bench": "bench",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "handbag": "handbag",
    "tie": "tie",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "sports ball",
    "ball": "sports ball",
    "kite": "kite",
    "baseball bat": "baseball bat",
    "baseball glove": "baseball glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
    "wine glass": "wine glass",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "doughnut": "donut",
    "cake": "cake",
    "toilet": "toilet",
    "mouse": "mouse",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "hair drier": "hair drier",
    "hair dryer": "hair drier",
    "toothbrush": "toothbrush",
}

ACTION_ALIASES = {
    "driving": "drive",
    "drives": "drive",
    "drove": "drive",
    "driven": "drive",
    "drive": "drive",
    "writing": "write",
    "writes": "write",
    "write": "write",
    "standing": "stand",
    "stands": "stand",
    "stand": "stand",
    "sitting": "sit",
    "sits": "sit",
    "sit": "sit",
    "walking": "walk",
    "walks": "walk",
    "walk": "walk",
    "pointing": "point to",
    "points": "point to",
    "point": "point to",
    "carrying": "carry/hold",
    "carries": "carry/hold",
    "carry": "carry/hold",
    "holding": "carry/hold",
    "holds": "carry/hold",
    "hold": "carry/hold",
    "opening": "open",
    "opens": "open",
    "open": "open",
    "listening": "listen to",
    "listens": "listen to",
    "listen": "listen to",
    "talking": "talk to",
    "talks": "talk to",
    "texting": "text on phone",
    "texts on phone": "text on phone",
    "shaking hands": "hand shake",
    "shakes hands": "hand shake",
    "handshake": "hand shake",
    "using a laptop": "work on computer",
    "uses a laptop": "work on computer",
    "watching a person": "watch person",
    "watching person": "watch person",
    "watch person": "watch person",
    "watching": "watch person",
    "watches": "watch person",
    "watch": "watch person",
    "working on a laptop": "work on computer",
    "working on the laptop": "work on computer",
    "working on computer": "work on computer",
    "working on the computer": "work on computer",
    "works on a laptop": "work on computer",
    "works on the laptop": "work on computer",
}
for _ava_action in AVA_V22_ACTIONS.values():
    ACTION_ALIASES.setdefault(_ava_action, _ava_action)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "containing",
    "discuss",
    "discussed",
    "discusses",
    "discussing",
    "displayed",
    "do",
    "does",
    "did",
    "each",
    "find",
    "for",
    "from",
    "hear",
    "heard",
    "in",
    "is",
    "it",
    "they",
    "locate",
    "me",
    "mention",
    "mentioned",
    "mentions",
    "of",
    "one",
    "on",
    "scene",
    "scenes",
    "segment",
    "segments",
    "show",
    "said",
    "say",
    "says",
    "screen",
    "shown",
    "sign",
    "spoken",
    "text",
    "visible",
    "written",
    "ocr",
    "talks",
    "the",
    "two",
    "to",
    "video",
    "videos",
    "was",
    "when",
    "where",
    "while",
    "with",
    "other",
}

SPEECH_CUE_PATTERN = re.compile(
    r"\b(?:say|said|says|mention|mentioned|mentions|discuss|discussed|"
    r"discusses|hear|heard|spoken)\b"
)
OCR_CUE_PATTERN = re.compile(
    r"\b(?:ocr|written|displayed|visible text|text on (?:the )?screen|"
    r"shown on (?:the )?screen|sign reads|subtitle|subtitles)\b"
)

PERSON_APPEARANCE_WORDS = {
    "shirt",
    "tshirt",
    "jacket",
    "top",
    "clothing",
    "clothes",
    "wearing",
    "wears",
    "dressed",
    "striped",
    "dark",
    "bright",
    "light",
    "glasses",
    "eyeglasses",
    "spectacles",
    "hat",
    "cap",
    "bag",
    "backpack",
}
OBJECT_APPEARANCE_MODIFIERS = {"small", "large", "bright", "dark", "light", "striped"}


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _inflected_action(token: str) -> str | None:
    """Resolve simple English inflections without enumerating every verb form."""

    candidates = [token]
    if token.endswith("ing") and len(token) > 4:
        stem = token[:-3]
        candidates.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
    if token.endswith("ied") and len(token) > 4:
        candidates.append(token[:-3] + "y")
    elif token.endswith("ed") and len(token) > 3:
        stem = token[:-2]
        candidates.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
    if token.endswith("ies") and len(token) > 4:
        candidates.append(token[:-3] + "y")
    elif token.endswith("es") and len(token) > 3:
        candidates.extend((token[:-2], token[:-1]))
    elif token.endswith("s") and len(token) > 2:
        candidates.append(token[:-1])
    for candidate in dict.fromkeys(candidates):
        action = ACTION_ALIASES.get(candidate)
        if action is not None:
            return action
    return None


class StructuredQueryParser:
    def parse(self, query: str) -> QueryPlan:
        normalized = " ".join(query.lower().replace("-", " ").split())
        speech_cued = bool(SPEECH_CUE_PATTERN.search(normalized))
        ocr_cued = bool(OCR_CUE_PATTERN.search(normalized))
        entities = self._extract_aliases(normalized, ENTITY_ALIASES)
        relationships = self._extract_aliases(normalized, RELATIONSHIP_ALIASES)
        relationship_tuples = self._extract_relationship_tuples(normalized)
        actions = self._extract_aliases(normalized, ACTION_ALIASES)
        appearance_constraints, appearance_tokens = self._extract_appearance_constraints(
            normalized, entities
        )
        inflected_action_tokens: set[str] = set()
        for token in _tokens(normalized):
            action = _inflected_action(token)
            if action is not None:
                if action not in actions:
                    actions.append(action)
                inflected_action_tokens.add(token)
        # Explicit linguistic/visible-text cues override an otherwise ambiguous
        # action homonym: "say drive" searches language, not observed driving.
        language_only_cued = (speech_cued or ocr_cued) and not re.search(
            r"\b(?:while|when|as)\b", normalized
        )
        if language_only_cued:
            actions = []
            inflected_action_tokens.clear()
            relationships = []
            relationship_tuples = []
        # In an explicit subject-predicate-object phrase, ``talking to`` and
        # ``writing on`` describe the edge.  Do not also require the segment's
        # person-centric AVA action label to be present.
        tuple_predicates = {item.predicate for item in relationship_tuples}
        if "talk_to" in tuple_predicates and "talk to" in actions:
            actions.remove("talk to")
        if "write_on" in tuple_predicates and "write" in actions:
            actions.remove("write")
        if "hold" in tuple_predicates and "carry/hold" in actions:
            actions.remove("carry/hold")
        if "speaking with each other" in normalized and not relationship_tuples:
            relationship_tuples.append(
                RelationshipQuery(
                    subject="person",
                    predicate="talk_to",
                    object="person",
                    directionality=relationship_directionality("talk_to"),
                )
            )
            if "talk_to" not in relationships:
                relationships.append("talk_to")
        speaker_match = re.search(r"\bspeaker[_ ]?(\d+)\b", normalized, re.IGNORECASE)
        speaker = f"SPEAKER_{int(speaker_match.group(1)):02d}" if speaker_match else None

        consumed: set[str] = set()
        for aliases, values in (
            (ENTITY_ALIASES, entities),
            (RELATIONSHIP_ALIASES, relationships),
            (ACTION_ALIASES, actions),
        ):
            for alias, canonical in aliases.items():
                if canonical in values and alias in normalized:
                    consumed.update(_tokens(alias))
        if speaker_match:
            consumed.update(_tokens(speaker_match.group(0)))
        consumed.update(inflected_action_tokens)
        consumed.update(appearance_tokens)

        concept_terms: list[str] = []
        for token in _tokens(normalized):
            if token in STOPWORDS or token in consumed or token.isdigit():
                continue
            if token not in concept_terms:
                concept_terms.append(token)

        ocr_terms = concept_terms if ocr_cued else []
        spoken_terms = [] if ocr_cued else concept_terms
        linguistic = bool(spoken_terms or ocr_terms or speaker)
        visual_modes = sum(
            bool(value)
            for value in (
                entities,
                actions,
                relationship_tuples or relationships,
                appearance_constraints,
            )
        )
        relation_only_entities = bool(relationship_tuples) and set(entities) == {
            item for relation in relationship_tuples for item in (relation.subject, relation.object)
        }
        action_only_person = bool(actions) and set(entities) <= {"person"}
        if linguistic and visual_modes:
            intent = "multimodal_search"
        elif (relationship_tuples or relationships) and (relation_only_entities or not entities):
            intent = "relationship_search"
        elif actions and action_only_person:
            intent = "action_search"
        elif visual_modes > 1:
            intent = "multimodal_search"
        elif relationship_tuples or relationships:
            intent = "relationship_search"
        elif actions:
            intent = "action_search"
        elif entities:
            intent = "visual_search"
        elif ocr_terms:
            intent = "ocr_search"
        elif speaker:
            intent = "speaker_search"
        else:
            intent = "transcript_search" if speech_cued else "multimodal_search"

        return QueryPlan(
            intent=intent,
            visual_entities=entities,
            relationships=relationships,
            relationship_tuples=relationship_tuples,
            actions=actions,
            spoken_terms=spoken_terms,
            ocr_terms=ocr_terms,
            speaker=speaker,
            appearance_constraints=appearance_constraints,
        )

    def _extract_appearance_constraints(
        self, text: str, entities: list[str]
    ) -> tuple[list[AppearanceConstraint], set[str]]:
        """Extract observable clothing/accessory/object appearance constraints.

        The deterministic layer deliberately supports only non-sensitive visual
        descriptions.  Unknown modifiers adjacent to a known entity are kept as
        an open-ended CLIP description instead of being invented as attributes.
        """

        constraints: list[AppearanceConstraint] = []
        consumed: set[str] = set()
        words = _tokens(text)
        color_pattern = "|".join(sorted(APPEARANCE_COLORS, key=len, reverse=True))

        person_mentioned = "person" in entities
        if person_mentioned:
            clothing = re.search(
                rf"\b(?:(?P<style>striped|dark|bright|light)\s+)?(?P<color>{color_pattern})\s+"
                r"(?P<garment>t\s*shirt|shirt|jacket|top|clothing|clothes)\b",
                text,
            )
            if clothing is None:
                clothing = re.search(
                    rf"\b(?P<color>{color_pattern})\s+"
                    r"(?:(?P<style>striped|dark|bright|light)\s+)"
                    r"(?P<garment>t\s*shirt|shirt|jacket|top|clothing|clothes)\b",
                    text,
                )
            reversed_clothing = re.search(
                rf"\b(?:wearing|wears|dressed in)\s+(?P<color>{color_pattern})\b",
                text,
            )
            color = None
            description_parts: list[str] = []
            if clothing:
                color = clothing.group("color")
                style = clothing.group("style")
                garment = clothing.group("garment").replace(" ", "")
                description_parts.extend(item for item in (style, color, garment) if item)
                consumed.update(_tokens(clothing.group(0)))
            elif reversed_clothing:
                color = reversed_clothing.group("color")
                description_parts.extend((color, "upper clothing"))
                consumed.update(_tokens(reversed_clothing.group(0)))
            if clothing is None and reversed_clothing is None:
                open_clothing = re.search(
                    r"\b(?P<phrase>(?:striped|dark|bright|light|small|large)\s+"
                    r"(?:t\s*shirt|shirt|jacket|top|clothing|clothes))\b",
                    text,
                )
                if open_clothing:
                    description_parts.extend(_tokens(open_clothing.group("phrase")))
                    consumed.update(_tokens(open_clothing.group(0)))

            has_glasses = bool(re.search(r"\b(?:glasses|eyeglasses|spectacles)\b", text))
            has_hat = bool(re.search(r"\b(?:hat|cap)\b", text))
            has_bag = bool(
                re.search(
                    r"\b(?:person|people|someone|somebody|man|woman|guy)\b"
                    r".{0,35}\b(?:with|carrying|wearing)\b.{0,20}\b(?:bag|backpack)\b",
                    text,
                )
            )
            for token in PERSON_APPEARANCE_WORDS:
                if re.search(rf"\b{re.escape(token)}\b", text):
                    consumed.add(token)
            if color:
                consumed.add(color)

            open_person_description = self._open_appearance_description(
                words,
                entity_tokens={"person", "people", "someone", "somebody", "man", "woman", "guy"},
                allowed_modifiers=OBJECT_APPEARANCE_MODIFIERS,
            )
            if open_person_description:
                description_parts.extend(_tokens(open_person_description))
                consumed.update(_tokens(open_person_description))

            if color or has_glasses or has_hat or has_bag or description_parts:
                description = " ".join(dict.fromkeys(description_parts)) or None
                visual_description = (
                    f"a person wearing {description}" if description else "a person"
                )
                if has_glasses:
                    visual_description += " wearing glasses"
                if has_hat:
                    visual_description += " wearing a hat"
                if has_bag:
                    visual_description += " carrying a bag or backpack"
                constraints.append(
                    AppearanceConstraint(
                        entity_class="person",
                        upper_clothing_color=color,
                        upper_clothing_description=description,
                        glasses=True if has_glasses else None,
                        hat=True if has_hat else None,
                        bag=True if has_bag else None,
                        description=visual_description,
                    )
                )

        for start, _end, entity_class in self._mentions(text, ENTITY_ALIASES):
            if entity_class == "person":
                continue
            prefix = text[max(0, start - 45) : start].strip()
            prefix_words = _tokens(prefix)
            modifiers: list[str] = []
            allowed = set(APPEARANCE_COLORS) | OBJECT_APPEARANCE_MODIFIERS
            for word in reversed(prefix_words):
                if word not in allowed:
                    break
                modifiers.append(word)
                if len(modifiers) == 3:
                    break
            modifiers.reverse()
            phrase = " ".join(modifiers)
            color = next((word for word in modifiers if word in APPEARANCE_COLORS), None)
            if not phrase:
                continue
            description = f"{phrase} {entity_class}".strip()
            constraints.append(
                AppearanceConstraint(
                    entity_class=entity_class,
                    color=color,
                    description=description,
                )
            )
            consumed.update(_tokens(phrase))

        unique: dict[str, AppearanceConstraint] = {}
        for constraint in constraints:
            unique[constraint.entity_class] = constraint
        if any(item.color == "orange" for item in unique.values()):
            entities[:] = [item for item in entities if item != "orange"]
        return list(unique.values()), consumed

    @staticmethod
    def _open_appearance_description(
        words: list[str], *, entity_tokens: set[str], allowed_modifiers: set[str]
    ) -> str | None:
        for index, word in enumerate(words):
            if word not in entity_tokens:
                continue
            nearby = [
                item
                for item in words[max(0, index - 4) : min(len(words), index + 5)]
                if item in allowed_modifiers
            ]
            return " ".join(nearby) or None
        return None

    @staticmethod
    def _extract_aliases(text: str, aliases: dict[str, str]) -> list[str]:
        values: list[str] = []
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                canonical = aliases[alias]
                if canonical not in values:
                    values.append(canonical)
        return values

    @staticmethod
    def _mentions(text: str, aliases: dict[str, str]) -> list[tuple[int, int, str]]:
        candidates: list[tuple[int, int, str]] = []
        for alias in sorted(aliases, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(alias)}\b", text):
                candidates.append((match.start(), match.end(), aliases[alias]))
        selected: list[tuple[int, int, str]] = []
        for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
            if any(candidate[0] < end and candidate[1] > start for start, end, _ in selected):
                continue
            selected.append(candidate)
        return sorted(selected)

    def _extract_relationship_tuples(self, text: str) -> list[RelationshipQuery]:
        entities = self._mentions(text, ENTITY_ALIASES)
        relationships = self._mentions(text, RELATIONSHIP_ALIASES)
        tuples: list[RelationshipQuery] = []
        seen: set[tuple[str, str, str]] = set()
        for start, end, predicate in relationships:
            subjects = [item for item in entities if item[1] <= start]
            objects = [item for item in entities if item[0] >= end]
            if not subjects or not objects:
                continue
            subject = max(subjects, key=lambda item: item[1])[2]
            object_class = min(objects, key=lambda item: item[0])[2]
            key = (subject, predicate, object_class)
            if key in seen:
                continue
            seen.add(key)
            tuples.append(
                RelationshipQuery(
                    subject=subject,
                    predicate=predicate,
                    object=object_class,
                    directionality=relationship_directionality(predicate),
                )
            )
        return tuples


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    cypher: str
    parameters: dict


class GraphQueryRunner(Protocol):
    def run_query(self, query: str, parameters: dict | None = None) -> list[dict]: ...


class SafeCypherCompiler:
    CYPHER = """
    MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)
    WHERE (size($video_ids) = 0 OR v.video_id IN $video_ids)
      AND all(entity IN $visual_entities WHERE EXISTS {
        MATCH (s)-[:HAS_ENTITY]->(occ:EntityOccurrence)
        WHERE toLower(occ.class_label) = entity
      })
      AND all(action IN $actions WHERE EXISTS {
        MATCH (s)-[:HAS_ACTION]->(a:Action)
        WHERE toLower(a.name) = action
      })
      AND all(predicate IN $relationships WHERE EXISTS {
        MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
        WHERE toLower(e.predicate) = predicate
      })
      AND all(relation IN $relationship_tuples WHERE EXISTS {
        MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
        MATCH (e)-[:FROM_ENTITY]->(source:EntityOccurrence)
        MATCH (e)-[:TO_ENTITY]->(target:EntityOccurrence)
        WHERE toLower(e.predicate) = relation.predicate
          AND (
            (toLower(source.class_label) = relation.subject
             AND toLower(target.class_label) = relation.object)
            OR (relation.symmetric
                AND toLower(source.class_label) = relation.object
                AND toLower(target.class_label) = relation.subject)
          )
      })
      AND ($speaker IS NULL OR EXISTS {
        MATCH (s)-[:HAS_SPEAKER]->(sp:Speaker)
        WHERE sp.label = $speaker
      })
      AND all(term IN $spoken_terms WHERE toLower(s.transcript) CONTAINS term)
      AND all(term IN $ocr_terms WHERE EXISTS {
        MATCH (s)-[:HAS_OCR]->(ocr:OCRText)
        WHERE toLower(ocr.text) CONTAINS term
      })
    OPTIONAL MATCH (s)-[:HAS_RELATIONSHIP]->(matched:RelationshipEvidence)
    OPTIONAL MATCH (matched)-[:FROM_ENTITY]->(matched_source:EntityOccurrence)
    OPTIONAL MATCH (matched)-[:TO_ENTITY]->(matched_target:EntityOccurrence)
    WITH v, s, matched, matched_source, matched_target
    WHERE (
      size($relationship_tuples) > 0
      AND any(relation IN $relationship_tuples WHERE
        toLower(matched.predicate) = relation.predicate
        AND (
          (toLower(matched_source.class_label) = relation.subject
           AND toLower(matched_target.class_label) = relation.object)
          OR (relation.symmetric
              AND toLower(matched_source.class_label) = relation.object
              AND toLower(matched_target.class_label) = relation.subject)
        )
      )
    ) OR (
      size($relationship_tuples) = 0
      AND (size($relationships) = 0 OR toLower(matched.predicate) IN $relationships)
    )
    WITH v, s, [item IN collect(CASE WHEN matched IS NULL THEN NULL ELSE {
      relationship_id: matched.relationship_id,
      source_id: matched.source_id,
      source_class: matched_source.class_label,
      predicate: matched.predicate,
      target_id: matched.target_id,
      target_class: matched_target.class_label,
      directionality: matched.directionality,
      confidence: matched.confidence,
      source_method: matched.source_method,
      timestamp: matched.timestamp,
      model_version: matched.model_version,
      start_time: matched.start_time,
      end_time: matched.end_time,
      source_track_id: matched.source_track_id,
      target_track_id: matched.target_track_id,
      observation_count: coalesce(matched.observation_count, 1),
      temporal_smoothed: coalesce(matched.temporal_smoothed, false)
    } END) WHERE item IS NOT NULL] AS relationship_tuples
    RETURN v.video_id AS video_id, v.display_name AS video_title,
           s.segment_id AS segment_id,
           s.start_time AS start_time, s.end_time AS end_time,
           s.transcript AS transcript, relationship_tuples
    ORDER BY s.start_time ASC
    LIMIT $limit
    """.strip()

    def compile(self, plan: QueryPlan, video_ids: list[str], limit: int) -> CompiledQuery:
        return CompiledQuery(
            cypher=self.CYPHER,
            parameters={
                "video_ids": list(video_ids),
                "visual_entities": list(plan.visual_entities),
                "actions": list(plan.actions),
                "relationships": list(plan.relationships),
                "relationship_tuples": [
                    {
                        "subject": item.subject,
                        "predicate": item.predicate,
                        "object": item.object,
                        "symmetric": item.directionality == "symmetric",
                    }
                    for item in plan.relationship_tuples
                ],
                "spoken_terms": list(plan.spoken_terms),
                "ocr_terms": list(plan.ocr_terms),
                "speaker": plan.speaker,
                "limit": int(limit),
            },
        )


class HashingTextEncoder:
    """Deterministic, dependency-free lexical embedding baseline.

    This is intentionally documented as a lexical baseline, not as a learned
    semantic model.  A SentenceTransformer adapter can replace it without
    changing the ranking contract.
    """

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
        self.model_name = "signed_hashing_256"
        self.is_semantic = False

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimensions do not match")
        return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=False))))


class SentenceTransformerTextEncoder:
    """Local learned sentence embeddings with normalized cosine retrieval."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "auto",
        allow_downloads: bool = False,
    ) -> None:
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.is_semantic = True
        self.model: Any = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=not allow_downloads,
        )
        self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(value) for value in vector]

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimensions do not match")
        similarity = sum(a * b for a, b in zip(left, right, strict=False))
        return max(0.0, min(1.0, similarity))


def sentence_embedding_status(model_name: str) -> str:
    try:
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is None:
            return "unavailable_missing_dependency"
        configured = Path(model_name)
        if configured.is_dir():
            return "available_local"
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return "unavailable_missing_checkpoint"
    return "available_cached"


def configured_text_encoder(settings: Any) -> Any:
    mode = settings.semantic_retrieval_mode.strip().lower()
    if mode == "hashing":
        return HashingTextEncoder()
    if mode != "sentence_transformer":
        raise ValueError(f"Unknown semantic retrieval mode: {mode}")
    try:
        return SentenceTransformerTextEncoder(
            settings.sentence_embedding_model,
            device=settings.sentence_embedding_device,
            allow_downloads=settings.allow_model_downloads,
        )
    except Exception as exc:
        LOGGER.warning(
            "Sentence embedding model is unavailable; using deterministic hashing fallback: %s",
            type(exc).__name__,
        )
        return HashingTextEncoder()


class LocalHybridSearchEngine:
    def __init__(
        self,
        repository: SQLiteRepository,
        parser: StructuredQueryParser | None = None,
        encoder: Any | None = None,
        semantic_min_similarity: float = 0.20,
        appearance_settings: Any | None = None,
        appearance_encoder: Any | None = None,
    ):
        self.repository = repository
        self.parser = parser or StructuredQueryParser()
        self.encoder = encoder or HashingTextEncoder()
        if not 0 <= semantic_min_similarity <= 1:
            raise ValueError("semantic_min_similarity must be between zero and one")
        self.semantic_min_similarity = semantic_min_similarity
        self.appearance_settings = appearance_settings
        self._configured_appearance_encoder = appearance_encoder

    def search(
        self,
        request: SearchRequest,
        *,
        candidate_segment_ids: set[str] | None = None,
        plan_override: QueryPlan | None = None,
    ) -> SearchResponse:
        plan = plan_override or self.parser.parse(request.query)
        query_vector = self.encoder.encode(request.query)
        videos = {video.video_id: video for video in self.repository.list_videos()}
        scored: list[SearchResult] = []
        appearance_by_id: dict[str, EntityAppearance] = {}
        appearance_by_track: dict[tuple[str, str], EntityAppearance] = {}
        appearance_text_vectors: dict[str, list[float]] = {}
        appearance_encoder = None
        appearance_enabled = (
            bool(
                self.appearance_settings is not None
                and self.appearance_settings.enable_appearance_features
            )
            or self._configured_appearance_encoder is not None
        )
        if plan.appearance_constraints and appearance_enabled:
            appearances = self.repository.list_entity_appearances(request.video_ids or None)
            appearance_by_id = {item.appearance_id: item for item in appearances}
            appearance_by_track = {(item.video_id, item.track_id): item for item in appearances}
            descriptions = list(
                dict.fromkeys(
                    item.description for item in plan.appearance_constraints if item.description
                )
            )
            if descriptions:
                try:
                    appearance_encoder = self._get_appearance_encoder()
                    if appearance_encoder is not None:
                        appearance_text_vectors = dict(
                            zip(
                                descriptions,
                                appearance_encoder.encode_texts(descriptions),
                                strict=True,
                            )
                        )
                except Exception as exc:
                    LOGGER.warning(
                        "Appearance text encoding unavailable; using explicit attributes: %s",
                        type(exc).__name__,
                    )

        for segment in self.repository.list_segments(request.video_ids or None):
            if (
                candidate_segment_ids is not None
                and segment.segment_id not in candidate_segment_ids
            ):
                continue
            video = videos.get(segment.video_id)
            if video is None:
                continue
            score, reasons, matched_relationships, appearance_matches = self._score_with_appearance(
                plan,
                query_vector,
                segment,
                appearance_by_id=appearance_by_id,
                appearance_by_track=appearance_by_track,
                appearance_text_vectors=appearance_text_vectors,
                appearance_encoder=appearance_encoder,
                appearance_enabled=appearance_enabled,
            )
            if score <= 0:
                continue
            scored.append(
                SearchResult(
                    video_id=segment.video_id,
                    video_title=video.display_name,
                    segment_id=segment.segment_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    score=round(score, 4),
                    transcript=segment.transcript,
                    speakers=segment.speakers,
                    entities=segment.entities,
                    actions=segment.actions,
                    action_evidence=[
                        ActionPredictionEvidence.model_validate(item)
                        for item in segment.processing_metadata.get("gnn_action_instances", [])
                    ],
                    ocr_evidence=segment.ocr_evidence,
                    relationships=segment.relationships,
                    matched_relationships=matched_relationships,
                    appearance_matches=appearance_matches,
                    thumbnail_url=(
                        f"/api/videos/{segment.video_id}/thumbnail?timestamp={segment.start_time}"
                    ),
                    stream_url=f"/api/videos/{segment.video_id}/stream",
                    match_reason="Matched " + ", ".join(reasons) + ".",
                )
            )

        scored.sort(key=lambda item: (-item.score, item.start_time, item.video_id))
        return SearchResponse(
            query=request.query,
            parsed_query=plan,
            results=scored[: request.limit],
        )

    def _get_appearance_encoder(self) -> Any | None:
        if self._configured_appearance_encoder is None and self.appearance_settings is not None:
            from .appearance import configured_appearance_encoder

            self._configured_appearance_encoder = configured_appearance_encoder(
                self.appearance_settings
            )
        return self._configured_appearance_encoder

    def _score(
        self, plan: QueryPlan, query_vector: list[float], segment: Any
    ) -> tuple[float, list[str], list[MatchedRelationship]]:
        """Backward-compatible scoring seam used by focused modality tests."""

        score, reasons, relationships, _ = self._score_with_appearance(plan, query_vector, segment)
        return score, reasons, relationships

    def _score_with_appearance(
        self,
        plan: QueryPlan,
        query_vector: list[float],
        segment,
        *,
        appearance_by_id: dict[str, EntityAppearance] | None = None,
        appearance_by_track: dict[tuple[str, str], EntityAppearance] | None = None,
        appearance_text_vectors: dict[str, list[float]] | None = None,
        appearance_encoder: Any | None = None,
        appearance_enabled: bool = False,
    ) -> tuple[
        float,
        list[str],
        list[MatchedRelationship],
        list[AppearanceMatchEvidence],
    ]:
        components: list[tuple[float, float, str]] = []
        matched_relationship_evidence: list[MatchedRelationship] = []
        appearance_match_evidence: list[AppearanceMatchEvidence] = []
        transcript_tokens = set(_tokens(segment.transcript))
        ocr_tokens = {
            token for evidence in segment.ocr_evidence for token in _tokens(evidence.text)
        }

        if plan.spoken_terms:
            matched_transcript = [term for term in plan.spoken_terms if term in transcript_tokens]
            matched_ocr_for_general_concept = (
                []
                if plan.intent == "transcript_search"
                else [term for term in plan.spoken_terms if term in ocr_tokens]
            )
            matched_concepts = list(
                dict.fromkeys([*matched_transcript, *matched_ocr_for_general_concept])
            )
            lexical = len(matched_concepts) / len(plan.spoken_terms)
            searchable_text = " ".join(
                [
                    segment.transcript,
                    *(
                        item.text
                        for item in segment.ocr_evidence
                        if plan.intent != "transcript_search"
                    ),
                ]
            )
            cached_embedding = segment.embedding
            if (
                plan.intent == "transcript_search"
                or cached_embedding is None
                or len(cached_embedding) != len(query_vector)
            ):
                cached_embedding = self.encoder.encode(searchable_text)
            vector_score = self.encoder.cosine(query_vector, cached_embedding)
            if self.encoder.is_semantic and vector_score < self.semantic_min_similarity:
                vector_score = 0.0
            spoken_reason = "; ".join(
                reason
                for reason in (
                    (f"spoken terms {'/'.join(matched_transcript)}" if matched_transcript else ""),
                    (
                        f"OCR terms {'/'.join(matched_ocr_for_general_concept)} (easyocr)"
                        if matched_ocr_for_general_concept
                        else ""
                    ),
                )
                if reason
            )
            vector_reason = (
                f"sentence embedding similarity ({self.encoder.model_name})"
                if vector_score and self.encoder.is_semantic
                else ("hashing embedding similarity" if vector_score else "")
            )
            semantic_weight = 0.65 if self.encoder.is_semantic else 0.3
            components.append((1.0 - semantic_weight, lexical, spoken_reason))
            components.append((semantic_weight, vector_score, vector_reason))

        if plan.ocr_terms:
            matched_ocr = [term for term in plan.ocr_terms if term in ocr_tokens]
            lexical = len(matched_ocr) / len(plan.ocr_terms)
            ocr_text = " ".join(item.text for item in segment.ocr_evidence)
            vector_score = self.encoder.cosine(query_vector, self.encoder.encode(ocr_text))
            if self.encoder.is_semantic and vector_score < self.semantic_min_similarity:
                vector_score = 0.0
            semantic_weight = 0.65 if self.encoder.is_semantic else 0.3
            components.append(
                (
                    1.0 - semantic_weight,
                    lexical,
                    f"OCR terms {'/'.join(matched_ocr)} (easyocr)" if matched_ocr else "",
                )
            )
            components.append(
                (
                    semantic_weight,
                    vector_score,
                    (
                        f"OCR sentence embedding similarity ({self.encoder.model_name})"
                        if vector_score and self.encoder.is_semantic
                        else ("OCR hashing embedding similarity" if vector_score else "")
                    ),
                )
            )

        if plan.visual_entities:
            entity_set = {item.lower() for item in segment.entities}
            matched = [item for item in plan.visual_entities if item in entity_set]
            if len(matched) != len(plan.visual_entities):
                return 0.0, [], [], []
            components.append(
                (
                    1.0,
                    len(matched) / len(plan.visual_entities),
                    f"visual entities {'/'.join(matched)}" if matched else "",
                )
            )

        if plan.actions:
            action_set = {item.lower() for item in segment.actions}
            matched = [item for item in plan.actions if item in action_set]
            if len(matched) != len(plan.actions):
                return 0.0, [], [], []
            action_sources = segment.processing_metadata.get("action_sources", {})
            action_confidences = segment.processing_metadata.get("action_confidences", {})
            mean_action_confidence = sum(
                float(action_confidences.get(action, 1.0)) for action in matched
            ) / len(matched)
            provenance = sorted({str(action_sources.get(action, "unknown")) for action in matched})
            components.append(
                (
                    1.0,
                    (len(matched) / len(plan.actions)) * mean_action_confidence,
                    (
                        f"actions {'/'.join(matched)} ({'/'.join(provenance)}, "
                        f"confidence {mean_action_confidence:.2f})"
                        if matched
                        else ""
                    ),
                )
            )

        if plan.relationship_tuples:
            confidences: list[float] = []
            relationship_reasons: list[str] = []
            seen_relationship_ids: set[str] = set()
            for relation_query in plan.relationship_tuples:
                matches = matching_relationships(segment, relation_query)
                if not matches:
                    return 0.0, [], [], []
                best = matches[0]
                confidences.append(best.confidence)
                if best.relationship_id not in seen_relationship_ids:
                    matched_relationship_evidence.append(best)
                    seen_relationship_ids.add(best.relationship_id)
                relationship_reasons.append(
                    f"relationship {best.source_class}[{best.source_id}] "
                    f"{best.predicate} {best.target_class}[{best.target_id}] "
                    f"({best.directionality}, confidence {best.confidence:.2f}, "
                    f"{best.source_method.value}"
                    + (f", model {best.model_version}" if best.model_version else "")
                    + ")"
                )
            components.append(
                (1.0, sum(confidences) / len(confidences), "; ".join(relationship_reasons))
            )
        elif plan.relationships:
            relation_set = {normalize_predicate(item.predicate) for item in segment.relationships}
            matched = [item for item in plan.relationships if item in relation_set]
            if len(matched) != len(plan.relationships):
                return 0.0, [], [], []
            components.append(
                (
                    1.0,
                    len(matched) / len(plan.relationships),
                    f"relationships {'/'.join(matched)}" if matched else "",
                )
            )

        if plan.speaker:
            speaker_matched = plan.speaker in segment.speakers
            if not speaker_matched:
                return 0.0, [], [], []
            components.append((1.0, 1.0, f"speaker {plan.speaker}"))

        if plan.appearance_constraints and appearance_enabled:
            appearance_score = self._appearance_score(
                plan.appearance_constraints,
                segment,
                appearance_by_id or {},
                appearance_by_track or {},
                appearance_text_vectors or {},
                appearance_encoder,
            )
            if appearance_score is None:
                return 0.0, [], [], []
            for value, reason, evidence in appearance_score:
                components.append((1.25, value, reason))
                if evidence is not None:
                    appearance_match_evidence.append(evidence)

        if not components:
            return 0.0, [], [], []
        total_weight = sum(weight for weight, _, _ in components)
        score = sum(weight * value for weight, value, _ in components) / total_weight
        reasons = [reason for _, value, reason in components if value > 0 and reason]
        return score, reasons, matched_relationship_evidence, appearance_match_evidence

    def _appearance_score(
        self,
        constraints: list[AppearanceConstraint],
        segment: Any,
        appearance_by_id: dict[str, EntityAppearance],
        appearance_by_track: dict[tuple[str, str], EntityAppearance],
        text_vectors: dict[str, list[float]],
        encoder: Any | None,
    ) -> list[tuple[float, str, AppearanceMatchEvidence | None]] | None:
        minimum_similarity = float(
            getattr(self.appearance_settings, "appearance_min_similarity", 0.20)
        )
        minimum_attribute = float(
            getattr(self.appearance_settings, "attribute_min_confidence", 0.45)
        )
        records: dict[str, EntityAppearance] = {}
        for detection in segment.detections:
            record = (
                appearance_by_id.get(detection.appearance_id) if detection.appearance_id else None
            )
            if record is None and detection.track_id:
                record = appearance_by_track.get((segment.video_id, detection.track_id))
            if record is not None:
                records[record.appearance_id] = record

        results: list[tuple[float, str, AppearanceMatchEvidence | None]] = []
        for constraint in constraints:
            candidates = [
                item for item in records.values() if item.class_label == constraint.entity_class
            ]
            # Backward compatibility: a legacy segment with no appearance cache
            # remains searchable, but ranks below a validated appearance match.
            if not candidates:
                results.append((0.15, "appearance evidence unavailable (fallback ranking)", None))
                continue

            explicit = self._explicit_appearance_requirements(constraint)
            ranked: list[tuple[int, int, float, float | None, EntityAppearance, list[str]]] = []
            for candidate in candidates:
                matched = 0
                mismatched = 0
                confidences: list[float] = []
                vias: list[str] = []
                for name, expected in explicit.items():
                    evidence = candidate.attributes.get(name)
                    if evidence is None or evidence.confidence < minimum_attribute:
                        continue
                    if evidence.value == expected:
                        matched += 1
                        confidences.append(evidence.confidence)
                    else:
                        mismatched += 1

                similarity: float | None = None
                if constraint.description and constraint.description in text_vectors:
                    if encoder is not None and (
                        candidate.appearance_embedding_model == encoder.model_name
                        and candidate.appearance_embedding_version == encoder.model_version
                    ):
                        similarity = cosine_similarity(
                            text_vectors[constraint.description], candidate.embedding
                        )
                if matched:
                    vias.append("explicit_attribute")
                if similarity is not None and similarity >= minimum_similarity:
                    vias.append("appearance_similarity")
                ranked.append(
                    (
                        matched,
                        mismatched,
                        sum(confidences) / len(confidences) if confidences else 0.0,
                        similarity,
                        candidate,
                        vias,
                    )
                )

            explicit_matches = [item for item in ranked if item[0] and item[1] == 0]
            usable = explicit_matches
            if not usable:
                usable = [
                    item
                    for item in ranked
                    if item[1] == 0 and item[3] is not None and item[3] >= minimum_similarity
                ]
            if not usable:
                # All sufficiently confident explicit evidence contradicts the
                # query: filter. Otherwise evidence is weak/missing and the old
                # retrieval score is retained as an honest fallback.
                if explicit and ranked and all(item[1] > 0 for item in ranked):
                    return None
                if not explicit and text_vectors and any(item[3] is not None for item in ranked):
                    return None
                results.append(
                    (0.15, "appearance evidence weak or unavailable (fallback ranking)", None)
                )
                continue

            best = max(
                usable,
                key=lambda item: (
                    item[0],
                    item[2],
                    item[3] if item[3] is not None else -1.0,
                    item[4].observation_count,
                ),
            )
            matched, _, confidence, similarity, candidate, vias = best
            value = confidence if matched else max(0.0, similarity or 0.0)
            description = constraint.description
            reasons: list[str] = []
            if matched:
                labels = [f"{key}={value}" for key, value in explicit.items()]
                reasons.append(
                    f"appearance attributes {'/'.join(labels)} "
                    f"({candidate.source_method.value}, confidence {confidence:.2f})"
                )
            if "appearance_similarity" in vias and similarity is not None:
                reasons.append(
                    f"appearance match '{description}' "
                    f"({candidate.appearance_embedding_model}, cosine {similarity:.3f})"
                )
            results.append(
                (
                    value,
                    "; ".join(reasons),
                    AppearanceMatchEvidence(
                        appearance_id=candidate.appearance_id,
                        track_id=candidate.track_id,
                        entity_class=candidate.class_label,
                        attributes=candidate.attributes,
                        description=description,
                        similarity=similarity,
                        matched_via=vias,
                        appearance_embedding_model=candidate.appearance_embedding_model,
                        appearance_embedding_version=candidate.appearance_embedding_version,
                        observation_count=candidate.observation_count,
                    ),
                )
            )
        return results

    @staticmethod
    def _explicit_appearance_requirements(
        constraint: AppearanceConstraint,
    ) -> dict[str, str | bool]:
        requirements: dict[str, str | bool] = {}
        if constraint.color is not None:
            requirements["color"] = constraint.color
        if constraint.upper_clothing_color is not None:
            requirements["upper_clothing_color"] = constraint.upper_clothing_color
        for name in ("glasses", "hat", "bag"):
            value = getattr(constraint, name)
            if value is not None:
                requirements[name] = value
        return requirements


class Neo4jGraphSearchEngine:
    """Allowlisted graph filtering with canonical evidence hydration from SQLite."""

    def __init__(
        self,
        repository: SQLiteRepository,
        query_runner: GraphQueryRunner,
        parser: StructuredQueryParser | None = None,
        compiler: SafeCypherCompiler | None = None,
        appearance_settings: Any | None = None,
    ):
        self.parser = parser or StructuredQueryParser()
        self.compiler = compiler or SafeCypherCompiler()
        self.query_runner = query_runner
        self.local = LocalHybridSearchEngine(
            repository,
            parser=self.parser,
            appearance_settings=appearance_settings,
        )

    def search(
        self, request: SearchRequest, *, plan_override: QueryPlan | None = None
    ) -> SearchResponse:
        plan = plan_override or self.parser.parse(request.query)
        compiled = self.compiler.compile(plan, request.video_ids, request.limit)
        rows = self.query_runner.run_query(compiled.cypher, compiled.parameters)
        candidate_segment_ids = {
            str(row["segment_id"]) for row in rows if row.get("segment_id") is not None
        }
        response = self.local.search(
            request,
            candidate_segment_ids=candidate_segment_ids,
            plan_override=plan,
        )
        return response.model_copy(update={"retrieval_backend": "neo4j"})
