"""Constrained, local-first natural-language query planning.

Groq is consulted only when the deterministic parser cannot confidently choose
between supported retrieval modes.  It receives ontologies, never data, and its
output is validated before it can reach either SQLite or Neo4j retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from .appearance import APPEARANCE_COLORS
from .ava_labels import AVA_V22_ACTIONS
from .config import Settings
from .domain import AppearanceConstraint, QueryPlan, RelationshipQuery, StrictModel
from .rag import (
    GROQ_CHAT_COMPLETIONS_URL,
    ProviderTransport,
    UrllibProviderTransport,
)
from .relationships import RELATIONSHIP_ALIASES, normalize_predicate, relationship_directionality
from .search import (
    ACTION_ALIASES,
    ENTITY_ALIASES,
    OCR_CUE_PATTERN,
    SPEECH_CUE_PATTERN,
    STOPWORDS,
    StructuredQueryParser,
)
from .vidor_relation import VIDOR_PREDICATES

LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
SPEAKER_PATTERN = re.compile(r"^SPEAKER_\d{2,}$")
PlannerIntent = Literal[
    "transcript_search",
    "visual_search",
    "action_search",
    "relationship_search",
    "speaker_search",
    "ocr_search",
    "multimodal_search",
]


class ProviderQueryPlan(StrictModel):
    intent: PlannerIntent
    actions: list[str] = Field(default_factory=list, max_length=10)
    entities: list[str] = Field(default_factory=list, max_length=10)
    relations: list[str] = Field(default_factory=list, max_length=10)
    spoken_concepts: list[str] = Field(default_factory=list, max_length=20)
    ocr_concepts: list[str] = Field(default_factory=list, max_length=20)
    speakers: list[str] = Field(default_factory=list, max_length=2)
    appearance: list[ProviderAppearanceConstraint] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0, le=1)


class ProviderAppearanceConstraint(StrictModel):
    entity_class: str
    color: str | None = None
    upper_clothing_color: str | None = None
    upper_clothing_description: str | None = Field(default=None, max_length=200)
    glasses: bool | None = None
    hat: bool | None = None
    bag: bool | None = None
    description: str | None = Field(default=None, max_length=200)


@dataclass(frozen=True, slots=True)
class QueryPlanningOutcome:
    plan: QueryPlan
    alternatives: tuple[QueryPlan, ...] = ()
    status: str = "deterministic_clear"
    ambiguities: tuple[str, ...] = ()


def query_planner_component_status(settings: Settings) -> str:
    if not settings.enable_query_planner:
        return "disabled"
    if not settings.groq_api_key:
        return "unavailable_missing_api_key"
    if not settings.query_planner_model.strip():
        return "unavailable_missing_model"
    return "available_configured_local_first"


class QueryPlanningService:
    def __init__(
        self,
        settings: Settings,
        *,
        parser: StructuredQueryParser | None = None,
        transport: ProviderTransport | None = None,
    ) -> None:
        self.enabled = settings.enable_query_planner
        self.api_key = settings.groq_api_key
        self.model = settings.query_planner_model.strip()
        self.min_confidence = max(0.0, min(1.0, settings.query_planner_min_confidence))
        self.timeout = max(1.0, min(60.0, settings.query_planner_timeout_seconds))
        self.parser = parser or StructuredQueryParser()
        self.transport = transport or UrllibProviderTransport()
        self.allowed_actions = frozenset(AVA_V22_ACTIONS.values())
        self.allowed_entities = frozenset(ENTITY_ALIASES.values())
        self.allowed_relations = frozenset(
            normalize_predicate(value)
            for value in (*RELATIONSHIP_ALIASES.values(), *VIDOR_PREDICATES)
        )

    def plan(self, query: str) -> QueryPlanningOutcome:
        local = self.parser.parse(query)
        reasons = self._ambiguity_reasons(query, local)
        if not reasons:
            return QueryPlanningOutcome(plan=local)

        bare_action = self._bare_action_homonym(query, local)
        fallback = self._safe_fallback(query, local, bare_action, reasons)
        if not self.enabled:
            return fallback
        if not self.api_key or not self.model:
            return QueryPlanningOutcome(
                plan=fallback.plan,
                alternatives=fallback.alternatives,
                status="fallback_provider_unconfigured",
                ambiguities=fallback.ambiguities,
            )

        try:
            envelope = self.transport.post_json(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "VidQuery/0.2 constrained-query-planner",
                },
                payload=self._request_payload(query),
                timeout=self.timeout,
            )
            provider = self._parse_provider_plan(envelope)
            planned = self._validated_plan(provider)
        except Exception as exc:
            LOGGER.warning("Constrained query planning unavailable: %s", type(exc).__name__)
            return QueryPlanningOutcome(
                plan=fallback.plan,
                alternatives=fallback.alternatives,
                status="fallback_provider_unavailable",
                ambiguities=fallback.ambiguities,
            )

        if provider.confidence < self.min_confidence:
            return QueryPlanningOutcome(
                plan=fallback.plan,
                alternatives=fallback.alternatives,
                status="fallback_low_confidence",
                ambiguities=fallback.ambiguities,
            )
        if bare_action:
            # A model confidence score cannot remove a genuine semantic ambiguity.
            return QueryPlanningOutcome(
                plan=fallback.plan,
                alternatives=fallback.alternatives,
                status="groq_ambiguous_merged",
                ambiguities=fallback.ambiguities,
            )
        return QueryPlanningOutcome(
            plan=planned,
            status="groq_planned_validated",
            ambiguities=tuple(reasons),
        )

    @staticmethod
    def _bare_action_homonym(query: str, plan: QueryPlan) -> bool:
        if SPEECH_CUE_PATTERN.search(query.lower()) or OCR_CUE_PATTERN.search(query.lower()):
            return False
        core = [token for token in TOKEN_PATTERN.findall(query.lower()) if token not in STOPWORDS]
        return len(core) == 1 and core[0] in ACTION_ALIASES and bool(plan.actions)

    def _ambiguity_reasons(self, query: str, plan: QueryPlan) -> list[str]:
        if self._bare_action_homonym(query, plan):
            return ["The query may refer either to an observed action or to a spoken concept."]
        explicitly_linguistic = bool(
            SPEECH_CUE_PATTERN.search(query.lower()) or OCR_CUE_PATTERN.search(query.lower())
        )
        structured = bool(
            plan.visual_entities
            or plan.actions
            or plan.relationships
            or plan.relationship_tuples
            or plan.speaker
            or plan.appearance_constraints
        )
        if plan.spoken_terms and not structured and not explicitly_linguistic:
            return [
                "The deterministic parser could not classify the requested concept as "
                "speech, visible text, an entity, an action, or a relationship."
            ]
        if plan.spoken_terms and (
            plan.visual_entities or plan.relationships or plan.relationship_tuples
        ):
            return [
                "The deterministic parser found unclassified language alongside visual constraints."
            ]
        return []

    @staticmethod
    def _safe_fallback(
        query: str,
        local: QueryPlan,
        bare_action: bool,
        reasons: list[str],
    ) -> QueryPlanningOutcome:
        if not bare_action:
            return QueryPlanningOutcome(
                plan=local,
                status="fallback_deterministic",
                ambiguities=tuple(reasons),
            )
        action = local.actions[0]
        action_plan = local.model_copy(
            update={
                "intent": "action_search",
                "relationships": [],
                "relationship_tuples": [],
                "spoken_terms": [],
                "ocr_terms": [],
            }
        )
        transcript_plan = QueryPlan(
            intent="transcript_search",
            spoken_terms=[action],
        )
        return QueryPlanningOutcome(
            plan=action_plan,
            alternatives=(transcript_plan,),
            status="fallback_ambiguous_merged",
            ambiguities=tuple(reasons),
        )

    def _validated_plan(self, provider: ProviderQueryPlan) -> QueryPlan:
        actions = self._validated_values(provider.actions, self.allowed_actions, "action")
        entities = self._validated_values(provider.entities, self.allowed_entities, "entity")
        normalized_relations = [normalize_predicate(value) for value in provider.relations]
        relations = self._validated_values(normalized_relations, self.allowed_relations, "relation")
        appearance = self._validated_appearance(provider.appearance)
        for constraint in appearance:
            if constraint.entity_class not in entities:
                entities.append(constraint.entity_class)
        if len(provider.speakers) > 1:
            raise ValueError("Only one speaker constraint is supported")
        speakers = [value.upper() for value in provider.speakers]
        if any(not SPEAKER_PATTERN.fullmatch(value) for value in speakers):
            raise ValueError("Invalid speaker label")
        spoken = self._concept_tokens(provider.spoken_concepts)
        ocr = self._concept_tokens(provider.ocr_concepts)
        tuples: list[RelationshipQuery] = []
        if relations and len(entities) >= 2:
            tuples.append(
                RelationshipQuery(
                    subject=entities[0],
                    predicate=relations[0],
                    object=entities[1],
                    directionality=relationship_directionality(relations[0]),
                )
            )
        return QueryPlan(
            intent=provider.intent,
            visual_entities=entities,
            relationships=relations,
            relationship_tuples=tuples,
            actions=actions,
            spoken_terms=spoken,
            ocr_terms=ocr,
            speaker=speakers[0] if speakers else None,
            appearance_constraints=appearance,
        )

    def _validated_appearance(
        self, values: list[ProviderAppearanceConstraint]
    ) -> list[AppearanceConstraint]:
        output: list[AppearanceConstraint] = []
        allowed_colors = frozenset(APPEARANCE_COLORS)
        forbidden = re.compile(
            r"\b(?:race|racial|ethnicity|religion|disabled|disability|gender|"
            r"identity|face recognition|exact age)\b",
            re.IGNORECASE,
        )
        for item in values:
            entity_class = item.entity_class.strip().lower()
            if entity_class not in self.allowed_entities:
                raise ValueError("Unsupported appearance entity label")
            color = item.color.strip().lower() if item.color else None
            clothing_color = (
                item.upper_clothing_color.strip().lower() if item.upper_clothing_color else None
            )
            if color is not None and color not in allowed_colors:
                raise ValueError("Unsupported appearance color")
            if clothing_color is not None and clothing_color not in allowed_colors:
                raise ValueError("Unsupported upper-clothing color")
            descriptions = (
                item.upper_clothing_description,
                item.description,
            )
            if any(value and forbidden.search(value) for value in descriptions):
                raise ValueError("Unsupported sensitive appearance description")
            description = item.description
            if description is None and item.upper_clothing_description:
                description = (
                    f"a person wearing {item.upper_clothing_description}"
                    if entity_class == "person"
                    else item.upper_clothing_description
                )
            if description is None and entity_class == "person":
                phrases = []
                if item.glasses:
                    phrases.append("wearing glasses")
                if item.hat:
                    phrases.append("wearing a hat")
                if item.bag:
                    phrases.append("carrying a bag or backpack")
                description = "a person " + " ".join(phrases) if phrases else None
            output.append(
                AppearanceConstraint(
                    entity_class=entity_class,
                    color=color,
                    upper_clothing_color=clothing_color,
                    upper_clothing_description=item.upper_clothing_description,
                    glasses=item.glasses,
                    hat=item.hat,
                    bag=item.bag,
                    description=description,
                )
            )
        return output

    @staticmethod
    def _validated_values(values: list[str], allowed: frozenset[str], kind: str) -> list[str]:
        normalized = list(dict.fromkeys(value.strip().lower() for value in values))
        invalid = [value for value in normalized if value not in allowed]
        if invalid:
            raise ValueError(f"Unsupported {kind} labels")
        return normalized

    @staticmethod
    def _concept_tokens(concepts: list[str]) -> list[str]:
        tokens: list[str] = []
        for concept in concepts:
            for token in TOKEN_PATTERN.findall(concept.lower()):
                if token not in STOPWORDS and token not in tokens:
                    tokens.append(token)
        return tokens

    def _request_payload(self, query: str) -> dict[str, Any]:
        properties = {
            "intent": {
                "type": "string",
                "enum": [
                    "transcript_search",
                    "visual_search",
                    "action_search",
                    "relationship_search",
                    "speaker_search",
                    "ocr_search",
                    "multimodal_search",
                ],
            },
            "actions": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {"type": "string"}},
            "relations": {"type": "array", "items": {"type": "string"}},
            "spoken_concepts": {"type": "array", "items": {"type": "string"}},
            "ocr_concepts": {"type": "array", "items": {"type": "string"}},
            "speakers": {"type": "array", "items": {"type": "string"}},
            "appearance": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_class": {"type": "string"},
                        "color": {"type": ["string", "null"]},
                        "upper_clothing_color": {"type": ["string", "null"]},
                        "upper_clothing_description": {"type": ["string", "null"]},
                        "glasses": {"type": ["boolean", "null"]},
                        "hat": {"type": ["boolean", "null"]},
                        "bag": {"type": ["boolean", "null"]},
                        "description": {"type": ["string", "null"]},
                    },
                    "required": [
                        "entity_class",
                        "color",
                        "upper_clothing_color",
                        "upper_clothing_description",
                        "glasses",
                        "hat",
                        "bag",
                        "description",
                    ],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        return {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 700,
            "tool_choice": "none",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Convert natural language into the supplied fixed VidQuery schema. "
                        "Use only allowlisted action, entity, and relation labels. Put words "
                        "the user wants heard in spoken_concepts and visible written words in "
                        "ocr_concepts. Appearance may contain only observable clothing color, "
                        "clothing description, glasses, hat, bag, object color, and an open-ended "
                        "visual description. Never infer identity or sensitive traits and never "
                        "invent an appearance fact not requested by the user. Never retrieve data, "
                        "rank results, access storage, or create database queries."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "allowed_actions": sorted(self.allowed_actions),
                            "allowed_entities": sorted(self.allowed_entities),
                            "allowed_relations": sorted(self.allowed_relations),
                            "allowed_appearance_colors": list(APPEARANCE_COLORS),
                            "aliases": {
                                "actions": ACTION_ALIASES,
                                "entities": ENTITY_ALIASES,
                                "relations": RELATIONSHIP_ALIASES,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vidquery_query_plan",
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    @staticmethod
    def _parse_provider_plan(envelope: dict[str, Any]) -> ProviderQueryPlan:
        content = envelope["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        return ProviderQueryPlan.model_validate(parsed)
