"""Constrained, local-first natural-language query planning.

Groq is consulted only when the deterministic parser cannot confidently choose
between supported retrieval modes.  It receives ontologies, never data, and its
output is validated before it can reach either SQLite or Neo4j retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import Field

from .appearance import APPEARANCE_COLORS
from .ava_labels import AVA_V22_ACTIONS
from .config import Settings
from .domain import (
    AppearanceConstraint,
    QueryHypothesis,
    QueryPlan,
    RelationshipQuery,
    StrictModel,
)
from .hypotheses import generate_hypotheses, is_bare_action_homonym
from .rag import (
    GROQ_CHAT_COMPLETIONS_URL,
    ProviderTransport,
    UrllibProviderTransport,
    reasoning_options,
)
from .relationships import RELATIONSHIP_ALIASES, normalize_predicate, relationship_directionality
from .search import (
    ACTION_ALIASES,
    ENTITY_ALIASES,
    OCR_CUE_PATTERN,
    SPEECH_CUE_PATTERN,
    STOPWORDS,
    StructuredQueryParser,
    route_visual_concepts,
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
    # Review-2: every distinct validated interpretation, first one primary.
    # Empty when the query is unambiguous.
    hypotheses: tuple[QueryHypothesis, ...] = ()


def _with_hypotheses(
    hypotheses: list[QueryHypothesis], *, status: str, ambiguities: tuple[str, ...]
) -> QueryPlanningOutcome:
    return QueryPlanningOutcome(
        plan=hypotheses[0].plan,
        alternatives=tuple(item.plan for item in hypotheses[1:]),
        status=status,
        ambiguities=ambiguities,
        hypotheses=tuple(hypotheses),
    )


def _aliases_in(query: str, aliases: dict[str, str]) -> dict[str, str]:
    normalized = " ".join(query.lower().replace("-", " ").split())
    return {
        alias: label
        for alias, label in aliases.items()
        if re.search(rf"\b{re.escape(alias)}\b", normalized)
    }


def query_planner_component_status(settings: Settings) -> str:
    if not settings.enable_query_planner:
        return "disabled"
    if not settings.groq_api_key:
        return "unavailable_missing_api_key"
    if not settings.query_planner_model.strip():
        return "unavailable_missing_model"
    return "available_configured_local_first"


class QueryPlanningService:
    _PLAN_CACHE_SIZE = 128

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
        self.parser = parser or StructuredQueryParser(
            route_visual_concepts=settings.visual_concept_routing
        )
        self.transport = transport or UrllibProviderTransport()
        self._plan_cache: OrderedDict[str, QueryPlanningOutcome] = OrderedDict()
        self.response_cache_path: Path | None = settings.query_plan_cache_path
        self._response_cache_lock = Lock()
        self.allowed_actions = frozenset(AVA_V22_ACTIONS.values())
        self.allowed_entities = frozenset(ENTITY_ALIASES.values())
        self.allowed_relations = frozenset(
            normalize_predicate(value)
            for value in (*RELATIONSHIP_ALIASES.values(), *VIDOR_PREDICATES)
        )

    def plan(self, query: str) -> QueryPlanningOutcome:
        cache_key = " ".join(query.lower().split())
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            self._plan_cache.move_to_end(cache_key)
            return cached

        def finish(outcome: QueryPlanningOutcome) -> QueryPlanningOutcome:
            self._plan_cache[cache_key] = outcome
            self._plan_cache.move_to_end(cache_key)
            if len(self._plan_cache) > self._PLAN_CACHE_SIZE:
                self._plan_cache.popitem(last=False)
            return outcome

        local = self.parser.parse(query)
        reasons = self._ambiguity_reasons(query, local)
        hypotheses = generate_hypotheses(query, local)
        if not reasons:
            if not hypotheses:
                return finish(QueryPlanningOutcome(plan=local))
            # Structurally clear to the parser but semantically open (for
            # example "find the laptop scene"): no provider call is needed,
            # every interpretation is retrieved and fused downstream.
            return finish(_with_hypotheses(
                hypotheses,
                status="deterministic_hypotheses",
                ambiguities=(
                    "The query names a concept that may be visible, spoken, or written; "
                    "all interpretations are retrieved and compared.",
                ),
            ))

        bare_action = self._bare_action_homonym(query, local)
        fallback = self._safe_fallback(query, local, bare_action, reasons, hypotheses)
        if not self.enabled:
            return finish(fallback)
        if not self.api_key or not self.model:
            return finish(self._restatus(fallback, "fallback_provider_unconfigured"))

        payload = self._request_payload(query)
        try:
            envelope = self._cached_response(payload)
            if envelope is None:
                envelope = self.transport.post_json(
                    GROQ_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "VidQuery/0.2 constrained-query-planner",
                    },
                    payload=payload,
                    timeout=self.timeout,
                )
                self._store_response(payload, envelope)
        except Exception as exc:
            LOGGER.warning(
                "Constrained query planning unavailable (%s: %s); using the local parser",
                type(exc).__name__,
                exc,
            )
            # Not cached: once the provider recovers (for example after a rate-limit
            # window), the same query is planned by it again.
            return self._restatus(fallback, "fallback_provider_unavailable")
        try:
            provider = self._parse_provider_plan(envelope)
            planned = self._validated_plan(provider)
            if self.parser.route_visual_concepts:
                planned = route_visual_concepts(planned, query)
        except Exception as exc:
            LOGGER.warning(
                "Constrained query plan rejected (%s: %s); using the local parser",
                type(exc).__name__,
                exc,
            )
            return finish(self._restatus(fallback, "fallback_provider_unavailable"))

        if provider.confidence < self.min_confidence:
            return finish(self._restatus(fallback, "fallback_low_confidence"))
        if bare_action:
            # A model confidence score cannot remove a genuine semantic ambiguity.
            return finish(self._restatus(fallback, "groq_ambiguous_merged"))
        provider_hypothesis = QueryHypothesis(
            hypothesis_id="planner_validated",
            label=self._describe_plan(planned),
            description="Constrained planner interpretation validated against the ontology.",
            plan=planned,
            prior=round(max(0.0, min(1.0, provider.confidence)), 4),
        )
        remaining = max(0.0, 1.0 - provider_hypothesis.prior)
        deterministic = [
            item
            for item in (hypotheses or self._literal_hypotheses(local))
            if item.plan != planned
        ]
        shared = sum(item.prior for item in deterministic) or 1.0
        scaled = [
            item.model_copy(update={"prior": round(remaining * item.prior / shared, 4)})
            for item in deterministic
        ]
        return finish(_with_hypotheses(
            [provider_hypothesis, *scaled],
            status="groq_planned_validated",
            ambiguities=tuple(reasons),
        ))

    @staticmethod
    def _restatus(outcome: QueryPlanningOutcome, status: str) -> QueryPlanningOutcome:
        return QueryPlanningOutcome(
            plan=outcome.plan,
            alternatives=outcome.alternatives,
            status=status,
            ambiguities=outcome.ambiguities,
            hypotheses=outcome.hypotheses,
        )

    @staticmethod
    def _describe_plan(plan: QueryPlan) -> str:
        parts: list[str] = []
        if plan.actions:
            parts.append("action " + "/".join(plan.actions))
        if plan.relationship_tuples:
            parts.extend(
                f"{item.subject} {item.predicate} {item.object}"
                for item in plan.relationship_tuples
            )
        elif plan.visual_entities:
            parts.append("visible " + "/".join(plan.visual_entities))
        if plan.appearance_constraints:
            parts.append("appearance match")
        if plan.spoken_terms:
            parts.append("spoken " + " ".join(plan.spoken_terms))
        if plan.ocr_terms:
            parts.append("on-screen text " + " ".join(plan.ocr_terms))
        if plan.speaker:
            parts.append(plan.speaker)
        return ", ".join(parts) or plan.intent.replace("_", " ")

    @staticmethod
    def _literal_hypotheses(local: QueryPlan) -> list[QueryHypothesis]:
        """The parser's literal reading, kept as a low-prior alternative."""

        if not (local.spoken_terms or local.ocr_terms or local.visual_entities or local.actions):
            return []
        return [
            QueryHypothesis(
                hypothesis_id="literal_terms",
                label="literal terms " + " ".join(local.spoken_terms or local.ocr_terms)
                if (local.spoken_terms or local.ocr_terms)
                else "literal parse",
                description="The deterministic parser's literal reading of the query.",
                plan=local,
                prior=0.3,
            )
        ]

    @staticmethod
    def _bare_action_homonym(query: str, plan: QueryPlan) -> bool:
        return is_bare_action_homonym(query, plan)

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
        hypotheses: list[QueryHypothesis],
    ) -> QueryPlanningOutcome:
        if bare_action or hypotheses:
            return _with_hypotheses(
                hypotheses,
                status="fallback_ambiguous_merged",
                ambiguities=tuple(reasons),
            )
        return QueryPlanningOutcome(
            plan=local,
            status="fallback_deterministic",
            ambiguities=tuple(reasons),
        )

    def _validated_plan(self, provider: ProviderQueryPlan) -> QueryPlan:
        actions = self._validated_values(
            provider.actions, self.allowed_actions, "action", ACTION_ALIASES
        )
        entities = self._validated_values(
            provider.entities, self.allowed_entities, "entity", ENTITY_ALIASES
        )
        normalized_relations = [
            normalize_predicate(RELATIONSHIP_ALIASES.get(value.strip().lower(), value))
            for value in provider.relations
        ]
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

    def _response_key(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _read_response_cache(self) -> dict[str, str]:
        if self.response_cache_path is None or not self.response_cache_path.is_file():
            return {}
        try:
            value = json.loads(self.response_cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOGGER.warning("Ignoring unreadable query plan cache %s", self.response_cache_path)
            return {}
        return value if isinstance(value, dict) else {}

    def _cached_response(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """A stored provider response for exactly this request, if any."""

        if self.response_cache_path is None:
            return None
        with self._response_cache_lock:
            content = self._read_response_cache().get(self._response_key(payload))
        if not isinstance(content, str):
            return None
        return {"choices": [{"message": {"content": content}}]}

    def _store_response(self, payload: dict[str, Any], envelope: dict[str, Any]) -> None:
        if self.response_cache_path is None:
            return
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return
        if not isinstance(content, str):
            return
        with self._response_cache_lock:
            cache = self._read_response_cache()
            cache[self._response_key(payload)] = content
            temporary = self.response_cache_path.with_name(self.response_cache_path.name + ".tmp")
            try:
                self.response_cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, self.response_cache_path)
            except OSError as exc:
                # The plan itself is fine; only the cache could not be saved.
                LOGGER.warning("Could not write query plan cache (%s)", exc)

    @staticmethod
    def _validated_values(
        values: list[str],
        allowed: frozenset[str],
        kind: str,
        aliases: dict[str, str] | None = None,
    ) -> list[str]:
        # A provider may answer with a known alias ("computer"); map it to the label.
        normalized = list(
            dict.fromkeys(
                (aliases or {}).get(value.strip().lower(), value.strip().lower())
                for value in values
            )
        )
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
            **reasoning_options(self.model),
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
                            # Only aliases that occur in the query: the full tables were
                            # over half of the request (about 1,700 tokens) and the
                            # provider's labels are mapped through them locally anyway.
                            "aliases": {
                                "actions": _aliases_in(query, ACTION_ALIASES),
                                "entities": _aliases_in(query, ENTITY_ALIASES),
                                "relations": _aliases_in(query, RELATIONSHIP_ALIASES),
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
