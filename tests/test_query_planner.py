from __future__ import annotations

import json

import pytest

from vidquery.query_planner import QueryPlanningService
from vidquery.search import StructuredQueryParser


class MockPlannerTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
        )
        if self.error:
            raise self.error
        return {
            "choices": [
                {"message": {"content": json.dumps(self.response)}}
            ]
        }


def configured_planner(settings_factory, transport):
    return QueryPlanningService(
        settings_factory(
            enable_query_planner=True,
            groq_api_key="mock-key",
            query_planner_model="mock-model",
        ),
        transport=transport,
    )


def provider_plan(**overrides):
    value = {
        "intent": "multimodal_search",
        "actions": [],
        "entities": [],
        "relations": [],
        "spoken_concepts": [],
        "ocr_concepts": [],
        "speakers": [],
        "appearance": [],
        "confidence": 0.91,
    }
    value.update(overrides)
    return value


def test_local_parser_distinguishes_driving_from_saying_drive():
    parser = StructuredQueryParser()

    observed = parser.parse("show me someone driving")
    spoken = parser.parse("where did they say drive?")

    assert observed.actions == ["drive"]
    assert observed.spoken_terms == []
    assert spoken.intent == "transcript_search"
    assert spoken.actions == []
    assert spoken.spoken_terms == ["drive"]


def test_local_parser_distinguishes_holding_from_spoken_holding():
    parser = StructuredQueryParser()

    observed = parser.parse("someone holding a mug")
    spoken = parser.parse("where did they say holding?")

    assert observed.visual_entities == ["person", "cup"]
    assert observed.relationship_tuples[0].predicate == "hold"
    assert spoken.actions == []
    assert spoken.relationships == []
    assert spoken.spoken_terms == ["holding"]


def test_local_parser_handles_regular_action_inflection_but_not_spoken_cue():
    parser = StructuredQueryParser()

    observed = parser.parse("find me the kissing scene")
    spoken = parser.parse("where did they say kissing?")

    assert observed.intent == "action_search"
    assert observed.actions == ["kiss"]
    assert observed.spoken_terms == []
    assert spoken.intent == "transcript_search"
    assert spoken.actions == []
    assert spoken.spoken_terms == ["kissing"]


def test_local_parser_distinguishes_ocr_from_transcript():
    parser = StructuredQueryParser()

    ocr = parser.parse("Kubernetes written on screen")
    transcript = parser.parse("where was Kubernetes mentioned?")

    assert ocr.intent == "ocr_search"
    assert ocr.ocr_terms == ["kubernetes"]
    assert ocr.spoken_terms == []
    assert transcript.intent == "transcript_search"
    assert transcript.spoken_terms == ["kubernetes"]
    assert transcript.ocr_terms == []


def test_local_aliases_plural_and_relationship_paraphrase_skip_groq(settings_factory):
    transport = MockPlannerTransport()
    planner = configured_planner(settings_factory, transport)

    plural = planner.plan("cars near people")
    interaction = planner.plan("two people speaking with each other")

    assert plural.plan.visual_entities == ["person", "car"]
    assert plural.plan.relationship_tuples[0].predicate == "near"
    assert interaction.plan.relationship_tuples[0].model_dump() == {
        "subject": "person",
        "predicate": "talk_to",
        "object": "person",
        "directionality": "directed",
    }
    assert transport.calls == []


def test_ambiguous_bare_action_is_exposed_and_merged(settings_factory):
    transport = MockPlannerTransport(provider_plan(intent="action_search", actions=["drive"]))
    outcome = configured_planner(settings_factory, transport).plan("find drive")

    assert outcome.status == "groq_ambiguous_merged"
    assert outcome.plan.actions == ["drive"]
    assert outcome.alternatives[0].spoken_terms == ["drive"]
    assert outcome.ambiguities
    sent = json.loads(transport.calls[0]["payload"]["messages"][1]["content"])
    assert "allowed_actions" in sent
    assert "database" not in json.dumps(sent).lower()
    assert "cypher" not in json.dumps(transport.calls[0]["payload"]).lower()


def test_invalid_groq_label_is_rejected_to_safe_fallback(settings_factory):
    transport = MockPlannerTransport(
        provider_plan(intent="action_search", actions=["teleport"])
    )
    outcome = configured_planner(settings_factory, transport).plan("find drive")

    assert outcome.status == "fallback_provider_unavailable"
    assert outcome.plan.actions == ["drive"]
    assert outcome.alternatives[0].spoken_terms == ["drive"]


def test_groq_unavailable_preserves_deterministic_fallback(settings_factory):
    transport = MockPlannerTransport(error=TimeoutError("offline"))
    outcome = configured_planner(settings_factory, transport).plan("find drive")

    assert outcome.status == "fallback_provider_unavailable"
    assert outcome.plan.actions == ["drive"]
    assert outcome.alternatives[0].spoken_terms == ["drive"]


def test_unknown_multimodal_phrase_accepts_only_validated_groq_plan(settings_factory):
    transport = MockPlannerTransport(
        provider_plan(
            intent="multimodal_search",
            actions=["drive"],
            entities=["car"],
        )
    )
    outcome = configured_planner(settings_factory, transport).plan(
        "locate automobile movement in progress"
    )

    assert outcome.status == "groq_planned_validated"
    assert outcome.plan.actions == ["drive"]
    assert outcome.plan.visual_entities == ["car"]


def test_unclassified_action_like_request_invokes_constrained_groq(settings_factory):
    transport = MockPlannerTransport(
        provider_plan(intent="action_search", actions=["kiss"])
    )

    outcome = configured_planner(settings_factory, transport).plan(
        "find me the smooching scene"
    )

    assert outcome.status == "groq_planned_validated"
    assert outcome.plan.intent == "action_search"
    assert outcome.plan.actions == ["kiss"]
    assert outcome.plan.spoken_terms == []
    assert len(transport.calls) == 1


def test_explicit_unknown_spoken_concept_does_not_invoke_groq(settings_factory):
    transport = MockPlannerTransport()

    outcome = configured_planner(settings_factory, transport).plan(
        "where did they say smooching?"
    )

    assert outcome.status == "deterministic_clear"
    assert outcome.plan.intent == "transcript_search"
    assert outcome.plan.spoken_terms == ["smooching"]
    assert transport.calls == []


def test_low_confidence_groq_plan_is_not_used(settings_factory):
    transport = MockPlannerTransport(
        provider_plan(intent="action_search", actions=["drive"], confidence=0.2)
    )
    outcome = configured_planner(settings_factory, transport).plan(
        "locate automobile movement in progress"
    )

    assert outcome.status == "fallback_low_confidence"
    assert outcome.plan.actions == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("driving", "drive"),
        ("cars", "car"),
        ("people", "person"),
        ("someone holding a mug", "cup"),
    ],
)
def test_local_morphology_and_synonyms(query, expected):
    dump = json.dumps(StructuredQueryParser().parse(query).model_dump())
    assert expected in dump
