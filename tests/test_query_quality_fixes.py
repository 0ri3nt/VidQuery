"""Regression tests for the confirmed issues in docs/QUERY_QUALITY_AUDIT_2026-10-26.md."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vidquery.config import Settings
from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    EvidenceAssessment,
    ProcessingState,
    QueryPlan,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TemporalLocalization,
    VideoRecord,
    utc_now,
)
from vidquery.query_planner import QueryPlanningService
from vidquery.rag import GroundedAnswerService, RAGProviderError, verify_answer_claims
from vidquery.reliability import UNMATCHED_CONSTRAINT, reliability_rank_score
from vidquery.search import LocalHybridSearchEngine, StructuredQueryParser
from vidquery.storage import SQLiteRepository

# --- Query parsing: function words are not speech-search terms -------------


@pytest.mark.parametrize(
    ("query", "forbidden"),
    [
        ("he walks into the room.", {"he", "into"}),
        ("person puts their shoes on.", {"their"}),
        ("person takes a towel out of the closet.", {"out"}),
        ("a person turns off its light.", {"its", "off"}),
        ("she opens the door then sits down.", {"she", "then", "down"}),
        ("person's phone is on the table.", {"s"}),
        ("person starts sneezing.", {"starts"}),
    ],
)
def test_function_words_are_not_spoken_terms(query, forbidden):
    plan = StructuredQueryParser().parse(query)
    assert not forbidden & set(plan.spoken_terms), plan.spoken_terms


@pytest.mark.parametrize(
    ("query", "spoken", "speaker"),
    [
        ("Find where SPEAKER_01 discusses architecture.", ["architecture"], "SPEAKER_01"),
        ("when does someone say deployment", ["deployment"], None),
        ("Find where someone talks about Kubernetes.", ["kubernetes"], None),
        ("Find where deployment is discussed.", ["deployment"], None),
        ("SQLite metadata", ["sqlite", "metadata"], None),
    ],
)
def test_speech_queries_keep_their_content_words(query, spoken, speaker):
    plan = StructuredQueryParser().parse(query)
    assert plan.spoken_terms == spoken
    assert plan.speaker == speaker


# --- Verdicts: unmatched constraints count, ranking does not move ------------


def _segment(segment_id: str, start: float, entities, actions) -> CanonicalSegment:
    detections = [
        Detection(
            detection_id=f"{segment_id}-d{index}", frame_id="f0", video_id="v1", timestamp=start,
            class_id=index, class_label=label, confidence=confidence,
            pixel_bbox=BoundingBox(x1=64, y1=48, x2=320, y2=432),
            normalized_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.9),
            centroid=(192.0, 240.0), centroid_normalized=(0.3, 0.5),
        )
        for index, (label, confidence) in enumerate(entities)
    ]
    return CanonicalSegment(
        segment_id=segment_id, video_id="v1", start_time=start, end_time=start + 5,
        transcript="", frame_ids=["f0"], detections=detections,
        entities=[label for label, _ in entities], actions=actions,
        processing_metadata={
            "action_sources": {action: "gnn_action_model" for action in actions},
            "action_confidences": {action: 0.6 for action in actions},
        },
    )


@pytest.fixture
def repository(tmp_path):
    repo = SQLiteRepository(tmp_path / "verdicts.sqlite3")
    now = utc_now()
    repo.create_video(VideoRecord(
        video_id="v1", display_name="v1.mp4", original_filename="v1.mp4",
        stored_path=str(tmp_path / "v1.mp4"), content_sha256="0" * 64, duration=30.0,
        width=640, height=480, fps=25.0, has_audio=False, upload_time=now,
        updated_time=now, state=ProcessingState.READY,
    ))
    repo.replace_segments("v1", [
        _segment("person-only", 0.0, [("person", 0.91)], []),
        _segment("person-cook", 5.0, [("person", 0.91)], ["cook"]),
        _segment("person-laptop", 10.0, [("person", 0.91), ("laptop", 0.8)], []),
        _segment("laptop-only", 15.0, [("laptop", 0.8)], []),
    ])
    return repo


def _by_segment(engine, query):
    response = engine.search(SearchRequest(query=query, video_ids=["v1"], limit=10,
                                           ranking="reliability", localize=False))
    return response, {item.segment_id: item for item in response.results}


def test_missing_action_no_longer_outscores_a_matched_action(repository):
    legacy = LocalHybridSearchEngine(repository, soft_constraints=True)
    fixed = LocalHybridSearchEngine(repository, soft_constraints=True,
                                    count_unmatched_constraints=True)
    _, before = _by_segment(legacy, "person cooking.")
    _, after = _by_segment(fixed, "person cooking.")
    # The audited inversion: no cooking at all was "supported", real cooking "weak".
    assert before["person-only"].evidence.verdict == "supported"
    assert before["person-only"].evidence.confidence > before["person-cook"].evidence.confidence
    assert after["person-only"].evidence.confidence < after["person-cook"].evidence.confidence
    assert after["person-only"].evidence.verdict != "supported"
    unmatched = [c for c in after["person-only"].evidence.contributions
                 if c.source_method == UNMATCHED_CONSTRAINT]
    assert [(c.modality, c.raw_score, c.weighted_score) for c in unmatched] == [("action", 0, 0)]
    assert "Unmatched query constraints (1 action)" in after["person-only"].evidence.explanation


def test_missing_object_lowers_confidence(repository):
    fixed = LocalHybridSearchEngine(repository, soft_constraints=True,
                                    count_unmatched_constraints=True)
    _, results = _by_segment(fixed, "person with a laptop.")
    assert results["person-only"].evidence.confidence < results["person-laptop"].evidence.confidence


@pytest.mark.parametrize("query", ["person cooking.", "person with a laptop.",
                                   "person cooking near a laptop.", "laptop."])
def test_ranking_and_scores_are_unchanged(repository, query):
    legacy = LocalHybridSearchEngine(repository, soft_constraints=True)
    fixed = LocalHybridSearchEngine(repository, soft_constraints=True,
                                    count_unmatched_constraints=True)
    before, _ = _by_segment(legacy, query)
    after, _ = _by_segment(fixed, query)
    assert [r.segment_id for r in before.results] == [r.segment_id for r in after.results]
    assert [r.score for r in before.results] == [r.score for r in after.results]
    for old, new in zip(before.results, after.results, strict=True):
        assert reliability_rank_score(old.score, old.evidence) == pytest.approx(
            reliability_rank_score(new.score, new.evidence))


def test_flag_off_and_strict_mode_are_unchanged(repository):
    legacy = LocalHybridSearchEngine(repository, soft_constraints=True)
    flag_off = LocalHybridSearchEngine(repository, soft_constraints=True,
                                       count_unmatched_constraints=False)
    strict = LocalHybridSearchEngine(repository)
    strict_fixed = LocalHybridSearchEngine(repository, count_unmatched_constraints=True)
    for query in ("person cooking.", "person with a laptop."):
        assert _by_segment(legacy, query)[0] == _by_segment(flag_off, query)[0]
        assert _by_segment(strict, query)[0] == _by_segment(strict_fixed, query)[0]
    _, results = _by_segment(flag_off, "person cooking.")
    assert all(item.evidence.ranking_confidence is None for item in results.values())


# --- Groq answers: claims and verdicts are checked against the evidence -----


class ScriptedTransport:
    def __init__(self, *answers, error: Exception | None = None):
        self.answers = list(answers)
        self.error = error
        self.calls = 0

    def post_json(self, url, *, headers, payload, timeout):
        self.calls += 1
        if self.error is not None and self.calls == 1:
            raise self.error
        return {"choices": [{"message": {"content": json.dumps(self.answers.pop(0))}}]}


def _result(start, verdict, confidence, *, entities=("person",), actions=(), transcript="",
            speakers=(), title="final_demo.mp4"):
    return SearchResult(
        video_id="v1", video_title=title, segment_id=f"v1-{start}", start_time=start,
        end_time=start + 5, score=0.6, transcript=transcript, speakers=list(speakers),
        entities=list(entities), actions=list(actions), relationships=[],
        matched_relationships=[], appearance_matches=[], thumbnail_url="/t", stream_url="/s",
        match_reason="Matched.",
        localization=TemporalLocalization(
            peak_time=start + 1, start_time=start, end_time=start + 5, precision="frame",
            source="detector", confidence=0.5),
        evidence=EvidenceAssessment(confidence=confidence, verdict=verdict),
    )


def _service(*answers):
    settings = replace(Settings(), enable_rag_generation=True, rag_provider="groq",
                       groq_api_key="scripted", rag_model="scripted")
    return GroundedAnswerService(settings, transport=ScriptedTransport(*answers))


def _response(results, interpretation="single"):
    return SearchResponse(query="q", parsed_query=QueryPlan(intent="multimodal_search"),
                          results=results, interpretation=interpretation)


ARCHITECTURE = [
    _result(0.0, "supported", 0.698, entities=("person", "laptop", "whiteboard"),
            transcript="We discuss the software architecture.", speakers=("SPEAKER_00",)),
    _result(15.0, "supported", 0.685, entities=("person", "laptop", "whiteboard"),
            transcript="This whiteboard shows the retrieval architecture.",
            speakers=("SPEAKER_01",)),
]


def test_real_grounded_answer_passes_verification():
    # A real Groq answer from the audit; its times and quote match the evidence.
    text = ("Architecture is discussed in the video at the beginning (0–5 s) where the "
            "speakers say, “We discuss the software architecture,” and again around "
            "15–20 s where they refer to the whiteboard showing the retrieval architecture.")
    assert verify_answer_claims(text, ARCHITECTURE) == []
    service = _service({"answer": text, "supported": True, "evidence_ids": ["E1", "E2"]})
    outcome = service.generate("Find where architecture is discussed.", _response(ARCHITECTURE))
    assert outcome.status == "generated_supported" and outcome.answer.supported


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Architecture is discussed at 00:42.", "timestamps outside every cited segment"),
        ("It is discussed between 30 and 35 seconds.", "timestamps outside every cited segment"),
        ("A person is cooking at the whiteboard.", "actions not in the cited evidence"),
        ("A dog sits near the laptop.", "objects not in the cited evidence"),
        ("SPEAKER_03 discusses architecture.", "speakers not in the cited evidence"),
        ("They say “we deploy with Kubernetes”.", "quotes not found"),
        ("The match has confidence 0.97.", "scores or confidences not in the cited evidence"),
        ("Architecture is discussed (92% confidence).", "scores or confidences not in the cited"),
    ],
)
def test_unbacked_claims_are_found(text, reason):
    assert any(item.startswith(reason) for item in verify_answer_claims(text, ARCHITECTURE))


def test_backed_numbers_times_and_speakers_pass():
    text = ("SPEAKER_01 discusses it at 00:16, between 15 and 20 s, with confidence 0.69; "
            "SPEAKER_00 says “we discuss the software architecture”.")
    assert verify_answer_claims(text, ARCHITECTURE) == []


def test_speech_claim_without_transcript_is_rejected():
    silent = [_result(15.0, "supported", 0.5, entities=("person", "cup"), actions=("drink",))]
    assert verify_answer_claims("The person says hello while drinking.", silent) == [
        "cites speech but no cited segment has a transcript"]


def test_claims_beyond_evidence_are_rejected_by_the_service():
    cup = [_result(15.0, "supported", 0.5, entities=("person", "cup"), actions=("drink",),
                   title="OM2NF.mp4")]
    for text in ("The person drinks at 00:42.", "The person is cooking at the stove."):
        service = _service({"answer": text, "supported": True, "evidence_ids": ["E1"]})
        outcome = service.generate("q", _response(cup))
        assert outcome.status == "rejected_unverified_claims"
        assert not outcome.answer.supported


def test_supported_claim_over_weak_verdicts_is_downgraded():
    # A real Groq answer from the audit: correct, but every cited verdict was weak.
    weak = [_result(start, "weak", conf, entities=("person", "cup"), actions=("carry/hold",),
                    title="I2DHG.mp4")
            for start, conf in ((15.0, 0.3778), (25.0, 0.3611), (20.0, 0.3688))]
    service = _service({"answer": "Yes, the person is holding a cup.", "supported": True,
                        "evidence_ids": ["E1", "E2", "E3"]})
    outcome = service.generate("person holding a cup.", _response(weak))
    assert outcome.status == "downgraded_weak_evidence"
    assert not outcome.answer.supported
    assert len(outcome.answer.citations) == 3
    assert "system verdict: weak" in outcome.answer.answer
    assert "I2DHG.mp4 00:15-00:20" in outcome.answer.answer


def test_supported_claim_over_insufficient_evidence_is_downgraded():
    poor = [_result(0.0, "insufficient", 0.05, entities=("person",), title="I2DHG.mp4")]
    service = _service({"answer": "Yes, a person is visible.", "supported": True,
                        "evidence_ids": ["E1"]})
    outcome = service.generate("q", _response(poor, interpretation="insufficient_evidence"))
    assert outcome.status == "downgraded_insufficient_evidence"
    assert outcome.answer.answer.startswith("No retrieved moment has sufficient evidence")


def test_answer_prompt_carries_the_system_interpretation():
    payload = _service()._request_payload("q", {}, "insufficient_evidence")
    user = json.loads(payload["messages"][1]["content"])
    assert user["system_interpretation"] == "insufficient_evidence"
    assert "evidence_verdict is 'supported'" in payload["messages"][0]["content"]


# --- Groq planner: fallback is logged and not cached on provider failure -----


def _planner(transport):
    settings = replace(Settings(), enable_query_planner=True, groq_api_key="scripted",
                       query_planner_model="scripted")
    return QueryPlanningService(settings, transport=transport)


def _provider_plan(actions):
    return {"intent": "action_search", "actions": actions, "entities": [], "relations": [],
            "spoken_concepts": [], "ocr_concepts": [], "speakers": [], "appearance": [],
            "confidence": 0.91}


def test_provider_failure_falls_back_visibly_and_retries_later(caplog):
    transport = ScriptedTransport(
        _provider_plan(["kiss"]),
        error=RAGProviderError("Groq request failed: HTTPError (HTTP 429)"),
    )
    planner = _planner(transport)
    with caplog.at_level("WARNING"):
        first = planner.plan("find me the smooching scene")
    assert first.status == "fallback_provider_unavailable"
    assert "HTTP 429" in caplog.text and "using the local parser" in caplog.text
    second = planner.plan("find me the smooching scene")
    assert second.status == "groq_planned_validated" and second.plan.actions == ["kiss"]
    assert transport.calls == 2


def test_rejected_provider_plan_is_still_cached():
    transport = ScriptedTransport(_provider_plan(["teleport"]))
    planner = _planner(transport)
    assert planner.plan("find drive").status == "fallback_provider_unavailable"
    assert planner.plan("find drive").status == "fallback_provider_unavailable"
    assert transport.calls == 1


def test_ui_shows_planner_and_answer_notices():
    script = Path("frontend/assets/app.js").read_text(encoding="utf-8")
    page = Path("frontend/index.html").read_text(encoding="utf-8")
    assert 'id="pipeline-notices"' in page
    for status in ("fallback_provider_unavailable", "provider_unavailable",
                   "rejected_unverified_claims", "downgraded_weak_evidence"):
        assert f"{status}:" in script
