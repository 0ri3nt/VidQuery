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
    TranscriptSegment,
    VideoRecord,
    utc_now,
)
from vidquery.pipeline import merge_preserved_evidence
from vidquery.query_planner import QueryPlanningService
from vidquery.rag import (
    DECLINED_ANSWER,
    GroundedAnswerService,
    RAGProviderError,
    verify_answer_claims,
)
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


def test_empty_refusal_is_an_unsupported_answer_not_a_provider_failure():
    # Live Groq output for "Find where SPEAKER_01 discusses Docker." over weak evidence.
    weak = [_result(5.0, "weak", 0.375, transcript="The laptop runs the API.",
                    speakers=("SPEAKER_01",))]
    service = _service({"answer": "", "supported": False, "evidence_ids": []})
    outcome = service.generate("Find where SPEAKER_01 discusses Docker.", _response(weak))
    assert outcome.status == "generated_unsupported"
    assert outcome.answer is not None and outcome.answer.supported is False
    assert outcome.answer.answer == DECLINED_ANSWER


def test_empty_supported_answer_is_still_invalid():
    service = _service({"answer": "", "supported": True, "evidence_ids": ["E1"]})
    outcome = service.generate("q", _response(ARCHITECTURE))
    assert outcome.status == "provider_unavailable" and outcome.answer is None


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


# --- Speakers: spoken words count only for the named speaker -----------------


def _utt(segment_id, start, end, speaker, text):
    return TranscriptSegment(segment_id=segment_id, video_id="v1", start_time=start,
                             end_time=end, text=text, speaker=speaker)


def test_named_speaker_scopes_spoken_terms(repository):
    both = CanonicalSegment(
        segment_id="both", video_id="v1", start_time=0.0, end_time=5.0,
        transcript="Deployment uses Docker. We discuss architecture.",
        transcript_segments=[_utt("a", 0.0, 2.5, "SPEAKER_00", "Deployment uses Docker."),
                             _utt("b", 2.5, 5.0, "SPEAKER_01", "We discuss architecture.")],
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    repository.replace_segments("v1", [both])
    engine = LocalHybridSearchEngine(repository)

    def top(query):
        return engine.search(SearchRequest(query=query, video_ids=["v1"], limit=1,
                                           ranking="reliability", localize=True)).results[0]

    wrong = top("Find where SPEAKER_01 discusses Docker.")
    right = top("Find where SPEAKER_00 discusses Docker.")
    assert "spoken terms docker" not in wrong.match_reason
    assert "spoken terms docker" in right.match_reason
    assert wrong.evidence.confidence < right.evidence.confidence
    # The localized moment is the named speaker's own utterance.
    assert 0.0 <= right.localization.peak_time <= 2.5


def test_refresh_keeps_labelled_utterances_and_speaker_names():
    labelled = _utt("manual:v:10:transcript", 10.0, 15.0, "SPEAKER_00",
                    "Deployment uses Docker for the backend.")
    stale = _utt("tx_old", 10.0, 15.0, "UNKNOWN", "Deployment uses Docker for the back-end.")
    preserved = CanonicalSegment(
        segment_id="p", video_id="v", start_time=10.0, end_time=15.0,
        transcript=labelled.text, transcript_segments=[labelled, stale],
        speakers=["SPEAKER_00"], processing_metadata={"manual_annotation_overlay": True},
    )
    fresh = CanonicalSegment(
        segment_id="f", video_id="v", start_time=10.0, end_time=15.0,
        transcript="Deployment uses Docker for the back-end.",
        transcript_segments=[_utt("tx_new", 10.7, 14.4, "SPEAKER_01",
                                  "Deployment uses Docker for the back-end.")],
        speakers=["SPEAKER_01"],
    )
    merged = merge_preserved_evidence([fresh], [preserved])[0]
    assert [item.segment_id for item in merged.transcript_segments] == [labelled.segment_id]
    assert merged.speakers == ["SPEAKER_00"]


def test_refresh_does_not_accumulate_duplicate_model_utterances():
    preserved = CanonicalSegment(
        segment_id="p", video_id="v", start_time=0.0, end_time=5.0, transcript="I love you.",
        transcript_segments=[_utt("tx_old", 0.0, 2.0, "UNKNOWN", "I love you."),
                             _utt("tx_unique", 2.0, 4.0, "UNKNOWN", "Only in the old run.")],
        processing_metadata={"source": "legacy_ava_fused_adapter"},
    )
    fresh = CanonicalSegment(
        segment_id="f", video_id="v", start_time=0.0, end_time=5.0, transcript="I love you.",
        transcript_segments=[_utt("tx_new", 0.0, 2.0, "UNKNOWN", "I love you.")],
    )
    merged = merge_preserved_evidence([fresh], [preserved])[0]
    assert sorted(item.segment_id for item in merged.transcript_segments) == ["tx_new", "tx_unique"]


def test_speech_implies_a_person_for_answer_checks():
    lecture = [_result(0.0, "supported", 0.7, entities=("laptop",),
                       transcript="We discuss the software architecture.",
                       speakers=("SPEAKER_00",))]
    text = "Someone says “we discuss the software architecture” at the start."
    assert verify_answer_claims(text, lecture) == []
    silent = [_result(0.0, "supported", 0.7, entities=("laptop",))]
    assert verify_answer_claims("Someone uses the laptop.", silent) == [
        "objects not in the cited evidence: ['person']"]


@pytest.mark.parametrize("text", [
    "The person is not cooking.",
    "There is no dog here, only a person near the laptop.",
    "The person isn’t drinking from a cup.",
])
def test_negated_mentions_are_not_claims(text):
    assert verify_answer_claims(text, ARCHITECTURE) == []


def test_unnegated_false_claims_are_still_rejected():
    assert verify_answer_claims("The person is cooking near a dog.", ARCHITECTURE) == [
        "objects not in the cited evidence: ['dog']",
        "actions not in the cited evidence: ['cook']",
    ]


# --- Planner budget: smaller requests, alias mapping, optional response cache -


def test_planner_request_sends_only_query_aliases_and_stays_small():
    planner = _planner(ScriptedTransport())
    payload = planner._request_payload("Find where someone discusses deployment near a laptop.")
    aliases = json.loads(payload["messages"][1]["content"])["aliases"]
    assert aliases["entities"] == {"someone": "person", "laptop": "laptop"}
    assert len(json.dumps(payload)) < 6000  # was about 12,400 characters with every alias


def test_provider_alias_labels_are_mapped_locally():
    plan = _provider_plan([])
    plan.update(intent="visual_search", entities=["computer"])
    outcome = _planner(ScriptedTransport(plan)).plan("find the workstation scene")
    assert outcome.status == "groq_planned_validated"
    assert outcome.plan.visual_entities == ["laptop"]


def test_planner_response_cache_survives_restarts(tmp_path):
    settings = replace(Settings(), enable_query_planner=True, groq_api_key="scripted",
                       query_planner_model="scripted",
                       query_plan_cache_path=tmp_path / "plans.json")
    first_transport = ScriptedTransport(_provider_plan(["kiss"]))
    first = QueryPlanningService(settings, transport=first_transport).plan(
        "find me the smooching scene")
    restarted_transport = ScriptedTransport()
    second = QueryPlanningService(settings, transport=restarted_transport).plan(
        "find me the smooching scene")
    assert first.status == second.status == "groq_planned_validated"
    assert second.plan == first.plan
    assert (first_transport.calls, restarted_transport.calls) == (1, 0)
    other = ScriptedTransport(_provider_plan(["hug"]))
    QueryPlanningService(settings, transport=other).plan("find me the cuddling scene")
    assert other.calls == 1  # a different request is never served from the cache


def test_unwritable_plan_cache_keeps_the_provider_plan(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    settings = replace(Settings(), enable_query_planner=True, groq_api_key="scripted",
                       query_planner_model="scripted",
                       query_plan_cache_path=blocker / "plans.json")
    outcome = QueryPlanningService(
        settings, transport=ScriptedTransport(_provider_plan(["kiss"]))
    ).plan("find me the smooching scene")
    assert outcome.status == "groq_planned_validated"


# --- Routing: visible things are not transcript searches ---------------------


@pytest.mark.parametrize(
    ("query", "concepts"),
    [
        ("person opens the closet door.", ["closet", "door"]),
        ("person takes a towel out of the closet.", ["takes", "towel", "closet"]),
        ("person starts sneezing.", ["sneezing"]),
        ("person puts their shoes on.", ["puts", "shoes"]),
    ],
)
def test_visible_words_are_visual_concepts_not_spoken_terms(query, concepts):
    plan = StructuredQueryParser(route_visual_concepts=True).parse(query)
    assert plan.visual_concepts == concepts
    assert not set(concepts) & set(plan.spoken_terms)


def test_routing_is_off_by_default():
    plan = StructuredQueryParser().parse("person opens the closet door.")
    assert plan.spoken_terms == ["closet", "door"] and plan.visual_concepts == []


def test_speech_cued_words_stay_spoken():
    plan = StructuredQueryParser(route_visual_concepts=True).parse("when does someone say door")
    assert plan.spoken_terms == ["door"] and plan.visual_concepts == []


def test_visual_concept_needs_no_transcript_evidence(repository):
    engine = LocalHybridSearchEngine(repository,
                                     parser=StructuredQueryParser(route_visual_concepts=True))
    result = engine.search(SearchRequest(query="the closet door", video_ids=["v1"], limit=1,
                                         ranking="reliability", localize=False)).results
    modalities = {c.modality for item in result for c in item.evidence.contributions}
    assert "transcript" not in modalities


def test_concept_only_query_keeps_a_spoken_reading():
    from vidquery.hypotheses import generate_hypotheses

    plan = StructuredQueryParser(route_visual_concepts=True).parse("the closet")
    readings = [item.hypothesis_id for item in generate_hypotheses("the closet", plan)]
    assert readings == ["visible_concept", "spoken_mention"]


def test_provider_plans_are_routed_too():
    from vidquery.search import route_visual_concepts

    provider_plan = QueryPlan(intent="multimodal_search", visual_entities=["person"],
                              spoken_terms=["closet", "deployment"])
    routed = route_visual_concepts(provider_plan, "person near the closet discussing deployment")
    assert routed.visual_concepts == ["closet"] and routed.spoken_terms == ["deployment"]
    cued = route_visual_concepts(provider_plan, "when does someone say closet")
    assert cued.spoken_terms == ["closet", "deployment"]


# --- Contradiction: a confidently observed different pose --------------------


def _posed(segment_id, start, poses):
    seg = _segment(segment_id, start, [("person", 0.9)], [])
    instances = [{"action": label, "confidence": confidence, "source_entity_id": "p1",
                  "timestamp": start + offset, "provenance": "gnn_action_model"}
                 for offset, label, confidence in poses]
    return seg.model_copy(update={"processing_metadata": {
        **seg.processing_metadata, "gnn_action_instances": instances}})


def test_pose_contradiction_rule():
    from vidquery.search import pose_contradicted

    standing = _posed("s", 0.0, [(0.0, "stand", 0.9), (1.0, "stand", 0.8)])
    assert pose_contradicted("sit", standing)
    assert not pose_contradicted("carry/hold", standing)  # not a pose
    assert not pose_contradicted("sit", _posed("one", 0.0, [(0.0, "stand", 0.9)]))
    assert not pose_contradicted("sit", _posed("low", 0.0, [(0.0, "stand", 0.9),
                                                             (1.0, "stand", 0.4)]))
    assert not pose_contradicted("sit", _posed("mixed", 0.0, [(0.0, "stand", 0.9),
                                                               (1.0, "sit", 0.7)]))


def test_contradiction_lowers_the_verdict_but_not_the_ranking(repository):
    repository.replace_segments("v1", [
        _posed("standing", 0.0, [(0.0, "stand", 0.9), (1.0, "stand", 0.85)]),
        _posed("unclear", 5.0, [(5.0, "stand", 0.5), (6.0, "sit", 0.4)]),
    ])
    base = LocalHybridSearchEngine(repository, soft_constraints=True,
                                   count_unmatched_constraints=True)
    fixed = LocalHybridSearchEngine(repository, soft_constraints=True,
                                    count_unmatched_constraints=True, count_contradictions=True)
    before, by_before = _by_segment(base, "person sitting.")
    after, by_after = _by_segment(fixed, "person sitting.")
    assert [r.segment_id for r in before.results] == [r.segment_id for r in after.results]
    assert by_after["standing"].evidence.confidence < by_before["standing"].evidence.confidence
    assert by_after["unclear"].evidence.confidence == by_before["unclear"].evidence.confidence
    assert "contradicted by another observed pose" in by_after["standing"].evidence.explanation


# --- Memory: the appearance text-embedding cache is bounded -------------------


def test_text_embedding_cache_is_bounded(monkeypatch):
    import threading
    from collections import OrderedDict

    import torch

    import vidquery.appearance as appearance

    class Model:
        def encode_text(self, tokens):
            return torch.ones((tokens.shape[0], 4))

    encoder = appearance.OpenCLIPAppearanceEncoder.__new__(appearance.OpenCLIPAppearanceEncoder)
    encoder._lock = threading.RLock()
    encoder._text_cache = OrderedDict()
    encoder.tokenizer = lambda texts: torch.zeros((len(texts), 1))
    encoder.model = Model()
    encoder.device = torch.device("cpu")
    monkeypatch.setattr(appearance, "TEXT_CACHE_SIZE", 50)
    for start in range(0, 200, 20):
        vectors = encoder.encode_texts([f"query {index}" for index in range(start, start + 20)])
        assert len(vectors) == 20
    assert len(encoder._text_cache) == 50
    assert "query 199" in encoder._text_cache and "query 0" not in encoder._text_cache
