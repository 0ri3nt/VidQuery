from __future__ import annotations

from fastapi.testclient import TestClient
from test_search_and_storage import evidence_segment, video_record

from vidquery.api import create_app
from vidquery.domain import QueryPlan, SearchRequest
from vidquery.hypotheses import fuse_hypotheses, generate_hypotheses
from vidquery.models import parse_whisper_result
from vidquery.query_planner import QueryPlanningService
from vidquery.reliability import (
    RawContribution,
    assess_evidence,
    reliability_prior,
    reliability_rank_score,
)
from vidquery.search import LocalHybridSearchEngine, StructuredQueryParser
from vidquery.storage import SQLiteRepository

# --------------------------------------------------------------------------
# Reliability-aware fusion


def test_reliability_priors_rank_transcript_above_weak_gnns():
    assert reliability_prior("whisper_transcript") > reliability_prior("yolo_detection")
    assert reliability_prior("yolo_detection") > reliability_prior("relationship_gnn")
    assert reliability_prior("relationship_gnn") > reliability_prior("gnn_action_model")
    assert reliability_prior("sentence-transformers/all-MiniLM-L6-v2") == reliability_prior(
        "sentence_embedding"
    )
    assert reliability_prior("something_new") == 0.5


def test_exact_transcript_hit_is_supported_but_gnn_only_action_is_not():
    transcript = assess_evidence(
        QueryPlan(intent="transcript_search", spoken_terms=["deployment"]),
        [
            RawContribution("transcript", 1.0, "whisper_transcript"),
            RawContribution("semantic", 0.4, "hashing_embedding"),
        ],
    )
    assert transcript.verdict == "supported"
    assert transcript.confidence > 0.5

    gnn_only = assess_evidence(
        QueryPlan(intent="action_search", actions=["drive"]),
        [RawContribution("action", 0.4, "gnn_action_model")],
    )
    assert gnn_only.verdict == "insufficient"
    assert "down-weighted" in gnn_only.explanation

    with_person = assess_evidence(
        QueryPlan(intent="action_search", actions=["drive"], visual_entities=["person"]),
        [
            RawContribution("action", 0.4, "gnn_action_model"),
            RawContribution("entity", 0.85, "yolo_detection"),
        ],
    )
    assert with_person.verdict == "weak"
    assert gnn_only.confidence < with_person.confidence < transcript.confidence


def test_ground_truth_actions_are_fully_trusted():
    assessment = assess_evidence(
        QueryPlan(intent="action_search", actions=["stand"]),
        [RawContribution("action", 1.0, "ava_ground_truth")],
    )
    assert assessment.verdict == "supported"
    assert assessment.confidence == 1.0
    assert reliability_rank_score(0.5, assessment) == 0.75


def test_search_exposes_evidence_assessment_and_reliability_ranking(tmp_path):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    strong = evidence_segment()
    weak = strong.model_copy(
        update={
            "segment_id": "video-1:0:5000",
            "start_time": 0.0,
            "end_time": 5.0,
            "actions": ["drive"],
            "transcript": "",
            "transcript_segments": [],
            "processing_metadata": {
                "action_sources": {"drive": "gnn_action_model"},
                "action_confidences": {"drive": 0.35},
            },
        }
    )
    repository.replace_segments("video-1", [strong, weak])
    engine = LocalHybridSearchEngine(repository)

    response = engine.search(SearchRequest(query="driving", video_ids=["video-1"]))
    assert response.results[0].evidence is not None
    assert response.results[0].evidence.verdict == "insufficient"
    assert response.interpretation == "insufficient_evidence"
    assert response.results[0].evidence.contributions[0].source_method == "gnn_action_model"

    # A visible person adds corroboration: still not "supported", but no longer abstained.
    with_person = engine.search(SearchRequest(query="someone driving", video_ids=["video-1"]))
    assert with_person.results[0].evidence.verdict == "weak"
    assert with_person.interpretation == "single"

    supported = engine.search(
        SearchRequest(query="where is deployment mentioned", video_ids=["video-1"])
    )
    assert supported.results[0].evidence.verdict == "supported"
    assert supported.interpretation == "single"


# --------------------------------------------------------------------------
# Hypothesis generation


def test_hypotheses_for_bare_action_entity_only_and_bare_concept():
    parser = StructuredQueryParser()

    drive = generate_hypotheses("find drive", parser.parse("find drive"))
    assert [item.hypothesis_id for item in drive] == ["visible_action", "spoken_mention"]
    assert drive[0].plan.actions == ["drive"] and drive[1].plan.spoken_terms == ["drive"]

    laptop = generate_hypotheses("find the laptop scene", parser.parse("find the laptop scene"))
    assert [item.hypothesis_id for item in laptop] == ["visible_object", "spoken_mention"]
    assert laptop[0].plan.visual_entities == ["laptop"]
    assert laptop[1].plan.spoken_terms == ["laptop"]

    concept = generate_hypotheses("deployment", parser.parse("deployment"))
    assert [item.hypothesis_id for item in concept] == ["spoken_mention", "visible_text"]
    assert concept[1].plan.intent == "ocr_search"


def test_cued_and_structured_queries_produce_no_hypotheses():
    parser = StructuredQueryParser()
    for query in (
        "where did they say drive?",
        "Kubernetes written on screen",
        "find a person near a laptop",
        "find the person",
    ):
        assert generate_hypotheses(query, parser.parse(query)) == [], query


def test_planner_returns_hypotheses_without_calling_groq(settings_factory):
    planner = QueryPlanningService(settings_factory(enable_query_planner=False))

    outcome = planner.plan("find the laptop scene")
    assert outcome.status == "deterministic_hypotheses"
    assert [item.hypothesis_id for item in outcome.hypotheses] == [
        "visible_object",
        "spoken_mention",
    ]
    assert outcome.plan == outcome.hypotheses[0].plan
    assert outcome.alternatives == (outcome.hypotheses[1].plan,)

    bare = planner.plan("find drive")
    assert bare.status == "fallback_ambiguous_merged"
    assert [item.hypothesis_id for item in bare.hypotheses] == [
        "visible_action",
        "spoken_mention",
    ]

    clear = planner.plan("find a person near a laptop")
    assert clear.status == "deterministic_clear" and clear.hypotheses == ()


# --------------------------------------------------------------------------
# Fusion and the API contract


def _fused_repository(tmp_path) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    spoken = evidence_segment().model_copy(
        update={
            "transcript": "we will drive to the office later",
            "transcript_segments": [
                evidence_segment().transcript_segments[0].model_copy(
                    update={"text": "we will drive to the office later"}
                )
            ],
            "actions": [],
            "processing_metadata": {},
        }
    )
    repository.replace_segments("video-1", [spoken])
    return repository


def test_bare_action_resolves_to_spoken_reading_when_only_speech_supports_it(tmp_path):
    repository = _fused_repository(tmp_path)
    engine = LocalHybridSearchEngine(repository)
    hypotheses = generate_hypotheses("find drive", StructuredQueryParser().parse("find drive"))
    retrieved = {
        item.hypothesis_id: engine.search(
            SearchRequest(query="find drive", video_ids=["video-1"]), plan_override=item.plan
        ).results
        for item in hypotheses
    }

    fused = fuse_hypotheses(hypotheses, retrieved, limit=5)

    assert fused.interpretation == "resolved"
    assert fused.selected_hypothesis_id == "spoken_mention"
    assert fused.clarification_prompt is None
    assert fused.results[0].supporting_hypotheses == ["spoken_mention"]
    by_id = {item.hypothesis_id: item for item in fused.hypotheses}
    assert by_id["visible_action"].result_count == 0
    assert by_id["spoken_mention"].posterior > by_id["visible_action"].posterior


def test_balanced_support_asks_for_clarification_and_pinning_resolves(tmp_path):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    base = evidence_segment()
    spoken = base.model_copy(
        update={
            "segment_id": "video-1:0:5000",
            "start_time": 0.0,
            "end_time": 5.0,
            "transcript": "drive",
            "transcript_segments": [
                base.transcript_segments[0].model_copy(
                    update={"start_time": 0.0, "end_time": 1.0, "text": "drive"}
                )
            ],
            "actions": [],
            "detections": [],
            "entities": [],
            "relationships": [],
            "processing_metadata": {},
        }
    )
    acted = base.model_copy(
        update={
            "actions": ["drive"],
            "transcript": "",
            "transcript_segments": [],
            "processing_metadata": {
                "action_sources": {"drive": "ava_ground_truth"},
                "action_confidences": {"drive": 1.0},
            },
        }
    )
    repository.replace_segments("video-1", [spoken, acted])
    application = create_app(settings_factory_stub(tmp_path), repository)
    client = TestClient(application)

    payload = client.post("/api/search", json={"query": "find drive"}).json()
    assert payload["interpretation"] == "clarification_suggested"
    assert payload["clarification_prompt"].startswith("Did you mean")
    assert {item["hypothesis_id"] for item in payload["hypotheses"]} == {
        "visible_action",
        "spoken_mention",
    }
    assert all(item["result_count"] == 1 for item in payload["hypotheses"])
    assert len(payload["results"]) == 2

    pinned = client.post(
        "/api/search", json={"query": "find drive", "hypothesis_id": "spoken_mention"}
    ).json()
    assert pinned["interpretation"] == "resolved"
    assert pinned["selected_hypothesis_id"] == "spoken_mention"
    assert [item["supporting_hypotheses"] for item in pinned["results"]] == [["spoken_mention"]]
    assert pinned["parsed_query"]["intent"] == "transcript_search"
    # Every reading stays in the response so the client can switch back.
    assert [item["hypothesis_id"] for item in pinned["hypotheses"]] == [
        "visible_action",
        "spoken_mention",
    ]

    unknown = client.post("/api/search", json={"query": "find drive", "hypothesis_id": "nope"})
    assert unknown.status_code == 422


def settings_factory_stub(tmp_path):
    from vidquery.config import Settings

    settings = Settings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        generated_dir=tmp_path / "generated",
        database_path=tmp_path / "vidquery.sqlite3",
        whisper_cache_dir=tmp_path / "models" / "whisper",
        pyannote_cache_dir=tmp_path / "models" / "pyannote",
        enable_neo4j=False,
        enable_yolo=False,
        enable_whisper=False,
        enable_diarization=False,
        require_diarization=False,
        allow_model_downloads=False,
        enable_relation_gnn=False,
        semantic_retrieval_mode="hashing",
        enable_rag_generation=False,
        enable_query_planner=False,
        enable_ocr=False,
    )
    settings.ensure_directories()
    return settings


def test_unambiguous_api_query_keeps_single_interpretation(tmp_path):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    repository.replace_segments("video-1", [evidence_segment()])
    client = TestClient(create_app(settings_factory_stub(tmp_path), repository))

    payload = client.post(
        "/api/search",
        json={"query": "find a person near a laptop", "ranking": "reliability"},
    ).json()

    assert payload["interpretation"] == "single"
    assert payload["hypotheses"] == []
    assert payload["results"][0]["localization"]["peak_time"] == 5.0
    assert payload["results"][0]["evidence"]["verdict"] in {"supported", "weak"}


# --------------------------------------------------------------------------
# Whisper word timestamps


def test_whisper_words_are_preserved_and_optional():
    payload = {
        "segments": [
            {
                "start": 40.0,
                "end": 43.0,
                "text": " So the deployment",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.05,
                "words": [
                    {"word": " So", "start": 40.0, "end": 40.3, "probability": 0.9},
                    {"word": " the", "start": 40.3, "end": 40.5, "probability": 0.8},
                    {"word": " deployment", "start": 42.3, "end": 42.9, "probability": 0.95},
                ],
            },
            {"start": 43.0, "end": 44.0, "text": " legacy segment"},
        ]
    }
    segments = parse_whisper_result(payload, "video-1")
    assert [word.word for word in segments[0].words] == ["So", "the", "deployment"]
    assert segments[0].words[2].start_time == 42.3
    assert segments[1].words == []
    # Round-trips through the canonical JSON payload without schema changes.
    assert segments[0].model_validate_json(segments[0].model_dump_json()) == segments[0]
