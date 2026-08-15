from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from vidquery.api import create_app
from vidquery.domain import (
    OCREvidence,
    QueryPlan,
    Relationship,
    RelationshipSource,
    SearchResponse,
    SearchResult,
)
from vidquery.rag import GroundedAnswerService
from vidquery.storage import SQLiteRepository


class MockTransport:
    def __init__(self, answer: dict | None = None, error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return {
            "choices": [
                {"message": {"content": json.dumps(self.answer)}}
            ]
        }


def result(index: int, score: float) -> SearchResult:
    return SearchResult(
        video_id=f"video-{index}",
        video_title=f"Video {index}",
        segment_id=f"video-{index}:{index * 5000}:{(index + 1) * 5000}",
        start_time=float(index * 5),
        end_time=float((index + 1) * 5),
        score=score,
        transcript=f"Architecture deployment evidence {index}",
        speakers=[f"SPEAKER_0{index}"],
        entities=["person", "laptop"],
        actions=["talk to"],
        ocr_evidence=[
            OCREvidence(
                ocr_id=f"ocr-{index}",
                text=f"deployment {index}",
                start_time=float(index * 5),
                end_time=float((index + 1) * 5),
                confidence=0.9,
                source_method=RelationshipSource.EASY_OCR,
            )
        ],
        relationships=[],
        matched_relationships=[],
        thumbnail_url=f"/api/videos/video-{index}/thumbnail?timestamp={index * 5}",
        stream_url=f"/api/videos/video-{index}/stream",
        match_reason="Matched test evidence.",
    )


def search_response(*items: SearchResult) -> SearchResponse:
    return SearchResponse(
        query="Where is deployment discussed?",
        parsed_query=QueryPlan(spoken_terms=["deployment"]),
        results=list(items),
    )


def configured_service(settings_factory, transport, **overrides):
    settings = settings_factory(
        enable_rag_generation=True,
        rag_provider="groq",
        rag_model="test-groq-model",
        rag_top_k=2,
        rag_min_evidence_score=0.5,
        rag_timeout_seconds=7,
        groq_api_key="test-key-not-real",
        **overrides,
    )
    return GroundedAnswerService(settings, transport=transport)


def test_groq_receives_only_top_k_allowlisted_multimodal_evidence(settings_factory):
    transport = MockTransport(
        {"answer": "Deployment is discussed first.", "supported": True, "evidence_ids": ["E1"]}
    )
    service = configured_service(settings_factory, transport)

    outcome = service.generate(
        "Where is deployment discussed?",
        search_response(result(0, 0.9), result(1, 0.8), result(2, 0.7)),
    )

    assert outcome.status == "generated_supported"
    assert outcome.answer is not None and outcome.answer.supported is True
    assert outcome.answer.citations[0].start_time == 0.0
    assert outcome.answer.citations[0].end_time == 5.0
    sent = json.loads(transport.calls[0]["payload"]["messages"][1]["content"])
    assert list(sent["retrieved_evidence"]) == ["E1", "E2"]
    evidence = sent["retrieved_evidence"]["E1"]
    assert set(evidence) == {
        "segment_id",
        "video_id",
        "video_title",
        "start_time",
        "end_time",
        "retrieval_score",
        "transcript",
        "speakers",
        "ocr",
        "entities",
        "actions",
        "relationships",
        "appearance",
    }
    assert "cypher" not in json.dumps(transport.calls[0]["payload"]).lower()
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer test-key-not-real"
    assert transport.calls[0]["headers"]["User-Agent"] == "VidQuery/0.2 grounded-rag"
    assert transport.calls[0]["payload"]["response_format"]["type"] == "json_schema"


def test_rag_caps_unmatched_relationship_projection_to_provider_safe_size(
    settings_factory,
):
    transport = MockTransport(
        {"answer": "Grounded.", "supported": True, "evidence_ids": ["E1"]}
    )
    service = configured_service(settings_factory, transport)
    ranked = result(0, 0.9).model_copy(
        update={
            "relationships": [
                Relationship(
                    relationship_id=f"rel-{index}",
                    source_id="person",
                    target_id=f"object-{index}",
                    predicate="near",
                    confidence=0.8,
                    source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
                    timestamp=float(index),
                )
                for index in range(40)
            ]
        }
    )

    outcome = service.generate("query", search_response(ranked))
    sent = json.loads(transport.calls[0]["payload"]["messages"][1]["content"])

    assert outcome.status == "generated_supported"
    assert len(sent["retrieved_evidence"]["E1"]["relationships"]) == 8


def test_invalid_provider_evidence_id_is_rejected(settings_factory):
    transport = MockTransport(
        {"answer": "Unsupported citation.", "supported": True, "evidence_ids": ["E99"]}
    )
    service = configured_service(settings_factory, transport)

    outcome = service.generate("query", search_response(result(0, 0.9)))

    assert outcome.status == "rejected_invalid_evidence_ids"
    assert outcome.answer is not None and outcome.answer.supported is False
    assert outcome.answer.evidence_ids == []
    assert outcome.answer.citations == []


def test_insufficient_evidence_skips_groq_and_returns_unsupported(settings_factory):
    transport = MockTransport()
    service = configured_service(settings_factory, transport)

    outcome = service.generate("query", search_response(result(0, 0.49)))

    assert outcome.status == "insufficient_evidence"
    assert outcome.answer is not None and outcome.answer.supported is False
    assert transport.calls == []


def test_provider_failure_preserves_normal_search_results(settings_factory):
    transport = MockTransport(error=TimeoutError("quota or network unavailable"))
    service = configured_service(settings_factory, transport)
    response = search_response(result(0, 0.9))

    outcome = service.generate("query", response)

    assert outcome.answer is None
    assert outcome.status == "provider_unavailable"
    assert response.results[0].segment_id == "video-0:0:5000"


def test_search_api_adds_grounded_answer_without_hiding_ranked_results(
    tmp_path, settings_factory
):
    repository = SQLiteRepository(tmp_path / "rag-api.sqlite3")
    application = create_app(settings_factory(), repository)
    ranked = search_response(result(0, 0.9))
    application.state.search = SimpleNamespace(search=lambda _payload: ranked)
    application.state.rag = configured_service(
        settings_factory,
        MockTransport(
            {
                "answer": "Deployment is discussed at the start.",
                "supported": True,
                "evidence_ids": ["E1"],
            }
        ),
    )

    payload = TestClient(application).post("/api/search", json={"query": "deployment"}).json()

    assert payload["results"][0]["segment_id"] == "video-0:0:5000"
    assert payload["generated_answer"]["supported"] is True
    assert payload["generated_answer"]["citations"][0] == {
        "evidence_id": "E1",
        "segment_id": "video-0:0:5000",
        "video_id": "video-0",
        "video_title": "Video 0",
        "start_time": 0.0,
        "end_time": 5.0,
        "stream_url": "/api/videos/video-0/stream",
    }
    assert payload["rag_status"] == "generated_supported"


def test_frontend_keeps_results_and_seeks_answer_citations():
    javascript = open("frontend/assets/app.js", encoding="utf-8").read()
    html = open("frontend/index.html", encoding="utf-8").read()

    assert 'id="generated-answer"' in html
    assert "renderGeneratedAnswer(payload.generated_answer" in javascript
    assert "rankedResults.find" in javascript
    assert "openPlayer(ranked ||" in javascript
    assert '$("#results")' in javascript
    assert "generated_unsupported" not in javascript
    assert not any(value in javascript for value in ("Â", "Ã", "â†", "â€¦", "â€“"))
