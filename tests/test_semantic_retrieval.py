from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from vidquery.search import (
    HashingTextEncoder,
    LocalHybridSearchEngine,
    SentenceTransformerTextEncoder,
    StructuredQueryParser,
    configured_text_encoder,
    sentence_embedding_status,
)


def test_sentence_embedding_handles_deployment_paraphrase():
    model = "sentence-transformers/all-MiniLM-L6-v2"
    if sentence_embedding_status(model) != "available_cached":
        pytest.skip("cached sentence-transformer checkpoint is unavailable")
    semantic = SentenceTransformerTextEncoder(model, device="cpu")
    hashing = HashingTextEncoder()
    query = "Find where deployment is discussed"
    transcript = "We will release the application to production tomorrow."

    semantic_score = semantic.cosine(semantic.encode(query), semantic.encode(transcript))
    hashing_score = hashing.cosine(hashing.encode(query), hashing.encode(transcript))

    assert semantic_score > hashing_score
    assert semantic_score > 0.2


def test_search_scoring_reuses_cached_segment_embedding():
    class CountingEncoder:
        model_name = "test-semantic"
        is_semantic = True

        def __init__(self):
            self.calls = 0

        def encode(self, _text):
            self.calls += 1
            return [1.0, 0.0]

        @staticmethod
        def cosine(left, right):
            return sum(a * b for a, b in zip(left, right, strict=True))

    encoder = CountingEncoder()
    engine = LocalHybridSearchEngine(cast(Any, SimpleNamespace()), encoder=encoder)
    plan = StructuredQueryParser().parse("deployment")
    segment = SimpleNamespace(
        transcript="release to production",
        entities=[],
        actions=[],
        relationships=[],
        ocr_evidence=[],
        speakers=[],
        processing_metadata={},
        embedding=[1.0, 0.0],
    )

    score, reasons, _ = engine._score(plan, [1.0, 0.0], segment)

    assert encoder.calls == 0
    assert score > 0
    assert any("sentence embedding similarity" in reason for reason in reasons)


def test_low_similarity_without_lexical_evidence_is_rejected():
    class WeakSemanticEncoder:
        model_name = "test-semantic"
        is_semantic = True

        @staticmethod
        def encode(_text):
            return [1.0, 0.0]

        @staticmethod
        def cosine(_left, _right):
            return 0.19

    engine = LocalHybridSearchEngine(
        cast(Any, SimpleNamespace()), encoder=WeakSemanticEncoder()
    )
    plan = StructuredQueryParser().parse("zzzxylophone quantum penguin")
    segment = SimpleNamespace(
        transcript="architecture deployment graph database",
        entities=[],
        actions=[],
        relationships=[],
        ocr_evidence=[],
        speakers=[],
        processing_metadata={},
        embedding=[1.0, 0.0],
    )

    score, reasons, matched = engine._score(plan, [1.0, 0.0], segment)

    assert score == 0.0
    assert reasons == []
    assert matched == []


def test_configured_encoder_falls_back_when_checkpoint_is_missing(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("checkpoint is unavailable")

    monkeypatch.setattr("vidquery.search.SentenceTransformerTextEncoder", unavailable)
    settings = SimpleNamespace(
        semantic_retrieval_mode="sentence_transformer",
        sentence_embedding_model="missing",
        sentence_embedding_device="cpu",
        allow_model_downloads=False,
    )

    assert isinstance(configured_text_encoder(settings), HashingTextEncoder)
