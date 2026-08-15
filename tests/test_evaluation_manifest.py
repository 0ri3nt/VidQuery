from __future__ import annotations

from scripts.validate_multivideo_evaluation import _query_plan_satisfies, _satisfies
from vidquery.domain import CanonicalSegment, OCREvidence


def test_manifest_validator_accepts_ocr_as_text_evidence():
    segment = CanonicalSegment(
        segment_id="ocr",
        video_id="video",
        start_time=0,
        end_time=5,
        ocr_evidence=[
            OCREvidence(
                ocr_id="ocr-1",
                text="Kubernetes deployment",
                start_time=0,
                end_time=3,
                confidence=0.99,
            )
        ],
    )
    query = {
        "expected_entities": [],
        "expected_action": None,
        "expected_speaker": None,
        "expected_spoken_concept": "kubernetes",
        "expected_relation": None,
    }

    assert _satisfies(segment, query)


def test_negative_validation_uses_parsed_constraints_not_vacuous_expectations():
    segment = CanonicalSegment(
        segment_id="visual",
        video_id="video",
        start_time=0,
        end_time=5,
        entities=["person"],
    )

    assert _query_plan_satisfies(segment, "person")
    assert not _query_plan_satisfies(segment, "dog")
