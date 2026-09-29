from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_search_and_storage import evidence_segment, video_record

from vidquery.challenge_evaluation import (
    ChallengeManifestError,
    evaluate_challenge,
    validate_manifest,
    write_challenge_results,
)
from vidquery.domain import TranscriptWord
from vidquery.query_planner import QueryPlanningService
from vidquery.storage import SQLiteRepository

TEMPLATE = Path(__file__).resolve().parents[1] / "evaluation" / "queries_challenge.template.json"


def _manifest(**overrides) -> dict:
    manifest = {
        "schema_version": "challenge-1.0",
        "queries": [
            {
                "id": "ts-deployment",
                "query": "when is deployment mentioned",
                "category": "timestamp",
                "source_video_id": "meeting",
                "expected_intents": ["transcript_search"],
                "relevant": [
                    {
                        "source_video_id": "meeting",
                        "peak_time": 6.3,
                        "start_time": 6.2,
                        "end_time": 6.9,
                    }
                ],
                "tolerance_seconds": 0.5,
            },
            {
                "id": "amb-drive",
                "query": "find drive",
                "category": "ambiguous",
                "source_video_id": "meeting",
                "expected_intents": ["transcript_search"],
                "expected_hypothesis_ids": ["spoken_mention"],
                "relevant": [
                    {
                        "source_video_id": "meeting",
                        "peak_time": 5.7,
                        "start_time": 5.6,
                        "end_time": 6.0,
                    }
                ],
                "tolerance_seconds": 0.5,
            },
            {
                "id": "neg-bicycle",
                "query": "find someone riding a bicycle",
                "category": "negative",
                "source_video_id": "meeting",
                "negative": True,
            },
            {
                "id": "neg-gnn-only",
                "query": "kissing",
                "category": "negative",
                "source_video_id": "meeting",
                "negative": True,
            },
        ],
    }
    manifest.update(overrides)
    return manifest


def _repository(tmp_path) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    base = evidence_segment()
    utterance = base.transcript_segments[0].model_copy(
        update={
            "text": "We drive the deployment plan",
            "start_time": 5.0,
            "end_time": 7.5,
            "words": [
                TranscriptWord(word="We", start_time=5.0, end_time=5.2),
                TranscriptWord(word="drive", start_time=5.7, end_time=6.0),
                TranscriptWord(word="the", start_time=6.0, end_time=6.2),
                TranscriptWord(word="deployment", start_time=6.3, end_time=6.9),
                TranscriptWord(word="plan", start_time=7.0, end_time=7.4),
            ],
        }
    )
    # A weak GNN-only "kiss" prediction that Review-1 would answer confidently.
    segment = base.model_copy(
        update={
            "transcript": utterance.text,
            "transcript_segments": [utterance],
            "actions": ["kiss"],
            "processing_metadata": {
                "frame_sample_rate": 1.0,
                "action_sources": {"kiss": "gnn_action_model"},
                "action_confidences": {"kiss": 0.3},
                "action_instances": [
                    {
                        "action": "kiss",
                        "confidence": 0.3,
                        "timestamp": 8.0,
                        "source_method": "gnn_action_model",
                    }
                ],
            },
        }
    )
    repository.replace_segments("video-1", [segment])
    return repository


def test_template_is_rejected_until_examples_are_replaced():
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    with pytest.raises(ChallengeManifestError, match="template example"):
        validate_manifest(payload)
    for item in payload["queries"]:
        item.pop("status")
    warnings = validate_manifest(payload)
    assert any("ambiguous" in warning for warning in warnings)
    assert any("40-60" in warning for warning in warnings)


def test_manifest_structural_errors_are_explicit():
    with pytest.raises(ChallengeManifestError, match="schema_version"):
        validate_manifest({"schema_version": "1", "queries": [{}]})
    broken = _manifest()
    broken["queries"][0]["relevant"][0]["peak_time"] = 99.0
    with pytest.raises(ChallengeManifestError, match="peak outside"):
        validate_manifest(broken)
    negative_with_relevant = _manifest()
    negative_with_relevant["queries"][2]["relevant"] = [
        {"source_video_id": "meeting", "peak_time": 1, "start_time": 0, "end_time": 2}
    ]
    with pytest.raises(ChallengeManifestError, match="must not list relevant"):
        validate_manifest(negative_with_relevant)


def test_review2_beats_review1_on_timestamp_ambiguity_and_abstention(tmp_path, settings_factory):
    repository = _repository(tmp_path)
    dataset = tmp_path / "challenge.json"
    dataset.write_text(json.dumps(_manifest()), encoding="utf-8")
    planner = QueryPlanningService(settings_factory(enable_query_planner=False))

    result = evaluate_challenge(repository, dataset, planner=planner, limit=5)
    write_challenge_results(result, tmp_path / "out.json")

    review1 = result["summary"]["review1_segment_start"]["overall"]
    localized = result["summary"]["review2_localized"]["overall"]
    review2 = result["summary"]["review2_full"]["overall"]

    # Both systems find the right five-second window ...
    assert review1["precision_at_1"] == 1.0 and review2["precision_at_1"] == 1.0
    # ... but only the localized systems land within +/-0.5 s of the word onset.
    assert review1["within_tolerance_rate"] == 0.0
    assert localized["within_tolerance_rate"] == 1.0
    assert review2["within_tolerance_rate"] == 1.0
    assert review2["mean_abs_timestamp_error_on_hits"] < 0.05
    assert review1["mean_abs_timestamp_error_on_hits"] > 0.3
    assert review2["returned_precision_histogram"] == {"word": 2}

    # Review-1 has no intent decision for ambiguous queries; Review-2 resolves it.
    assert review2["intent_accuracy_strict"] == 1.0
    assert review1["intent_accuracy_strict"] == 0.5

    # Negatives: a bicycle never matches; the GNN-only "kiss" is answered
    # confidently by Review-1 but abstained by Review-2.
    assert review1["negative_rejection_rate"] == 0.5
    assert review2["negative_rejection_rate"] == 1.0
    assert review1["confident_wrong_rate"] > review2["confident_wrong_rate"] == 0.0

    comparison = result["review1_vs_review2"]
    assert comparison["within_tolerance_rate"]["delta"] == 1.0
    assert comparison["negative_rejection_rate"]["delta"] == 0.5
    assert result["category_counts"] == {"ambiguous": 1, "negative": 2, "timestamp": 1}
    assert (tmp_path / "out.json").is_file()


def test_unindexed_video_reference_fails_fast(tmp_path, settings_factory):
    repository = _repository(tmp_path)
    manifest = _manifest()
    manifest["queries"][0]["source_video_id"] = "missing"
    manifest["queries"][0]["relevant"][0]["source_video_id"] = "missing"
    dataset = tmp_path / "challenge.json"
    dataset.write_text(json.dumps(manifest), encoding="utf-8")
    planner = QueryPlanningService(settings_factory(enable_query_planner=False))
    with pytest.raises(ChallengeManifestError, match="not indexed: missing"):
        evaluate_challenge(repository, dataset, planner=planner)
