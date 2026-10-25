from __future__ import annotations

import json
from collections import Counter

import pytest
from test_search_and_storage import evidence_segment, video_record

from scripts.build_ava_holdout_benchmark import (
    _runs,
    assert_no_training_overlap,
    build_video_queries,
)
from vidquery.challenge_evaluation import (
    MethodRun,
    RankedMoment,
    _query_metrics,
    _selective_metrics,
    _without_abstention,
    evaluate_challenge,
)
from vidquery.domain import QueryPlan, SearchRequest
from vidquery.phonetics import find_phonetic_matches, phonetic_key, phonetic_similarity
from vidquery.query_planner import QueryPlanningService
from vidquery.reliability import (
    SOURCE_RELIABILITY,
    canonical_source,
    fit_reliability_priors,
    reliability_prior,
)
from vidquery.search import LocalHybridSearchEngine, configured_search_engine
from vidquery.storage import SQLiteRepository

# --------------------------------------------------------------------------
# Phonetic matching


def test_phonetic_key_merges_asr_confusable_sounds():
    assert phonetic_key("binary") == "PNR"
    assert phonetic_key("vandri") == "PNTR"
    assert phonetic_similarity("binary", "vandri") == pytest.approx(0.75)
    assert phonetic_key("") == ""


def test_phonetic_matches_only_unmatched_terms_and_known_variants():
    tokens = ["vandri", "search", "tree", "in", "4", "js"]
    matches = find_phonetic_matches(["binary", "search", "neo4j", "tree"], tokens)
    assert set(matches) == {"binary", "neo4j"}
    assert matches["binary"].surface_tokens == ("vandri",)
    assert matches["binary"].method == "phonetic_key"
    assert matches["neo4j"].surface_tokens == ("4", "js")
    assert matches["neo4j"].method == "known_variant"


def test_phonetic_matching_ignores_short_terms_and_different_onsets():
    assert find_phonetic_matches(["tree"], ["three"]) == {}
    # Same skeleton tail but a different first sound class is rejected.
    assert find_phonetic_matches(["binary"], ["canary"]) == {}
    with pytest.raises(ValueError):
        find_phonetic_matches(["binary"], ["vandri"], threshold=0)


def _misheard_repository(tmp_path) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    base = evidence_segment()
    utterance = base.transcript_segments[0].model_copy(
        update={"text": "today we build a vandri search tree"}
    )
    repository.replace_segments(
        "video-1",
        [
            base.model_copy(
                update={
                    "transcript": utterance.text,
                    "transcript_segments": [utterance],
                }
            )
        ],
    )
    return repository


def test_phonetic_match_is_off_by_default_and_tagged_when_enabled(tmp_path, settings_factory):
    repository = _misheard_repository(tmp_path)
    plan = QueryPlan(intent="transcript_search", spoken_terms=["binary"])
    request = SearchRequest(query="binary", limit=5, ranking="reliability", localize=True)

    default = LocalHybridSearchEngine(repository).search(request, plan_override=plan)
    assert default.results == []

    enabled = configured_search_engine(
        repository, settings_factory(enable_phonetic_match=True), reliability_priors=None
    ).search(request, plan_override=plan)
    assert len(enabled.results) == 1
    result = enabled.results[0]
    sources = {item.source_method for item in result.evidence.contributions}
    assert "phonetic_transcript" in sources
    assert "phonetic match binary~vandri" in result.match_reason
    # A sound-alike hit is weaker than an exact one and is never "supported" alone.
    assert result.evidence.verdict != "supported"


# --------------------------------------------------------------------------
# Reliability fitting


def test_model_names_map_to_prior_keys_for_fitting_and_lookup():
    assert canonical_source("sentence-transformers/all-MiniLM-L6-v2") == "sentence_embedding"
    assert canonical_source("ViT-B-32:openai") == "openclip_appearance"
    assert canonical_source("no_such_model") == "unknown"
    fitted = fit_reliability_priors(
        [
            {"source_method": "sentence-transformers/all-MiniLM-L6-v2", "correct": False,
             "raw_score": 0.4}
            for _ in range(10)
        ]
    )
    assert fitted["sentence_embedding"] == round(5 * 0.6 / 15, 6)


def test_fitter_ignores_zero_score_contributions():
    observations = [
        {"source_method": "whisper_transcript", "correct": False, "raw_score": 0.0}
        for _ in range(20)
    ]
    assert fit_reliability_priors(observations)["whisper_transcript"] == 0.9


def test_partial_learned_priors_fall_back_per_source():
    learned = {"gnn_action_model": 0.1}
    assert reliability_prior("gnn_action_model", learned) == 0.1
    assert reliability_prior("whisper_transcript", learned) == SOURCE_RELIABILITY[
        "whisper_transcript"
    ]


# --------------------------------------------------------------------------
# Selective-prediction metrics


def _moment(verdict: str | None, confidence: float | None) -> RankedMoment:
    return RankedMoment(
        video_id="v",
        segment_start=0.0,
        segment_end=5.0,
        time_point=1.0,
        interval_start=0.0,
        interval_end=2.0,
        score=0.5,
        verdict=verdict,
        confidence=confidence,
        precision="frame",
    )


def _row(item: dict, run: MethodRun) -> dict:
    return {
        "negative": bool(item.get("negative")),
        "metrics": _query_metrics(item, run, {"v": "src"}),
    }


NEGATIVE = {"id": "n", "query": "q", "source_video_id": "src", "negative": True}
POSITIVE = {
    "id": "p",
    "query": "q",
    "source_video_id": "src",
    "relevant": [
        {"source_video_id": "src", "peak_time": 1.0, "start_time": 0.5, "end_time": 1.5},
        {"source_video_id": "src", "peak_time": 30.0, "start_time": 29.0, "end_time": 31.0},
    ],
}


def test_weak_answer_to_negative_is_not_a_rejection():
    weak = MethodRun(ranked=[_moment("weak", 0.2)], selected_intent=None,
                     selected_hypothesis_id=None)
    row = _row(NEGATIVE, weak)
    assert row["metrics"]["weak_answer"] is True
    assert row["metrics"]["negative_rejected"] is False
    assert row["metrics"]["confident_wrong"] is False
    summary = _selective_metrics([row])
    # Answering a negative at all is an error, whatever the verdict.
    assert summary["coverage"] == 1.0
    assert summary["selective_risk"] == 1.0
    assert summary["rejection_recall"] == 0.0


def test_abstention_on_negative_counts_as_rejection():
    abstained = MethodRun(
        ranked=[_moment("insufficient", 0.05)],
        selected_intent=None,
        selected_hypothesis_id=None,
        interpretation="insufficient_evidence",
    )
    summary = _selective_metrics([_row(NEGATIVE, abstained)])
    assert summary["coverage"] == 0.0
    assert summary["selective_risk"] is None
    assert summary["rejection_f1"] == 1.0


def test_recall_aurc_and_auroc_definitions():
    good = MethodRun(ranked=[_moment("supported", 0.9)], selected_intent=None,
                     selected_hypothesis_id=None)
    negative = MethodRun(ranked=[_moment("weak", 0.2)], selected_intent=None,
                         selected_hypothesis_id=None)
    positive_row = _row(POSITIVE, good)
    assert positive_row["metrics"]["precision_at_1"] == 1.0
    assert positive_row["metrics"]["recall_at_cutoff"] == 0.5
    summary = _selective_metrics([positive_row, _row(NEGATIVE, negative)])
    # Ranked by confidence: correct positive first (risk 0), then the negative (risk 0.5).
    assert summary["risk_coverage"] == [
        {"coverage": 0.5, "risk": 0.0},
        {"coverage": 1.0, "risk": 0.5},
    ]
    assert summary["aurc"] == 0.25
    assert summary["answerability_auroc"] == 1.0


def test_methods_without_confidence_have_undefined_ranking_metrics():
    review1 = MethodRun(ranked=[_moment(None, None)], selected_intent=None,
                        selected_hypothesis_id=None)
    summary = _selective_metrics([_row(POSITIVE, review1), _row(NEGATIVE, review1)])
    assert summary["aurc"] is None
    assert summary["answerability_auroc"] is None
    assert summary["selective_risk"] == 0.5


def test_disabling_abstention_answers_with_original_confidence():
    abstained = MethodRun(
        ranked=[_moment("insufficient", 0.05)],
        selected_intent="action_search",
        selected_hypothesis_id=None,
        interpretation="insufficient_evidence",
    )
    answered = _without_abstention(abstained)
    assert answered.interpretation == "single"
    assert answered.ranked[0].verdict is None
    assert answered.ranked[0].confidence == 0.05
    metrics = _row(NEGATIVE, answered)["metrics"]
    assert metrics["negative_rejected"] is False and metrics["confident_wrong"] is True


def test_challenge_results_include_no_abstention_ablation_and_metadata(
    tmp_path, settings_factory
):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    repository.replace_segments("video-1", [evidence_segment()])
    manifest = {
        "schema_version": "challenge-1.0",
        "queries": [
            {
                "id": "ts",
                "query": "deployment",
                "category": "timestamp",
                "source_video_id": "meeting",
                "relevant": [
                    {"source_video_id": "meeting", "peak_time": 6.0, "start_time": 5.0,
                     "end_time": 7.0}
                ],
            },
            {
                "id": "neg",
                "query": "someone riding a bicycle",
                "category": "negative",
                "source_video_id": "meeting",
                "negative": True,
            },
        ],
    }
    dataset = tmp_path / "challenge.json"
    dataset.write_text(json.dumps(manifest), encoding="utf-8")
    planner = QueryPlanningService(settings_factory(enable_query_planner=False))
    result = evaluate_challenge(
        repository, dataset, planner=planner, run_metadata={"reliability_priors": "x"}
    )
    assert set(result["summary"]) == {
        "review1_segment_start",
        "review2_localized",
        "review2_full",
        "review2_full_no_abstention",
    }
    assert result["run_metadata"] == {"reliability_priors": "x"}
    assert result["planner_status_histogram"]
    overall = result["summary"]["review2_full"]["overall"]
    for key in ("aurc", "rejection_f1", "answerability_auroc", "recall_at_cutoff"):
        assert key in overall
    assert result["summary"]["review1_segment_start"]["overall"]["aurc"] is None


# --------------------------------------------------------------------------
# Held-out AVA benchmark builder


def test_runs_merge_consecutive_keyframes():
    assert _runs([905, 903, 904, 910, 911, 920]) == [(903, 905), (910, 911), (920, 920)]


def test_builder_selects_positives_negatives_and_excludes_synonyms():
    video_labels = {
        12: set(range(902, 1798)),  # stand: ubiquitous, excluded
        15: {1000, 1001, 1002, 1100},  # answer phone
        14: {950, 951, 952},  # walk
        37: {1200},  # listen: too rare to be positive but present
    }
    frequency = Counter({74: 100, 80: 90, 8: 80, 57: 70, 29: 60})
    queries, coverage = build_video_queries(
        "vid",
        "ava_vid_w900",
        video_labels,
        frequency,
        positives=5,
        negatives=3,
        max_coverage=0.25,
    )
    positives = [item for item in queries if not item["negative"]]
    negatives = [item for item in queries if item["negative"]]
    assert [item["id"] for item in positives] == ["ava-vid-a15", "ava-vid-a14"]
    phone = positives[0]["relevant"]
    assert phone[0] == {"source_video_id": "ava_vid_w900", "peak_time": 101.0,
                        "start_time": 99.5, "end_time": 102.5}
    assert len(phone) == 2
    negative_labels = [int(item["id"][-2:]) for item in negatives]
    # 74 (listen to) is a synonym of present 37; 57 (text on phone) of present 15.
    assert 74 not in negative_labels and 57 not in negative_labels
    assert negative_labels[:2] == [80, 8]
    assert coverage["excluded_as_ubiquitous"] == ["stand"]


def test_builder_refuses_training_overlap(tmp_path):
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"split": {"train": ["abc"]}}), encoding="utf-8")
    assert assert_no_training_overlap(["xyz"], [metadata])[metadata.as_posix()][
        "overlap_with_benchmark"
    ] == []
    with pytest.raises(SystemExit, match="used to train"):
        assert_no_training_overlap(["abc"], [metadata])
