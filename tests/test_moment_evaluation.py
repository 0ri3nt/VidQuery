from __future__ import annotations

import pytest

from vidquery.challenge_evaluation import MethodRun, RankedMoment
from vidquery.moment_evaluation import (
    _query_row,
    content_blind_baselines,
    predicted_intervals,
    summarize_rows,
    temporal_iou,
)


def _moment(start, end, score=0.5, confidence=0.4, verdict="weak", interval=None):
    interval = interval or (start, end)
    return RankedMoment(
        video_id="v",
        segment_start=start,
        segment_end=end,
        time_point=(interval[0] + interval[1]) / 2,
        interval_start=interval[0],
        interval_end=interval[1],
        score=score,
        verdict=verdict,
        confidence=confidence,
        precision="segment",
    )


def _run(*moments, interpretation="single"):
    return MethodRun(ranked=list(moments), selected_intent=None, selected_hypothesis_id=None,
                     interpretation=interpretation)


POSITIVE = {"id": "p", "kind": "positive", "answerable": True, "video": "V", "start": 5.0,
            "end": 12.0}
NEGATIVE = {"id": "n", "kind": "in_domain_negative", "answerable": False, "video": "V"}


def test_temporal_iou():
    assert temporal_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert temporal_iou((0, 5), (5, 10)) == 0.0
    assert temporal_iou((2, 2), (2, 2)) == 0.0


def test_merge_extends_top_segment_over_adjacent_strong_segments():
    ranked = [_moment(5, 10, score=1.0), _moment(10, 15, score=0.9), _moment(0, 5, score=0.2)]
    assert predicted_intervals(ranked, span="merge", merge_ratio=0.8)[0] == (5, 15)
    assert predicted_intervals(ranked, span="segment")[0] == (5, 10)
    with pytest.raises(ValueError):
        predicted_intervals(ranked, span="other")


def test_stages_and_official_metrics():
    correct = _query_row(POSITIVE, _run(_moment(5, 10)), span="segment", merge_ratio=0.8)
    assert correct["metrics"]["stage"] == "correct"  # IoU 5/7 = 0.71
    assert correct["metrics"]["r1_iou0.7"] == 1.0
    localization = _query_row(
        POSITIVE, _run(_moment(5, 10, interval=(9.8, 10.0))), span="native", merge_ratio=0.8
    )
    assert localization["metrics"]["stage"] == "localization"
    ranking = _query_row(POSITIVE, _run(_moment(20, 25), _moment(5, 10)), span="segment",
                         merge_ratio=0.8)
    assert ranking["metrics"]["stage"] == "ranking"
    assert ranking["metrics"]["r5_iou0.5"] == 1.0 and ranking["metrics"]["r1_iou0.5"] == 0.0
    retrieval = _query_row(POSITIVE, _run(_moment(20, 25)), span="segment", merge_ratio=0.8)
    assert retrieval["metrics"]["stage"] == "retrieval"
    missing = _query_row(POSITIVE, _run(), span="segment", merge_ratio=0.8)
    assert missing["metrics"]["stage"] == "no_result" and missing["metrics"]["top_iou"] == 0.0


def test_abstained_positive_counts_for_ranking_but_not_p_at_1():
    row = _query_row(POSITIVE, _run(_moment(5, 10, verdict="insufficient"),
                                    interpretation="insufficient_evidence"),
                     span="segment", merge_ratio=0.8)
    assert row["metrics"]["r1_iou0.5"] == 1.0
    assert row["metrics"]["precision_at_1"] == 0.0


def test_weak_answer_to_negative_is_not_a_rejection():
    weak = _query_row(NEGATIVE, _run(_moment(0, 5)), span="segment", merge_ratio=0.8)
    rejected = _query_row(NEGATIVE, _run(interpretation="insufficient_evidence"),
                          span="segment", merge_ratio=0.8)
    assert weak["metrics"]["negative_rejected"] is False
    assert rejected["metrics"]["negative_rejected"] is True
    positive = _query_row(POSITIVE, _run(_moment(5, 10, confidence=0.9)), span="segment",
                          merge_ratio=0.8)
    summary = summarize_rows([positive, weak, rejected])
    assert summary["negative_rejection_rate_by_kind"] == {"in_domain_negative": 0.5}
    assert summary["precision_at_1"] == 1.0
    assert summary["answerability_auroc"] == 1.0


def test_content_blind_baselines_choose_window_on_validation_only():
    validation = [{"video": "a", "start": 0.0, "end": 10.0}] * 3
    test = [{"video": "b", "start": 20.0, "end": 30.0}]
    result = content_blind_baselines(validation, test, {"a": 30.0, "b": 30.0})
    fixed = result["fixed_window_from_validation"]
    assert fixed["validation"]["r1_iou0.5"] == 1.0
    assert fixed["test"]["r1_iou0.5"] == 0.0  # the validation prior does not transfer here
    # Two of the six 5 s segments reach IoU 0.5 with a 10 s moment.
    assert result["random_segment"]["test"]["r1_iou0.5"] == pytest.approx(2 / 6, abs=1e-4)
