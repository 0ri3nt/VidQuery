from collections import Counter

from vidquery.ava_coverage import VideoCoverage, select_coverage_videos


def _coverage(video_id: str, counts: dict[int, int]) -> VideoCoverage:
    return VideoCoverage(video_id, Counter(counts), "train")


def test_coverage_selector_prioritizes_missing_and_rare_actions_deterministically():
    coverage = {
        "current": _coverage("current", {1: 10, 2: 1}),
        "broad": _coverage("broad", {2: 4, 3: 5}),
        "narrow": _coverage("narrow", {2: 4}),
        "finish": _coverage("finish", {3: 5}),
    }

    first = select_coverage_videos(
        coverage,
        {"current"},
        rare_max_support=1,
        target_support=5,
        max_additional_videos=3,
    )
    second = select_coverage_videos(
        coverage,
        {"current"},
        rare_max_support=1,
        target_support=5,
        max_additional_videos=3,
    )

    assert first["selected_video_ids"] == second["selected_video_ids"]
    assert first["selected_video_ids"] == ["broad"]
    action_two = next(row for row in first["actions"] if row["action_id"] == 2)
    action_three = next(row for row in first["actions"] if row["action_id"] == 3)
    assert action_two["current_support"] == 1
    assert action_two["expected_support_after_download"] == 5
    assert action_three["expected_support_after_download"] == 5


def test_coverage_selector_records_unreachable_targets_without_inventing_support():
    coverage = {
        "current": _coverage("current", {1: 2}),
        "candidate": _coverage("candidate", {1: 1}),
    }
    result = select_coverage_videos(
        coverage,
        {"current"},
        rare_max_support=2,
        target_support=5,
        max_additional_videos=2,
    )
    assert 1 in result["priority_actions_below_target_after_selection"]
    action = next(row for row in result["actions"] if row["action_id"] == 1)
    assert action["expected_support_after_download"] == 3
