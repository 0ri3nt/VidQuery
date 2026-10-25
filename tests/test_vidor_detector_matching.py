from __future__ import annotations

import pytest

from vidquery.vidor_improved import match_detections_to_annotations


def _row(tid: int, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"tid": tid, "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}}


OBSERVATIONS = [_row(1, 10, 10, 50, 90), _row(2, 60, 40, 90, 90), _row(3, 0, 0, 5, 5)]
CATEGORIES = {1: "child", 2: "chair", 3: "cup"}


def test_matches_by_iou_and_compatible_class_using_detector_values():
    detections = [
        ([0.11, 0.1, 0.5, 0.9], "person", 0.77),  # matches the child
        ([0.6, 0.4, 0.9, 0.9], "chair", 0.66),  # matches the chair
        ([0.6, 0.4, 0.9, 0.9], "dog", 0.99),  # wrong class: ignored
    ]
    matched = match_detections_to_annotations(
        OBSERVATIONS, CATEGORIES, detections, width=100, height=100
    )
    assert set(matched) == {1, 2}
    person_features, person_class = matched[1]
    assert person_class == "adult"  # YOLO 'person' maps to adult, as at inference
    assert person_features[6] == 0.77  # detector confidence, not 1.0
    assert person_features[7] == 1.0
    assert matched[2][1] == "chair" and matched[2][0][6] == 0.66


def test_low_overlap_unsupported_and_duplicate_detections_do_not_match():
    detections = [
        ([0.0, 0.0, 0.05, 0.05], "mouse", 0.9),  # unsupported COCO class
        ([0.6, 0.4, 0.9, 0.9], "chair", 0.5),
        ([0.61, 0.41, 0.9, 0.9], "chair", 0.6),  # second chair: one-to-one
        ([0.3, 0.3, 0.4, 0.4], "person", 0.9),  # IoU with child far below 0.5
    ]
    matched = match_detections_to_annotations(
        OBSERVATIONS, CATEGORIES, detections, width=100, height=100
    )
    assert set(matched) == {2}
    assert matched[2][0][6] in {0.5, 0.6}


def test_minimum_iou_is_respected():
    detections = [([0.6, 0.4, 0.9, 0.9], "chair", 0.5)]
    assert match_detections_to_annotations(
        OBSERVATIONS, CATEGORIES, detections, width=100, height=100, minimum_iou=1.01
    ) == {}
    with pytest.raises(KeyError):
        match_detections_to_annotations(
            [_row(9, 0, 0, 1, 1)], CATEGORIES, detections, width=100, height=100
        )
