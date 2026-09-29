from __future__ import annotations

from math import hypot

from .domain import BoundingBox


def normalize_bbox(box: BoundingBox, width: int, height: int) -> BoundingBox:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return BoundingBox(
        x1=min(1.0, max(0.0, box.x1 / width)),
        y1=min(1.0, max(0.0, box.y1 / height)),
        x2=min(1.0, max(0.0, box.x2 / width)),
        y2=min(1.0, max(0.0, box.y2 / height)),
    )


def centroid(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def bbox_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a < start_a or end_b < start_b:
        raise ValueError("interval end must be greater than or equal to start")
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def normalized_centroid_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax, ay = centroid(box_a)
    bx, by = centroid(box_b)
    return hypot(ax - bx, ay - by)
