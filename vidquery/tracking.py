from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from .domain import Detection, Relationship
from .geometry import bbox_iou


@dataclass(slots=True)
class _ActiveTrack:
    track_id: str
    detection: Detection


def _track_id(video_id: str, class_label: str, ordinal: int) -> str:
    raw = f"{video_id}:{class_label.strip().lower()}:{ordinal}"
    return f"trk_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _centroid_distance(left: Detection, right: Detection) -> float:
    return (
        (left.centroid_normalized[0] - right.centroid_normalized[0]) ** 2
        + (left.centroid_normalized[1] - right.centroid_normalized[1]) ** 2
    ) ** 0.5


def assign_detection_tracks(
    detections: list[Detection],
    *,
    iou_threshold: float = 0.3,
    centroid_threshold: float = 0.15,
    max_gap_seconds: float = 1.5,
) -> list[Detection]:
    """Assign deterministic, class-aware tracks without changing detection IDs."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    if centroid_threshold <= 0:
        raise ValueError("centroid_threshold must be positive")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")

    assigned: dict[str, str] = {}
    by_video: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_video[detection.video_id].append(detection)

    for video_id, video_detections in sorted(by_video.items()):
        by_timestamp: dict[float, list[Detection]] = defaultdict(list)
        for detection in video_detections:
            by_timestamp[detection.timestamp].append(detection)

        active: dict[str, _ActiveTrack] = {}
        next_ordinal = 0
        for timestamp, frame_detections in sorted(by_timestamp.items()):
            active = {
                track_id: track
                for track_id, track in active.items()
                if timestamp - track.detection.timestamp <= max_gap_seconds
            }
            candidates: list[tuple[float, str, str]] = []
            for detection in sorted(frame_detections, key=lambda item: item.detection_id):
                label = detection.class_label.strip().lower()
                for track_id, track in active.items():
                    if track.detection.class_label.strip().lower() != label:
                        continue
                    overlap = bbox_iou(
                        detection.normalized_bbox, track.detection.normalized_bbox
                    )
                    distance = _centroid_distance(detection, track.detection)
                    if overlap < iou_threshold and distance > centroid_threshold:
                        continue
                    proximity = max(0.0, 1.0 - distance / centroid_threshold)
                    candidates.append(
                        (overlap + 0.25 * proximity, track_id, detection.detection_id)
                    )

            matched_tracks: set[str] = set()
            matched_detections: set[str] = set()
            frame_by_id = {item.detection_id: item for item in frame_detections}
            for _, track_id, detection_id in sorted(
                candidates, key=lambda item: (-item[0], item[1], item[2])
            ):
                if track_id in matched_tracks or detection_id in matched_detections:
                    continue
                assigned[detection_id] = track_id
                active[track_id] = _ActiveTrack(track_id, frame_by_id[detection_id])
                matched_tracks.add(track_id)
                matched_detections.add(detection_id)

            for detection in sorted(frame_detections, key=lambda item: item.detection_id):
                if detection.detection_id in matched_detections:
                    continue
                track_id = _track_id(video_id, detection.class_label, next_ordinal)
                next_ordinal += 1
                assigned[detection.detection_id] = track_id
                active[track_id] = _ActiveTrack(track_id, detection)

    return [
        detection.model_copy(update={"track_id": assigned[detection.detection_id]})
        for detection in detections
    ]


def _relationship_track_key(
    relationship: Relationship,
    detection_tracks: dict[str, str],
) -> tuple[str, str, str, str, str]:
    source_track = detection_tracks.get(relationship.source_id, relationship.source_id)
    target_track = detection_tracks.get(relationship.target_id, relationship.target_id)
    predicate = relationship.predicate.strip().lower()
    if predicate in {"near", "overlaps_with", "touches"}:
        source_track, target_track = sorted((source_track, target_track))
    return (
        source_track,
        target_track,
        predicate,
        relationship.source_method.value,
        relationship.model_version or "",
    )


def smooth_relationship_intervals(
    relationships: list[Relationship],
    detections: list[Detection],
    *,
    sample_interval: float,
    max_gap_seconds: float = 1.5,
) -> list[Relationship]:
    """Merge adjacent observations for the same tracked tuple into intervals."""

    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")

    detection_tracks = {
        item.detection_id: item.track_id
        for item in detections
        if item.track_id is not None
    }
    grouped: dict[tuple[str, str, str, str, str], list[Relationship]] = defaultdict(list)
    for relationship in relationships:
        grouped[_relationship_track_key(relationship, detection_tracks)].append(
            relationship
        )

    smoothed: list[Relationship] = []
    for key, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: (item.timestamp, item.relationship_id))
        episodes: list[list[Relationship]] = []
        for relationship in ordered:
            if (
                not episodes
                or relationship.timestamp - episodes[-1][-1].timestamp > max_gap_seconds
            ):
                episodes.append([relationship])
            else:
                episodes[-1].append(relationship)

        for episode in episodes:
            representative = max(
                episode, key=lambda item: (item.confidence, -item.timestamp)
            )
            start_time = min(
                item.start_time if item.start_time is not None else item.timestamp
                for item in episode
            )
            end_time = max(
                item.end_time
                if item.end_time is not None
                else item.timestamp + sample_interval
                for item in episode
            )
            source_track = detection_tracks.get(
                representative.source_id, representative.source_id
            )
            target_track = detection_tracks.get(
                representative.target_id, representative.target_id
            )
            raw_id = ":".join(
                [*key, f"{start_time:.3f}", f"{end_time:.3f}"]
            )
            smoothed.append(
                representative.model_copy(
                    update={
                        "relationship_id": (
                            "rel_"
                            + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]
                        ),
                        "source_track_id": source_track,
                        "target_track_id": target_track,
                        "confidence": sum(item.confidence for item in episode)
                        / len(episode),
                        "timestamp": start_time,
                        "start_time": start_time,
                        "end_time": end_time,
                        "observation_count": sum(
                            item.observation_count for item in episode
                        ),
                        "temporal_smoothed": len(episode) > 1,
                    }
                )
            )
    return sorted(smoothed, key=lambda item: (item.timestamp, item.relationship_id))
