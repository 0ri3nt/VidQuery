from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

from .domain import (
    CanonicalSegment,
    Detection,
    FrameRecord,
    OCREvidence,
    Relationship,
    RelationshipSource,
    TranscriptSegment,
)
from .geometry import bbox_iou, normalized_centroid_distance


def _relationship_id(
    source_id: str, target_id: str, predicate: str, timestamp: float, source_method: str
) -> str:
    raw = f"{source_id}:{target_id}:{predicate}:{timestamp:.3f}:{source_method}"
    return f"rel_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _segment_id(video_id: str, start: float, end: float) -> str:
    return f"{video_id}:{int(round(start * 1000))}:{int(round(end * 1000))}"


def _localized_relationship_id(
    relationship: Relationship,
    source_id: str,
    target_id: str,
    segment_start: float,
) -> str:
    raw = (
        f"{relationship.relationship_id}:{source_id}:{target_id}:"
        f"{segment_start:.3f}"
    )
    return f"rel_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _localize_relationship(
    relationship: Relationship,
    detections: list[Detection],
    *,
    segment_start: float,
    segment_end: float,
) -> Relationship | None:
    detections_by_id = {item.detection_id: item for item in detections}
    if not relationship.source_track_id or not relationship.target_track_id:
        if (
            relationship.source_id in detections_by_id
            and relationship.target_id in detections_by_id
        ):
            return relationship
        return None

    sources = [
        item for item in detections if item.track_id == relationship.source_track_id
    ]
    targets = [
        item for item in detections if item.track_id == relationship.target_track_id
    ]
    if not sources or not targets:
        return None

    relation_start = (
        relationship.start_time
        if relationship.start_time is not None
        else relationship.timestamp
    )
    relation_end = (
        relationship.end_time
        if relationship.end_time is not None
        else relationship.timestamp
    )
    overlap_start = max(segment_start, relation_start)
    overlap_end = min(segment_end, relation_end)
    if overlap_end <= overlap_start:
        return None
    focus = (overlap_start + overlap_end) / 2
    source, target = min(
        ((source, target) for source in sources for target in targets),
        key=lambda pair: (
            abs(pair[0].timestamp - pair[1].timestamp),
            abs(((pair[0].timestamp + pair[1].timestamp) / 2) - focus),
            pair[0].detection_id,
            pair[1].detection_id,
        ),
    )
    return relationship.model_copy(
        update={
            "relationship_id": _localized_relationship_id(
                relationship, source.detection_id, target.detection_id, segment_start
            ),
            "source_id": source.detection_id,
            "target_id": target.detection_id,
            "timestamp": source.timestamp,
            "start_time": overlap_start,
            "end_time": overlap_end,
        }
    )


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    timestamp: float
    label: str
    source_method: RelationshipSource
    confidence: float | None = None
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    resolved_predicate: str | None = None


def resolved_action_relationship(action: ActionEvidence) -> Relationship | None:
    if not (
        action.source_entity_id
        and action.target_entity_id
        and action.resolved_predicate
        and action.confidence is not None
    ):
        return None
    return Relationship(
        relationship_id=_relationship_id(
            action.source_entity_id,
            action.target_entity_id,
            action.resolved_predicate,
            action.timestamp,
            RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER.value,
        ),
        source_id=action.source_entity_id,
        target_id=action.target_entity_id,
        predicate=action.resolved_predicate,
        confidence=action.confidence,
        source_method=RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER,
        timestamp=action.timestamp,
    )


def build_spatial_relationships(
    detections: list[Detection], near_threshold: float = 0.22
) -> list[Relationship]:
    if near_threshold <= 0:
        raise ValueError("near_threshold must be positive")

    by_frame: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame_id].append(detection)

    relationships: list[Relationship] = []
    for frame_detections in by_frame.values():
        for left_index, source in enumerate(frame_detections):
            for target in frame_detections[left_index + 1 :]:
                distance = normalized_centroid_distance(
                    source.normalized_bbox, target.normalized_bbox
                )
                overlap = bbox_iou(source.normalized_bbox, target.normalized_bbox)
                timestamp = source.timestamp

                if overlap > 0:
                    predicate = "overlaps_with"
                    relationships.append(
                        Relationship(
                            relationship_id=_relationship_id(
                                source.detection_id,
                                target.detection_id,
                                predicate,
                                timestamp,
                                RelationshipSource.BOUNDING_BOX_GEOMETRY.value,
                            ),
                            source_id=source.detection_id,
                            target_id=target.detection_id,
                            predicate=predicate,
                            confidence=min(1.0, overlap),
                            source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
                            timestamp=timestamp,
                        )
                    )

                if distance <= near_threshold:
                    predicate = "near"
                    confidence = max(0.0, 1.0 - (distance / near_threshold))
                    relationships.append(
                        Relationship(
                            relationship_id=_relationship_id(
                                source.detection_id,
                                target.detection_id,
                                predicate,
                                timestamp,
                                RelationshipSource.BOUNDING_BOX_GEOMETRY.value,
                            ),
                            source_id=source.detection_id,
                            target_id=target.detection_id,
                            predicate=predicate,
                            confidence=confidence,
                            source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
                            timestamp=timestamp,
                        )
                    )

                source_x, source_y = source.centroid_normalized
                target_x, target_y = target.centroid_normalized
                if abs(source_x - target_x) >= 0.05:
                    left, right = (source, target) if source_x < target_x else (target, source)
                    relationships.append(
                        Relationship(
                            relationship_id=_relationship_id(
                                left.detection_id,
                                right.detection_id,
                                "left_of",
                                timestamp,
                                RelationshipSource.BOUNDING_BOX_GEOMETRY.value,
                            ),
                            source_id=left.detection_id,
                            target_id=right.detection_id,
                            predicate="left_of",
                            confidence=min(1.0, abs(source_x - target_x)),
                            source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
                            timestamp=timestamp,
                        )
                    )
                if abs(source_y - target_y) >= 0.05:
                    above, below = (source, target) if source_y < target_y else (target, source)
                    relationships.append(
                        Relationship(
                            relationship_id=_relationship_id(
                                above.detection_id,
                                below.detection_id,
                                "above",
                                timestamp,
                                RelationshipSource.BOUNDING_BOX_GEOMETRY.value,
                            ),
                            source_id=above.detection_id,
                            target_id=below.detection_id,
                            predicate="above",
                            confidence=min(1.0, abs(source_y - target_y)),
                            source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
                            timestamp=timestamp,
                        )
                    )
    return relationships


def build_canonical_segments(
    *,
    video_id: str,
    duration: float,
    segment_duration: float,
    frames: list[FrameRecord],
    detections: list[Detection],
    relationships: list[Relationship],
    transcripts: list[TranscriptSegment],
    actions: list[ActionEvidence] | None = None,
    ocr_evidence: list[OCREvidence] | None = None,
    processing_metadata: dict | None = None,
) -> list[CanonicalSegment]:
    if segment_duration <= 0:
        raise ValueError("segment_duration must be positive")
    if duration < 0:
        raise ValueError("duration must not be negative")

    action_values = actions or []
    ocr_values = ocr_evidence or []
    maximum_event_end = max(
        [duration]
        + [frame.timestamp for frame in frames]
        + [item.end_time for item in transcripts]
        + [
            item.end_time if item.end_time is not None else item.timestamp
            for item in relationships
        ]
        + [item.timestamp for item in action_values]
        + [item.end_time for item in ocr_values]
    )
    effective_duration = max(segment_duration, maximum_event_end)
    segment_count = max(1, int(math.ceil(effective_duration / segment_duration)))

    frames_by_index: dict[int, list[FrameRecord]] = defaultdict(list)
    detections_by_index: dict[int, list[Detection]] = defaultdict(list)
    relationships_by_index: dict[int, list[Relationship]] = defaultdict(list)
    actions_by_index: dict[int, list[ActionEvidence]] = defaultdict(list)
    ocr_by_index: dict[int, list[OCREvidence]] = defaultdict(list)

    def index_for(timestamp: float) -> int:
        return min(segment_count - 1, max(0, int(timestamp // segment_duration)))

    for frame in frames:
        frames_by_index[index_for(frame.timestamp)].append(frame)
    for detection in detections:
        detections_by_index[index_for(detection.timestamp)].append(detection)
    for relationship in relationships:
        if (
            relationship.source_track_id
            and relationship.target_track_id
            and relationship.start_time is not None
            and relationship.end_time is not None
        ):
            first = index_for(relationship.start_time)
            last = index_for(
                max(relationship.start_time, relationship.end_time - 1e-9)
            )
            for index in range(first, last + 1):
                relationships_by_index[index].append(relationship)
        else:
            relationships_by_index[index_for(relationship.timestamp)].append(
                relationship
            )
    for action in action_values:
        actions_by_index[index_for(action.timestamp)].append(action)
        resolved = resolved_action_relationship(action)
        if resolved is not None:
            relationships_by_index[index_for(action.timestamp)].append(resolved)
    for evidence in ocr_values:
        first = index_for(evidence.start_time)
        last = index_for(max(evidence.start_time, evidence.end_time - 1e-9))
        for index in range(first, last + 1):
            ocr_by_index[index].append(evidence)

    transcripts_by_index: dict[int, list[TranscriptSegment]] = defaultdict(list)
    for transcript in transcripts:
        first = index_for(transcript.start_time)
        last = index_for(max(transcript.start_time, transcript.end_time - 1e-9))
        for index in range(first, last + 1):
            window_start = index * segment_duration
            window_end = window_start + segment_duration
            if min(window_end, transcript.end_time) - max(window_start, transcript.start_time) > 0:
                transcripts_by_index[index].append(transcript)

    segments: list[CanonicalSegment] = []
    for index in range(segment_count):
        start = index * segment_duration
        end = min(effective_duration, start + segment_duration)
        if end <= start:
            end = start + segment_duration

        segment_frames = sorted(frames_by_index[index], key=lambda item: item.timestamp)
        segment_detections = detections_by_index[index]
        segment_relationships = [
            localized
            for relationship in relationships_by_index[index]
            if (
                localized := _localize_relationship(
                    relationship,
                    segment_detections,
                    segment_start=start,
                    segment_end=end,
                )
            )
            is not None
        ]
        segment_transcripts = transcripts_by_index[index]
        segment_actions = actions_by_index[index]
        segment_ocr = ocr_by_index[index]
        transcript_text = " ".join(
            item.text.strip() for item in segment_transcripts if item.text.strip()
        ).strip()

        metadata = dict(processing_metadata or {})
        if segment_frames:
            metadata["thumbnail_path"] = segment_frames[0].path
        if segment_actions:
            metadata["action_sources"] = {
                item.label: item.source_method.value for item in segment_actions
            }
            metadata["action_confidences"] = {
                label: max(
                    item.confidence
                    for item in segment_actions
                    if item.label == label and item.confidence is not None
                )
                for label in {item.label for item in segment_actions}
                if any(
                    item.label == label and item.confidence is not None
                    for item in segment_actions
                )
            }
            metadata["gnn_action_instances"] = [
                {
                    "action": item.label,
                    "confidence": item.confidence,
                    "source_entity_id": item.source_entity_id,
                    "target_entity_id": item.target_entity_id,
                    "resolved_predicate": item.resolved_predicate,
                    "provenance": (
                        RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER.value
                        if item.target_entity_id
                        else item.source_method.value
                    ),
                    "timestamp": item.timestamp,
                }
                for item in segment_actions
                if item.source_method is RelationshipSource.GNN_ACTION_MODEL
            ]

        segments.append(
            CanonicalSegment(
                segment_id=_segment_id(video_id, start, end),
                video_id=video_id,
                start_time=start,
                end_time=end,
                frame_ids=[frame.frame_id for frame in segment_frames],
                detections=segment_detections,
                entities=sorted({item.class_label.lower() for item in segment_detections}),
                relationships=segment_relationships,
                actions=sorted({item.label.lower() for item in segment_actions}),
                transcript=transcript_text,
                transcript_segments=segment_transcripts,
                speakers=sorted(
                    {
                        item.speaker
                        for item in segment_transcripts
                        if item.speaker and item.speaker != "UNKNOWN"
                    }
                ),
                ocr_evidence=segment_ocr,
                processing_metadata=metadata,
            )
        )
    return segments
