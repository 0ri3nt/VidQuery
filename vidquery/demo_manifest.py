from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    Relationship,
    RelationshipSource,
    TranscriptSegment,
    VideoRecord,
)
from .storage import SQLiteRepository

_CLASS_IDS = {"person": 0, "laptop": 63, "whiteboard": -1}


def load_demo_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "duration_seconds", "entities", "relationships", "segments"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Demo manifest is missing fields: {', '.join(sorted(missing))}")
    if not manifest["segments"]:
        raise ValueError("Demo manifest must contain at least one labelled segment")
    return manifest


def _manual_detection(
    video: VideoRecord,
    segment: CanonicalSegment,
    entity: dict[str, Any],
) -> Detection:
    try:
        x1, y1, x2, y2 = (float(value) for value in entity["normalized_bbox"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid normalized_bbox for demo entity {entity!r}") from exc
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
        raise ValueError(f"Demo entity bbox must be normalized and ordered: {entity!r}")

    entity_id = str(entity["id"])
    class_label = str(entity["class"]).strip().lower()
    start_ms = int(round(segment.start_time * 1000))
    detection_id = f"manual:{video.video_id}:{start_ms}:{entity_id}"
    frame_id = (
        segment.frame_ids[0] if segment.frame_ids else f"manual:{video.video_id}:{start_ms}:frame"
    )
    pixel = BoundingBox(
        x1=x1 * video.width,
        y1=y1 * video.height,
        x2=x2 * video.width,
        y2=y2 * video.height,
    )
    normalized = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    center_normalized = ((x1 + x2) / 2, (y1 + y2) / 2)
    return Detection(
        detection_id=detection_id,
        frame_id=frame_id,
        video_id=video.video_id,
        timestamp=segment.start_time,
        class_id=_CLASS_IDS.get(class_label, -1),
        class_label=class_label,
        confidence=1.0,
        pixel_bbox=pixel,
        normalized_bbox=normalized,
        centroid=(
            center_normalized[0] * video.width,
            center_normalized[1] * video.height,
        ),
        centroid_normalized=center_normalized,
    )


def _annotate_segment(
    *,
    video: VideoRecord,
    segment: CanonicalSegment,
    manifest_segment: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    manifest_path: Path,
) -> CanonicalSegment:
    automatic_detections = [
        item for item in segment.detections if not item.detection_id.startswith("manual:")
    ]
    manual_detections = [_manual_detection(video, segment, entity) for entity in entities]
    detections = automatic_detections + manual_detections
    detection_ids = {
        str(entity["id"]): detection.detection_id
        for entity, detection in zip(entities, manual_detections, strict=True)
    }

    manual_relationships: list[Relationship] = []
    for relation in relationships:
        subject = str(relation["subject"])
        target = str(relation["object"])
        predicate = str(relation["predicate"]).strip().lower()
        if subject not in detection_ids or target not in detection_ids:
            raise ValueError(f"Demo relationship refers to an unknown entity: {relation!r}")
        start_ms = int(round(segment.start_time * 1000))
        manual_relationships.append(
            Relationship(
                relationship_id=(
                    f"manual:{video.video_id}:{start_ms}:{subject}:{predicate}:{target}"
                ),
                source_id=detection_ids[subject],
                target_id=detection_ids[target],
                predicate=predicate,
                confidence=1.0,
                source_method=RelationshipSource.MANUAL_ANNOTATION,
                timestamp=segment.start_time,
            )
        )

    transcript = str(manifest_segment["transcript"]).strip()
    speaker = str(manifest_segment["speaker"]).strip()
    transcript_segment = TranscriptSegment(
        segment_id=f"manual:{video.video_id}:{int(round(segment.start_time * 1000))}:transcript",
        video_id=video.video_id,
        start_time=float(manifest_segment["start_time"]),
        end_time=float(manifest_segment["end_time"]),
        text=transcript,
        confidence=1.0,
        speaker=speaker,
    )
    metadata = dict(segment.processing_metadata)
    if segment.transcript and segment.transcript != transcript:
        metadata["automatic_transcript_before_manifest"] = segment.transcript
    metadata.update(
        {
            "controlled_demo_manifest": str(manifest_path),
            "manual_annotation_overlay": True,
            "manual_entity_count": len(manual_detections),
            "manual_relationship_count": len(manual_relationships),
            "transcript_source": RelationshipSource.MANUAL_ANNOTATION.value,
        }
    )
    automatic_relationships = [
        item
        for item in segment.relationships
        if item.source_method != RelationshipSource.MANUAL_ANNOTATION
    ]
    return segment.model_copy(
        update={
            "detections": detections,
            "entities": sorted({item.class_label.lower() for item in detections}),
            "relationships": automatic_relationships + manual_relationships,
            "transcript": transcript,
            "transcript_segments": [transcript_segment],
            "speakers": [speaker],
            "processing_metadata": metadata,
        }
    )


def apply_demo_manifest(
    repository: SQLiteRepository,
    *,
    video_id: str,
    manifest_path: Path,
    tolerance_seconds: float = 0.05,
) -> list[CanonicalSegment]:
    """Apply transparent, deterministic ground truth to a controlled demo video.

    This does not masquerade as model output: every added relation uses the
    ``manual_annotation`` source and the processing metadata records the overlay.
    """

    video = repository.get_video(video_id)
    manifest_path = manifest_path.resolve()
    manifest = load_demo_manifest(manifest_path)
    segments = repository.list_segments([video_id])
    if not segments:
        raise ValueError(f"Video {video_id} has no canonical segments to annotate")

    labelled_by_start = {float(item["start_time"]): item for item in manifest["segments"]}
    annotated: list[CanonicalSegment] = []
    matched_starts: set[float] = set()
    for segment in segments:
        match = next(
            (
                (start, item)
                for start, item in labelled_by_start.items()
                if abs(segment.start_time - start) <= tolerance_seconds
                and abs(segment.end_time - float(item["end_time"])) <= tolerance_seconds
            ),
            None,
        )
        if match is None:
            annotated.append(segment)
            continue
        start, manifest_segment = match
        matched_starts.add(start)
        annotated.append(
            _annotate_segment(
                video=video,
                segment=segment,
                manifest_segment=manifest_segment,
                entities=list(manifest["entities"]),
                relationships=list(manifest["relationships"]),
                manifest_path=manifest_path,
            )
        )

    missing = sorted(set(labelled_by_start) - matched_starts)
    if missing:
        raise ValueError(f"No canonical segment matched manifest start times: {missing}")
    repository.replace_segments(video_id, annotated)
    return annotated
