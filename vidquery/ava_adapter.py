from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .ava_labels import AVA_V22_ACTIONS
from .config import PROJECT_ROOT, Settings
from .domain import (
    BoundingBox,
    Detection,
    FrameRecord,
    ProcessingState,
    Relationship,
    RelationshipSource,
    TranscriptSegment,
    VideoRecord,
)
from .geometry import centroid, normalize_bbox
from .segments import ActionEvidence, build_canonical_segments
from .storage import SQLiteRepository

SPATIAL_RELATIONS = {
    "overlaps_with",
    "touches",
    "near",
    "left_of",
    "right_of",
    "above",
    "below",
    "contains",
    "inside",
}


def _stable_id(prefix: str, *parts: object) -> str:
    raw = ":".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _canonical_action(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ")
    match = re.fullmatch(r"(?:action )?(\d+)", normalized)
    if match:
        action_id = int(match.group(1))
        return AVA_V22_ACTIONS.get(action_id, f"ava_action_{action_id}")
    return normalized


def import_fused_ava(
    *,
    video: VideoRecord,
    fused_path: Path,
    settings: Settings,
    repository: SQLiteRepository,
) -> int:
    payload = json.loads(Path(fused_path).read_text(encoding="utf-8"))
    frame_root = (
        PROJECT_ROOT
        / "data"
        / "ava"
        / "extracted_frames"
        / str(payload.get("split", "train"))
        / payload["video_id"]
    )
    frames: list[FrameRecord] = []
    detections: list[Detection] = []
    relationships: list[Relationship] = []
    actions: list[ActionEvidence] = []
    transcripts_by_key: dict[tuple, TranscriptSegment] = {}

    for frame_index, frame in enumerate(payload.get("frames", [])):
        timestamp = max(0.0, float(frame.get("timestamp", 0.0)))
        frame_file = str(
            frame.get("frame_file")
            or f"{payload['video_id']}_{int(timestamp):04d}.jpg"
        )
        frame_id = f"{video.video_id}_{int(round(timestamp * 1000)):012d}"
        frame_path = frame_root / frame_file
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                video_id=video.video_id,
                frame_index=frame_index,
                timestamp=timestamp,
                path=str(frame_path),
                width=video.width,
                height=video.height,
            )
        )

        node_lookup: dict[int, Detection] = {}
        for node in frame.get("nodes", []):
            node_id = int(node.get("node_id", len(node_lookup)))
            values = [float(value) for value in node.get("bbox", [0, 0, 0, 0])]
            pixel_box = BoundingBox(x1=values[0], y1=values[1], x2=values[2], y2=values[3])
            normalized_box = normalize_bbox(pixel_box, video.width, video.height)
            detection = Detection(
                detection_id=_stable_id("ava_det", video.video_id, timestamp, node_id),
                frame_id=frame_id,
                video_id=video.video_id,
                timestamp=timestamp,
                class_id=-1,
                class_label=str(node.get("class_name", "unknown")).lower(),
                confidence=min(1.0, max(0.0, float(node.get("confidence", 0.0)))),
                pixel_bbox=pixel_box,
                normalized_bbox=normalized_box,
                centroid=centroid(pixel_box),
                centroid_normalized=centroid(normalized_box),
            )
            detections.append(detection)
            node_lookup[node_id] = detection

            annotation = node.get("matched_annotation")
            if isinstance(annotation, dict):
                # ``action_ids`` are the authoritative AVA v2.2 labels. Some
                # historical fused files contain human-readable action_labels
                # produced by an incorrect local map, so only use those as a
                # compatibility fallback when the official numeric IDs are absent.
                action_ids = annotation.get("action_ids", [])
                raw_labels = (
                    [f"action_{int(action_id)}" for action_id in action_ids]
                    if action_ids
                    else annotation.get("action_labels", [])
                )
                for label in raw_labels:
                    actions.append(
                        ActionEvidence(
                            timestamp=timestamp,
                            label=_canonical_action(str(label)),
                            source_method=RelationshipSource.AVA_GROUND_TRUTH,
                        )
                    )

        for edge in frame.get("edges", []):
            source = node_lookup.get(int(edge.get("source", -1)))
            target = node_lookup.get(int(edge.get("target", -1)))
            if source is None or target is None:
                continue
            predicate = str(edge.get("type", "related_to")).lower()
            if predicate in SPATIAL_RELATIONS:
                source_method = RelationshipSource.BOUNDING_BOX_GEOMETRY
            elif predicate.startswith("action_") or source.detection_id == target.detection_id:
                source_method = RelationshipSource.AVA_GROUND_TRUTH
            else:
                source_method = RelationshipSource.HEURISTIC_ACTION_MAPPING
            if "weight" in edge:
                confidence = min(1.0, max(0.0, float(edge["weight"])))
            elif "distance" in edge:
                diagonal = max(1.0, math.hypot(video.width, video.height))
                confidence = max(0.0, 1.0 - float(edge["distance"]) / diagonal)
            else:
                confidence = 0.65
            relationships.append(
                Relationship(
                    relationship_id=_stable_id(
                        "ava_rel", source.detection_id, target.detection_id, predicate, timestamp
                    ),
                    source_id=source.detection_id,
                    target_id=target.detection_id,
                    predicate=predicate,
                    confidence=confidence,
                    source_method=source_method,
                    timestamp=timestamp,
                )
            )

        for audio in frame.get("audio_segments", []):
            start = max(0.0, float(audio.get("start", timestamp)))
            end = max(start, float(audio.get("end", start)))
            text = str(audio.get("text", "")).strip()
            speaker = str(audio.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
            key = (round(start, 3), round(end, 3), speaker, text)
            transcripts_by_key.setdefault(
                key,
                TranscriptSegment(
                    segment_id=_stable_id("ava_tx", video.video_id, *key),
                    video_id=video.video_id,
                    start_time=start,
                    end_time=end,
                    text=text,
                    confidence=None,
                    speaker=speaker.upper() if speaker.lower() == "unknown" else speaker,
                ),
            )

    segments = build_canonical_segments(
        video_id=video.video_id,
        duration=video.duration,
        segment_duration=settings.segment_duration,
        frames=frames,
        detections=detections,
        relationships=relationships,
        transcripts=list(transcripts_by_key.values()),
        actions=actions,
        processing_metadata={
            "source": "legacy_ava_fused_adapter",
            "source_file": str(Path(fused_path)),
            "relationship_mode": "legacy_geometry_and_ava_annotations",
            "gnn_predictions_used": False,
            "learned_action_model": "not_used_ava_ground_truth",
            "learned_action_predictions_used": False,
        },
    )
    evidence_segments = [
        segment
        for segment in segments
        if segment.frame_ids
        or segment.transcript_segments
        or segment.detections
        or segment.relationships
        or segment.actions
    ]
    repository.replace_segments(video.video_id, evidence_segments)
    repository.update_state(
        video.video_id,
        ProcessingState.READY,
        current_stage="legacy AVA import complete",
        warnings=[
            "Imported existing AVA artifacts; no fresh model inference was performed",
            "AVA annotations and geometry heuristics are not GNN predictions",
        ],
    )
    return len(evidence_segments)
