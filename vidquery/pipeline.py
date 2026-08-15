from __future__ import annotations

import hashlib
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from threading import RLock

from .config import Settings
from .domain import (
    CanonicalSegment,
    Detection,
    EntityAppearance,
    FrameRecord,
    ProcessingState,
    Relationship,
    RelationshipSource,
    TranscriptSegment,
    VideoRecord,
)
from .media import AudioTrackExtractor, FrameExtractor
from .models import (
    Diarizer,
    OptionalComponentUnavailable,
    PyannoteDiarizer,
    Transcriber,
    VisualDetector,
    WhisperTranscriber,
    YoloVisualDetector,
    assign_speakers,
)
from .segments import (
    ActionEvidence,
    build_canonical_segments,
    build_spatial_relationships,
    resolved_action_relationship,
)
from .storage import SQLiteRepository

LOGGER = logging.getLogger(__name__)


class ProcessingConflictError(RuntimeError):
    pass


def _preserves_annotation_evidence(segment: CanonicalSegment) -> bool:
    metadata = segment.processing_metadata
    return bool(
        metadata.get("manual_annotation_overlay")
        or metadata.get("source") == "legacy_ava_fused_adapter"
    )


def nearest_cached_frame(
    frames_by_millisecond: dict[int, Path],
    timestamp_millisecond: int,
    *,
    tolerance_millisecond: int,
) -> Path | None:
    """Return an exact or safely nearby sampled frame for an inference timestamp."""

    exact = frames_by_millisecond.get(timestamp_millisecond)
    if exact is not None:
        return exact
    if not frames_by_millisecond:
        return None
    nearest_time = min(
        frames_by_millisecond,
        key=lambda candidate: abs(candidate - timestamp_millisecond),
    )
    if abs(nearest_time - timestamp_millisecond) > tolerance_millisecond:
        return None
    return frames_by_millisecond[nearest_time]


def merge_preserved_evidence(
    fresh_segments: list[CanonicalSegment],
    existing_segments: list[CanonicalSegment],
) -> list[CanonicalSegment]:
    """Merge fresh inference with trusted manual/AVA evidence on matching windows."""

    preserved_by_start = {
        round(segment.start_time, 3): segment
        for segment in existing_segments
        if _preserves_annotation_evidence(segment)
    }
    merged: list[CanonicalSegment] = []
    for fresh in fresh_segments:
        preserved = preserved_by_start.get(round(fresh.start_time, 3))
        if preserved is None:
            merged.append(fresh)
            continue

        is_legacy_ava = preserved.processing_metadata.get("source") == (
            "legacy_ava_fused_adapter"
        )
        learned_sources = {
            RelationshipSource.GNN_PREDICTION,
            RelationshipSource.LEARNED_ACTION_MODEL,
            RelationshipSource.GNN_ACTION_MODEL,
            RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER,
            RelationshipSource.RELATIONSHIP_GNN,
        }
        if is_legacy_ava:
            trusted_relationships = [
                item
                for item in preserved.relationships
                if item.source_method not in learned_sources
            ]
            trusted_detections = preserved.detections
        else:
            trusted_relationships = [
                item
                for item in preserved.relationships
                if item.source_method is RelationshipSource.MANUAL_ANNOTATION
            ]
            trusted_detection_ids = {
                endpoint
                for item in trusted_relationships
                for endpoint in (item.source_id, item.target_id)
            }
            trusted_detections = [
                item
                for item in preserved.detections
                if item.detection_id.startswith("manual:")
                or item.detection_id in trusted_detection_ids
            ]

        detections = {item.detection_id: item for item in fresh.detections}
        detections.update({item.detection_id: item for item in trusted_detections})
        relationships = {item.relationship_id: item for item in fresh.relationships}
        relationships.update(
            {item.relationship_id: item for item in trusted_relationships}
        )
        transcript_segments = {
            item.segment_id: item for item in fresh.transcript_segments
        }
        transcript_segments.update(
            {item.segment_id: item for item in preserved.transcript_segments}
        )
        ocr_evidence = {item.ocr_id: item for item in fresh.ocr_evidence}

        metadata = {**preserved.processing_metadata, **fresh.processing_metadata}
        old_action_sources = preserved.processing_metadata.get("action_sources", {})
        fresh_action_sources = fresh.processing_metadata.get("action_sources", {})
        metadata.update(
            {
                "action_sources": {**fresh_action_sources, **old_action_sources},
                "preserved_evidence": True,
                "preserved_evidence_source": (
                    preserved.processing_metadata.get("source")
                    or "manual_annotation"
                ),
                "relationship_mode": "fresh_models_plus_preserved_annotations",
            }
        )
        if fresh.transcript and preserved.transcript:
            metadata["fresh_whisper_transcript"] = fresh.transcript
        speakers = {*fresh.speakers, *preserved.speakers}
        if speakers - {"UNKNOWN"}:
            speakers.discard("UNKNOWN")

        merged.append(
            fresh.model_copy(
                update={
                    "frame_ids": list(dict.fromkeys([*fresh.frame_ids, *preserved.frame_ids])),
                    "detections": list(detections.values()),
                    "entities": sorted({*fresh.entities, *preserved.entities}),
                    "relationships": list(relationships.values()),
                    "actions": sorted({*fresh.actions, *preserved.actions}),
                    "transcript": preserved.transcript or fresh.transcript,
                    "transcript_segments": list(transcript_segments.values()),
                    "speakers": sorted(speakers),
                    "ocr_evidence": list(ocr_evidence.values()),
                    "embedding": preserved.embedding or fresh.embedding,
                    "processing_metadata": metadata,
                }
            )
        )
    return merged


class VideoProcessingService:
    def __init__(
        self,
        settings: Settings,
        repository: SQLiteRepository,
        *,
        frame_extractor: FrameExtractor | None = None,
        audio_extractor: AudioTrackExtractor | None = None,
        visual_detector: VisualDetector | None = None,
        transcriber: Transcriber | None = None,
        diarizer: Diarizer | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.frame_extractor = frame_extractor or FrameExtractor(settings.frame_sample_rate)
        self.audio_extractor = audio_extractor or AudioTrackExtractor()
        self.visual_detector = visual_detector or YoloVisualDetector(settings)
        self.transcriber = transcriber or WhisperTranscriber(settings)
        self.diarizer = diarizer or PyannoteDiarizer(settings)
        self._active: set[str] = set()
        self._active_lock = RLock()

    def _begin(self, video_id: str) -> None:
        with self._active_lock:
            if video_id in self._active:
                raise ProcessingConflictError(f"Video {video_id} is already processing")
            self._active.add(video_id)

    def _finish(self, video_id: str) -> None:
        with self._active_lock:
            self._active.discard(video_id)

    def process(self, video_id: str) -> VideoRecord:
        self._begin(video_id)
        stage = "initialization"
        warnings: list[str] = []
        try:
            video = self.repository.get_video(video_id)
            existing_segments = self.repository.list_segments([video_id])
            work_dir = self.settings.generated_dir / video_id

            stage = "frame extraction"
            self.repository.update_state(
                video_id, ProcessingState.PREPROCESSING, current_stage=stage, warnings=warnings
            )
            frames = self.frame_extractor.extract(video, work_dir / "frames")
            if not frames:
                raise RuntimeError("Frame extraction produced no readable frames")

            stage = "visual detection"
            self.repository.update_state(
                video_id,
                ProcessingState.VISUAL_PROCESSING,
                current_stage=stage,
                warnings=warnings,
            )
            visual_detector_status = "not_started"
            try:
                detections = self.visual_detector.detect(frames)
                visual_detector_status = "completed"
            except OptionalComponentUnavailable as exc:
                detections = []
                visual_detector_status = "unavailable"
                warnings.append(str(exc))
            except Exception as exc:  # model failure should not discard transcript indexing
                LOGGER.exception("Visual detector failed for %s", video_id)
                detections = []
                visual_detector_status = f"failed_{type(exc).__name__}"
                warnings.append(f"Visual detection unavailable: {type(exc).__name__}")

            tracking_status = "disabled"
            if self.settings.enable_temporal_tracking:
                try:
                    from .tracking import assign_detection_tracks

                    detections = assign_detection_tracks(
                        detections,
                        iou_threshold=self.settings.track_iou_threshold,
                        centroid_threshold=self.settings.track_centroid_threshold,
                        max_gap_seconds=self.settings.track_max_gap_seconds,
                    )
                    tracking_status = "completed"
                except Exception as exc:
                    LOGGER.exception("Temporal tracking failed for %s", video_id)
                    tracking_status = f"failed_{type(exc).__name__}"
                    warnings.append(f"Temporal tracking unavailable: {type(exc).__name__}")

            appearance_records, detections, appearance_status = (
                self._appearance_features(video, frames, detections, warnings)
            )

            ocr_evidence = []
            ocr_status = "disabled"
            if self.settings.enable_ocr:
                try:
                    from .ocr import EasyOCRExtractor

                    ocr_evidence = EasyOCRExtractor(self.settings).extract(frames)
                    ocr_status = "completed"
                except Exception as exc:
                    LOGGER.exception("OCR extraction failed for %s", video_id)
                    ocr_status = f"failed_{type(exc).__name__}"
                    warnings.append(f"OCR unavailable: {type(exc).__name__}")

            stage = "audio processing"
            self.repository.update_state(
                video_id,
                ProcessingState.AUDIO_PROCESSING,
                current_stage=stage,
                warnings=warnings,
            )
            transcripts = []
            speakers = []
            transcription_status = "not_started"
            diarization_status = "not_started"
            audio_path = self.audio_extractor.extract(video, work_dir / "audio.wav")
            if audio_path is None:
                transcription_status = "no_audio"
                diarization_status = "no_audio"
                warnings.append("No usable audio track; continuing with visual evidence")
            else:
                try:
                    transcripts = self.transcriber.transcribe(audio_path, video_id)
                    transcription_status = "completed"
                except OptionalComponentUnavailable as exc:
                    transcription_status = "unavailable"
                    warnings.append(str(exc))
                except Exception as exc:
                    LOGGER.exception("Transcription failed for %s", video_id)
                    transcription_status = f"failed_{type(exc).__name__}"
                    warnings.append(f"Transcription unavailable: {type(exc).__name__}")

                if self.settings.enable_diarization:
                    try:
                        speakers = self.diarizer.diarize(audio_path, video_id)
                        diarization_status = "completed"
                    except OptionalComponentUnavailable as exc:
                        diarization_status = "unavailable"
                        warnings.append(str(exc))
                        if self.settings.require_diarization:
                            raise RuntimeError(
                                "Required Pyannote diarization is unavailable"
                            ) from exc
                    except Exception as exc:
                        LOGGER.exception("Diarization failed for %s", video_id)
                        diarization_status = f"failed_{type(exc).__name__}"
                        warnings.append(f"Diarization unavailable: {type(exc).__name__}")
                        if self.settings.require_diarization:
                            raise RuntimeError(
                                "Required Pyannote diarization failed"
                            ) from exc
                else:
                    diarization_status = "disabled"
                    if self.settings.require_diarization:
                        raise RuntimeError(
                            "Pyannote diarization is required but disabled"
                        )
                transcripts = assign_speakers(transcripts, speakers)

            stage = "scene relationship generation"
            self.repository.update_state(
                video_id,
                ProcessingState.SCENE_GRAPH_GENERATION,
                current_stage=stage,
                warnings=warnings,
            )
            relationships = build_spatial_relationships(
                detections, near_threshold=self.settings.near_threshold
            )
            learned_relationships, learned_relationship_status = (
                self._learned_relationships(video, frames, detections, warnings)
            )
            relationships.extend(learned_relationships)
            raw_relationship_count = len(relationships)
            relationship_smoothing_status = "disabled"
            if self.settings.enable_temporal_tracking:
                try:
                    from .tracking import smooth_relationship_intervals

                    relationships = smooth_relationship_intervals(
                        relationships,
                        detections,
                        sample_interval=1.0
                        / max(self.settings.frame_sample_rate, 0.001),
                        max_gap_seconds=(
                            self.settings.relationship_smoothing_gap_seconds
                        ),
                    )
                    relationship_smoothing_status = "completed"
                except Exception as exc:
                    LOGGER.exception(
                        "Temporal relationship smoothing failed for %s", video_id
                    )
                    relationship_smoothing_status = f"failed_{type(exc).__name__}"
                    warnings.append(
                        "Temporal relationship smoothing unavailable: "
                        f"{type(exc).__name__}"
                    )
            learned_actions, learned_action_status = self._learned_actions(
                video, detections, transcripts, warnings
            )

            stage = "multimodal alignment"
            self.repository.update_state(
                video_id, ProcessingState.ALIGNING, current_stage=stage, warnings=warnings
            )
            segments = build_canonical_segments(
                video_id=video_id,
                duration=video.duration,
                segment_duration=self.settings.segment_duration,
                frames=frames,
                detections=detections,
                relationships=relationships,
                transcripts=transcripts,
                actions=learned_actions,
                ocr_evidence=ocr_evidence,
                processing_metadata={
                    "frame_sample_rate": self.settings.frame_sample_rate,
                    "segment_duration": self.settings.segment_duration,
                    "yolo_model": self.settings.yolo_model if self.settings.enable_yolo else None,
                    "visual_detector_status": visual_detector_status,
                    "visual_detection_count": len(detections),
                    "temporal_tracking_enabled": self.settings.enable_temporal_tracking,
                    "temporal_tracking_status": tracking_status,
                    "detection_track_count": len(
                        {item.track_id for item in detections if item.track_id}
                    ),
                    "appearance_features_enabled": (
                        self.settings.enable_appearance_features
                    ),
                    "appearance_status": appearance_status,
                    "appearance_embedding_model": (
                        self.settings.appearance_model
                        if self.settings.enable_appearance_features
                        else None
                    ),
                    "appearance_track_count": len(appearance_records),
                    "appearance_embeddings_cached": bool(appearance_records),
                    "yolo_inference_used": (
                        self.settings.enable_yolo and visual_detector_status == "completed"
                    ),
                    "whisper_model": (
                        self.settings.whisper_model if self.settings.enable_whisper else None
                    ),
                    "transcription_status": transcription_status,
                    "transcript_segment_count": len(transcripts),
                    "whisper_inference_used": (
                        self.settings.enable_whisper and transcription_status == "completed"
                    ),
                    "diarization_enabled": self.settings.enable_diarization,
                    "diarization_required": self.settings.require_diarization,
                    "diarization_status": diarization_status,
                    "pyannote_model": (
                        self.settings.pyannote_model
                        if self.settings.enable_diarization
                        else None
                    ),
                    "speaker_segment_count": len(speakers),
                    "relationship_mode": (
                        "bounding_box_geometry_plus_vidor_gnn"
                        if learned_relationships
                        else "bounding_box_geometry"
                    ),
                    "relationship_gnn_status": learned_relationship_status,
                    "relationship_gnn_model_version": (
                        self.settings.relation_gnn_model_version
                        if self.settings.enable_relation_gnn
                        else None
                    ),
                    "relationship_gnn_predictions_used": bool(learned_relationships),
                    "relationship_raw_observation_count": raw_relationship_count,
                    "relationship_interval_count": len(relationships),
                    "relationship_smoothing_status": relationship_smoothing_status,
                    "ocr_enabled": self.settings.enable_ocr,
                    "ocr_status": ocr_status,
                    "ocr_source_method": (
                        RelationshipSource.EASY_OCR.value
                        if self.settings.enable_ocr
                        else None
                    ),
                    "ocr_evidence_count": len(ocr_evidence),
                    "learned_action_model": learned_action_status,
                    "learned_action_architecture": (
                        "pyg_gat80" if self.settings.enable_gnn_actions else "mlp"
                    ),
                    "gnn_model_version": (
                        self.settings.gnn_model_version
                        if self.settings.enable_gnn_actions
                        else None
                    ),
                    "gnn_action_inference_enabled": self.settings.enable_gnn_actions,
                    "learned_action_predictions_used": bool(learned_actions),
                },
            )
            segments = merge_preserved_evidence(segments, existing_segments)
            segments = self._embed_segments(segments, warnings)

            stage = "local indexing"
            self.repository.update_state(
                video_id, ProcessingState.INDEXING, current_stage=stage, warnings=warnings
            )
            self.repository.replace_entity_appearances(video_id, appearance_records)
            self.repository.replace_segments(video_id, segments)

            if self.settings.enable_neo4j:
                try:
                    from .neo4j import CanonicalNeo4jIndexer

                    indexer = CanonicalNeo4jIndexer.from_settings(self.settings)
                    try:
                        schema_path = (
                            Path(__file__).resolve().parents[1]
                            / "database"
                            / "schema.cypher"
                        )
                        indexer.apply_schema(schema_path)
                        indexer.ingest(video, segments, appearance_records)
                    finally:
                        indexer.close()
                except Exception as exc:
                    LOGGER.exception("Neo4j indexing failed for %s", video_id)
                    warnings.append(f"Neo4j indexing unavailable: {type(exc).__name__}")

            return self.repository.update_state(
                video_id,
                ProcessingState.READY,
                current_stage="complete",
                failure_message=None,
                failure_detail=None,
                warnings=warnings,
            )
        except Exception as exc:
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.repository.update_state(
                video_id,
                ProcessingState.FAILED,
                current_stage=stage,
                failure_message=f"Processing failed during {stage}",
                failure_detail=detail,
                warnings=warnings,
            )
            raise
        finally:
            self._finish(video_id)

    def _learned_relationships(
        self,
        video: VideoRecord,
        frames: list[FrameRecord],
        detections: list[Detection],
        warnings: list[str],
    ) -> tuple[list[Relationship], str]:
        if not self.settings.enable_relation_gnn:
            return [], "disabled"
        try:
            import torch

            from .relationships import normalize_predicate
            from .vidor_improved import (
                ImprovedVidORRelationPredictor,
                configured_relation_checkpoint_status,
            )
            from .vidor_relation import VidORRelationPredictor

            status = configured_relation_checkpoint_status(
                self.settings.relation_gnn_checkpoint,
                self.settings.relation_gnn_metadata,
                self.settings.relation_gnn_model_version,
            )
            if status != "available_validated":
                warnings.append(f"Relationship GNN inference unavailable: {status}")
                return [], status
            device = self.settings.relation_gnn_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            pair_visual = "pair-visual" in self.settings.relation_gnn_model_version
            predictor: VidORRelationPredictor | ImprovedVidORRelationPredictor
            if pair_visual:
                predictor = ImprovedVidORRelationPredictor(
                    self.settings.relation_gnn_checkpoint,
                    self.settings.relation_gnn_metadata,
                    device=device,
                    expected_model_version=self.settings.relation_gnn_model_version,
                )
            else:
                predictor = VidORRelationPredictor(
                    self.settings.relation_gnn_checkpoint,
                    self.settings.relation_gnn_metadata,
                    device=device,
                    expected_model_version=self.settings.relation_gnn_model_version,
                )
            by_timestamp: dict[float, list[Detection]] = defaultdict(list)
            for detection in detections:
                by_timestamp[detection.timestamp].append(detection)
            frames_by_millisecond = {
                int(round(frame.timestamp * 1000)): Path(frame.path) for frame in frames
            }
            relationships: list[Relationship] = []
            interval = 1.0 / max(self.settings.frame_sample_rate, 0.001)
            previous_detections: list[Detection] | None = None
            missing_visual_frame = False
            for timestamp, frame_detections in sorted(by_timestamp.items()):
                if pair_visual:
                    frame_path = nearest_cached_frame(
                        frames_by_millisecond,
                        int(round(timestamp * 1000)),
                        tolerance_millisecond=max(
                            100,
                            int(550 / max(self.settings.frame_sample_rate, 0.001)),
                        ),
                    )
                    if frame_path is None or not frame_path.is_file():
                        missing_visual_frame = True
                        previous_detections = frame_detections
                        continue
                    assert isinstance(predictor, ImprovedVidORRelationPredictor)
                    frame_predictions = predictor.predict_detections(
                        frame_detections,
                        frame_path=frame_path,
                        previous_detections=previous_detections,
                    )
                else:
                    assert isinstance(predictor, VidORRelationPredictor)
                    frame_predictions = predictor.predict_detections(frame_detections)
                previous_detections = frame_detections
                for prediction in frame_predictions:
                    source_id = str(prediction["source_id"])
                    target_id = str(prediction["target_id"])
                    predicate = normalize_predicate(str(prediction["predicate"]))
                    raw_id = (
                        f"{source_id}:{target_id}:{predicate}:{timestamp:.3f}:"
                        f"{RelationshipSource.RELATIONSHIP_GNN.value}:"
                        f"{predictor.model_version}"
                    )
                    relationships.append(
                        Relationship(
                            relationship_id=(
                                "rel_"
                                + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]
                            ),
                            source_id=source_id,
                            target_id=target_id,
                            predicate=predicate,
                            confidence=float(prediction["confidence"]),
                            source_method=RelationshipSource.RELATIONSHIP_GNN,
                            timestamp=timestamp,
                            model_version=predictor.model_version,
                            start_time=timestamp,
                            end_time=min(video.duration, timestamp + interval),
                        )
                    )
            if missing_visual_frame:
                warnings.append(
                    "Pair-visual relationship GNN skipped one or more timestamps "
                    "without a readable extracted frame; no zero visual fallback was used"
                )
            return relationships, status
        except Exception as exc:
            LOGGER.exception("Relationship GNN inference failed for %s", video.video_id)
            status = f"unavailable_{type(exc).__name__}"
            warnings.append(f"Relationship GNN inference unavailable: {type(exc).__name__}")
            return [], status

    def _appearance_features(
        self,
        video: VideoRecord,
        frames: list[FrameRecord],
        detections: list[Detection],
        warnings: list[str],
    ) -> tuple[list[EntityAppearance], list[Detection], str]:
        if not self.settings.enable_appearance_features:
            return [], detections, "disabled"
        existing = self.repository.list_entity_appearances([video.video_id])
        try:
            from .appearance import (
                TrackAppearanceExtractor,
                configured_appearance_encoder,
            )

            encoder = configured_appearance_encoder(self.settings)
            if encoder is None:
                return existing, detections, "disabled"
            extractor = TrackAppearanceExtractor(
                encoder,
                attribute_min_confidence=self.settings.attribute_min_confidence,
                max_observations=self.settings.appearance_max_observations,
            )
            appearances, attached = extractor.extract(
                video_id=video.video_id,
                frames=frames,
                detections=detections,
                existing=existing,
            )
            return appearances, attached, "completed_cached_track_embeddings"
        except Exception as exc:
            LOGGER.exception("Appearance extraction failed for %s", video.video_id)
            from .appearance import attach_cached_appearances

            attached = attach_cached_appearances(detections, existing)
            status = f"failed_{type(exc).__name__}"
            warnings.append(f"Appearance features unavailable: {type(exc).__name__}")
            return existing, attached, status

    def _embed_segments(
        self,
        segments: list[CanonicalSegment],
        warnings: list[str],
    ) -> list[CanonicalSegment]:
        if self.settings.semantic_retrieval_mode == "hashing":
            return segments
        try:
            from .search import configured_text_encoder

            encoder = configured_text_encoder(self.settings)
            embedded: list[CanonicalSegment] = []
            for segment in segments:
                text = " ".join(
                    [
                        segment.transcript,
                        *segment.entities,
                        *segment.actions,
                        *(relationship.predicate for relationship in segment.relationships),
                        *(evidence.text for evidence in segment.ocr_evidence),
                    ]
                ).strip()
                metadata = dict(segment.processing_metadata)
                metadata.update(
                    {
                        "semantic_retrieval_mode": "sentence_transformer",
                        "sentence_embedding_model": encoder.model_name,
                        "sentence_embedding_dimensions": encoder.dimensions,
                        "sentence_embedding_cached": True,
                    }
                )
                embedded.append(
                    segment.model_copy(
                        update={
                            "embedding": encoder.encode(text),
                            "processing_metadata": metadata,
                        }
                    )
                )
            return embedded
        except Exception as exc:
            LOGGER.exception("Sentence embedding generation failed")
            warnings.append(f"Sentence embeddings unavailable: {type(exc).__name__}")
            return segments

    def backfill_gnn_actions(self, video_id: str) -> int:
        """Apply the validated GNN to stored evidence without rerunning media models."""

        video = self.repository.get_video(video_id)
        segments = self.repository.list_segments([video_id])
        detections = {
            detection.detection_id: detection
            for segment in segments
            for detection in segment.detections
        }
        transcripts = {
            transcript.segment_id: transcript
            for segment in segments
            for transcript in segment.transcript_segments
        }
        warnings: list[str] = []
        actions, status = self._gnn_actions(
            video,
            list(detections.values()),
            list(transcripts.values()),
            warnings,
        )
        if status != "available_validated":
            raise RuntimeError(f"GNN action backfill unavailable: {status}")
        by_window: dict[int, list[ActionEvidence]] = defaultdict(list)
        for action in actions:
            by_window[int(action.timestamp // self.settings.segment_duration)].append(action)

        refreshed: list[CanonicalSegment] = []
        for segment in segments:
            index = int(segment.start_time // self.settings.segment_duration)
            predictions = by_window.get(index, [])
            metadata = dict(segment.processing_metadata)
            sources = dict(metadata.get("action_sources", {}))
            confidences = dict(metadata.get("action_confidences", {}))
            stale_gnn_actions = {
                action
                for action, source in sources.items()
                if source == RelationshipSource.GNN_ACTION_MODEL.value
            }
            action_names = set(segment.actions).difference(stale_gnn_actions)
            for action in stale_gnn_actions:
                sources.pop(action, None)
                confidences.pop(action, None)
            for prediction in predictions:
                action_names.add(prediction.label)
                existing_source = sources.get(prediction.label)
                if existing_source not in {
                    RelationshipSource.AVA_GROUND_TRUTH.value,
                    RelationshipSource.MANUAL_ANNOTATION.value,
                }:
                    sources[prediction.label] = RelationshipSource.GNN_ACTION_MODEL.value
                    if prediction.confidence is not None:
                        confidences[prediction.label] = max(
                            float(confidences.get(prediction.label, 0.0)),
                            prediction.confidence,
                        )
            metadata.update(
                {
                    "action_sources": sources,
                    "action_confidences": confidences,
                    "learned_action_model": status,
                    "learned_action_architecture": "pyg_gat80",
                    "gnn_model_version": self.settings.gnn_model_version,
                    "gnn_action_inference_enabled": True,
                    "learned_action_predictions_used": bool(predictions),
                    "gnn_action_backfill": True,
                    "gnn_action_instances": [
                        {
                            "action": item.label,
                            "confidence": item.confidence,
                            "source_entity_id": item.source_entity_id,
                            "target_entity_id": item.target_entity_id,
                            "resolved_predicate": item.resolved_predicate,
                            "provenance": (
                                RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER.value
                                if item.target_entity_id
                                else RelationshipSource.GNN_ACTION_MODEL.value
                            ),
                            "timestamp": item.timestamp,
                        }
                        for item in predictions
                    ],
                }
            )
            retained_relationships = [
                relationship
                for relationship in segment.relationships
                if relationship.source_method
                is not RelationshipSource.GNN_ACTION_PLUS_TARGET_RESOLVER
            ]
            retained_relationships.extend(
                relationship
                for prediction in predictions
                if (relationship := resolved_action_relationship(prediction)) is not None
            )
            refreshed.append(
                segment.model_copy(
                    update={
                        "actions": sorted(action_names),
                        "relationships": retained_relationships,
                        "processing_metadata": metadata,
                    }
                )
            )
        self.repository.replace_segments(video_id, refreshed)
        if self.settings.enable_neo4j:
            from .neo4j import CanonicalNeo4jIndexer

            CanonicalNeo4jIndexer.from_settings(self.settings).ingest(video, refreshed)
        self.repository.update_state(
            video_id,
            ProcessingState.READY,
            current_stage="complete",
            failure_message=None,
            failure_detail=None,
            warnings=[*video.warnings, *warnings],
        )
        return len(actions)

    def _learned_actions(
        self,
        video: VideoRecord,
        detections: list[Detection],
        transcripts: list[TranscriptSegment],
        warnings: list[str],
    ) -> tuple[list[ActionEvidence], str]:
        if self.settings.enable_gnn_actions:
            return self._gnn_actions(video, detections, transcripts, warnings)
        if not self.settings.enable_learned_actions:
            return [], "disabled"
        try:
            from .action_model import (
                PersonActionPredictor,
                action_model_status,
                extract_person_feature_rows,
            )

            status = action_model_status(
                self.settings.action_model_checkpoint,
                self.settings.action_model_metadata,
            )
            if status != "available_validated":
                warnings.append(f"Learned action inference unavailable: {status}")
                return [], status
            predictor = PersonActionPredictor(
                self.settings.action_model_checkpoint,
                self.settings.action_model_metadata,
            )
            by_timestamp: dict[float, list[Detection]] = defaultdict(list)
            for detection in detections:
                by_timestamp[detection.timestamp].append(detection)
            predictions: dict[tuple[float, str], float] = {}
            for timestamp, frame_detections in by_timestamp.items():
                nodes = [
                    {
                        "node_id": index,
                        "class_name": detection.class_label,
                        "confidence": detection.confidence,
                        "bbox": detection.pixel_bbox.as_list(),
                        "center": list(detection.centroid),
                    }
                    for index, detection in enumerate(frame_detections)
                ]
                audio_segments = [
                    {
                        "start": transcript.start_time,
                        "end": transcript.end_time,
                        "speaker": transcript.speaker,
                        "text": transcript.text,
                    }
                    for transcript in transcripts
                    if transcript.start_time <= timestamp <= transcript.end_time
                ]
                for _, features in extract_person_feature_rows(
                    nodes,
                    audio_segments,
                    width=video.width,
                    height=video.height,
                ):
                    for prediction in predictor.predict(list(features)):
                        key = (timestamp, str(prediction["label"]))
                        predictions[key] = max(
                            predictions.get(key, 0.0),
                            float(prediction["confidence"]),
                        )
            return [
                ActionEvidence(
                    timestamp=timestamp,
                    label=label,
                    source_method=RelationshipSource.LEARNED_ACTION_MODEL,
                    confidence=confidence,
                )
                for (timestamp, label), confidence in sorted(predictions.items())
            ], status
        except Exception as exc:
            LOGGER.exception("Learned action inference failed for %s", video.video_id)
            status = f"unavailable_{type(exc).__name__}"
            warnings.append(f"Learned action inference unavailable: {type(exc).__name__}")
            return [], status

    def _gnn_actions(
        self,
        video: VideoRecord,
        detections: list[Detection],
        transcripts: list[TranscriptSegment],
        warnings: list[str],
    ) -> tuple[list[ActionEvidence], str]:
        try:
            import torch

            from .ava_gnn80 import (
                AVA80GNNPredictor,
                ava80_gnn_status,
                build_pyg_timestamp_graph,
                resolve_action_targets,
            )

            status = ava80_gnn_status(
                self.settings.gnn_action_checkpoint,
                self.settings.gnn_action_metadata,
                self.settings.gnn_model_version,
            )
            if status != "available_validated":
                warnings.append(f"GNN action inference unavailable: {status}")
                return [], status
            device = self.settings.gnn_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = AVA80GNNPredictor(
                self.settings.gnn_action_checkpoint,
                self.settings.gnn_action_metadata,
                device=device,
                expected_model_version=self.settings.gnn_model_version,
            )
            by_timestamp: dict[float, list[Detection]] = defaultdict(list)
            for detection in detections:
                by_timestamp[detection.timestamp].append(detection)
            predictions: list[ActionEvidence] = []
            try:
                import cv2

                cached_frames_dir = self.settings.generated_dir / video.video_id / "frames"
                cached_frames = {
                    int(path.stem.rsplit("_", 1)[-1]): path
                    for path in cached_frames_dir.glob("*.jpg")
                    if path.stem.rsplit("_", 1)[-1].isdigit()
                }
                source_stem = Path(video.stored_path).stem
                ava_frames = {
                    int(path.stem.rsplit("_", 1)[-1]): path
                    for path in Path("data/ava/extracted_frames").rglob(
                        f"{source_stem}_*.jpg"
                    )
                }
                capture = (
                    cv2.VideoCapture(video.stored_path)
                    if Path(video.stored_path).is_file()
                    else None
                )
            except ImportError:
                cached_frames_dir = Path("__opencv_unavailable__")
                cached_frames = {}
                capture = None
            visual_warning_added = False
            for timestamp, frame_detections in sorted(by_timestamp.items()):
                nodes = [
                    {
                        "node_id": detection.detection_id,
                        "class_name": detection.class_label,
                        "confidence": detection.confidence,
                        "bbox": detection.pixel_bbox.as_list(),
                        "center": list(detection.centroid),
                    }
                    for detection in frame_detections
                ]
                audio_segments = [
                    {
                        "start": transcript.start_time,
                        "end": transcript.end_time,
                        "speaker": transcript.speaker,
                        "text": transcript.text,
                    }
                    for transcript in transcripts
                    if transcript.start_time <= timestamp <= transcript.end_time
                ]
                frame_image = None
                if cached_frames:
                    cached_frame = nearest_cached_frame(
                        cached_frames,
                        int(round(timestamp * 1000)),
                        tolerance_millisecond=max(
                            100,
                            int(550 / max(self.settings.frame_sample_rate, 0.001)),
                        ),
                    )
                    if cached_frame is not None and cached_frame.is_file():
                        bgr_frame = cv2.imread(str(cached_frame))
                        if bgr_frame is not None:
                            frame_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if frame_image is None:
                    ava_frame = ava_frames.get(int(round(timestamp)))
                    if ava_frame is not None:
                        bgr_frame = cv2.imread(str(ava_frame))
                        if bgr_frame is not None:
                            frame_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if frame_image is None and capture is not None and capture.isOpened():
                    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ok, bgr_frame = capture.read()
                    if ok:
                        frame_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if frame_image is None and not visual_warning_added:
                    warnings.append(
                        "GNN visual frame features unavailable for one or more timestamps; "
                        "zero visual descriptors were used"
                    )
                    visual_warning_added = True
                graph = build_pyg_timestamp_graph(
                    video_id=video.video_id,
                    timestamp=timestamp,
                    nodes=nodes,
                    audio_segments=audio_segments,
                    width=video.width,
                    height=video.height,
                    frame_image=frame_image,
                    require_targets=False,
                )
                if graph is None:
                    continue
                graph_predictions = predictor.predict_graph(graph, frame_image=frame_image)
                resolved = {
                    (item["source_id"], item["action"]): item
                    for item in resolve_action_targets(
                        nodes=nodes, predictions=graph_predictions
                    )
                }
                for source_id, node_predictions in graph_predictions.items():
                    for prediction in node_predictions:
                        label = str(prediction["label"])
                        target = resolved.get((source_id, label))
                        predictions.append(
                            ActionEvidence(
                                timestamp=timestamp,
                                label=label,
                                source_method=RelationshipSource.GNN_ACTION_MODEL,
                                confidence=float(prediction["confidence"]),
                                source_entity_id=source_id,
                                target_entity_id=(
                                    str(target["target_id"]) if target is not None else None
                                ),
                                resolved_predicate=(
                                    str(target["predicate"]) if target is not None else None
                                ),
                            )
                        )
            if capture is not None:
                capture.release()
            return predictions, status
        except Exception as exc:
            LOGGER.exception("GNN action inference failed for %s", video.video_id)
            status = f"unavailable_{type(exc).__name__}"
            warnings.append(f"GNN action inference unavailable: {type(exc).__name__}")
            return [], status
