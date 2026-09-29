from __future__ import annotations

from pathlib import Path

from .config import Settings
from .domain import CanonicalSegment, EntityAppearance, VideoRecord
from .relationships import normalize_predicate, relationship_directionality


class CanonicalNeo4jIndexer:
    def __init__(self, uri: str, username: str, password: str):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("neo4j Python driver is not installed") from exc
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()

    @classmethod
    def from_settings(cls, settings: Settings) -> CanonicalNeo4jIndexer:
        if not settings.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is required when ENABLE_NEO4J=true")
        return cls(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)

    def close(self) -> None:
        self.driver.close()

    def run_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def health_check(self) -> bool:
        result = self.run_query("RETURN 1 AS ok")
        return bool(result and result[0].get("ok") == 1)

    def apply_schema(self, schema_path: Path) -> None:
        statements = [
            statement.strip()
            for statement in schema_path.read_text(encoding="utf-8").split(";")
            if statement.strip()
        ]
        for statement in statements:
            self.run_query(statement)

    def ingest(
        self,
        video: VideoRecord,
        segments: list[CanonicalSegment],
        appearances: list[EntityAppearance] | None = None,
    ) -> None:
        self.run_query(
            """
            MERGE (v:Video {video_id: $video_id})
            SET v.display_name = $display_name,
                v.stored_path = $stored_path,
                v.duration = $duration,
                v.width = $width,
                v.height = $height,
                v.fps = $fps,
                v.has_audio = $has_audio,
                v.state = $state
            """,
            {
                "video_id": video.video_id,
                "display_name": video.display_name,
                "stored_path": video.stored_path,
                "duration": video.duration,
                "width": video.width,
                "height": video.height,
                "fps": video.fps,
                "has_audio": video.has_audio,
                "state": video.state.value,
            },
        )

        # SQLite is authoritative. Remove only the previously mirrored evidence
        # owned by this video before recreating it, so changed actions or
        # relationships cannot survive as stale graph edges after reprocessing.
        self.run_query(
            """
            MATCH (owned)
            WHERE owned.video_id = $video_id
              AND (owned:EntityOccurrence
                   OR owned:TranscriptSegment
                   OR owned:Speaker
                   OR owned:RelationshipEvidence
                   OR owned:OCRText
                   OR owned:EntityAppearance)
            DETACH DELETE owned
            """,
            {"video_id": video.video_id},
        )
        self.run_query(
            """
            MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
            DETACH DELETE s
            """,
            {"video_id": video.video_id},
        )
        self.run_query(
            """
            MATCH (action:Action)
            WHERE NOT (action)<-[:HAS_ACTION]-(:Segment)
            DELETE action
            """
        )

        rows = [self._segment_row(segment) for segment in segments]
        self.run_query(
            """
            UNWIND $segments AS row
            MATCH (v:Video {video_id: $video_id})
            MERGE (s:Segment {segment_id: row.segment_id})
            SET s.video_id = $video_id,
                s.start_time = row.start_time,
                s.end_time = row.end_time,
                s.transcript = row.transcript
            MERGE (v)-[:HAS_SEGMENT]->(s)

            FOREACH (entity IN row.entities |
                MERGE (occ:EntityOccurrence {detection_id: entity.detection_id})
                SET occ.video_id = $video_id,
                    occ.frame_id = entity.frame_id,
                    occ.timestamp = entity.timestamp,
                    occ.class_id = entity.class_id,
                    occ.class_label = entity.class_label,
                    occ.confidence = entity.confidence,
                    occ.track_id = entity.track_id,
                    occ.appearance_id = entity.appearance_id,
                    occ.pixel_bbox = entity.pixel_bbox,
                    occ.normalized_bbox = entity.normalized_bbox
                MERGE (s)-[:HAS_ENTITY]->(occ)
            )

            FOREACH (transcript IN row.transcripts |
                MERGE (tx:TranscriptSegment {transcript_segment_id: transcript.segment_id})
                SET tx.video_id = $video_id,
                    tx.start_time = transcript.start_time,
                    tx.end_time = transcript.end_time,
                    tx.text = transcript.text,
                    tx.confidence = transcript.confidence,
                    tx.speaker = transcript.speaker
                MERGE (s)-[:HAS_TRANSCRIPT]->(tx)
            )

            FOREACH (speaker IN row.speakers |
                MERGE (sp:Speaker {speaker_key: $video_id + ':' + speaker})
                SET sp.video_id = $video_id, sp.label = speaker
                MERGE (s)-[:HAS_SPEAKER]->(sp)
            )

            FOREACH (action IN row.actions |
                MERGE (a:Action {name: action})
                MERGE (s)-[:HAS_ACTION]->(a)
            )

            FOREACH (ocr IN row.ocr_evidence |
                MERGE (text:OCRText {ocr_id: ocr.ocr_id})
                SET text.video_id = $video_id,
                    text.text = ocr.text,
                    text.start_time = ocr.start_time,
                    text.end_time = ocr.end_time,
                    text.confidence = ocr.confidence,
                    text.frame_ids = ocr.frame_ids,
                    text.source_method = ocr.source_method
                MERGE (s)-[:HAS_OCR]->(text)
            )

            FOREACH (relationship IN row.relationships |
                MERGE (e:RelationshipEvidence {relationship_id: relationship.relationship_id})
                SET e.video_id = $video_id,
                    e.source_id = relationship.source_id,
                    e.target_id = relationship.target_id,
                    e.source_class = relationship.source_class,
                    e.target_class = relationship.target_class,
                    e.predicate = relationship.predicate,
                    e.directionality = relationship.directionality,
                    e.confidence = relationship.confidence,
                    e.source_method = relationship.source_method,
                    e.timestamp = relationship.timestamp,
                    e.model_version = relationship.model_version,
                    e.start_time = relationship.start_time,
                    e.end_time = relationship.end_time,
                    e.source_track_id = relationship.source_track_id,
                    e.target_track_id = relationship.target_track_id,
                    e.observation_count = relationship.observation_count,
                    e.temporal_smoothed = relationship.temporal_smoothed
                MERGE (s)-[:HAS_RELATIONSHIP]->(e)
            )
            """,
            {"video_id": video.video_id, "segments": rows},
        )

        self.run_query(
            """
            UNWIND $appearances AS row
            MERGE (appearance:EntityAppearance {appearance_id: row.appearance_id})
            SET appearance.video_id = $video_id,
                appearance.track_id = row.track_id,
                appearance.class_label = row.class_label,
                appearance.appearance_embedding_model = row.appearance_embedding_model,
                appearance.appearance_embedding_version = row.appearance_embedding_version,
                appearance.observation_count = row.observation_count,
                appearance.source_method = row.source_method,
                appearance.color = row.color,
                appearance.color_confidence = row.color_confidence,
                appearance.upper_clothing_color = row.upper_clothing_color,
                appearance.upper_clothing_color_confidence =
                    row.upper_clothing_color_confidence,
                appearance.upper_clothing_description =
                    row.upper_clothing_description,
                appearance.glasses = row.glasses,
                appearance.glasses_confidence = row.glasses_confidence,
                appearance.hat = row.hat,
                appearance.hat_confidence = row.hat_confidence,
                appearance.bag = row.bag,
                appearance.bag_confidence = row.bag_confidence
            WITH appearance
            MATCH (occ:EntityOccurrence {appearance_id: appearance.appearance_id})
            MERGE (occ)-[:HAS_APPEARANCE]->(appearance)
            """,
            {
                "video_id": video.video_id,
                "appearances": [
                    self._appearance_row(item) for item in (appearances or [])
                ],
            },
        )

        self.run_query(
            """
            MATCH (e:RelationshipEvidence {video_id: $video_id})
            OPTIONAL MATCH (source:EntityOccurrence {detection_id: e.source_id})
            OPTIONAL MATCH (target:EntityOccurrence {detection_id: e.target_id})
            FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END |
                MERGE (e)-[:FROM_ENTITY]->(source)
            )
            FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
                MERGE (e)-[:TO_ENTITY]->(target)
            )
            """,
            {"video_id": video.video_id},
        )

    @staticmethod
    def _segment_row(segment: CanonicalSegment) -> dict:
        detection_classes = {
            item.detection_id: item.class_label.lower() for item in segment.detections
        }
        return {
            "segment_id": segment.segment_id,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "transcript": segment.transcript,
            "entities": [
                {
                    "detection_id": item.detection_id,
                    "frame_id": item.frame_id,
                    "timestamp": item.timestamp,
                    "class_id": item.class_id,
                    "class_label": item.class_label,
                    "confidence": item.confidence,
                    "track_id": item.track_id,
                    "appearance_id": item.appearance_id,
                    "pixel_bbox": item.pixel_bbox.as_list(),
                    "normalized_bbox": item.normalized_bbox.as_list(),
                }
                for item in segment.detections
            ],
            "transcripts": [
                {
                    "segment_id": item.segment_id,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "text": item.text,
                    "confidence": item.confidence,
                    "speaker": item.speaker,
                }
                for item in segment.transcript_segments
            ],
            "speakers": segment.speakers,
            "actions": segment.actions,
            "ocr_evidence": [
                {
                    "ocr_id": item.ocr_id,
                    "text": item.text,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "confidence": item.confidence,
                    "frame_ids": item.frame_ids,
                    "source_method": item.source_method.value,
                }
                for item in segment.ocr_evidence
            ],
            "relationships": [
                {
                    "relationship_id": item.relationship_id,
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "source_class": detection_classes.get(item.source_id),
                    "target_class": detection_classes.get(item.target_id),
                    "predicate": normalize_predicate(item.predicate),
                    "directionality": relationship_directionality(item.predicate),
                    "confidence": item.confidence,
                    "source_method": item.source_method.value,
                    "timestamp": item.timestamp,
                    "model_version": item.model_version,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "source_track_id": item.source_track_id,
                    "target_track_id": item.target_track_id,
                    "observation_count": item.observation_count,
                    "temporal_smoothed": item.temporal_smoothed,
                }
                for item in segment.relationships
            ],
        }

    @staticmethod
    def _appearance_row(item: EntityAppearance) -> dict:
        def value(name: str):
            evidence = item.attributes.get(name)
            return evidence.value if evidence is not None else None

        def confidence(name: str):
            evidence = item.attributes.get(name)
            return evidence.confidence if evidence is not None else None

        return {
            "appearance_id": item.appearance_id,
            "track_id": item.track_id,
            "class_label": item.class_label,
            "appearance_embedding_model": item.appearance_embedding_model,
            "appearance_embedding_version": item.appearance_embedding_version,
            "observation_count": item.observation_count,
            "source_method": item.source_method.value,
            "color": value("color"),
            "color_confidence": confidence("color"),
            "upper_clothing_color": value("upper_clothing_color"),
            "upper_clothing_color_confidence": confidence(
                "upper_clothing_color"
            ),
            "upper_clothing_description": value("upper_clothing_description"),
            "glasses": value("glasses"),
            "glasses_confidence": confidence("glasses"),
            "hat": value("hat"),
            "hat_confidence": confidence("hat"),
            "bag": value("bag"),
            "bag_confidence": confidence("bag"),
        }
