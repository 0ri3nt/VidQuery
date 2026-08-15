import hashlib
import re
from pathlib import Path

from .neo4j_client import Neo4jClient


class GraphIngestor:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    def apply_schema(self, schema_path: Path) -> None:
        statements = [
            stmt.strip()
            for stmt in schema_path.read_text(encoding="utf-8").split(";")
            if stmt.strip()
        ]
        for stmt in statements:
            self.neo4j.run_query(stmt)

    def ingest_video_metadata(self, video_dict: dict) -> None:
        video_id = str(video_dict.get("video_id", "")).strip()
        if not video_id:
            return

        query = """
        MERGE (v:Video {video_id: $video_id})
        SET v.file_path = coalesce($file_path, v.file_path),
            v.split = coalesce($split, v.split),
            v.source = coalesce($source, v.source)
        """
        self.neo4j.run_query(
            query,
            {
                "video_id": video_id,
                "file_path": video_dict.get("file_path"),
                "split": video_dict.get("split"),
                "source": video_dict.get("source"),
            },
        )

    @staticmethod
    def _sanitize_relation(rel: str) -> str:
        rel = str(rel or "RELATED_TO").upper().replace("-", "_")
        rel = re.sub(r"[^A-Z0-9_]", "_", rel)
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", rel):
            return "RELATED_TO"
        return rel

    @staticmethod
    def _concept_from_action(action: str) -> str:
        return str(action or "unknown_action").strip().lower().replace("_", " ")

    def _merge_frame_scene(
        self,
        video_id: str,
        split: str | None,
        frame_data: dict,
    ) -> tuple[int, str]:
        timestamp = int(frame_data.get("timestamp", 0))
        frame_file = str(frame_data.get("frame_file", ""))
        scene_key = f"{video_id}:{timestamp}"

        self.neo4j.run_query(
            """
            MERGE (v:Video {video_id: $video_id})
            SET v.split = coalesce($split, v.split)

            MERGE (s:Scene {scene_key: $scene_key})
            SET s.video_id = $video_id,
                s.timestamp = $timestamp

            MERGE (f:Frame {video_id: $video_id, timestamp: $timestamp})
            SET f.frame_file = $frame_file

            MERGE (v)-[:HAS_SCENE]->(s)
            MERGE (s)-[:HAS_FRAME]->(f)
            MERGE (v)-[:HAS_FRAME]->(f)
            """,
            {
                "video_id": video_id,
                "split": split,
                "scene_key": scene_key,
                "timestamp": timestamp,
                "frame_file": frame_file,
            },
        )
        return timestamp, scene_key

    def _merge_visual_node(self, video_id: str, timestamp: int, node: dict) -> tuple[str, bool]:
        node_id = int(node.get("node_id", 0))
        class_name = str(node.get("class_name", "object"))
        confidence = float(node.get("confidence", 0.0))
        bbox = [float(v) for v in node.get("bbox", [])]
        center = [float(v) for v in node.get("center", [])]

        if class_name.lower() == "person":
            person_key = f"{video_id}:{timestamp}:{node_id}"
            self.neo4j.run_query(
                """
                MERGE (p:Person {person_key: $person_key})
                SET p.class_name = $class_name,
                    p.confidence = $confidence,
                    p.node_id = $node_id,
                    p.video_id = $video_id,
                    p.timestamp = $timestamp,
                    p.bbox = $bbox,
                    p.center = $center
                WITH p
                MATCH (f:Frame {video_id: $video_id, timestamp: $timestamp})
                MERGE (f)-[:HAS_PERSON]->(p)
                """,
                {
                    "person_key": person_key,
                    "class_name": class_name,
                    "confidence": confidence,
                    "node_id": node_id,
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "center": center,
                },
            )
            return person_key, True

        object_key = f"{video_id}:{timestamp}:{node_id}"
        self.neo4j.run_query(
            """
            MERGE (o:Object {object_key: $object_key})
            SET o.class_name = $class_name,
                o.confidence = $confidence,
                o.node_id = $node_id,
                o.video_id = $video_id,
                o.timestamp = $timestamp,
                o.bbox = $bbox,
                o.center = $center
            WITH o
            MATCH (f:Frame {video_id: $video_id, timestamp: $timestamp})
            MERGE (f)-[:HAS_OBJECT]->(o)
            """,
            {
                "object_key": object_key,
                "class_name": class_name,
                "confidence": confidence,
                "node_id": node_id,
                "video_id": video_id,
                "timestamp": timestamp,
                "bbox": bbox,
                "center": center,
            },
        )
        return object_key, False

    def _merge_performs(
        self,
        person_key: str,
        concept_name: str,
        video_id: str,
        timestamp: int,
        event_key: str | None = None,
    ) -> None:
        concept_name = self._concept_from_action(concept_name)
        if not concept_name:
            return

        if event_key:
            self.neo4j.run_query(
                """
                MATCH (p:Person {person_key: $person_key})
                MERGE (c:Concept {name: $concept_name})
                MERGE (p)-[r:PERFORMS {event_key: $event_key}]->(c)
                SET r.video_id = $video_id,
                    r.timestamp = $timestamp
                """,
                {
                    "person_key": person_key,
                    "concept_name": concept_name,
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "event_key": event_key,
                },
            )
            return

        self.neo4j.run_query(
            """
            MATCH (p:Person {person_key: $person_key})
            MERGE (c:Concept {name: $concept_name})
            MERGE (p)-[r:PERFORMS]->(c)
            SET r.video_id = $video_id,
                r.timestamp = $timestamp
            """,
            {
                "person_key": person_key,
                "concept_name": concept_name,
                "video_id": video_id,
                "timestamp": timestamp,
            },
        )

    @staticmethod
    def _build_audio_segment_key(
        video_id: str,
        timestamp: int,
        speaker_label: str,
        start: float,
        end: float,
        text: str,
    ) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        text_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"{video_id}:{timestamp}:{speaker_label}:{start:.3f}:{end:.3f}:{text_hash}"

    def _merge_audio_speaker(self, video_id: str, timestamp: int, segment: dict) -> None:
        speaker_label = str(segment.get("speaker", "UNKNOWN_SPEAKER"))
        text = str(segment.get("text", "")).strip()
        start = float(segment.get("start", float(timestamp)))
        end = float(segment.get("end", float(timestamp) + 1.0))
        duration = max(0.0, end - start)
        speaker_key = f"{video_id}:speaker:{speaker_label}"
        segment_key = self._build_audio_segment_key(
            video_id=video_id,
            timestamp=timestamp,
            speaker_label=speaker_label,
            start=start,
            end=end,
            text=text,
        )

        self.neo4j.run_query(
            """
            MERGE (sp:Person {person_key: $speaker_key})
            SET sp.class_name = 'speaker',
                sp.speaker_label = $speaker_label,
                sp.video_id = $video_id

            MERGE (seg:AudioSegment {segment_key: $segment_key})
            SET seg.video_id = $video_id,
                seg.timestamp = $timestamp,
                seg.speaker_label = $speaker_label,
                seg.start = $start,
                seg.end = $end,
                seg.duration = $duration,
                seg.text = $text

            WITH sp, seg
            MATCH (f:Frame {video_id: $video_id, timestamp: $timestamp})
            MERGE (f)-[:HAS_AUDIO_SEGMENT]->(seg)
            MERGE (seg)-[:SPOKEN_BY]->(sp)
            MERGE (f)-[r:SPOKEN_BY {segment_key: $segment_key}]->(sp)
            SET r.start = $start,
                r.end = $end,
                r.text = $text
            """,
            {
                "speaker_key": speaker_key,
                "segment_key": segment_key,
                "speaker_label": speaker_label,
                "video_id": video_id,
                "timestamp": timestamp,
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
            },
        )

        self._merge_performs(
            person_key=speaker_key,
            concept_name="speech",
            video_id=video_id,
            timestamp=timestamp,
            event_key=segment_key,
        )

    def _ingest_frame(self, video_id: str, split: str | None, frame_data: dict) -> None:
        timestamp, _ = self._merge_frame_scene(video_id, split, frame_data)

        nodes = frame_data.get("nodes", [])
        edges = frame_data.get("edges", [])
        node_lookup: dict[int, tuple[str, bool]] = {}

        for node in nodes:
            key, is_person = self._merge_visual_node(video_id, timestamp, node)
            node_lookup[int(node.get("node_id", 0))] = (key, is_person)

            if is_person:
                ann = node.get("matched_annotation")
                if isinstance(ann, dict):
                    for action_label in ann.get("action_labels", []):
                        self._merge_performs(key, str(action_label), video_id, timestamp)

        for edge in edges:
            src_id = int(edge.get("source", -1))
            dst_id = int(edge.get("target", -1))
            if src_id not in node_lookup or dst_id not in node_lookup:
                continue

            source_key, source_is_person = node_lookup[src_id]
            target_key, _ = node_lookup[dst_id]

            relation_raw = str(edge.get("type", "RELATED_TO"))
            relation = self._sanitize_relation(relation_raw)

            query = f"""
            MATCH (s) WHERE (s:Object OR s:Person) AND
                  ((s.object_key = $source_key) OR (s.person_key = $source_key))
            MATCH (t) WHERE (t:Object OR t:Person) AND
                  ((t.object_key = $target_key) OR (t.person_key = $target_key))
            MERGE (s)-[r:{relation}]->(t)
            SET r.weight = $weight,
                r.distance = $distance
            """
            self.neo4j.run_query(
                query,
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "weight": float(edge.get("weight", 0.0)),
                    "distance": float(edge.get("distance", 0.0)),
                },
            )

            if source_is_person and src_id == dst_id:
                self._merge_performs(source_key, relation_raw, video_id, timestamp)

        for segment in frame_data.get("audio_segments", []):
            self._merge_audio_speaker(video_id, timestamp, segment)

    def ingest_scene_graph(self, fused_json: dict) -> None:
        video_id = str(fused_json.get("video_id", "")).strip()
        split = fused_json.get("split")

        if isinstance(fused_json.get("frames"), list):
            self.ingest_video_metadata(
                {
                    "video_id": video_id,
                    "split": split,
                    "source": "fused",
                }
            )
            for frame in fused_json.get("frames", []):
                frame_video_id = str(frame.get("video_id", video_id)).strip() or video_id
                self._ingest_frame(frame_video_id, split, frame)
            return

        if not video_id:
            video_id = str(fused_json.get("video_id", "unknown_video"))

        self.ingest_video_metadata(
            {
                "video_id": video_id,
                "split": split,
                "source": "scene_graph",
            }
        )
        self._ingest_frame(video_id, split, fused_json)
