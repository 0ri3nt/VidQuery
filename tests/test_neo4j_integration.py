from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vidquery.config import get_settings
from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    ProcessingState,
    Relationship,
    RelationshipSource,
    SearchRequest,
    VideoRecord,
)
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.search import Neo4jGraphSearchEngine, SafeCypherCompiler, StructuredQueryParser
from vidquery.storage import SQLiteRepository

pytestmark = pytest.mark.integration

INTEGRATION_VIDEO_IDS = ["integration-video-a", "integration-video-b"]


def _cleanup_integration_records(indexer: CanonicalNeo4jIndexer) -> None:
    parameters = {"video_ids": INTEGRATION_VIDEO_IDS}
    indexer.run_query(
        """
        MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)-[:HAS_RELATIONSHIP]->(e)
        WHERE v.video_id IN $video_ids
        DETACH DELETE e
        """,
        parameters,
    )
    indexer.run_query(
        """
        MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)-[:HAS_ENTITY]->(occ)
        WHERE v.video_id IN $video_ids
        DETACH DELETE occ
        """,
        parameters,
    )
    indexer.run_query(
        """
        MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)
        WHERE v.video_id IN $video_ids
        DETACH DELETE s
        """,
        parameters,
    )
    indexer.run_query(
        "MATCH (v:Video) WHERE v.video_id IN $video_ids DETACH DELETE v",
        parameters,
    )


@pytest.fixture
def live_neo4j():
    settings = get_settings()
    if not settings.neo4j_password:
        pytest.skip("Neo4j integration skipped: NEO4J_PASSWORD is not configured")
    try:
        indexer = CanonicalNeo4jIndexer.from_settings(settings)
    except Exception as exc:
        pytest.skip(f"Neo4j integration skipped: service unavailable ({type(exc).__name__})")
    try:
        _cleanup_integration_records(indexer)
        yield indexer
    finally:
        _cleanup_integration_records(indexer)
        indexer.close()


def _video(video_id: str, hash_character: str) -> VideoRecord:
    now = datetime.now(UTC)
    return VideoRecord(
        video_id=video_id,
        display_name=f"{video_id}.mp4",
        original_filename=f"{video_id}.mp4",
        stored_path=f"{video_id}.mp4",
        content_sha256=hash_character * 64,
        duration=10,
        width=100,
        height=100,
        fps=10,
        has_audio=False,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
    )


def _detection(video_id: str, identifier: str, label: str, timestamp: float) -> Detection:
    return Detection(
        detection_id=identifier,
        frame_id=f"{video_id}-frame",
        video_id=video_id,
        timestamp=timestamp,
        class_id=0,
        class_label=label,
        confidence=0.9,
        pixel_bbox=BoundingBox(x1=10, y1=10, x2=30, y2=30),
        normalized_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3),
        centroid=(20, 20),
        centroid_normalized=(0.2, 0.2),
    )


def _segment(
    video_id: str,
    target_label: str,
    *,
    include_chair_distractor: bool = False,
) -> CanonicalSegment:
    timestamp = 5.0
    person = _detection(video_id, f"{video_id}-person", "person", timestamp)
    target = _detection(video_id, f"{video_id}-target", target_label, timestamp)
    detections = [person, target]
    if include_chair_distractor:
        detections.append(_detection(video_id, f"{video_id}-chair", "chair", timestamp))
    relationships = [
        Relationship(
            relationship_id=f"{video_id}-near",
            source_id=person.detection_id,
            target_id=target.detection_id,
            predicate="near",
            confidence=0.8,
            source_method=RelationshipSource.MANUAL_ANNOTATION,
            timestamp=timestamp,
        )
    ]
    if target_label == "chair":
        relationships.append(
            Relationship(
                relationship_id=f"{video_id}-left-of",
                source_id=person.detection_id,
                target_id=target.detection_id,
                predicate="left_of",
                confidence=0.7,
                source_method=RelationshipSource.MANUAL_ANNOTATION,
                timestamp=timestamp,
            )
        )
    return CanonicalSegment(
        segment_id=f"{video_id}:5000:10000",
        video_id=video_id,
        start_time=5,
        end_time=10,
        frame_ids=[f"{video_id}-frame"],
        detections=detections,
        entities=sorted({item.class_label for item in detections}),
        relationships=relationships,
    )


def test_live_schema_ingestion_idempotency_and_exact_graph_search(live_neo4j, tmp_path):
    live_neo4j.apply_schema(Path("database/schema.cypher"))
    constraint_names = {
        row["name"] for row in live_neo4j.run_query("SHOW CONSTRAINTS YIELD name RETURN name")
    }
    index_names = {
        row["name"] for row in live_neo4j.run_query("SHOW INDEXES YIELD name RETURN name")
    }
    assert "segment_id_unique" in constraint_names
    assert "relationship_evidence_id_unique" in constraint_names
    assert "segment_video_time_idx" in index_names
    assert "relationship_predicate_idx" in index_names

    repository = SQLiteRepository(tmp_path / "neo4j-integration.sqlite3")
    videos = [
        _video("integration-video-a", "c"),
        _video("integration-video-b", "d"),
    ]
    segments = [
        _segment("integration-video-a", "chair"),
        _segment("integration-video-b", "dining table", include_chair_distractor=True),
    ]
    for video, segment in zip(videos, segments, strict=True):
        repository.create_video(video)
        repository.replace_segments(video.video_id, [segment])
        live_neo4j.ingest(video, [segment])
        live_neo4j.ingest(video, [segment])

    counts = live_neo4j.run_query(
        """
        MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)
        WHERE v.video_id IN $video_ids
        OPTIONAL MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
        RETURN count(DISTINCT v) AS videos,
               count(DISTINCT s) AS segments,
               count(DISTINCT e) AS relationship_evidence
        """,
        {"video_ids": [video.video_id for video in videos]},
    )[0]
    assert counts == {"videos": 2, "segments": 2, "relationship_evidence": 3}
    assert live_neo4j.health_check()

    engine = Neo4jGraphSearchEngine(repository, live_neo4j)
    symmetric = engine.search(
        SearchRequest(query="chair near person", retrieval_backend="neo4j")
    )
    assert symmetric.retrieval_backend == "neo4j"
    assert [item.segment_id for item in symmetric.results] == [
        "integration-video-a:5000:10000"
    ]
    assert symmetric.results[0].start_time == 5
    assert symmetric.results[0].end_time == 10
    assert symmetric.results[0].matched_relationships[0].target_class == "chair"

    reversed_direction = engine.search(
        SearchRequest(query="chair left of person", retrieval_backend="neo4j")
    )
    assert reversed_direction.results == []

    compiled = SafeCypherCompiler().compile(
        StructuredQueryParser().parse("person near chair"), [], 5
    )
    rows = live_neo4j.run_query(compiled.cypher, compiled.parameters)
    integration_rows = [
        row for row in rows if row["segment_id"] == "integration-video-a:5000:10000"
    ]
    assert integration_rows[0]["start_time"] == 5
    assert integration_rows[0]["relationship_tuples"][0]["source_class"] == "person"
    assert integration_rows[0]["relationship_tuples"][0]["target_class"] == "chair"
