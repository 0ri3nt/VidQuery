from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from vidquery.config import get_settings
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.storage import SQLiteRepository

VIDEO_ID = "aa03a76b-b03b-4f97-8542-d89736ab29e7"


def _counts(indexer: CanonicalNeo4jIndexer) -> dict:
    return indexer.run_query(
        """
        MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
        OPTIONAL MATCH (s)-[:HAS_ENTITY]->(entity:EntityOccurrence)
        OPTIONAL MATCH (s)-[:HAS_ACTION]->(action:Action)
        OPTIONAL MATCH (s)-[:HAS_RELATIONSHIP]->(evidence:RelationshipEvidence)
        RETURN count(DISTINCT s) AS segments,
               count(DISTINCT entity) AS entity_occurrences,
               count(DISTINCT action) AS actions,
               count(DISTINCT evidence) AS relationship_evidence
        """,
        {"video_id": VIDEO_ID},
    )[0]


def main() -> None:
    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    video = repository.get_video(VIDEO_ID)
    segments = repository.list_segments([VIDEO_ID])
    indexer = CanonicalNeo4jIndexer.from_settings(settings)
    try:
        indexer.apply_schema(Path("database/schema.cypher"))
        indexer.ingest(video, segments)
        first = _counts(indexer)
        indexer.ingest(video, segments)
        second = _counts(indexer)
        gnn_only_action = indexer.run_query(
            """
            MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
                  -[:HAS_ACTION]->(action:Action {name: $action})
            RETURN s.segment_id AS segment_id,
                   s.start_time AS start_time,
                   s.end_time AS end_time,
                   action.name AS action
            ORDER BY s.start_time
            LIMIT 5
            """,
            {"video_id": VIDEO_ID, "action": "write"},
        )
        resolved_relationships = indexer.run_query(
            """
            MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
                  -[:HAS_RELATIONSHIP]->(evidence:RelationshipEvidence)
            WHERE evidence.source_method = $source_method
            RETURN s.segment_id AS segment_id,
                   evidence.predicate AS predicate,
                   evidence.confidence AS confidence,
                   evidence.source_method AS provenance,
                   evidence.timestamp AS timestamp
            ORDER BY evidence.confidence DESC
            LIMIT 5
            """,
            {
                "video_id": VIDEO_ID,
                "source_method": "gnn_action_plus_target_resolver",
            },
        )
        constraints = indexer.run_query(
            "SHOW CONSTRAINTS YIELD name RETURN name ORDER BY name"
        )
        indexes = indexer.run_query("SHOW INDEXES YIELD name RETURN name ORDER BY name")
        report = {
            "schema_version": "ava80-scale30-neo4j-proof-v1",
            "captured_at": datetime.now(UTC).isoformat(),
            "video_id": VIDEO_ID,
            "model_version": settings.gnn_model_version,
            "health": indexer.health_check(),
            "constraint_count": len(constraints),
            "index_count": len(indexes),
            "counts_after_first_ingestion": first,
            "counts_after_second_ingestion": second,
            "idempotent": first == second,
            "gnn_only_action_write": gnn_only_action,
            "gnn_target_resolver_relationships": resolved_relationships,
            "verified": {
                "schema": bool(constraints and indexes),
                "action_node": bool(gnn_only_action),
                "relationship_provenance": bool(resolved_relationships),
                "duplicate_prevention": first == second,
            },
        }
    finally:
        indexer.close()
    output = Path("evaluation/results/ava80-scale30-neo4j-proof.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
