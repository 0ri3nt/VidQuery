from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from vidquery.config import get_settings
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.storage import SQLiteRepository

BASE_URL = "http://127.0.0.1:8000"
FINAL_DEMO_ID = "4d508fd2-cb26-4caf-8c06-f7589c23764b"
OCR_DEMO_ID = "291cb37c-6bb5-410f-9c09-437cfcbd80cc"


def _request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _search(query: str, video_id: str, backend: str = "sqlite") -> dict[str, Any]:
    response = _request(
        "/api/search",
        {
            "query": query,
            "video_ids": [video_id],
            "limit": 10,
            "retrieval_backend": backend,
        },
    )
    if not response.get("results"):
        raise RuntimeError(f"proof query returned no result: {query} ({backend})")
    return response


def main() -> None:
    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    health = _request("/api/health")
    final_video = repository.get_video(FINAL_DEMO_ID)
    final_segments = repository.list_segments([FINAL_DEMO_ID])
    ocr_segments = repository.list_segments([OCR_DEMO_ID])
    if not final_segments or not ocr_segments:
        raise RuntimeError("controlled proof fixtures are not indexed")

    sqlite_relation = _search("person in front of person", FINAL_DEMO_ID)
    neo4j_relation = _search(
        "person in front of person", FINAL_DEMO_ID, backend="neo4j"
    )
    semantic = _search(
        "Find where the application is released to production", FINAL_DEMO_ID
    )
    ocr = _search("Kubernetes", OCR_DEMO_ID)

    relation_matches = [
        match
        for result in sqlite_relation["results"]
        for match in result["matched_relationships"]
        if match["source_method"] == "relationship_gnn"
    ]
    if not relation_matches:
        raise RuntimeError("no relationship-GNN tuple appeared in SQLite search")

    metadata = [segment.processing_metadata for segment in final_segments]
    learned_relation_count = sum(
        relationship.source_method.value == "relationship_gnn"
        for segment in final_segments
        for relationship in segment.relationships
    )
    smoothed_relation_count = sum(
        relationship.temporal_smoothed
        for segment in final_segments
        for relationship in segment.relationships
    )
    stale_untracked_count = sum(
        relationship.source_method.value == "relationship_gnn"
        and not relationship.source_track_id
        for segment in final_segments
        for relationship in segment.relationships
    )

    indexer = CanonicalNeo4jIndexer.from_settings(settings)
    try:
        neo4j_counts = indexer.run_query(
            """
            MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
            OPTIONAL MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
            OPTIONAL MATCH (s)-[:HAS_OCR]->(ocr:OCRText)
            RETURN count(DISTINCT s) AS segments,
                   count(DISTINCT e) AS relationships,
                   count(DISTINCT CASE
                     WHEN e.source_method = $learned_source THEN e END
                   ) AS learned_relationships,
                   count(DISTINCT CASE
                     WHEN e.temporal_smoothed = true THEN e END
                   ) AS smoothed_relationships,
                   count(DISTINCT ocr) AS ocr_nodes
            """,
            {
                "video_id": FINAL_DEMO_ID,
                "learned_source": "relationship_gnn",
            },
        )[0]
        ocr_neo4j = indexer.run_query(
            """
            MATCH (:Video {video_id: $video_id})-[:HAS_SEGMENT]->(:Segment)
                  -[:HAS_OCR]->(ocr:OCRText)
            RETURN count(DISTINCT ocr) AS ocr_nodes,
                   collect(DISTINCT toLower(ocr.text)) AS texts
            """,
            {"video_id": OCR_DEMO_ID},
        )[0]
    finally:
        indexer.close()

    proof = {
        "schema_version": "vidquery-full-fresh-video-proof-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "video": {
            "video_id": final_video.video_id,
            "display_name": final_video.display_name,
            "duration": final_video.duration,
            "state": final_video.state.value,
            "warnings": final_video.warnings,
            "segments": len(final_segments),
        },
        "health": health,
        "fresh_processing": {
            "diarization_statuses": sorted(
                {str(item.get("diarization_status")) for item in metadata}
            ),
            "tracking_statuses": sorted(
                {str(item.get("temporal_tracking_status")) for item in metadata}
            ),
            "smoothing_statuses": sorted(
                {
                    str(item.get("relationship_smoothing_status"))
                    for item in metadata
                }
            ),
            "ocr_statuses": sorted({str(item.get("ocr_status")) for item in metadata}),
            "embedding_dimensions": sorted(
                {len(segment.embedding or []) for segment in final_segments}
            ),
            "learned_relationships": learned_relation_count,
            "smoothed_relationships": smoothed_relation_count,
            "stale_untracked_learned_relationships": stale_untracked_count,
            "maximum_observation_count": max(
                (item["observation_count"] for item in relation_matches), default=0
            ),
        },
        "queries": {
            "sqlite_exact_relation": sqlite_relation,
            "neo4j_exact_relation": neo4j_relation,
            "semantic_paraphrase": semantic,
            "ocr": ocr,
        },
        "neo4j": {
            "final_demo": neo4j_counts,
            "ocr_demo": ocr_neo4j,
        },
        "verified": {
            "application_healthy": health["application"] == "ok",
            "sqlite_healthy": health["database"] == "available",
            "neo4j_healthy": health["neo4j"] == "available",
            "compulsory_diarization_configured": health["models"]["diarization"]
            == "available_configured_required",
            "action_gnn_validated": health["models"][
                "gnn_person_action_classifier"
            ]
            == "available_validated_enabled",
            "relation_gnn_validated": health["models"][
                "gnn_relationship_prediction"
            ]
            == "available_validated_enabled",
            "sentence_embeddings_cached": health["models"][
                "sentence_embedding_retrieval"
            ]
            == "available_cached_enabled",
            "easyocr_cached": health["models"]["ocr"] == "available_cached",
            "fresh_diarization_completed": all(
                item.get("diarization_status") == "completed" for item in metadata
            ),
            "fresh_tracking_completed": all(
                item.get("temporal_tracking_status") == "completed"
                for item in metadata
            ),
            "fresh_smoothing_completed": all(
                item.get("relationship_smoothing_status") == "completed"
                for item in metadata
            ),
            "fresh_embeddings_are_384d": all(
                len(segment.embedding or []) == 384 for segment in final_segments
            ),
            "no_stale_learned_relationships": stale_untracked_count == 0,
            "temporally_smoothed_search_evidence": any(
                item["temporal_smoothed"] and item["observation_count"] > 1
                for item in relation_matches
            ),
            "sqlite_neo4j_tuple_parity": (
                sqlite_relation["results"][0]["start_time"]
                == neo4j_relation["results"][0]["start_time"]
            ),
            "semantic_query_ranked_deployment_moment": semantic["results"][0][
                "start_time"
            ]
            == 10.0,
            "ocr_query_returned_easyocr": any(
                evidence["source_method"] == "easyocr"
                for evidence in ocr["results"][0]["ocr_evidence"]
            ),
            "neo4j_contains_smoothed_relationships": neo4j_counts[
                "smoothed_relationships"
            ]
            > 0,
            "neo4j_contains_ocr": ocr_neo4j["ocr_nodes"] > 0,
        },
    }
    verified = cast(dict[str, bool], proof["verified"])
    failures = [name for name, value in verified.items() if not value]
    if failures:
        raise RuntimeError(f"full-pipeline proof failed: {failures}")

    output = Path("evaluation/results/full-fresh-video-e2e-proof.json")
    output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "verified": proof["verified"],
                "fresh_processing": proof["fresh_processing"],
                "neo4j": proof["neo4j"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
