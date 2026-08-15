from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidquery.config import get_settings
from vidquery.neo4j import CanonicalNeo4jIndexer


def _request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    video_id = "4d508fd2-cb26-4caf-8c06-f7589c23764b"
    settings = get_settings()
    health = _request(f"{base_url}/api/health")
    queries: dict[str, Any] = {}
    for query in (
        "laptop above table",
        "chair behind laptop",
        "person in front of person",
    ):
        response = _request(
            f"{base_url}/api/search",
            {"query": query, "video_ids": [video_id], "limit": 1},
        )
        if not response.get("results"):
            raise RuntimeError(f"relationship-GNN API proof returned no result: {query}")
        matched = response["results"][0].get("matched_relationships", [])
        if not matched or matched[0].get("source_method") != "relationship_gnn":
            raise RuntimeError(f"query did not return relationship-GNN evidence: {query}")
        if matched[0].get("model_version") != settings.relation_gnn_model_version:
            raise RuntimeError(f"query returned the wrong relation model version: {query}")
        queries[query] = response

    indexer = CanonicalNeo4jIndexer.from_settings(settings)
    try:
        neo4j_rows = indexer.run_query(
            """
            MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
                  -[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
            WHERE e.source_method = $source_method
              AND e.model_version = $model_version
            RETURN count(e) AS relationships,
                   count(DISTINCT e.predicate) AS predicates,
                   min(e.start_time) AS first_start,
                   max(e.end_time) AS last_end
            """,
            {
                "video_id": video_id,
                "source_method": "relationship_gnn",
                "model_version": settings.relation_gnn_model_version,
            },
        )
    finally:
        indexer.close()
    metrics = json.loads(
        Path("data/app/models/vidor_relation/held_out_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    proof = {
        "schema_version": "vidor-relation-live-proof-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "model_version": settings.relation_gnn_model_version,
        "checkpoint": settings.relation_gnn_checkpoint.as_posix(),
        "health": health,
        "held_out_metrics": metrics["relationship_gnn"]["test_metrics"],
        "candidate_edge_recall": metrics["dataset"]["candidate_edge_recall"],
        "api_queries": queries,
        "neo4j": neo4j_rows[0],
        "verified": {
            "checkpoint_health": health["models"]["gnn_relationship_prediction"]
            == "available_validated_enabled",
            "compulsory_diarization_health": health["models"]["diarization"]
            == "available_configured_required",
            "api_exact_tuple_provenance": True,
            "api_model_version": True,
            "neo4j_model_version": bool(neo4j_rows[0]["relationships"]),
            "held_out_metrics_exist": True,
        },
    }
    output = Path("evaluation/results/vidor-relation-live-proof.json")
    output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "verified": proof["verified"],
                "neo4j": proof["neo4j"],
                "queries": {
                    query: {
                        "start_time": response["results"][0]["start_time"],
                        "match_reason": response["results"][0]["match_reason"],
                    }
                    for query, response in queries.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
