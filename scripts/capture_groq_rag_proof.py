from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_request(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
        return json.loads(response.read())


def capture(base_url: str, query: str, video_name: str | None) -> dict[str, Any]:
    health = _json_request(f"{base_url}/api/health")
    video_ids: list[str] = []
    if video_name:
        videos = _json_request(f"{base_url}/api/videos")
        selected = next(
            (item for item in videos if item["display_name"].lower() == video_name.lower()),
            None,
        )
        if selected is None:
            raise RuntimeError(f"video not found: {video_name}")
        video_ids = [selected["video_id"]]

    response = _json_request(
        f"{base_url}/api/search",
        {"query": query, "video_ids": video_ids, "limit": 10},
    )
    answer = response.get("generated_answer")
    if not answer or not answer.get("supported"):
        raise RuntimeError(f"Groq did not return a supported answer: {response.get('rag_status')}")

    result_by_segment = {item["segment_id"]: item for item in response["results"]}
    citation_ids = [item["evidence_id"] for item in answer["citations"]]
    all_citations_exact = all(
        citation["segment_id"] in result_by_segment
        and citation["start_time"]
        == result_by_segment[citation["segment_id"]]["start_time"]
        and citation["end_time"]
        == result_by_segment[citation["segment_id"]]["end_time"]
        for citation in answer["citations"]
    )
    if answer["evidence_ids"] != citation_ids or not all_citations_exact:
        raise RuntimeError("answer citation validation failed")
    cited_results = [
        result_by_segment[item["segment_id"]] for item in answer["citations"]
    ]
    relationship_sources = sorted(
        {
            relation["source_method"]
            for result in cited_results
            for relation in result["relationships"]
        }
    )
    relationship_models = sorted(
        {
            relation["model_version"]
            for result in cited_results
            for relation in result["relationships"]
            if relation.get("model_version")
        }
    )
    action_provenance = sorted(
        {
            action["provenance"]
            for result in cited_results
            for action in result["action_evidence"]
        }
    )

    return {
        "schema_version": "vidquery-query-planner-rag-proof-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "query": query,
        "video_name": video_name,
        "retrieval_backend": response["retrieval_backend"],
        "ranked_result_count": len(response["results"]),
        "rag_health": health.get("models", {}).get("rag_generation"),
        "rag_status": response["rag_status"],
        "query_planner_health": health.get("models", {}).get("query_planner"),
        "query_planner_status": response.get("query_planner_status"),
        "query_ambiguities": response.get("query_ambiguities", []),
        "parsed_query": response.get("parsed_query"),
        "alternative_query_plans": response.get("alternative_query_plans", []),
        "answer": answer,
        "cited_graph_evidence": {
            "action_provenance": action_provenance,
            "relationship_sources": relationship_sources,
            "relationship_model_versions": relationship_models,
        },
        "verified": {
            "raw_ranked_results_preserved": bool(response["results"]),
            "all_evidence_ids_resolved": answer["evidence_ids"] == citation_ids,
            "all_citation_intervals_match_ranked_results": all_citations_exact,
            "planner_produced_validated_structured_plan": bool(
                response.get("query_planner_status")
            ),
            "cited_result_contains_gnn_or_graph_evidence": bool(
                "gnn_action_model" in action_provenance
                or "gnn_action_plus_target_resolver" in action_provenance
                or "relationship_gnn" in relationship_sources
            ),
            "no_api_key_in_artifact": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one live Groq grounded-RAG proof")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default="find architecture")
    parser.add_argument("--video-name", default="final_demo.mp4")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/groq-rag-live-proof.json"),
    )
    args = parser.parse_args()
    proof = capture(args.base_url.rstrip("/"), args.query, args.video_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rag_status": proof["rag_status"],
                "supported": proof["answer"]["supported"],
                "evidence_ids": proof["answer"]["evidence_ids"],
            },
            indent=2,
        )
    )
    print(f"Wrote sanitized proof to {args.output}")


if __name__ == "__main__":
    main()
