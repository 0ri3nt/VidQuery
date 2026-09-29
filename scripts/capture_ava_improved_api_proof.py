from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vidquery.config import get_settings
from vidquery.storage import SQLiteRepository


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Capture an unseen-video ROI-GAT API proof")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--video-id", default="aa03a76b-b03b-4f97-8542-d89736ab29e7"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/ava80-scale30-api-unseen-proof.json"),
    )
    args = parser.parse_args()
    settings = get_settings()
    split_path = Path("data/app/models/ava80_scaling/30/split_manifest.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_video_id = "hbYvDvJrpNk"
    if source_video_id in split["train"]:
        raise RuntimeError("API proof video unexpectedly appears in the training split")
    stored_segments = SQLiteRepository(settings.database_path).list_segments([args.video_id])
    stored_versions = sorted(
        {
            str(segment.processing_metadata.get("gnn_model_version"))
            for segment in stored_segments
        }
    )
    if stored_versions != [settings.gnn_model_version]:
        raise RuntimeError(
            "stored API proof segments do not all use the configured checkpoint: "
            f"{stored_versions}"
        )
    health = _json_request(f"{args.base_url}/api/health")
    video = _json_request(f"{args.base_url}/api/videos/{args.video_id}")
    queries = {}
    for query in ("write", "talk to"):
        response = _json_request(
            f"{args.base_url}/api/search",
            {"query": query, "video_ids": [args.video_id], "limit": 1},
        )
        results = response.get("results", [])
        if not results:
            raise RuntimeError(f"unseen-video API proof query returned no result: {query}")
        evidence = results[0].get("action_evidence", [])
        if not any(
            item.get("provenance") in {
                "gnn_action_model", "gnn_action_plus_target_resolver"
            }
            for item in evidence
        ):
            raise RuntimeError(f"query has no GNN provenance in API evidence: {query}")
        queries[query] = response
    if "gnn_action_model" not in queries["write"]["results"][0]["match_reason"]:
        raise RuntimeError("GNN-only action query did not expose GNN provenance in match_reason")
    proof = {
        "schema_version": "ava80-scale30-unseen-api-proof-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "model_version": settings.gnn_model_version,
        "training_video_ids": split["train"],
        "validation_video_ids": split["validation"],
        "test_video_ids": split["test"],
        "unseen_video_source_id": source_video_id,
        "unseen_video_id": args.video_id,
        "unseen_video_was_in_training": source_video_id in split["train"],
        "unseen_video_was_in_test": source_video_id in split["test"],
        "stored_segment_count": len(stored_segments),
        "stored_gnn_model_versions": stored_versions,
        "health": health,
        "video": video,
        "queries": queries,
        "verified": {
            "checkpoint_health": health.get("models", {}).get(
                "gnn_person_action_classifier"
            ) == "available_validated_enabled",
            "compulsory_diarization_health": health.get("models", {}).get(
                "diarization"
            ) == "available_configured_required",
            "held_out_from_training": source_video_id not in split["train"],
            "stored_checkpoint_version": stored_versions
            == [settings.gnn_model_version],
            "sqlite_search": all(
                response.get("retrieval_backend") == "sqlite"
                for response in queries.values()
            ),
            "gnn_provenance": True,
            "gnn_only_action_match_reason": "gnn_action_model"
            in queries["write"]["results"][0]["match_reason"],
            "target_resolver_provenance": any(
                item.get("provenance") == "gnn_action_plus_target_resolver"
                for response in queries.values()
                for item in response["results"][0].get("action_evidence", [])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "verified": proof["verified"],
        "top_results": {
            query: {
                "start_time": response["results"][0]["start_time"],
                "end_time": response["results"][0]["end_time"],
                "match_reason": response["results"][0]["match_reason"],
            }
            for query, response in queries.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
