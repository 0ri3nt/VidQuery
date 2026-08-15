from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def _check(report: dict[str, Any], name: str, condition: bool, detail: Any) -> None:
    report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})


def _search(client: httpx.Client, query: str, **overrides: Any) -> httpx.Response:
    payload: dict[str, Any] = {"query": query, "limit": 10}
    payload.update(overrides)
    return client.post("/api/search", json=payload)


def run_http_qa(base_url: str, video_id: str, demo_video: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "run_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "video_id": video_id,
        "checks": [],
    }
    with httpx.Client(base_url=base_url, timeout=20, follow_redirects=True) as client:
        root = client.get("/")
        _check(
            report,
            "packaged_frontend",
            root.status_code == 200 and "VidQuery" in root.text,
            {"status": root.status_code, "bytes": len(root.content)},
        )
        refresh = client.get("/")
        _check(
            report,
            "page_refresh",
            refresh.status_code == 200 and refresh.content == root.content,
            {"status": refresh.status_code},
        )
        for asset in ("/assets/app.css", "/assets/app.js"):
            response = client.get(asset)
            _check(
                report,
                f"static_delivery:{asset}",
                response.status_code == 200 and len(response.content) > 100,
                {"status": response.status_code, "bytes": len(response.content)},
            )
        javascript = client.get("/assets/app.js")
        _check(
            report,
            "video_filter_control",
            'id="video-filter"' in root.text
            and "video_ids: selectedVideo ? [selectedVideo] : []" in javascript.text,
            {"selector_in_html": 'id="video-filter"' in root.text},
        )

        health = client.get("/api/health")
        health_payload = health.json()
        _check(
            report,
            "health",
            health.status_code == 200
            and health_payload.get("database") == "available"
            and health_payload.get("neo4j") == "available",
            health_payload,
        )
        videos = client.get("/api/videos")
        video_payload = videos.json()
        selected = next((item for item in video_payload if item["video_id"] == video_id), None)
        _check(
            report,
            "library_and_processing_status",
            videos.status_code == 200 and selected is not None and selected["state"] == "READY",
            selected,
        )
        _check(
            report,
            "multiple_video_library",
            len(video_payload) >= 2,
            {"video_count": len(video_payload)},
        )
        invalid_upload = client.post(
            "/api/videos", files={"file": ("invalid.txt", b"not an mp4", "text/plain")}
        )
        _check(
            report,
            "upload_validation_error",
            invalid_upload.status_code == 415,
            invalid_upload.text,
        )
        if demo_video is not None:
            with demo_video.open("rb") as stream:
                duplicate_upload = client.post(
                    "/api/videos",
                    files={"file": (demo_video.name, stream, "video/mp4")},
                )
            duplicate_payload = duplicate_upload.json()
            _check(
                report,
                "upload_deduplication",
                duplicate_upload.status_code == 202
                and duplicate_payload.get("duplicate") is True
                and duplicate_payload.get("video_id") == video_id,
                duplicate_payload,
            )

        relation = _search(client, "person near laptop", video_ids=[video_id])
        relation_payload = relation.json()
        relation_results = relation_payload.get("results", [])
        first_relation = relation_results[0] if relation_results else {}
        matched = first_relation.get("matched_relationships", [])
        exact_tuple = matched[0] if matched else {}
        _check(
            report,
            "exact_relationship_display_and_reason",
            relation.status_code == 200
            and exact_tuple.get("source_class") == "person"
            and exact_tuple.get("predicate") == "near"
            and exact_tuple.get("target_class") == "laptop"
            and exact_tuple.get("source_method") == "manual_annotation"
            and "person[" in first_relation.get("match_reason", ""),
            {"start_time": first_relation.get("start_time"), "matched_tuple": exact_tuple},
        )
        thumbnail = client.get(first_relation.get("thumbnail_url", "/missing"))
        _check(
            report,
            "result_thumbnail",
            thumbnail.status_code == 200
            and thumbnail.headers.get("content-type", "").startswith("image/jpeg"),
            {"status": thumbnail.status_code, "bytes": len(thumbnail.content)},
        )

        architecture = _search(client, "architecture", video_ids=[video_id]).json()
        architecture_starts = [item["start_time"] for item in architecture.get("results", [])]
        _check(
            report,
            "transcript_search",
            bool(architecture_starts) and architecture_starts[0] in {0.0, 15.0},
            architecture_starts,
        )
        speaker = _search(client, "SPEAKER_01 architecture", video_ids=[video_id]).json()
        speaker_starts = [item["start_time"] for item in speaker.get("results", [])]
        _check(report, "speaker_search", speaker_starts[:1] == [15.0], speaker_starts)
        multimodal = _search(client, "deployment person near laptop", video_ids=[video_id]).json()
        multimodal_starts = [item["start_time"] for item in multimodal.get("results", [])]
        _check(
            report,
            "multimodal_search",
            multimodal_starts[:1] == [10.0],
            multimodal_starts,
        )

        graph = _search(
            client,
            "person near laptop",
            video_ids=[video_id],
            retrieval_backend="neo4j",
        )
        graph_payload = graph.json()
        graph_results = graph_payload.get("results", [])
        _check(
            report,
            "neo4j_exact_graph_search",
            graph.status_code == 200
            and bool(graph_results)
            and bool(graph_results[0].get("matched_relationships")),
            {
                "status": graph.status_code,
                "start_times": [item["start_time"] for item in graph_results],
            },
        )

        no_results = _search(client, "zzzxylophone quantum penguin", video_ids=[video_id]).json()
        _check(report, "no_result_state", no_results.get("results") == [], no_results)
        repeated = _search(client, "person near laptop", video_ids=[video_id]).json()
        first_ids = [item["segment_id"] for item in relation_results]
        repeated_ids = [item["segment_id"] for item in repeated.get("results", [])]
        _check(report, "repeated_query", first_ids == repeated_ids, repeated_ids)
        filtered = _search(client, "person", video_ids=[video_id]).json()
        filtered_ids = {item["video_id"] for item in filtered.get("results", [])}
        _check(report, "video_filter", filtered_ids == {video_id}, sorted(filtered_ids))
        missing_filter = _search(client, "person", video_ids=["missing-video-id"]).json()
        _check(
            report,
            "missing_video_filter",
            missing_filter.get("results") == [],
            missing_filter,
        )

        stream_response = client.get(
            f"/api/videos/{video_id}/stream", headers={"Range": "bytes=0-127"}
        )
        _check(
            report,
            "range_video_playback",
            stream_response.status_code == 206
            and len(stream_response.content) == 128
            and stream_response.headers.get("accept-ranges") == "bytes",
            {
                "status": stream_response.status_code,
                "content_range": stream_response.headers.get("content-range"),
                "bytes": len(stream_response.content),
            },
        )
        bad_range = client.get(
            f"/api/videos/{video_id}/stream", headers={"Range": "bytes=999999999-"}
        )
        _check(report, "invalid_range_error", bad_range.status_code == 416, bad_range.text)
        missing_video = client.get("/api/videos/missing-video-id")
        _check(report, "api_not_found", missing_video.status_code == 404, missing_video.text)
        invalid_search = client.post("/api/search", json={"query": "", "limit": 10})
        _check(report, "validation_error", invalid_search.status_code == 422, invalid_search.text)

    report["passed"] = sum(item["passed"] for item in report["checks"])
    report["failed"] = len(report["checks"]) - report["passed"]
    report["all_passed"] = report["failed"] == 0
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible HTTP QA against VidQuery")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/http-ui-qa.json"))
    args = parser.parse_args()
    report = run_http_qa(args.base_url, args.video_id, args.demo_video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}, indent=2))
    print(f"Wrote QA evidence to {args.output}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
