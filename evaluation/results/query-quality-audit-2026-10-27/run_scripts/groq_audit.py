"""Groq audit: grounded answers and LLM query plans versus the no-Groq fallback.

  collect --queries Q.json --output OUT.json   (run under run_with_profile for one library)
  check   --input OUT.json [...] --output CHECKS.json

`collect` runs every query through the real API app three times: twice with
Groq (each in a fresh app, so the in-process plan cache cannot hide
nondeterminism) and once with the Groq key removed (planner and answer
fallback). Secrets are never written: only answers, plans and evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

PEAK_TOLERANCE = 0.75
COCO = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "bench", "bird", "cat", "dog",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake", "chair", "couch", "sofa", "potted plant", "bed", "dining table",
    "table", "toilet", "tv", "television", "laptop", "mouse", "remote", "keyboard", "phone",
    "cell phone", "microwave", "oven", "stove", "toaster", "sink", "refrigerator", "fridge",
    "book", "clock", "vase", "scissors", "teddy bear", "toothbrush", "whiteboard", "door",
    "closet", "towel", "blanket", "glasses", "newspaper", "shoes", "jacket", "shirt", "window",
]
ACTIONS = [
    "drink", "drinking", "eat", "eating", "cook", "cooking", "sit", "sitting", "stand", "standing",
    "walk", "walking", "open", "opening", "opens", "close", "closing", "hold", "holding", "read",
    "reading", "watch", "watching", "smile", "smiling", "talk", "talking", "discuss", "discusses",
    "discussing", "run", "running", "lie", "lying", "put", "putting", "take", "taking", "carry",
    "carrying", "pick", "picks", "wear", "wearing", "dress", "dressing", "undress", "undressing",
]
TIME_PATTERNS = [
    re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d))?\b"),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", re.I),
]


def collect(args: argparse.Namespace) -> None:
    from fastapi.testclient import TestClient

    from vidquery.api import create_app
    from vidquery.config import get_settings
    from vidquery.storage import SQLiteRepository

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    videos = {v.display_name: v.video_id for v in repository.list_videos()}

    def fresh_client(groq: bool) -> TestClient:
        active = settings if groq else replace(settings, groq_api_key="")
        return TestClient(create_app(active, repository))

    records = []
    for run in ("groq_1", "groq_2", "fallback"):
        client = fresh_client(run != "fallback")
        for item in queries:
            body = {"query": item["query"], "limit": 5, "ranking": "reliability",
                    "localize": True, "collapse_duplicate_evidence": True,
                    "video_ids": [videos[item["video"]]] if item.get("video") else []}
            started = time.perf_counter()
            response = client.post("/api/search", json=body)
            elapsed = round(time.perf_counter() - started, 2)
            payload = response.json()
            results = payload.get("results", [])
            records.append({
                "run": run, "query": item["query"], "video": item.get("video"),
                "kind": item.get("kind"), "truth": item.get("truth"), "http": response.status_code,
                "seconds": elapsed, "planner_status": payload.get("query_planner_status"),
                "plan": payload.get("parsed_query"), "interpretation": payload.get("interpretation"),
                "rag_status": payload.get("rag_status"), "answer": payload.get("generated_answer"),
                "results": [{
                    "rank": index + 1, "segment_id": r["segment_id"], "video_title": r["video_title"],
                    "start": r["start_time"], "end": r["end_time"],
                    "peak": (r.get("localization") or {}).get("peak_time"),
                    "score": r["score"],
                    "verdict": (r.get("evidence") or {}).get("verdict"),
                    "confidence": (r.get("evidence") or {}).get("confidence"),
                    "match_reason": r["match_reason"], "entities": r["entities"],
                    "actions": r["actions"], "transcript": r["transcript"],
                    "speakers": r["speakers"],
                    "relationships": sorted({f"{x['predicate']}" for x in r["relationships"]})[:20],
                    "matched_relationships": [
                        f"{x['source_class']} {x['predicate']} {x['target_class']}"
                        for x in r["matched_relationships"]],
                    "ocr": [x["text"] for x in r.get("ocr_evidence", [])],
                } for index, r in enumerate(results)],
            })
            print(run, records[-1]["planner_status"], records[-1]["rag_status"], item["query"][:60],
                  flush=True)
            if run != "fallback":
                time.sleep(args.pause)
    args.output.write_text(json.dumps(records, indent=1), encoding="utf-8")


def _times(text: str) -> list[float]:
    found = []
    for match in TIME_PATTERNS[0].finditer(text):
        a, b, c, tenth = match.groups()
        seconds = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
        found.append(seconds + (int(tenth) / 10 if tenth else 0))
    for match in TIME_PATTERNS[1].finditer(text):
        found.append(float(match.group(1)))
    return found


def _terms(text: str, vocabulary: list[str]) -> set[str]:
    lowered = text.lower()
    return {word for word in vocabulary if re.search(rf"\b{re.escape(word)}\b", lowered)}


def check_record(record: dict) -> dict:
    answer = record.get("answer") or {}
    text = answer.get("answer", "") if answer else ""
    cited_ids = {c["segment_id"] for c in answer.get("citations", [])} if answer else set()
    cited = [r for r in record["results"] if r["segment_id"] in cited_ids]
    issues = []
    if answer and answer.get("supported"):
        if record["interpretation"] == "insufficient_evidence":
            issues.append("supported answer although the system's interpretation is insufficient_evidence")
        if cited and all(r["verdict"] == "insufficient" for r in cited):
            issues.append("supported answer citing only 'insufficient' results")
        if cited and all(r["verdict"] != "supported" for r in cited):
            issues.append("supported answer citing only weak/insufficient results")
        windows = [(r["start"], r["end"]) for r in cited] + [
            (r["peak"] - PEAK_TOLERANCE, r["peak"] + PEAK_TOLERANCE) for r in cited if r["peak"] is not None]
        bad_times = [t for t in _times(text)
                     if not any(a - PEAK_TOLERANCE <= t <= b + PEAK_TOLERANCE for a, b in windows)]
        if bad_times:
            issues.append(f"time(s) {bad_times} outside every cited segment")
        evidence_text = " ".join(
            " ".join([*r["entities"], *r["actions"], r["transcript"], *r["speakers"],
                      *r["relationships"], *r["matched_relationships"], *r["ocr"]])
            for r in cited).lower()
        unsupported_objects = sorted(t for t in _terms(text, COCO)
                                     if not re.search(rf"\b{re.escape(t)}", evidence_text))
        unsupported_actions = sorted(t for t in _terms(text, ACTIONS)
                                     if not re.search(rf"\b{re.escape(t[:4])}", evidence_text))
        if unsupported_objects:
            issues.append(f"objects not in cited evidence: {unsupported_objects}")
        if unsupported_actions:
            issues.append(f"actions not in cited evidence: {unsupported_actions}")
    return {"query": record["query"], "video": record["video"], "run": record["run"],
            "kind": record["kind"], "rag_status": record["rag_status"],
            "supported": answer.get("supported") if answer else None,
            "answer": text, "cited_verdicts": [r["verdict"] for r in cited],
            "interpretation": record["interpretation"], "issues": issues}


def check(args: argparse.Namespace) -> None:
    records = [r for path in args.input for r in json.loads(Path(path).read_text(encoding="utf-8"))]
    checks = [check_record(r) for r in records]
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in records:
        by_key.setdefault((r["video"], r["query"]), {})[r["run"]] = r

    def top(r):
        return (r["results"][0]["segment_id"], r["results"][0]["verdict"]) if r["results"] else None

    comparisons = []
    for (video, query), runs in by_key.items():
        g1, g2, fb = runs.get("groq_1"), runs.get("groq_2"), runs.get("fallback")
        if not (g1 and g2 and fb):
            continue
        comparisons.append({
            "query": query, "video": video, "kind": g1["kind"],
            "planner_status": [g1["planner_status"], g2["planner_status"], fb["planner_status"]],
            "plan_same_across_groq_runs": g1["plan"] == g2["plan"],
            "plan_same_as_fallback": g1["plan"] == fb["plan"],
            "top_same_across_groq_runs": top(g1) == top(g2),
            "top_same_as_fallback": top(g1) == top(fb),
            "ranked_ids_same_as_fallback": [x["segment_id"] for x in g1["results"]]
            == [x["segment_id"] for x in fb["results"]],
            "answer_same_across_groq_runs": (g1["answer"] or {}).get("answer") == (g2["answer"] or {}).get("answer"),
            "supported_same_across_groq_runs": (g1["answer"] or {}).get("supported")
            == (g2["answer"] or {}).get("supported"),
            "groq_top": g1["results"][0] if g1["results"] else None,
            "fallback_top": fb["results"][0] if fb["results"] else None,
            "truth": g1["truth"], "groq_plan": g1["plan"], "fallback_plan": fb["plan"],
            "fallback_rag_status": fb["rag_status"], "fallback_answer": fb["answer"],
        })
    groq_checks = [c for c in checks if c["run"].startswith("groq")]
    summary = {
        "records": len(records),
        "rag_status": {run: _count(c["rag_status"] for c in checks if c["run"] == run)
                       for run in ("groq_1", "groq_2", "fallback")},
        "planner_status": {run: _count(r["planner_status"] for r in records if r["run"] == run)
                           for run in ("groq_1", "groq_2", "fallback")},
        "groq_answers_with_issues": sum(bool(c["issues"]) for c in groq_checks),
        "groq_supported_answers": sum(bool(c["supported"]) for c in groq_checks),
        "issue_types": _count(re.sub(r"[:(].*", "", issue).strip()
                              for c in groq_checks for issue in c["issues"]),
        "queries_compared": len(comparisons),
        "plan_differs_between_groq_runs": sum(not c["plan_same_across_groq_runs"] for c in comparisons),
        "top_differs_between_groq_runs": sum(not c["top_same_across_groq_runs"] for c in comparisons),
        "answer_text_differs_between_groq_runs": sum(not c["answer_same_across_groq_runs"] for c in comparisons),
        "supported_flag_differs_between_groq_runs": sum(not c["supported_same_across_groq_runs"] for c in comparisons),
        "plan_differs_groq_vs_fallback": sum(not c["plan_same_as_fallback"] for c in comparisons),
        "top_differs_groq_vs_fallback": sum(not c["top_same_as_fallback"] for c in comparisons),
    }
    args.output.write_text(json.dumps({"summary": summary, "checks": checks,
                                       "comparisons": comparisons}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def _count(values) -> dict:
    out: dict = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--queries", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--pause", type=float, default=2.0)
    k = sub.add_parser("check")
    k.add_argument("--input", nargs="+", required=True)
    k.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args) if args.mode == "collect" else check(args)


if __name__ == "__main__":
    main()
