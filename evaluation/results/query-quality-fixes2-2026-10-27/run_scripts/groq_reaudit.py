"""Live Groq re-audit after the fixes, sized for the account's 200k-tokens/day cap.

collect QUERIES.json OUT.json [--subset N] [--pause S]
  run "groq_1":   every query, Groq on, planner responses cached in a scratch file
  run "groq_2":   the first N queries again, Groq on, no cache (planner and answer determinism)
  run "fallback": every query, no key (local planner, no answer; free)
prompt PLANS.json OUT.json
  For the queries whose groq_1 plan came from Groq, ask Groq once more with the previous
  full-alias prompt and compare the validated plans with the reduced prompt's plans.
Token use is read from each Groq response's "usage" field (no headers or keys are kept),
paced to stay under --tpm tokens/minute, and the run stops at --budget total tokens.
Secrets are never written.
"""

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

USAGE: list[dict] = []
SPENT = {"tokens": 0}


def record_usage():
    """Wrap the Groq transport so each call's token usage (or HTTP error) is recorded."""
    import vidquery.rag as rag

    original = rag.UrllibProviderTransport.post_json

    def post_json(self, url, *, headers, payload, timeout):
        kind = "planner" if payload.get("max_completion_tokens") == 700 else "answer"
        try:
            envelope = original(self, url, headers=headers, payload=payload, timeout=timeout)
        except rag.RAGProviderError as exc:
            cause = exc.__cause__
            entry = {
                "kind": kind,
                "error": str(exc)[:80],
                "reason": repr(getattr(cause, "reason", cause))[:120],
            }
            try:  # the rate-limit message says which limit was hit
                message = json.loads(cause.read()).get("error", {}).get("message", "")
                entry["message"] = re.sub(r"org_\w+", "org_[redacted]", message)[:240]
            except Exception:
                pass
            USAGE.append(entry)
            raise
        usage = envelope.get("usage") or {}
        USAGE.append(
            {
                "kind": kind,
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
                "total": usage.get("total_tokens"),
            }
        )
        SPENT["tokens"] += int(usage.get("total_tokens") or 0)
        return envelope

    rag.UrllibProviderTransport.post_json = post_json


def pace(calls, pause, tpm):
    """Sleep long enough that this query's tokens stay under the per-minute limit."""
    tokens = sum(int(c.get("total") or 0) for c in calls)
    time.sleep(max(pause, 60.0 * tokens / tpm))


def run_queries(settings, repository, queries, run, pause, tpm=6000, budget=None):
    from fastapi.testclient import TestClient

    from vidquery.api import create_app

    videos = {v.display_name: v.video_id for v in repository.list_videos()}
    client = TestClient(create_app(settings, repository))
    records = []
    for item in queries:
        if budget is not None and SPENT["tokens"] >= budget:
            print(run, "budget reached; stopping", flush=True)
            break
        body = {
            "query": item["query"],
            "limit": 5,
            "ranking": "reliability",
            "localize": True,
            "collapse_duplicate_evidence": True,
            "video_ids": [videos[item["video"]]] if item.get("video") else [],
        }
        first, limited = len(USAGE), []
        payload = client.post("/api/search", json=body).json()
        if any("HTTP 429" in c.get("error", "") for c in USAGE[first:]):
            # Back off once and retry, so the record measures the answer, not the limit.
            limited = USAGE[first:]
            time.sleep(65)
            first = len(USAGE)
            payload = client.post("/api/search", json=body).json()
            if any("per day" in c.get("message", "") for c in USAGE[first:]):
                print(run, "daily token cap reached; stopping", flush=True)
                break
        records.append(
            {
                "run": run,
                "query": item["query"],
                "video": item.get("video"),
                "kind": item.get("kind"),
                "truth": item.get("truth"),
                "planner_status": payload.get("query_planner_status"),
                "plan": payload.get("parsed_query"),
                "interpretation": payload.get("interpretation"),
                "rag_status": payload.get("rag_status"),
                "answer": payload.get("generated_answer"),
                "groq_calls": USAGE[first:],
                "rate_limited_calls": limited,
                "results": [
                    {
                        "segment_id": r["segment_id"],
                        "video_title": r["video_title"],
                        "start": r["start_time"],
                        "end": r["end_time"],
                        "peak": (r.get("localization") or {}).get("peak_time"),
                        "score": r["score"],
                        "verdict": (r.get("evidence") or {}).get("verdict"),
                        "confidence": (r.get("evidence") or {}).get("confidence"),
                        "entities": r["entities"],
                        "actions": r["actions"],
                        "transcript": r["transcript"],
                        "speakers": r["speakers"],
                        "relationships": sorted({x["predicate"] for x in r["relationships"]})[:20],
                        "matched_relationships": [
                            f"{x['source_class']} {x['predicate']} {x['target_class']}"
                            for x in r["matched_relationships"]
                        ],
                        "ocr": [x["text"] for x in r.get("ocr_evidence", [])],
                    }
                    for r in payload.get("results", [])
                ],
            }
        )
        print(
            run,
            records[-1]["planner_status"],
            records[-1]["rag_status"],
            item["query"][:60],
            flush=True,
        )
        if run != "fallback":
            pace(USAGE[first:], pause, tpm)
    return records


def collect(args):
    from vidquery.config import get_settings
    from vidquery.storage import SQLiteRepository

    record_usage()
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    if args.only:
        queries = [queries[i] for i in args.only]
    base = get_settings()
    repository = SQLiteRepository(base.database_path)
    cache = args.output.with_suffix(".plan-cache.json")
    records = run_queries(
        replace(base, query_plan_cache_path=cache),
        repository,
        queries,
        "groq_1",
        args.pause,
        args.tpm,
        args.budget,
    )
    records += run_queries(
        replace(base, query_plan_cache_path=None),
        repository,
        queries[: args.subset],
        "groq_2",
        args.pause,
        args.tpm,
        args.budget,
    )
    records += run_queries(replace(base, groq_api_key=""), repository, queries, "fallback", 0)
    args.output.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print("tokens spent", SPENT["tokens"], "calls", len(USAGE), flush=True)


def prompt(args):
    import vidquery.query_planner as planner_module
    from vidquery.config import get_settings
    from vidquery.query_planner import QueryPlanningService
    from vidquery.search import ACTION_ALIASES, ENTITY_ALIASES, RELATIONSHIP_ALIASES

    records = [
        r for r in json.loads(args.records.read_text(encoding="utf-8")) if r["run"] == "groq_1"
    ]
    groq_planned = list(
        {r["query"]: r for r in records if str(r["planner_status"]).startswith("groq")}.values()
    )
    full_tables = {
        id(ACTION_ALIASES): ACTION_ALIASES,
        id(ENTITY_ALIASES): ENTITY_ALIASES,
        id(RELATIONSHIP_ALIASES): RELATIONSHIP_ALIASES,
    }
    planner_module._aliases_in = lambda query, aliases: full_tables[id(aliases)]  # old prompt
    record_usage()
    service = QueryPlanningService(replace(get_settings(), query_plan_cache_path=None))
    rows = []
    for record in groq_planned[: args.limit]:
        first = len(USAGE)
        outcome = service.plan(record["query"])
        rows.append(
            {
                "query": record["query"],
                "reduced_status": record["planner_status"],
                "full_status": outcome.status,
                "same_plan": outcome.plan.model_dump(mode="json") == record["plan"],
                "reduced_plan": record["plan"],
                "full_plan": outcome.plan.model_dump(mode="json"),
                "reduced_calls": [c for c in record["groq_calls"] if c["kind"] == "planner"],
                "full_calls": USAGE[first:],
            }
        )
        print(rows[-1]["same_plan"], outcome.status, record["query"][:60], flush=True)
        pace(USAGE[first:], args.pause, args.tpm)
    args.output.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("tokens spent", SPENT["tokens"], "calls", len(USAGE), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("collect")
    c.add_argument("queries", type=Path)
    c.add_argument("output", type=Path)
    c.add_argument("--subset", type=int, default=12)
    c.add_argument("--pause", type=float, default=4.0)
    c.add_argument("--tpm", type=float, default=6000.0)
    c.add_argument("--budget", type=int, default=None, help="stop Groq runs at this many tokens")
    c.add_argument("--only", type=int, nargs="*", help="indices of the queries to run")
    p = sub.add_parser("prompt")
    p.add_argument("records", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--pause", type=float, default=4.0)
    p.add_argument("--tpm", type=float, default=6000.0)
    args = parser.parse_args()
    collect(args) if args.mode == "collect" else prompt(args)


if __name__ == "__main__":
    main()
