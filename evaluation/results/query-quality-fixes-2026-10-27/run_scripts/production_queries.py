"""Run the documented final_demo queries on two copies of the production library and compare.

Usage: production_queries.py QUERIES.json OUT.json NAME=DB [NAME=DB ...]
Production settings from .env; the query planner and answer generation are off so
results are deterministic and no Groq quota is used.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

from vidquery.config import get_settings
from vidquery.domain import SearchRequest
from vidquery.query_planner import QueryPlanningService
from vidquery.search import configured_search_engine
from vidquery.storage import SQLiteRepository


def run(database: Path, queries: list[dict]) -> list[dict]:
    settings = replace(get_settings(), database_path=database, enable_query_planner=False,
                       enable_rag_generation=False)
    repository = SQLiteRepository(database)
    engine = configured_search_engine(repository, settings)
    planner = QueryPlanningService(settings)
    videos = {video.display_name: video.video_id for video in repository.list_videos()}
    rows = []
    for item in queries:
        planning = planner.plan(item["query"])
        request = SearchRequest(query=item["query"], video_ids=[videos[item["video"]]], limit=5,
                                ranking="reliability", localize=True,
                                collapse_duplicate_evidence=True)
        response = engine.search(request, plan_override=planning.plan)
        top = response.results[0] if response.results else None
        hit = bool(top and item["truth"] and any(
            start - 0.01 <= top.start_time and top.end_time <= end + 0.01
            for start, end in item["truth"]))
        rows.append({"query": item["query"], "kind": item["kind"], "truth": item["truth"],
                     "interpretation": response.interpretation,
                     "top": [top.start_time, top.end_time] if top else None,
                     "verdict": top.evidence.verdict if top and top.evidence else None,
                     "confidence": top.evidence.confidence if top and top.evidence else None,
                     "transcript": top.transcript if top else None, "top1_correct": hit})
    return rows


def main() -> None:
    queries = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    runs = {name: run(Path(db), queries) for name, db in (a.split("=", 1) for a in sys.argv[3:])}
    names = list(runs)
    for index, item in enumerate(queries):
        cells = " | ".join(
            f"{n}: {r[index]['top']} {r[index]['verdict']} {r[index]['confidence']} "
            f"{'OK' if r[index]['top1_correct'] else ('-' if not item['truth'] else 'MISS')}"
            for n, r in runs.items())
        print(f"{item['query'][:58]:58} {cells}")
    summary = {n: {"positives_top1_correct": sum(r["top1_correct"] for r in rows if r["kind"] == "positive"),
                   "positives": sum(r["kind"] == "positive" for r in rows),
                   "negatives_rejected": sum(r["interpretation"] == "insufficient_evidence"
                                             for r in rows if r["kind"] != "positive"),
                   "negatives": sum(r["kind"] != "positive" for r in rows)} for n, rows in runs.items()}
    identical = all(runs[names[0]][i]["top"] == runs[n][i]["top"] for n in names[1:]
                    for i in range(len(queries)))
    print(json.dumps(summary), "top-1 windows identical:", identical)
    Path(sys.argv[2]).write_text(json.dumps({"summary": summary, "identical_top1": identical,
                                             "runs": runs}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
