from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vidquery.config import get_settings  # noqa: E402
from vidquery.query_planner import QueryPlanningService  # noqa: E402


def _observed(outcome: Any) -> dict[str, Any]:
    plan = outcome.plan
    return {
        "intent": plan.intent,
        "actions": plan.actions,
        "visual_entities": plan.visual_entities,
        "spoken_terms": plan.spoken_terms,
        "ocr_terms": plan.ocr_terms,
        "tuple_predicate": (
            plan.relationship_tuples[0].predicate if plan.relationship_tuples else None
        ),
        "alternative_spoken_terms": (
            outcome.alternatives[0].spoken_terms if outcome.alternatives else []
        ),
        "ambiguous": bool(outcome.ambiguities),
        "status": outcome.status,
    }


def evaluate(dataset: Path) -> dict[str, Any]:
    source = json.loads(dataset.read_text(encoding="utf-8"))
    settings = replace(get_settings(), enable_query_planner=False)
    planner = QueryPlanningService(settings)
    rows = []
    for case in source["cases"]:
        observed = _observed(planner.plan(case["query"]))
        mismatches = {
            key: {"expected": expected, "observed": observed.get(key)}
            for key, expected in case["expected"].items()
            if observed.get(key) != expected
        }
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "passed": not mismatches,
                "expected": case["expected"],
                "observed": observed,
                "mismatches": mismatches,
            }
        )
    return {
        "schema_version": source["schema_version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "provider_calls": 0,
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic query planning")
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/query_planner_cases.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/query-planner-final.json"),
    )
    args = parser.parse_args()
    report = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"]}, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
