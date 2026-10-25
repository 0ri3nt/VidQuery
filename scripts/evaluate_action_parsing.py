"""Measure deterministic parser accuracy on AVA action labels and false positives.

Development sets only: the calibration split of an AVA benchmark manifest, the
80 canonical label strings, and teammate-written manifests whose queries carry
an ``expected_action`` field (``null`` means no action should be parsed).
The held-out ``test`` split is skipped unless ``--include-test`` is given, so
parser rules are never developed against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vidquery.ava_labels import AVA_V22_ACTIONS
from vidquery.search import StructuredQueryParser

FILLERS = {"something", "someone", "somebody", "anything", "object", "objects", "thing"}


def _ava_label(item: dict[str, Any]) -> str | None:
    identifier = str(item.get("id", ""))
    if identifier.startswith("ava-") and identifier[-3:-2] in {"a", "n"}:
        return AVA_V22_ACTIONS[int(identifier[-2:])]
    return None


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    parser = StructuredQueryParser()
    action_cases = [case for case in cases if case["expected"] is not None]
    other_cases = [case for case in cases if case["expected"] is None]
    rows = []
    for case in cases:
        plan = parser.parse(case["query"])
        rows.append(
            {
                **case,
                "actions": plan.actions,
                "intent": plan.intent,
                "spoken_terms": plan.spoken_terms,
                "correct": case["expected"] in plan.actions if case["expected"] else None,
                "exact": plan.actions == [case["expected"]] if case["expected"] else None,
                "false_positive": bool(plan.actions) if case["expected"] is None else None,
                "filler_spoken": sorted(set(plan.spoken_terms) & FILLERS),
            }
        )
    by_id = {row["query"]: row for row in rows}
    action_rows = [by_id[case["query"]] for case in action_cases]
    other_rows = [by_id[case["query"]] for case in other_cases]
    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "action_query_count": len(action_rows),
        "label_recall": rate([bool(row["correct"]) for row in action_rows]),
        "exact_single_label": rate([bool(row["exact"]) for row in action_rows]),
        "action_intent_rate": rate(
            [row["intent"] in {"action_search", "multimodal_search"} and bool(row["actions"])
             for row in action_rows]
        ),
        "filler_spoken_term_rate": rate([bool(row["filler_spoken"]) for row in action_rows]),
        "non_action_query_count": len(other_rows),
        "false_positive_rate": rate([bool(row["false_positive"]) for row in other_rows]),
        "failures": [
            {key: row[key] for key in ("set", "query", "expected", "actions", "spoken_terms")}
            for row in rows
            if row["correct"] is False or row["false_positive"]
        ],
    }


def load_cases(paths: list[Path], include_test: bool) -> dict[str, list[dict[str, Any]]]:
    sets: dict[str, list[dict[str, Any]]] = {
        "canonical_labels": [
            {"set": "canonical_labels", "query": f"person {label}", "expected": label}
            for label in AVA_V22_ACTIONS.values()
        ]
    }
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        videos = payload.get("videos") if isinstance(payload.get("videos"), dict) else {}
        cases = []
        seen: set[str] = set()
        for item in payload.get("queries", []):
            split = (videos.get(item.get("source_video_id")) or {}).get("split")
            if split == "test" and not include_test:
                continue
            expected = _ava_label(item)
            if expected is None and "expected_action" in item:
                expected = item["expected_action"]
            elif expected is None:
                continue
            if item["query"] in seen:
                continue
            seen.add(item["query"])
            cases.append({"set": path.stem, "query": item["query"], "expected": expected})
        sets[path.stem] = cases
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        type=Path,
        nargs="*",
        default=[
            Path("evaluation/queries_challenge_ava_holdout.json"),
            Path("evaluation/queries_independent.json"),
            Path("evaluation/queries_multivideo.json"),
        ],
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sets = load_cases(args.manifests, args.include_test)
    result = {name: evaluate(cases) for name, cases in sets.items()}
    summary = {
        name: {key: value for key, value in values.items() if key != "failures"}
        for name, values in result.items()
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
