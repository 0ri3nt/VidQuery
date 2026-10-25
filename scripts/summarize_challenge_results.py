"""Tabulate evaluate-challenge result files and paired bootstrap differences.

Paired intervals resample the *same* queries for both systems, which is the
right test when every system answers an identical query set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

KEYS = (
    "precision_at_1",
    "ranking_precision_at_1",
    "recall_at_cutoff",
    "mrr",
    "within_tolerance_rate",
    "intent_accuracy_strict",
    "positive_answered_rate",
    "negative_rejection_rate",
    "negative_weak_answer_rate",
    "false_abstention_rate",
    "coverage",
    "selective_risk",
    "aurc",
    "rejection_precision",
    "rejection_recall",
    "rejection_f1",
    "answerability_auroc",
    "mean_latency_ms",
)


def _per_query(result: dict[str, Any], method: str, key: str) -> dict[str, float]:
    return {
        row["id"]: float(row["metrics"][key])
        for row in result["queries"][method]
        if not row["negative"] and row["metrics"].get(key) is not None
    }


def paired_interval(
    left: dict[str, float], right: dict[str, float], *, samples: int = 2000, seed: int = 2026
) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    differences = [right[key] - left[key] for key in shared]
    rng = random.Random(seed)
    means = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "queries": len(shared),
        "mean_difference": round(sum(differences) / len(differences), 4),
        "ci95": [round(means[int(0.025 * samples)], 4), round(means[int(0.975 * samples) - 1], 4)],
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="NAME=PATH pairs or paths")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT[:METHOD[:RIGHT_METHOD]]",
        help="Compare two runs (same method) or two methods of runs on identical queries",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs: dict[str, dict[str, Any]] = {}
    for item in args.results:
        text = str(item)
        name, _, path = text.partition("=") if "=" in text else (Path(text).stem, "", text)
        runs[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, Any] = {}
    for name, result in runs.items():
        metadata = result.get("run_metadata", {})
        table[name] = {
            "query_count": result["query_count"],
            "split": result["split"],
            "dataset": result["dataset"],
            "reliability_priors": metadata.get("reliability_priors"),
            "thresholds": [
                metadata.get("evidence_weak_threshold"),
                metadata.get("evidence_supported_threshold"),
            ],
            "indexed_gnn": metadata.get("indexed_gnn_action_model_versions"),
            "text_encoder": metadata.get("text_encoder"),
            "methods": {
                method: {
                    **{key: values["overall"].get(key) for key in KEYS},
                    "positive_query_count": values["overall"]["positive_query_count"],
                    "negative_query_count": values["overall"]["negative_query_count"],
                    "ci95": values["overall"]["bootstrap_95_ci"],
                }
                for method, values in result["summary"].items()
            },
        }
    pairs = {}
    for spec in args.pair:
        left, right, *rest = spec.split(":")
        method = rest[0] if rest else "review2_full"
        right_method = rest[1] if len(rest) > 1 else method
        pairs[f"{left}[{method}] -> {right}[{right_method}]"] = {
            key: paired_interval(
                _per_query(runs[left], method, key), _per_query(runs[right], right_method, key)
            )
            for key in ("precision_at_1", "ranking_precision_at_1", "mrr")
        }
    output = {"schema_version": "challenge-summary-v1", "runs": table, "paired": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    main()
