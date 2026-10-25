"""Fit the abstention threshold for the moment benchmark on validation rows only.

Reads an ``evaluate_moment_benchmark.py`` result for the *validation* split,
one ``method|span`` key (with abstention behaviour of ``review2_full``), and
grid-searches the weak threshold on each query's maximum fused confidence:

* ``balanced`` (primary): maximise 0.5 * P@1(positives) + 0.5 * rejection
  rate (negatives). NA-VMR has about two negatives per positive, so a plain
  accuracy objective would favour rejecting everything.
* ``effective_reliability``: +1 per correct answer, -1 per wrong answer
  (any answer to a negative), 0 per abstention.
* ``p1_preserving``: the highest negative rejection whose validation P@1 is
  within ``--max-p1-loss`` of the no-abstention P@1. Added after the balanced
  objective proved degenerate on validation (in-domain negatives are
  inseparable), so report it as a post-hoc, validation-only choice.

Writes a priors file (hand-picked priors + fitted thresholds) usable with
``--reliability-priors``, plus the full trade-off curve.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from vidquery.reliability import SOURCE_RELIABILITY

DEFAULT_SUPPORTED = 0.45


def _answered(row: dict, threshold: float) -> bool:
    value = row.get("max_confidence")
    return value is not None and value >= threshold


def curve(rows: list[dict], thresholds: list[float]) -> list[dict]:
    positives = [row for row in rows if not row["negative"]]
    negatives = [row for row in rows if row["negative"]]
    output = []
    for threshold in thresholds:
        def answered(row: dict, threshold: float = threshold) -> bool:
            return _answered(row, threshold)

        correct = sum(
            answered(row) and row["metrics"]["ranking_precision_at_1"] for row in positives
        )
        rejected = sum(not answered(row) for row in negatives)
        wrong = sum(answered(row) and not row["metrics"]["ranking_precision_at_1"]
                    for row in positives) + sum(answered(row) for row in negatives)
        p1 = correct / len(positives) if positives else 0.0
        rejection = rejected / len(negatives) if negatives else 0.0
        output.append(
            {
                "threshold": round(threshold, 4),
                "p_at_1": round(p1, 4),
                "negative_rejection": round(rejection, 4),
                "balanced": round(0.5 * p1 + 0.5 * rejection, 4),
                "effective_reliability": round((correct - wrong) / len(rows), 4),
                "coverage": round(sum(answered(row) for row in rows) / len(rows), 4),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--key", required=True, help="e.g. review2_full|expand:10")
    parser.add_argument(
        "--objective",
        choices=("balanced", "effective_reliability", "p1_preserving"),
        default="balanced",
    )
    parser.add_argument("--max-p1-loss", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result["run_metadata"]["split"] != "validation":
        raise SystemExit("thresholds may only be fitted on the validation split")
    rows = result["rows"][args.key]
    values = sorted({round(row["max_confidence"], 4) for row in rows
                     if row.get("max_confidence") is not None})
    thresholds = [0.0, *values, (values[-1] + 1e-4) if values else 1.0]
    points = curve(rows, thresholds)
    if args.objective == "p1_preserving":
        ceiling = points[0]["p_at_1"]
        eligible = [point for point in points if point["p_at_1"] >= ceiling - args.max_p1_loss]
        best = max(eligible, key=lambda point: (point["negative_rejection"], -point["threshold"]))
    else:
        best = max(points, key=lambda point: (point[args.objective], -point["threshold"]))
    weak = min(1.0, best["threshold"])
    payload = {
        "schema_version": "reliability-priors-1.1",
        "priors": dict(SOURCE_RELIABILITY),
        "priors_source": "hand_picked",
        "thresholds": {
            "evidence_weak_threshold": weak,
            "evidence_supported_threshold": max(weak, DEFAULT_SUPPORTED),
            "objective": args.objective,
            "validation_point": best,
            "fitted_on": args.result.as_posix(),
            "key": args.key,
        },
        "split": "validation",
        "curve": points,
        "created_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(best))


if __name__ == "__main__":
    main()
