"""Fit learned reliability priors and, optionally, abstention thresholds.

Priors come from an observation file (``scripts/calibrate_reliability.py
--output``).  Thresholds come from a decision file (``--decisions-output``)
collected under the priors they will be used with.  Without an observation
file the hand-picked priors are kept, so "default priors + fitted thresholds"
is expressible.  Fitting on the test split is refused.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from vidquery.reliability import (
    SOURCE_RELIABILITY,
    canonical_source,
    fit_abstention_thresholds,
    fit_reliability_priors,
)


def _load(path: Path, key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get(key), list):
        raise SystemExit(f"{path} must contain a '{key}' list")
    if payload.get("split") == "test":
        raise SystemExit(f"{path} was collected on the test split; refusing to fit on it")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, nargs="?", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-observations", type=int, default=10)
    parser.add_argument("--strength", type=int, default=5)
    parser.add_argument("--decisions", type=Path, default=None)
    parser.add_argument("--supported-precision", type=float, default=0.8)
    parser.add_argument("--wrong-answer-cost", type=float, default=1.0)
    args = parser.parse_args()
    if args.observations is None and args.decisions is None:
        raise SystemExit("give an observation file and/or --decisions")
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")

    result: dict = {"schema_version": "reliability-priors-1.1"}
    if args.observations is not None:
        payload = _load(args.observations, "observations")
        observations = payload["observations"]
        priors = fit_reliability_priors(
            observations,
            minimum_observations=args.minimum_observations,
            strength=args.strength,
        )
        # Counts use the fitter's canonical keys and only non-zero evidence.
        counts = Counter(
            canonical_source(str(item["source_method"]))
            for item in observations
            if float(item.get("raw_score", 1.0)) > 0
        )
        result.update(
            {
                "priors": priors,
                "priors_source": "fitted",
                "observation_limit": payload.get("limit"),
                "observations_per_source": dict(sorted(counts.items())),
                "minimum_observations": args.minimum_observations,
                "strength": args.strength,
                "kept_hand_picked": sorted(
                    source
                    for source in priors
                    if counts.get(source, 0) < args.minimum_observations
                ),
                "manifest": payload.get("manifest"),
                "manifest_sha256": payload.get("manifest_sha256"),
                "split": payload.get("split", "calibration"),
            }
        )
    else:
        result.update({"priors": dict(SOURCE_RELIABILITY), "priors_source": "hand_picked"})
    if args.decisions is not None:
        decisions = _load(args.decisions, "decisions")
        result["thresholds"] = fit_abstention_thresholds(
            decisions["decisions"],
            supported_precision=args.supported_precision,
            wrong_answer_cost=args.wrong_answer_cost,
        )
        result["thresholds"]["decisions_file"] = args.decisions.as_posix()
        result["thresholds"]["decisions_priors"] = decisions.get("priors")
        result.setdefault("split", decisions.get("split", "calibration"))
        result.setdefault("manifest_sha256", decisions.get("manifest_sha256"))
    result["created_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reliability calibration to {args.output}")


if __name__ == "__main__":
    main()
