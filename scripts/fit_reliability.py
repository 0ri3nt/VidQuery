"""Fit learned reliability priors from calibration observations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from vidquery.reliability import canonical_source, fit_reliability_priors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-observations", type=int, default=10)
    parser.add_argument("--strength", type=int, default=5)
    args = parser.parse_args()

    payload = json.loads(args.observations.read_text(encoding="utf-8"))
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise SystemExit("calibration file must contain an 'observations' list")
    priors = fit_reliability_priors(
        observations,
        minimum_observations=args.minimum_observations,
        strength=args.strength,
    )
    # Report counts under the same canonical keys the fitter uses, and only for
    # contributions with non-zero evidence (the ones that influence the fit).
    counts = Counter(
        canonical_source(str(item["source_method"]))
        for item in observations
        if float(item.get("raw_score", 1.0)) > 0
    )
    result = {
        "schema_version": "reliability-priors-1.0",
        "priors": priors,
        "observations_per_source": dict(sorted(counts.items())),
        "minimum_observations": args.minimum_observations,
        "strength": args.strength,
        "kept_hand_picked": sorted(
            source for source in priors if counts.get(source, 0) < args.minimum_observations
        ),
        "manifest": payload.get("manifest"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "split": payload.get("split", "calibration"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote learned priors to {args.output}")


if __name__ == "__main__":
    main()
