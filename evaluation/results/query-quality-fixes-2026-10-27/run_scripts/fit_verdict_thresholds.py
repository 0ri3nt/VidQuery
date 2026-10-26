"""Fit verdict thresholds on a VALIDATION result, as pre-registered in ../preregistration.md.

weak:      scripts.fit_moment_abstention p1_preserving (max P@1 loss 0.01) on max_confidence.
supported: lowest top-1 confidence at which answered validation queries reach precision
           >= 0.5 (positive with top-1 IoU >= 0.5 counts correct, any negative wrong),
           with at least 20 answered; 1.0 if none does.

Usage: fit_verdict_thresholds.py VALIDATION_RESULT.json OUT_PRIORS.json
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.fit_moment_abstention import curve
from vidquery.reliability import SOURCE_RELIABILITY

KEY = "review2_full|expand:10"
MAX_P1_LOSS = 0.01
SUPPORTED_PRECISION = 0.5
MIN_ANSWERED = 20


def main(result_path: Path, out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["run_metadata"]["split"] != "validation":
        raise SystemExit("thresholds may only be fitted on the validation split")
    rows = result["rows"][KEY]
    values = sorted({round(r["max_confidence"], 4) for r in rows if r.get("max_confidence") is not None})
    points = curve(rows, [0.0, *values, values[-1] + 1e-4])
    ceiling = points[0]["p_at_1"]
    eligible = [p for p in points if p["p_at_1"] >= ceiling - MAX_P1_LOSS]
    weak_point = max(eligible, key=lambda p: (p["negative_rejection"], -p["threshold"]))
    weak = weak_point["threshold"]

    top1 = sorted({round(r["confidence"], 4) for r in rows if r.get("confidence") is not None})
    supported, supported_point = 1.0, None
    for threshold in top1:
        answered = [r for r in rows if r.get("confidence") is not None and r["confidence"] >= threshold]
        correct = sum(1 for r in answered if not r["negative"]
                      and r["metrics"]["ranking_precision_at_1"])
        if len(answered) >= MIN_ANSWERED and correct / len(answered) >= SUPPORTED_PRECISION:
            supported = threshold
            supported_point = {"threshold": threshold, "answered": len(answered),
                               "precision": round(correct / len(answered), 4)}
            break
    supported = max(supported, weak)
    payload = {
        "schema_version": "reliability-priors-1.1",
        "priors": dict(SOURCE_RELIABILITY),
        "priors_source": "hand_picked",
        "thresholds": {
            "evidence_weak_threshold": weak,
            "evidence_supported_threshold": supported,
            "objective": f"weak: p1_preserving(max_p1_loss={MAX_P1_LOSS}); supported: lowest "
                         f"top-1 confidence with precision >= {SUPPORTED_PRECISION} "
                         f"(>= {MIN_ANSWERED} answered)",
            "weak_validation_point": weak_point,
            "supported_validation_point": supported_point,
            "fitted_on": result_path.as_posix(),
            "key": KEY,
            "requires": "UNMATCHED_CONSTRAINT_EVIDENCE=true",
        },
        "split": "validation",
        "curve": points,
        "created_at": datetime.now(UTC).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"weak": weak_point, "supported": supported_point, "supported_threshold": supported}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
