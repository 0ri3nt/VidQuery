"""Compare Charades-STA / NA-VMR result files from ``evaluate_moment_benchmark.py``.

All runs must cover the same manifest and query sequence; rows are matched by
position (NA-VMR repeats some query IDs) with the id/kind sequence asserted
equal. Reports per run the localization and answerability metrics, paired
video-level bootstrap differences, and per-video P@1.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

POSITIVE_METRICS = ("precision_at_1", "r1_iou0.5", "r1_iou0.7", "r5_iou0.5")
NEGATIVE_KINDS = ("in_domain_negative", "out_of_domain_negative")


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _score(row: dict[str, Any]) -> float:
    value = row.get("max_confidence")
    return float(value) if value is not None else -1.0


def run_metrics(result: dict[str, Any], key: str) -> dict[str, Any]:
    summary = result["summary"][key]
    rows = result["rows"][key]
    positives = [_score(row) for row in rows if not row["negative"]]
    metrics = {name: summary[name] for name in (*POSITIVE_METRICS, "mean_iou",
                                                "positive_answered_rate", "coverage", "aurc",
                                                "rejection_precision", "rejection_recall",
                                                "rejection_f1", "answerability_auroc")}
    metrics["ci95"] = summary.get("ci95", {})
    for kind in NEGATIVE_KINDS:
        metrics[f"rejected_{kind}"] = summary["negative_rejection_rate_by_kind"].get(kind)
        value = auroc(positives, [_score(row) for row in rows if row["kind"] == kind])
        metrics[f"auroc_vs_{kind}"] = round(value, 4) if value is not None else None
    metrics["failure_stages"] = summary.get("failure_stages")
    return metrics


def _bootstrap(values: dict[str, list[float]], samples: int, seed: int) -> dict[str, Any]:
    videos = sorted(values)
    flat = [value for video in videos for value in values[video]]
    if not flat:
        return {"difference": None, "ci95": None, "ci_excludes_zero": False}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        picked = [value for video in rng.choices(videos, k=len(videos)) for value in values[video]]
        draws.append(sum(picked) / len(picked))
    draws.sort()
    low, high = draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]
    return {"difference": round(sum(flat) / len(flat), 4), "ci95": [round(low, 4), round(high, 4)],
            "ci_excludes_zero": bool(low > 0 or high < 0)}


def paired(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], videos: list[str],
           samples: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in POSITIVE_METRICS:
        by_video: dict[str, list[float]] = defaultdict(list)
        better = worse = 0
        for video, a, b in zip(videos, rows_a, rows_b, strict=True):
            if a["negative"]:
                continue
            delta = float(b["metrics"][metric]) - float(a["metrics"][metric])
            by_video[video].append(delta)
            better += delta > 0
            worse += delta < 0
        output[metric] = {**_bootstrap(by_video, samples, seed),
                          "queries_better": better, "queries_worse": worse}
    for kind in NEGATIVE_KINDS:
        by_video = defaultdict(list)
        for video, a, b in zip(videos, rows_a, rows_b, strict=True):
            if a["kind"] == kind:
                by_video[video].append(float(b["metrics"]["negative_rejected"])
                                       - float(a["metrics"]["negative_rejected"]))
        output[f"rejected_{kind}"] = _bootstrap(by_video, samples, seed)
    return output


def compare(results: dict[str, dict[str, Any]], manifest: dict[str, Any], *, key: str,
            pairs: list[tuple[str, str]], samples: int, seed: int) -> dict[str, Any]:
    split = next(iter(results.values()))["run_metadata"]["split"]
    queries = manifest["splits"][split]["queries"]
    expected = [(query["id"], query["kind"]) for query in queries]
    reference_plans = None
    for name, result in results.items():
        rows = result["rows"][key]
        if [(row["id"], row["kind"]) for row in rows] != expected:
            raise ValueError(f"{name} does not cover the manifest's {split} queries in order")
        if result["run_metadata"]["split"] != split:
            raise ValueError(f"{name} is a different split")
        reference_plans = reference_plans or result["plans"]
        if result["plans"] != reference_plans:
            raise ValueError(f"{name} parsed the queries differently")
    videos = [query["video"] for query in queries]
    counts: dict[str, int] = defaultdict(int)
    for query in queries:
        if query["answerable"]:
            counts[query["video"]] += 1
    table: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        for video, row in zip(videos, result["rows"][key], strict=True):
            if not row["negative"]:
                entry = table.setdefault(video, {"positives": counts[video]})
                entry[name] = entry.get(name, 0.0) + float(row["metrics"]["precision_at_1"])
    for entry in table.values():
        for name in results:
            entry[name] = round(entry.get(name, 0.0) / entry["positives"], 4)
    pair_reports = {}
    for a, b in pairs:
        report = paired(results[a]["rows"][key], results[b]["rows"][key], videos, samples, seed)
        report["videos_p1_better"] = sum(1 for entry in table.values() if entry[b] > entry[a])
        report["videos_p1_worse"] = sum(1 for entry in table.values() if entry[b] < entry[a])
        pair_reports[f"{b} - {a}"] = report
    return {
        "split": split,
        "key": key,
        "queries": len(queries),
        "videos": len(set(videos)),
        "metrics": {name: run_metrics(result, key) for name, result in results.items()},
        "indexed_models": {name: result["run_metadata"].get("indexed_models")
                           for name, result in results.items()},
        "paired_video_bootstrap": pair_reports,
        "per_video_precision_at_1": table,
        "resampling": {"unit": "video", "samples": samples, "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=result.json")
    parser.add_argument("--pair", action="append", default=[], help="A:B reports B - A")
    parser.add_argument("--key", default="review2_full|expand:10")
    parser.add_argument(
        "--manifest", type=Path, default=Path("evaluation/charades_sta_navmr_benchmark.json")
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    paths = dict(item.split("=", 1) for item in args.run)
    results = {name: json.loads(Path(path).read_text(encoding="utf-8"))
               for name, path in paths.items()}
    report = compare(results, json.loads(args.manifest.read_text(encoding="utf-8")),
                     key=args.key, pairs=[tuple(item.split(":", 1)) for item in args.pair],
                     samples=args.samples, seed=args.seed)
    report["runs"] = {name: Path(path).as_posix() for name, path in paths.items()}
    report["created_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    brief = {name: {key: value for key, value in values.items()
                    if key not in ("ci95", "failure_stages")}
             for name, values in report["metrics"].items()}
    print(json.dumps({"metrics": brief, "pairs": report["paired_video_bootstrap"]}, indent=1))


if __name__ == "__main__":
    main()
