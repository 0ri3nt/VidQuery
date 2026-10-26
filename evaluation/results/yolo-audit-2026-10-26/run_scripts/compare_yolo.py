"""Paired comparison of detector rebuilds on the 50-video validation subset.

Usage: compare_yolo.py OUT.json
Each detector is compared with the yolov8n rebuild (paired video bootstrap,
2,000 resamples, seed 2026). The yolov8n rebuild is also compared with the
original integrated index restricted to the same videos (rebuild noise).
Indexing latency comes from each rebuild's progress.jsonl; index statistics
(detections, entities, actions, relations per segment) from its database.
"""

import json
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RES = Path("evaluation/results/yolo-audit-2026-10-26")
ROOT = Path("data/app/benchmark/yolo_audit_2026-10-26")
ORIGINAL = Path("evaluation/results/charades-integrated-2026-10-26/det-ava8-vidor4")
DETECTORS = ["yolov8n", "yolov8s", "yolov8m", "yolo11s", "yolo11m", "yolo26s"]
SETTINGS = {"T1": "validation-T1-selected-soft-frame.json", "T0": "validation-T0-default-strict-noframe.json"}
KEY = "review2_full|expand:10"


def rows(path, keep_videos=None, video_of=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))["rows"][KEY]
    if keep_videos is None:
        return data
    return [r for r, v in zip(data, video_of) if v in keep_videos]


def metric_rows(data, videos):
    out = []
    for r, v in zip(data, videos):
        out.append({"video": v, "kind": r["kind"],
                    "p1": r["metrics"].get("precision_at_1") if r["kind"] == "positive" else None,
                    "r5": r["metrics"].get("r5_iou0.5") if r["kind"] == "positive" else None,
                    "rejected": r["abstained"] if r["kind"] != "positive" else None})
    return out


def summary(items):
    pos = [r for r in items if r["kind"] == "positive"]
    ind = [r for r in items if r["kind"] == "in_domain_negative"]
    return {"p1": sum(r["p1"] for r in pos) / len(pos), "r5": sum(r["r5"] for r in pos) / len(pos),
            "in_domain_rejected": sum(r["rejected"] for r in ind) / len(ind)}


def paired(a, b):
    groups = defaultdict(list)
    for i, r in enumerate(a):
        groups[r["video"]].append(i)
    videos = sorted(groups)
    rng = random.Random(2026)
    diffs = defaultdict(list)
    for _ in range(2000):
        idx = [i for v in (rng.choice(videos) for _ in videos) for i in groups[v]]
        sa, sb = summary([a[i] for i in idx]), summary([b[i] for i in idx])
        for k in sa:
            diffs[k].append(sb[k] - sa[k])
    point = {k: summary(b)[k] - summary(a)[k] for k in summary(a)}
    return {k: {"difference": round(point[k], 4),
                "ci95": [round(sorted(v)[50], 4), round(sorted(v)[1949], 4)],
                "significant": not (sorted(v)[50] <= 0 <= sorted(v)[1949])} for k, v in diffs.items()}


def index_stats(name):
    db = sqlite3.connect(f"file:{ROOT / name / 'index.sqlite3'}?mode=ro", uri=True)
    seg = [json.loads(p) for (p,) in db.execute("select payload_json from segments")]
    n = len(seg)
    return {"segments": n,
            "detections_per_segment": round(sum(len(s["detections"]) for s in seg) / n, 2),
            "distinct_entities_per_segment": round(sum(len(s["entities"]) for s in seg) / n, 2),
            "actions_per_segment": round(sum(len(s["actions"]) for s in seg) / n, 2),
            "relations_per_segment": round(sum(len(s["relationships"]) for s in seg) / n, 2)}


def latency(name):
    entries = [json.loads(line) for line in (ROOT / name / "progress.jsonl").read_text().splitlines()]
    seconds = [e["seconds"] for e in entries if e.get("state") == "READY"]
    return {"videos": len(seconds), "mean_seconds_per_video": round(statistics.mean(seconds), 2),
            "median_seconds_per_video": round(statistics.median(seconds), 2)}


def main():
    manifest = json.loads((RES / "validation50_manifest.json").read_text(encoding="utf-8"))
    subset_videos = [q["video"] for q in manifest["splits"]["validation"]["queries"]]
    full = json.loads(Path("evaluation/charades_sta_navmr_benchmark.json").read_text(encoding="utf-8"))
    full_videos = [q["video"] for q in full["splits"]["validation"]["queries"]]
    keep = set(manifest["splits"]["validation"]["videos"])
    report = {"subset": {"videos": len(keep), "queries": len(subset_videos)}, "settings": {}}
    available = [d for d in DETECTORS if all((RES / d / f).exists() for f in SETTINGS.values())]
    report["detectors_completed"] = available
    for tag, filename in SETTINGS.items():
        base = metric_rows(rows(RES / "yolov8n" / filename), subset_videos)
        block = {"metrics": {d: summary(metric_rows(rows(RES / d / filename), subset_videos))
                             for d in available}, "paired_vs_yolov8n_rebuild": {}}
        original = metric_rows(rows(ORIGINAL / filename, keep, full_videos), subset_videos)
        assert [r["kind"] for r in original] == [r["kind"] for r in base]
        block["metrics"]["original_index_yolov8n"] = summary(original)
        block["rebuild_noise_yolov8n_vs_original"] = paired(original, base)
        for d in available:
            if d != "yolov8n":
                block["paired_vs_yolov8n_rebuild"][d] = paired(
                    base, metric_rows(rows(RES / d / filename), subset_videos))
        report["settings"][tag] = block
    report["latency"] = {d: latency(d) for d in available}
    report["index_stats"] = {d: index_stats(d) for d in available}
    micro = RES / "detector_microbenchmark.json"
    if micro.exists():
        report["per_frame_microbenchmark"] = json.loads(micro.read_text())
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


main()
