"""Pairwise comparison of frozen-test runs on differently built Charades indexes.

Rows are matched by position (the manifest repeats some NA-VMR query IDs), with
the id/kind sequence asserted identical. Same queries, flags, and thresholds.
"""
import json, random, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path("evaluation/results/charades-2026-10-25")
INDEX = {
    "v2_original": (R / "test", "data/app/benchmark/charades/charades-snapshot-test.sqlite3"),
    "v2_rerun": (R / "test-v2relation-rerun", "data/app/benchmark/charades_v2_rerun/charades-v2-rerun-snapshot-test.sqlite3"),
    "v3": (R / "test-v3relation", "data/app/benchmark/charades_v3/charades-v3-snapshot-test.sqlite3"),
}
manifest = json.loads(Path("evaluation/charades_sta_navmr_benchmark.json").read_text())
mq = manifest["splits"]["test"]["queries"]
KEY = "review2_full|expand:10"
POS = ["precision_at_1", "r1_iou0.5", "r1_iou0.7", "r5_iou0.5"]

def auroc(pos, neg):
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))

def score(row):
    return row["max_confidence"] if row.get("max_confidence") is not None else -1.0

def boot(by, samples=2000, seed=2026):
    vids = sorted(by); allv = [x for v in vids for x in by[v]]
    rng = random.Random(seed); d = []
    for _ in range(samples):
        pick = [x for v in rng.choices(vids, k=len(vids)) for x in by[v]]
        d.append(sum(pick) / len(pick))
    d.sort()
    return {"difference": round(sum(allv) / len(allv), 4),
            "ci95": [round(d[int(0.025 * samples)], 4), round(d[int(0.975 * samples) - 1], 4)]}

def load(tag, run):
    r = json.loads((INDEX[tag][0] / f"{run}.json").read_text())
    rows = r["rows"][KEY]
    assert [(x["id"], x["kind"]) for x in rows] == [(q["id"], q["kind"]) for q in mq]
    return r, rows

def metrics(r, rows):
    s = r["summary"][KEY]
    pos = [score(x) for x in rows if not x["negative"]]
    out = {f: s[f] for f in POS + ["mean_iou", "positive_answered_rate", "coverage", "aurc", "rejection_f1", "answerability_auroc"]}
    for kind in ("in_domain_negative", "out_of_domain_negative"):
        out[f"rejected_{kind}"] = s["negative_rejection_rate_by_kind"][kind]
        out[f"auroc_vs_{kind}"] = round(auroc(pos, [score(x) for x in rows if x["kind"] == kind]), 4)
    out["ci95_precision_at_1"] = s["ci95"]["precision_at_1"]
    return out

def paired(rows_a, rows_b):
    res = {}
    for m in POS:
        by = defaultdict(list); up = down = 0
        for q, a, b in zip(mq, rows_a, rows_b):
            if a["negative"]: continue
            d = b["metrics"][m] - a["metrics"][m]; by[q["video"]].append(d); up += d > 0; down += d < 0
        res[m] = {**boot(by), "queries_improved": up, "queries_worsened": down}
    for kind in ("in_domain_negative", "out_of_domain_negative"):
        by = defaultdict(list)
        for q, a, b in zip(mq, rows_a, rows_b):
            if a["kind"] == kind:
                by[q["video"]].append(float(b["metrics"]["negative_rejected"]) - float(a["metrics"]["negative_rejected"]))
        res[f"rejected_{kind}"] = boot(by)
    res["queries_with_changed_top1_interval"] = sum((a.get("predicted") or [None])[:1] != (b.get("predicted") or [None])[:1] for a, b in zip(rows_a, rows_b))
    return res

def index(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    name = {v: Path(f).stem for v, f in db.execute("select video_id, original_filename from videos")}
    return {(name[v], round(s, 3)): json.loads(p) for v, s, p in db.execute("select video_id, start_time, payload_json from segments")}

def drift(a, b):
    test = set(manifest["splits"]["test"]["videos"])
    c = Counter()
    for k in sorted(k for k in b if k[0] in test):
        o, n = a.get(k), b[k]
        if o is None: c["segment_missing"] += 1; continue
        c["segments_compared"] += 1
        det = lambda p: [(d["class_label"], d["timestamp"], round(d["confidence"], 4)) for d in p["detections"]]
        c["detections_differ"] += det(o) != det(n)
        for f in ("transcript", "actions", "ocr_evidence", "entities"):
            c[f"{f}_differ"] += o.get(f) != n.get(f)
        c["speaker_labels_differ"] += [s.get("speaker_label") if isinstance(s, dict) else s for s in o["speakers"]] != [s.get("speaker_label") if isinstance(s, dict) else s for s in n["speakers"]]
        rel = lambda p: sorted((r["predicate"], round(r["start_time"], 2)) for r in p["relationships"] if r["source_method"] == "relationship_gnn")
        c["relation_gnn_evidence_differ"] += rel(o) != rel(n)
    return dict(c)

runs = ["T1-selected-soft-frame", "T2-selected-p1-preserving-thresholds", "T0-default-strict-noframe"]
loaded = {(t, r): load(t, r) for t in INDEX for r in runs}
for t in INDEX:
    assert loaded[(t, runs[0])][0]["run_metadata"]["manifest_sha256"] == loaded[("v2_original", runs[0])][0]["run_metadata"]["manifest_sha256"]
    for r in runs:
        assert loaded[(t, r)][0]["plans"] == loaded[("v2_original", r)][0]["plans"]
        assert loaded[(t, r)][0]["run_metadata"]["evidence_thresholds"] == loaded[("v2_original", r)][0]["run_metadata"]["evidence_thresholds"]
indexes = {t: index(p) for t, (_, p) in INDEX.items()}
versions = {}
for t, segs in indexes.items():
    v = Counter(); src = Counter(); meta = Counter()
    for k, p in segs.items():
        if k[0] not in set(manifest["splits"]["test"]["videos"]): continue
        meta[p["processing_metadata"].get("relationship_gnn_model_version")] += 1
        for r in p["relationships"]:
            src[r["source_method"]] += 1
            if r["source_method"] == "relationship_gnn": v[r["model_version"]] += 1
    versions[t] = {"evidence_model_version": dict(v), "segment_metadata_version": dict(meta), "relation_sources": dict(src)}
pairs = [("v2_original", "v2_rerun", "noise floor: same model, fresh rebuild"),
         ("v2_rerun", "v3", "relation model effect, both fresh rebuilds"),
         ("v2_original", "v3", "as requested: previous 41.4% index vs v3 index")]
out = {"schema_version": "charades-index-comparison-v2", "key": KEY, "row_matching": "by position; id/kind sequence asserted identical",
       "resampling": "300 test videos, 2000 resamples, seed 2026",
       "metrics": {r: {t: metrics(*loaded[(t, r)]) for t in INDEX} for r in runs},
       "paired_b_minus_a": {f"{a}->{b}": {"meaning": why, **{r: paired(loaded[(a, r)][1], loaded[(b, r)][1]) for r in runs}} for a, b, why in pairs},
       "index_drift": {f"{a}->{b}": drift(indexes[a], indexes[b]) for a, b, _ in pairs},
       "relation_evidence": versions}
target = R / "test-v3relation" / "comparison-v2-v2rerun-v3.json"
assert not target.exists()
target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
