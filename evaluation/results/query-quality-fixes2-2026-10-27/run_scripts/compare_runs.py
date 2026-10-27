"""Paired video-bootstrap comparison of evaluator runs that may use different plans.

Rows are matched by position; ids and kinds must agree (plans may differ).
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

KEY = "review2_full|expand:10"
PAIRED = ("p1", "p5_r5_iou05", "r1_iou07", "in_domain_rejected", "positives_abstained",
          "auroc_pos_vs_in_domain")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"][KEY]


def per_query(rows):
    out = []
    for r in rows:
        out.append({"id": r["id"], "kind": r["kind"], "video": r["id"],
                    "p1": r["metrics"].get("precision_at_1") if r["kind"] == "positive" else None,
                    "rejected": r["abstained"] if r["kind"] != "positive" else None,
                    "positive_abstained": r["abstained"] if r["kind"] == "positive" else None,
                    "r1_05": r["metrics"].get("r1_iou0.5") if r["kind"] == "positive" else None,
                    "r1_07": r["metrics"].get("r1_iou0.7") if r["kind"] == "positive" else None,
                    "r5_05": r["metrics"].get("r5_iou0.5") if r["kind"] == "positive" else None,
                    "r5_07": r["metrics"].get("r5_iou0.7") if r["kind"] == "positive" else None,
                    "confidence": r["max_confidence"] or 0.0,
                    "verdict": r["top_verdict"]})
    return out


def auroc(pos, neg):
    if not pos or not neg:
        return None
    # Mann-Whitney U with mid-ranks for ties (identical to the pairwise count).
    values = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(values):
        j = i
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        mid = (i + j + 1) / 2
        rank_sum += mid * sum(flag for _, flag in values[i:j])
        i = j
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def metrics(rows):
    pos = [r for r in rows if r["kind"] == "positive"]
    ind = [r for r in rows if r["kind"] == "in_domain_negative"]
    ood = [r for r in rows if r["kind"] == "out_of_domain_negative"]
    sup = [r for r in rows if r["verdict"] == "supported"]
    return {
        "p1": sum(r["p1"] for r in pos) / len(pos),
        "in_domain_rejected": sum(r["rejected"] for r in ind) / len(ind),
        "positives_abstained": sum(r["positive_abstained"] for r in pos) / len(pos),
        "p5_r5_iou05": sum(r["r5_05"] for r in pos) / len(pos),
        "r1_iou05": sum(r["r1_05"] for r in pos) / len(pos),
        "r1_iou07": sum(r["r1_07"] for r in pos) / len(pos),
        "r5_iou07": sum(r["r5_07"] for r in pos) / len(pos),
        "false_acceptance_in_domain": 1 - sum(r["rejected"] for r in ind) / len(ind),
        "false_acceptance_out_of_domain": 1 - sum(r["rejected"] for r in ood) / len(ood),
        "out_of_domain_rejected": sum(r["rejected"] for r in ood) / len(ood),
        "auroc_pos_vs_in_domain": auroc([r["confidence"] for r in pos], [r["confidence"] for r in ind]),
        "supported_count": len(sup),
        "supported_precision": (sum(1 for r in sup if r["kind"] == "positive" and r["p1"]) / len(sup)) if sup else None,
    }


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    split = sys.argv[2]
    video_of = {q["id"]: q["video"] for q in manifest["splits"][split]["queries"]}
    runs = {name: per_query(load(path)) for name, path in (a.split("=", 1) for a in sys.argv[4:])}
    names = list(runs)
    base = runs[names[0]]
    for name in names[1:]:
        assert [(r["id"], r["kind"]) for r in runs[name]] == [(r["id"], r["kind"]) for r in base], name
    groups = defaultdict(list)
    for index, r in enumerate(base):
        groups[video_of[r["id"]]].append(index)
    videos = sorted(groups)
    rng = random.Random(2026)
    report = {"key": KEY, "videos": len(videos), "queries": len(base),
              "metrics": {n: metrics(runs[n]) for n in names}, "paired_vs_" + names[0]: {}}
    for name in names[1:]:
        diffs = defaultdict(list)
        for _ in range(2000):
            idx = [i for v in (rng.choice(videos) for _ in videos) for i in groups[v]]
            a = metrics([base[i] for i in idx]); b = metrics([runs[name][i] for i in idx])
            for k in PAIRED:
                if a[k] is not None and b[k] is not None:
                    diffs[k].append(b[k] - a[k])
        point = {k: report["metrics"][name][k] - report["metrics"][names[0]][k]
                 for k in PAIRED}
        report["paired_vs_" + names[0]][name] = {
            k: {"difference": round(point[k], 4),
                "ci95": [round(sorted(v)[int(0.025 * len(v))], 4), round(sorted(v)[int(0.975 * len(v)) - 1], 4)]}
            for k, v in diffs.items()}
    Path(sys.argv[3]).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


main()
