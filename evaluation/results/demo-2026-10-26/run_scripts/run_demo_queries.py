"""Pre-registered OM2NF demo queries: run each N times with the UI's search settings."""
import json, sys, urllib.request
VIDEO = "9be43e27-698a-4997-9fc3-841ee51a2d96"  # OM2NF
# (query, accepted event intervals, interval source) - fixed before any query was run.
CANDIDATES = [
    ("person drinking coffee from a glass.", [(2.2, 8.8)], "Charades-STA"),
    ("person drinking from a cup.", [(2.2, 8.8), (15.5, 18.5)], "STA [2.2,8.8]; second drink 15.5-18.5 seen in frames"),
    ("person reading a newspaper.", [(0.0, 24.0)], "frames"),
    ("person sitting at the table.", [(0.0, 24.0)], "frames"),
    ("person opens the newspaper.", [(20.5, 23.0)], "frames"),
    ("person puts his shoes on.", [(23.7, 34.8)], "Charades-STA"),
    ("person putting on shoes.", [(23.7, 34.8)], "Charades-STA (paraphrase)"),
    ("person bending down.", [(24.0, 33.5)], "frames"),
    ("person stands up and walks away.", [(33.0, 36.0)], "frames"),
    ("person walks to the stove to cook breakfast.", [(34.1, 44.03)], "Charades-STA"),
    ("person cooking at the stove.", [(35.5, 44.03)], "STA + frames (camera cuts to stove at 36 s)"),
    ("person picks up something from the stove.", [(36.0, 44.03)], "frames"),
    ("person holding a cup.", [(3.5, 7.5), (15.0, 18.5)], "frames (mug lifted)"),
    ("a microwave in the kitchen.", [(35.5, 44.03)], "frames (microwave above stove after cut)"),
    ("person near a chair.", [(25.0, 35.0)], "frames (chair visible behind person)"),
]
def search(q):
    body = json.dumps({"query": q, "video_ids": [VIDEO], "limit": 10, "ranking": "reliability",
                       "localize": True, "collapse_duplicate_evidence": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:8001/api/search", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
out = []
for q, intervals, source in CANDIDATES:
    runs = []
    for _ in range(repeats):
        d = search(q)
        res = d.get("results") or []
        top = res[0] if res else None
        loc = (top or {}).get("localization") or {}
        ev = (top or {}).get("evidence") or {}
        peak = loc.get("peak_time")
        hit = peak is not None and d.get("interpretation") != "insufficient_evidence" and any(a <= peak <= b for a, b in intervals)
        runs.append({"interpretation": d.get("interpretation"), "segment": [top["start_time"], top["end_time"]] if top else None,
                     "peak": peak, "precision": loc.get("precision"), "loc_source": loc.get("source"),
                     "verdict": ev.get("verdict"), "confidence": ev.get("confidence"),
                     "match_reason": (top or {}).get("match_reason"), "hit": hit,
                     "rank2_peak": ((res[1].get("localization") or {}).get("peak_time") if len(res) > 1 else None)})
    stable = len({json.dumps({k: r[k] for k in ("segment", "peak", "verdict", "confidence")}) for r in runs}) == 1
    out.append({"query": q, "event_intervals": intervals, "interval_source": source, "runs": runs, "stable": stable,
                "success_all_runs": all(r["hit"] for r in runs)})
    r = runs[0]
    print(f"{'OK ' if out[-1]['success_all_runs'] else 'XX '}{'stable' if stable else 'UNSTABLE'} | {q} -> seg {r['segment']} peak {r['peak']} ({r['precision']}) {r['verdict']} {r['confidence'] and round(r['confidence'],3)} | event {intervals}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
