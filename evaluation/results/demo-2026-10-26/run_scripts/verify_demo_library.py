"""Re-run the 19 finalized demo queries against a server and compare with the recorded outcomes.

Usage: python verify_demo_library.py BASE_URL REPEATS OUT_JSON
Videos are resolved by display name (OM2NF.mp4, I2DHG.mp4, GG4UR.mp4) from BASE_URL/api/videos,
so the check works against any library holding those videos. Search settings match the UI.
"""
import json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
base, repeats, out_path = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
expected = []  # (video name, query, segment start, peak, verdict, confidence)
for query, outcomes in json.loads((HERE / "final-queries-5x.json").read_text()).items():
    start, peak, verdict, conf = outcomes[0]
    expected.append(("OM2NF", query, start, peak, verdict, conf))
for name, queries in json.loads((HERE / "final-queries-5x-I2DHG-GG4UR.json").read_text()).items():
    for query, record in queries.items():
        start, peak, _source, verdict, conf = record["outcomes_over_5_runs"][0]
        expected.append((name, query, start, peak, verdict, conf))

videos = json.load(urllib.request.urlopen(f"{base}/api/videos", timeout=60))
videos = videos if isinstance(videos, list) else videos["videos"]
ids = {v["display_name"].rsplit(".", 1)[0]: v["video_id"] for v in videos}

def search(video_id, query):
    body = json.dumps({"query": query, "video_ids": [video_id], "limit": 10, "ranking": "reliability",
                       "localize": True, "collapse_duplicate_evidence": True}).encode()
    request = urllib.request.Request(f"{base}/api/search", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=300))

report = []
for name, query, start, peak, verdict, conf in expected:
    runs = []
    for _ in range(repeats):
        top = (search(ids[name], query).get("results") or [None])[0] or {}
        loc, ev = top.get("localization") or {}, top.get("evidence") or {}
        runs.append({"segment_start": top.get("start_time"), "peak": loc.get("peak_time"),
                     "verdict": ev.get("verdict"), "confidence": ev.get("confidence")})
    match = all(r["peak"] is not None and abs(r["peak"] - peak) < 1e-3 and r["segment_start"] == start
                and r["verdict"] == verdict and abs(r["confidence"] - conf) < 5e-4 for r in runs)
    report.append({"video": name, "query": query,
                   "expected": {"segment_start": start, "peak": peak, "verdict": verdict, "confidence": conf},
                   "runs": runs, "match": match})
    r = runs[0]
    print(f"{'OK' if match else 'XX'} {name} | {query} -> peak {r['peak']} {r['verdict']} {r['confidence']} "
          f"(expected {peak} {verdict} {conf})", flush=True)
print(f"{sum(r['match'] for r in report)}/{len(report)} queries reproduce their recorded outcome on all {repeats} runs")
Path(out_path).write_text(json.dumps({"base_url": base, "repeats": repeats, "video_ids": ids,
                                      "queries": report}, indent=1) + "\n")
