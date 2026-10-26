"""Pre-registered demo queries for more Charades validation videos (UI search settings, N repeats)."""
import json, sys, urllib.request
VIDEOS = {"K87AC": "b4024896-9799-4c28-97bb-edb88da3c1ae", "GG4UR": "0c824a7b-9237-4fb6-b202-a96cb5810308",
          "I2DHG": "2a380720-4895-4799-b892-73b60eae8c5b", "R9382": "a2163c9b-c581-4b7f-b8dd-23766885a0a7"}
# (query, accepted intervals, source) - fixed before any query was run.
CANDIDATES = {
 "K87AC": [
  ("a person opens a door.", [(0.0, 4.9)], "STA"),
  ("person closes the door.", [(0.7, 10.2)], "STA"),
  ("person walks into the room.", [(1.0, 5.0)], "frames"),
  ("person takes off his shirt.", [(11.0, 14.5)], "frames"),
  ("person starts undressing.", [(7.6, 12.7), (11.0, 14.5)], "STA [7.6,12.7] + frames [11,14.5]"),
  ("person dressing with items from a cabinet.", [(7.6, 12.7)], "STA"),
  ("person picks up clothes from the bed.", [(14.5, 17.5)], "frames"),
  ("person holding a shirt.", [(15.5, 21.0)], "frames"),
  ("a shirtless man.", [(13.0, 20.5)], "frames"),
  ("person puts on a shirt.", [(17.0, 23.0)], "frames"),
  ("person buttoning his shirt.", [(22.5, 29.0)], "frames"),
  ("person walks to the door.", [(28.5, 32.0)], "frames"),
 ],
 "GG4UR": [
  ("a person is watching television on the sofa.", [(0.0, 3.6), (0.0, 7.5)], "STA [0,3.6] + frames [0,7.5]"),
  ("person sitting on the couch.", [(0.0, 7.5)], "frames"),
  ("a television screen.", [(0.0, 7.5)], "frames"),
  ("a bicycle in the room.", [(0.0, 7.5)], "frames"),
  ("person walks to the closet.", [(10.5, 13.0)], "frames"),
  ("person opens door to closet.", [(10.0, 15.4)], "STA"),
  ("person seen opening the door to a hallway closet.", [(10.0, 15.4)], "STA"),
  ("person takes a towel out of the closet.", [(13.5, 18.5)], "frames"),
  ("person holding a towel.", [(14.5, 18.5)], "frames"),
  ("person carrying a blanket.", [(14.5, 18.5)], "frames"),
  ("person leaves the room.", [(18.0, 19.78)], "frames"),
 ],
 "I2DHG": [
  ("a person walks through a doorway.", [(0.0, 5.2)], "STA"),
  ("person takes off his jacket.", [(2.5, 8.5)], "frames"),
  ("one person was undressing.", [(0.8, 11.4)], "STA"),
  ("person putting their clothes on the bed.", [(0.0, 11.4)], "STA"),
  ("person drinking from a cup.", [(13.1, 23.4)], "STA"),
  ("a woman drinking from a cup.", [(13.1, 23.4)], "STA"),
  ("person holding a cup.", [(14.0, 21.5), (24.0, 26.0)], "frames"),
  ("person smiling.", [(19.7, 30.86)], "STA"),
  ("person sitting on the bed.", [(14.0, 21.5), (24.0, 26.0)], "frames"),
  ("person wearing glasses.", [(14.0, 21.5), (24.0, 26.0)], "frames"),
 ],
 "R9382": [
  ("a person is cooking on the stove.", [(0.0, 12.0)], "STA"),
  ("person standing at the stove.", [(0.0, 10.5)], "frames"),
  ("person walks to the fridge.", [(10.0, 12.0)], "frames"),
  ("person opens the refrigerator.", [(10.5, 17.5)], "frames"),
  ("a refrigerator door.", [(11.0, 17.5)], "frames"),
  ("person pulls something out of the fridge.", [(22.8, 27.7), (11.0, 17.5)], "STA [22.8,27.7] + frames [11,17.5]"),
  ("person close the door.", [(13.2, 21.1)], "STA"),
  ("person eat it.", [(22.8, 27.7)], "STA"),
  ("person eating a sandwich.", [(19.0, 27.7)], "frames"),
  ("person holding food.", [(19.0, 27.7)], "frames"),
 ],
}
def search(video, q):
    body = json.dumps({"query": q, "video_ids": [video], "limit": 10, "ranking": "reliability",
                       "localize": True, "collapse_duplicate_evidence": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:8001/api/search", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
repeats = int(sys.argv[1]); only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CANDIDATES)
out = {}
for name in only:
    out[name] = []
    for q, intervals, source in CANDIDATES[name]:
        runs = []
        for _ in range(repeats):
            d = search(VIDEOS[name], q); res = d.get("results") or []
            top = res[0] if res else None; loc = (top or {}).get("localization") or {}; ev = (top or {}).get("evidence") or {}
            peak = loc.get("peak_time")
            hit = peak is not None and d.get("interpretation") != "insufficient_evidence" and any(a <= peak <= b for a, b in intervals)
            runs.append({"interpretation": d.get("interpretation"), "segment": [top["start_time"], top["end_time"]] if top else None,
                         "peak": peak, "precision": loc.get("precision"), "loc_source": loc.get("source"),
                         "verdict": ev.get("verdict"), "confidence": ev.get("confidence"),
                         "match_reason": (top or {}).get("match_reason"), "hit": hit})
        stable = len({json.dumps({k: r[k] for k in ("segment", "peak", "verdict", "confidence")}) for r in runs}) == 1
        out[name].append({"query": q, "event_intervals": intervals, "interval_source": source, "runs": runs,
                          "stable": stable, "success_all_runs": all(r["hit"] for r in runs)})
        r = runs[0]
        print(f"{name} {'OK ' if out[name][-1]['success_all_runs'] else 'XX '}{'stable' if stable else 'UNSTABLE'} | {q} -> peak {r['peak'] and round(r['peak'],2)} seg {r['segment']} {r['verdict']} {r['confidence'] and round(r['confidence'],3)} | event {intervals} | {(r['match_reason'] or '')[:110]}", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
