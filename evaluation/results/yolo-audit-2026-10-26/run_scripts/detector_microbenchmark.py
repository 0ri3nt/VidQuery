"""Per-frame detector latency and detection counts on fixed frames (no downstream effect)."""
import json, sys, time
from pathlib import Path
from collections import Counter
import torch
from ultralytics import YOLO

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated = Path(sys.argv[2])
videos = manifest["splits"]["validation"]["videos"][:20]
import sqlite3
db = sqlite3.connect(f"file:{sys.argv[3]}?mode=ro", uri=True)
ids = {Path(name).stem: vid for vid, name in db.execute("select video_id, original_filename from videos")}
frames = []
for v in videos:
    frames += sorted((generated / ids[v] / "frames").glob("*.jpg"))[:10]
print(len(frames), "frames", flush=True)
weights = sys.argv[4:]
out = {}
for w in weights:
    model = YOLO(w)
    for f in frames[:5]:
        model.predict(source=str(f), conf=0.25, iou=0.45, device=0, verbose=False)
    torch.cuda.synchronize()
    counts, classes, t = 0, Counter(), time.perf_counter()
    for f in frames:
        r = model.predict(source=str(f), conf=0.25, iou=0.45, device=0, verbose=False)[0]
        counts += len(r.boxes)
        classes.update(r.names[int(c)] for c in r.boxes.cls.tolist())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / len(frames) * 1000
    out[Path(w).name] = {"ms_per_frame": round(ms, 1), "detections_per_frame": round(counts / len(frames), 2),
                         "person_per_frame": round(classes["person"] / len(frames), 2),
                         "top_classes": classes.most_common(8)}
    print(Path(w).name, out[Path(w).name], flush=True)
Path(sys.argv[1]).with_name("detector_microbenchmark.json").write_text(json.dumps({"frames": len(frames), "conf": 0.25, "iou": 0.45, "results": out}, indent=1))
