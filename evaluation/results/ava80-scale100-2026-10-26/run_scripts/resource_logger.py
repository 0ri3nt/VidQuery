"""Append free RAM/disk and the largest Python process RSS every 30 s until STOP exists."""
import json, sys, time, shutil
from datetime import datetime, timezone
from pathlib import Path
import psutil
log, stop = Path(sys.argv[1]), Path(sys.argv[2])
while not stop.exists():
    py = [p for p in psutil.process_iter(["name", "memory_info"]) if "python" in (p.info["name"] or "").lower()]
    row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
           "disk_free_gb": round(shutil.disk_usage("C:/").free / 1e9, 2),
           "max_python_rss_gb": round(max((p.info["memory_info"].rss for p in py), default=0) / 1e9, 2),
           "gpu_note": None}
    with log.open("a") as f: f.write(json.dumps(row) + "\n")
    time.sleep(30)
