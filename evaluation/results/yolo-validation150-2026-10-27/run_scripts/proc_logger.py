"""Every 10 s: RSS of the indexing process, GPU memory, free RAM and disk. Stops when STOP exists."""
import json, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import psutil
log, stop = Path(sys.argv[1]), Path(sys.argv[2])
while not stop.exists():
    rss = [p.info["memory_info"].rss for p in psutil.process_iter(["cmdline", "memory_info"])
           if p.info["cmdline"] and any("index_videos_batch" in part for part in p.info["cmdline"])]
    try:
        gpu = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        gpu = None
    row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "indexer_rss_gb": round(max(rss, default=0) / 1e9, 3), "gpu_used_mib": gpu,
           "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
           "disk_free_gb": round(shutil.disk_usage("C:/").free / 1e9, 2)}
    with log.open("a") as f:
        f.write(json.dumps(row) + "\n")
    time.sleep(10)
