"""Isolated per-detector cost: run in a fresh process per detector.

Usage: detector_resources.py WEIGHTS FRAMES_DIR_LIST_FILE
Prints JSON: weights size, peak process RSS, peak CUDA memory allocated, ms/frame.
"""

import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO

weights, frame_list = Path(sys.argv[1]), Path(sys.argv[2])
frames = [line.strip() for line in frame_list.read_text().splitlines() if line.strip()]
process = psutil.Process(os.getpid())
model = YOLO(str(weights))
for frame in frames[:5]:
    model.predict(source=frame, conf=0.25, iou=0.45, device=0, verbose=False)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
peak_rss = process.memory_info().rss
started = time.perf_counter()
for frame in frames:
    model.predict(source=frame, conf=0.25, iou=0.45, device=0, verbose=False)
    peak_rss = max(peak_rss, process.memory_info().rss)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
print(json.dumps({
    "weights": weights.name,
    "weights_mb": round(weights.stat().st_size / 1e6, 1),
    "frames": len(frames),
    "ms_per_frame": round(1000 * elapsed / len(frames), 1),
    "peak_process_rss_gb": round(peak_rss / 1e9, 2),
    "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
}))
