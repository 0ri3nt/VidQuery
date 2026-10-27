"""Profile per-video memory growth of batch indexing (scratch library; sources read-only).

Usage: index_memory_profile.py OUT.json N_VIDEOS [--trace] [--release]
Indexes the first N validation videos of the Charades manifest into the library given by
DATABASE_PATH/UPLOAD_DIR/GENERATED_DIR (set them to a scratch location) in one process, the
way scripts/index_videos_batch.py does, and records after each video: committed (private)
memory, RSS, live Python objects, and with --trace the top tracemalloc growth since video 2.
"""

import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

import psutil

from scripts.index_videos_batch import release_memory
from vidquery.config import get_settings
from vidquery.media import VideoIngestionService
from vidquery.pipeline import VideoProcessingService
from vidquery.storage import SQLiteRepository


def main(out: Path, count: int, trace: bool, release: bool) -> None:
    manifest = json.loads(Path("evaluation/charades_sta_navmr_benchmark.json").read_text(encoding="utf-8"))
    videos = manifest["splits"]["validation"]["videos"][:count]
    settings = get_settings()
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    ingestion = VideoIngestionService(settings, repository)
    processor = VideoProcessingService(settings, repository)
    process = psutil.Process()
    rows, baseline = [], None
    if trace:
        tracemalloc.start(10)
    for index, video in enumerate(videos, start=1):
        started = time.perf_counter()
        record, _ = ingestion.register_existing(Path("data/source/charades/videos") / f"{video}.mp4")
        processor.process(record.video_id)
        if release:
            release_memory()  # what scripts/index_videos_batch.py now does per video
        gc.collect()
        info = process.memory_info()
        rows.append({"video": index, "seconds": round(time.perf_counter() - started, 1),
                     "private_gb": round(getattr(info, "private", info.vms) / 1e9, 3),
                     "rss_gb": round(info.rss / 1e9, 3), "python_objects": len(gc.get_objects())})
        print(json.dumps(rows[-1]), flush=True)
        if trace and index == 2:
            baseline = tracemalloc.take_snapshot()
    report = {"videos": count, "rows": rows}
    if trace and baseline is not None:
        growth = tracemalloc.take_snapshot().compare_to(baseline, "traceback")
        report["top_python_growth"] = [
            {"kib": round(stat.size_diff / 1024, 1), "count": stat.count_diff,
             "where": [f"{frame.filename}:{frame.lineno}" for frame in stat.traceback[-4:]]}
            for stat in growth[:15]
        ]
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), "--trace" in sys.argv, "--release" in sys.argv)
