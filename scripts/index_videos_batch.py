"""Index many videos in one process so detector/ASR models load once.

Resumable: videos already ``READY`` in the configured database are skipped.
Each attempt appends one JSON line (video, state, seconds) to ``--progress``.
Use an isolated ``DATABASE_PATH``/``UPLOAD_DIR``/``GENERATED_DIR`` for
benchmarks so the main library is untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from vidquery.config import get_settings
from vidquery.domain import ProcessingState
from vidquery.media import VideoIngestionService
from vidquery.pipeline import VideoProcessingService
from vidquery.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    videos = [video for split in args.splits for video in manifest["splits"][split]["videos"]]
    if args.limit is not None:
        videos = videos[: args.limit]
    settings = get_settings()
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    ingestion = VideoIngestionService(settings, repository)
    processor = VideoProcessingService(settings, repository)
    ready = {
        Path(video.original_filename).stem
        for video in repository.list_videos()
        if video.state == ProcessingState.READY
    }
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    for index, video in enumerate(videos, start=1):
        if video in ready:
            continue
        path = args.video_dir / f"{video}.mp4"
        started = time.perf_counter()
        entry = {"video": video, "index": index, "of": len(videos)}
        try:
            record, _duplicate = ingestion.register_existing(path, copy_into_uploads=False)
            final = processor.process(record.video_id)
            entry.update(
                {
                    "state": final.state.value,
                    "segments": len(repository.list_segments([final.video_id])),
                    "warnings": final.warnings[:3],
                }
            )
        except Exception as exc:  # recorded and retried on the next invocation
            entry.update({"state": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:300]})
        entry["seconds"] = round(time.perf_counter() - started, 1)
        entry["finished_at"] = datetime.now(UTC).isoformat()
        with args.progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
