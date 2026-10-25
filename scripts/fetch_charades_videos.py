"""Range-read only the benchmark's Charades videos from the official 480p archive.

Resumable: a video whose file already matches the archive member size is
skipped.  Files are written atomically and checked for an MP4 header.  Only
``*.mp4`` members named by the manifest are read; nothing is executed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from remotezip import RemoteZip  # type: ignore[import-untyped]

ARCHIVE = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("evaluation/charades_sta_navmr_benchmark.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/source/charades/videos"))
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = [video for split in args.splits for video in manifest["splits"][split]["videos"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fetched = skipped = 0
    transferred = 0
    missing: list[str] = []
    with RemoteZip(ARCHIVE) as archive:
        members = {Path(info.filename).stem: info for info in archive.infolist()
                   if info.filename.endswith(".mp4")}
        for video in wanted:
            info = members.get(video)
            if info is None:
                missing.append(video)
                continue
            target = args.output_dir / f"{video}.mp4"
            if target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            data = archive.read(info)
            if b"ftyp" not in data[:64]:
                raise SystemExit(f"{video}: archive member is not an MP4")
            temporary = target.with_suffix(".mp4.part")
            temporary.write_bytes(data)
            os.replace(temporary, target)
            fetched += 1
            transferred += info.compress_size
    report = {
        "archive": ARCHIVE,
        "requested": len(wanted),
        "fetched": fetched,
        "already_present": skipped,
        "missing_from_archive": missing,
        "bytes_transferred": transferred,
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
