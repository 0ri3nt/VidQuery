"""Build OpenCLIP frame embeddings for already-indexed videos (no media reprocessing).

Reads the frames kept under ``GENERATED_DIR/<video_id>/frames`` and writes
``frame_clip_embeddings.npz`` next to them. Resumable: valid files are skipped.
"""

from __future__ import annotations

import argparse
import json
import time

from vidquery.appearance import configured_appearance_encoder
from vidquery.config import get_settings
from vidquery.frame_semantics import (
    FrameEmbeddingError,
    build_frame_embeddings,
    embedding_path,
    load_frame_embeddings,
)
from vidquery.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    settings = get_settings()
    encoder = configured_appearance_encoder(settings)
    if encoder is None:
        raise SystemExit("ENABLE_APPEARANCE_FEATURES must be true (OpenCLIP encoder)")
    repository = SQLiteRepository(settings.database_path)
    started = time.perf_counter()
    built = skipped = failed = 0
    total_bytes = 0
    for video in repository.list_videos():
        output = embedding_path(settings.generated_dir, video.video_id)
        try:
            load_frame_embeddings(output, expected_model_version=encoder.model_version)
            skipped += 1
            continue
        except FrameEmbeddingError:
            pass
        try:
            report = build_frame_embeddings(
                video.video_id,
                settings.generated_dir / video.video_id / "frames",
                encoder,
                output,
                batch_size=args.batch_size,
            )
            built += 1
            total_bytes += int(report["bytes"])
        except FrameEmbeddingError as exc:
            failed += 1
            print(f"[WARN] {video.original_filename}: {exc}")
    print(
        json.dumps(
            {
                "built": built,
                "skipped_valid": skipped,
                "failed": failed,
                "bytes_written": total_bytes,
                "seconds": round(time.perf_counter() - started, 1),
                "model_version": encoder.model_version,
            }
        )
    )


if __name__ == "__main__":
    main()
