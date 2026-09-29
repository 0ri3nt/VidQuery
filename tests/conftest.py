from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from vidquery.config import Settings

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("tmp/test-ultralytics").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/test-matplotlib").resolve()))


@pytest.fixture
def settings_factory(tmp_path):
    def factory(**overrides) -> Settings:
        base = Settings(
            data_dir=tmp_path / "data",
            upload_dir=tmp_path / "uploads",
            generated_dir=tmp_path / "generated",
            database_path=tmp_path / "vidquery.sqlite3",
            whisper_cache_dir=tmp_path / "models" / "whisper",
            pyannote_cache_dir=tmp_path / "models" / "pyannote",
            enable_neo4j=False,
            enable_yolo=False,
            enable_whisper=False,
            enable_diarization=False,
            require_diarization=False,
            allow_model_downloads=False,
            enable_relation_gnn=False,
            semantic_retrieval_mode="hashing",
            enable_rag_generation=False,
            enable_ocr=False,
            max_upload_size=20 * 1024 * 1024,
            frame_sample_rate=2.0,
            segment_duration=1.0,
        )
        configured = replace(base, **overrides)
        configured.ensure_directories()
        return configured

    return factory


@pytest.fixture
def sample_video(tmp_path) -> Path:
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    for index in range(20):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 1] = index * 10
        cv2.rectangle(frame, (10 + index, 20), (60 + index, 100), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path
