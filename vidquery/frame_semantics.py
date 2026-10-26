"""Frame-level OpenCLIP embeddings for text-to-visual moment retrieval.

The structured parser only matches what it can name (COCO objects, AVA
actions, VidOR predicates). Free-form queries such as "person turns a light
on" need a general visual-language signal. This module stores one frozen
OpenCLIP image embedding per extracted frame (fp16, next to the frames) and
scores a segment by its best-matching frame for a text query.

Embeddings are built from frames already on disk, so existing indexes can be
backfilled without reprocessing media. Search uses them only when
``ENABLE_FRAME_SEMANTIC_RETRIEVAL=true``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

FRAME_EMBEDDING_SCHEMA = "openclip-frame-embeddings-v1"
FILENAME = "frame_clip_embeddings.npz"


class FrameEmbeddingError(ValueError):
    pass


def embedding_path(generated_dir: Path, video_id: str) -> Path:
    return Path(generated_dir) / video_id / FILENAME


def build_frame_embeddings(
    video_id: str,
    frames_dir: Path,
    encoder: Any,
    output: Path,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Embed every ``*.jpg`` in ``frames_dir`` and write ``output`` atomically."""

    from PIL import Image

    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise FrameEmbeddingError(f"no frames for {video_id} in {frames_dir}")
    vectors: list[list[float]] = []
    for start in range(0, len(frames), batch_size):
        images = []
        for path in frames[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        vectors.extend(encoder.encode_images(images))
    array = np.asarray(vectors, dtype=np.float16)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=array,
        frame_ids=np.asarray([path.stem for path in frames]),
        meta=np.asarray(
            json.dumps(
                {
                    "schema_version": FRAME_EMBEDDING_SCHEMA,
                    "video_id": video_id,
                    "model_version": getattr(encoder, "model_version", "unknown"),
                    "frames": len(frames),
                }
            )
        ),
    )
    os.replace(temporary, output)
    return {"video_id": video_id, "frames": len(frames), "bytes": output.stat().st_size}


def load_frame_embeddings(
    path: Path, *, expected_model_version: str | None = None
) -> dict[str, Any]:
    if not path.is_file():
        raise FrameEmbeddingError(f"missing frame embeddings: {path}")
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta"]))
        if meta.get("schema_version") != FRAME_EMBEDDING_SCHEMA:
            raise FrameEmbeddingError(f"unexpected frame embedding schema in {path}")
        if expected_model_version and meta.get("model_version") != expected_model_version:
            raise FrameEmbeddingError(
                f"{path} was built with {meta.get('model_version')}, not {expected_model_version}"
            )
        embeddings = payload["embeddings"].astype(np.float32)
        frame_ids = [str(value) for value in payload["frame_ids"]]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-6, None)
    return {"index": {frame_id: row for row, frame_id in enumerate(frame_ids)},
            "embeddings": embeddings, "meta": meta}


class FrameSemanticIndex:
    """Per-video lazy cache; scores a segment by its best frame for a text vector."""

    def __init__(self, generated_dir: Path, *, expected_model_version: str | None = None):
        self.generated_dir = Path(generated_dir)
        self.expected_model_version = expected_model_version
        self._videos: dict[str, dict[str, Any] | None] = {}

    def _video(self, video_id: str) -> dict[str, Any] | None:
        if video_id not in self._videos:
            try:
                self._videos[video_id] = load_frame_embeddings(
                    embedding_path(self.generated_dir, video_id),
                    expected_model_version=self.expected_model_version,
                )
            except FrameEmbeddingError:
                self._videos[video_id] = None
        return self._videos[video_id]

    def forget(self, video_id: str) -> None:
        """Drop the cached embeddings of a video whose files moved."""

        self._videos.pop(video_id, None)

    def best_frame(
        self, video_id: str, frame_ids: list[str], text_vector: np.ndarray
    ) -> tuple[float, str] | None:
        """``(max cosine, frame id)`` over the segment's frames, or ``None``."""

        video = self._video(video_id)
        if video is None:
            return None
        present = [frame for frame in frame_ids if frame in video["index"]]
        if not present:
            return None
        rows = [video["index"][frame] for frame in present]
        similarities = video["embeddings"][rows] @ text_vector
        best = int(np.argmax(similarities))
        return float(similarities[best]), present[best]
