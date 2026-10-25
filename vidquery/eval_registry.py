"""Evaluation-only video registry (``evaluation/eval_only_videos.json``).

Videos listed there may never be used to train weights.  Training and dataset
expansion code calls :func:`assert_not_trained_on` / :func:`eval_only_video_ids`
so held-out data cannot leak into a training split by accident.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "evaluation" / "eval_only_videos.json"
REGISTRY_SCHEMA = "eval-only-registry-v1"


class EvaluationLeakageError(ValueError):
    pass


def eval_only_video_ids(
    path: Path = DEFAULT_REGISTRY, *, groups: Iterable[str] | None = None
) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REGISTRY_SCHEMA:
        raise EvaluationLeakageError(f"unexpected registry schema in {path}")
    wanted = set(groups) if groups is not None else None
    identifiers: set[str] = set()
    for name, group in (payload.get("groups") or {}).items():
        if wanted is None or name in wanted:
            identifiers.update(str(video) for video in group.get("video_ids", []))
    return identifiers


def assert_not_trained_on(
    training_video_ids: Iterable[str], path: Path = DEFAULT_REGISTRY, *, context: str = "training"
) -> None:
    overlap = sorted(set(map(str, training_video_ids)) & eval_only_video_ids(path))
    if overlap:
        raise EvaluationLeakageError(
            f"{context} would use evaluation-only videos: {', '.join(overlap)}"
        )
