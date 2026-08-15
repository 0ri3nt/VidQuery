from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .ava_gnn80 import _bbox_iou

os.environ.setdefault(
    "TORCH_HOME", str(Path(__file__).resolve().parents[1] / "data" / "app" / "cache" / "torch")
)

IMPROVED_DATASET_SCHEMA = "ava80-pretrained-visual-motion-v1"
RESNET18_BACKBONE = "torchvision-resnet18-imagenet1k-v1"
TEMPORAL_CLIP_ENCODER = "frozen-resnet18-short-clip-pool-v1"
TEMPORAL_CLIP_SCHEMA = "ava80-frozen-temporal-clip-v1"
MOTION_FEATURE_NAMES = (
    "has_previous",
    "has_next",
    "previous_center_dx",
    "previous_center_dy",
    "next_center_dx",
    "next_center_dy",
    "previous_log_area_ratio",
    "next_log_area_ratio",
    "track_age_seconds",
    "track_duration_seconds",
)


def _motion_delta(
    box: tuple[float, ...], timestamp: float, item: Any
) -> tuple[float, float, float]:
    if item is None:
        return 0.0, 0.0, 0.0
    _, _, other_time, other_box = item
    seconds = max(1e-6, abs(timestamp - other_time))
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    other_center = (
        (other_box[0] + other_box[2]) / 2,
        (other_box[1] + other_box[3]) / 2,
    )
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    other_area = max(
        1e-6,
        (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]),
    )
    return (
        (center[0] - other_center[0]) / seconds,
        (center[1] - other_center[1]) / seconds,
        math.log(area / other_area),
    )


def coverage_balanced_video_split(
    graphs: Sequence[Any],
    *,
    seed: int = 2026,
    validation_fraction: float = 0.18,
    test_fraction: float = 0.18,
) -> dict[str, Any]:
    """Create a deterministic, video-disjoint split with broad held-out coverage."""
    import torch

    per_video: dict[str, Any] = {}
    for graph in graphs:
        video_id = str(graph.video_id)
        positives = graph.y[graph.loss_mask].sum(dim=0)
        per_video[video_id] = per_video.get(video_id, torch.zeros(80)) + positives
    video_ids = sorted(per_video)
    if len(video_ids) < 10:
        raise ValueError("balanced AVA split requires at least ten processed videos")
    validation_size = max(2, round(len(video_ids) * validation_fraction))
    test_size = max(2, round(len(video_ids) * test_fraction))
    global_support = sum(per_video.values(), torch.zeros(80))
    eligible = global_support >= 3
    rng = random.Random(seed)
    tie_break = {video_id: rng.random() for video_id in video_ids}

    def pick(count: int, candidates: set[str], already_held_out: set[str]) -> list[str]:
        selected: list[str] = []
        covered = torch.zeros(80, dtype=torch.bool)
        for _ in range(count):
            best: tuple[float, float, float] | None = None
            best_video = ""
            for video_id in candidates.difference(selected):
                proposed_held_out = already_held_out.union(selected, {video_id})
                train_support = sum(
                    (
                        per_video[item]
                        for item in video_ids
                        if item not in proposed_held_out
                    ),
                    torch.zeros(80),
                )
                if bool(((train_support == 0) & eligible).any()):
                    continue
                present = per_video[video_id] > 0
                rare_reward = float(
                    ((present & ~covered) / torch.sqrt(global_support.clamp(min=1))).sum()
                )
                breadth = float((present & eligible).sum())
                candidate = (rare_reward, breadth, tie_break[video_id])
                if best is None or candidate > best:
                    best = candidate
                    best_video = video_id
            if not best_video:
                remaining = sorted(
                    candidates.difference(selected), key=lambda item: tie_break[item]
                )
                if not remaining:
                    break
                best_video = remaining[-1]
            selected.append(best_video)
            covered |= per_video[best_video] > 0
        return sorted(selected)

    candidates = set(video_ids)
    test = pick(test_size, candidates, set())
    validation = pick(validation_size, candidates.difference(test), set(test))
    train = sorted(candidates.difference(test, validation))
    split_sets = [set(train), set(validation), set(test)]
    if any(
        left.intersection(right)
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    ):
        raise RuntimeError("video leakage detected in coverage-balanced split")

    def support(ids: Sequence[str]) -> list[int]:
        values = sum((per_video[item] for item in ids), torch.zeros(80))
        return [int(value) for value in values]

    supports = {name: support(ids) for name, ids in (
        ("train", train), ("validation", validation), ("test", test)
    )}
    return {
        "schema_version": "ava80-video-disjoint-balanced-v2",
        "seed": seed,
        "train": train,
        "validation": validation,
        "test": test,
        "class_support": supports,
        "classes_with_positive_examples": {
            name: sum(value > 0 for value in values) for name, values in supports.items()
        },
        "unsupported_training_action_ids": [
            index + 1 for index, value in enumerate(supports["train"]) if value == 0
        ],
        "insufficient_validation_action_ids": [
            index + 1 for index, value in enumerate(supports["validation"]) if value < 2
        ],
        "insufficient_test_action_ids": [
            index + 1 for index, value in enumerate(supports["test"]) if value < 1
        ],
    }


def attach_motion_features(graphs: Sequence[Any]) -> list[Any]:
    """Assign local tracks and deterministic adjacent-timestamp motion features."""
    import torch

    output = [graph.clone() for graph in graphs]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for graph in output:
        grouped[str(graph.video_id)].append(graph)
    for video_id, video_graphs in grouped.items():
        video_graphs.sort(key=lambda graph: float(graph.timestamp))
        previous: dict[str, tuple[float, tuple[float, ...]]] = {}
        next_track_number = 0
        records: dict[str, list[tuple[Any, int, float, tuple[float, ...]]]] = defaultdict(list)
        for graph in video_graphs:
            timestamp = float(graph.timestamp)
            assigned: list[str] = [""] * int(graph.num_nodes)
            used_tracks: set[str] = set()
            for node_index in range(int(graph.num_nodes)):
                if not bool(graph.person_mask[node_index]):
                    continue
                node_id = str(graph.node_ids[node_index])
                box = tuple(float(value) for value in graph.x[node_index, :4])
                if node_id.startswith("ava:"):
                    track_id = f"{video_id}:{node_id}"
                else:
                    candidates = [
                        (track_id, _bbox_iou(box, prior_box))
                        for track_id, (prior_time, prior_box) in previous.items()
                        if timestamp - prior_time <= 2.1 and track_id not in used_tracks
                    ]
                    best = max(candidates, key=lambda item: item[1], default=("", 0.0))
                    if best[1] >= 0.3:
                        track_id = best[0]
                    else:
                        track_id = f"{video_id}:local:{next_track_number}"
                        next_track_number += 1
                assigned[node_index] = track_id
                used_tracks.add(track_id)
                previous[track_id] = (timestamp, box)
                records[track_id].append((graph, node_index, timestamp, box))
            graph.track_ids = assigned
            graph.motion_x = torch.zeros((int(graph.num_nodes), len(MOTION_FEATURE_NAMES)))

        for track_records in records.values():
            track_records.sort(key=lambda item: item[2])
            duration = track_records[-1][2] - track_records[0][2]
            for index, (graph, node_index, timestamp, box) in enumerate(track_records):
                previous_item = track_records[index - 1] if index else None
                next_item = track_records[index + 1] if index + 1 < len(track_records) else None

                previous_dx, previous_dy, previous_area = _motion_delta(
                    box, timestamp, previous_item
                )
                next_dx, next_dy, next_area = _motion_delta(box, timestamp, next_item)
                graph.motion_x[node_index] = torch.tensor(
                    [
                        float(previous_item is not None),
                        float(next_item is not None),
                        previous_dx,
                        previous_dy,
                        next_dx,
                        next_dy,
                        previous_area,
                        next_area,
                        timestamp - track_records[0][2],
                        duration,
                    ],
                    dtype=torch.float32,
                )
    return output


class FrozenResNet18Embedder:
    def __init__(self, device_name: str = "auto") -> None:
        import torch
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.model = backbone.eval().to(self.device)
        self.transform = weights.transforms()
        self.dimension = 512

    def encode(self, images: Sequence[Any], batch_size: int = 64) -> Any:
        import torch

        if not images:
            return torch.empty((0, self.dimension), dtype=torch.float32)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = torch.stack(
                    [self.transform(image) for image in images[start : start + batch_size]]
                ).to(self.device)
                outputs.append(self.model(batch).cpu())
        return torch.cat(outputs, dim=0)


def attach_pretrained_visual_features(
    graphs: Sequence[Any],
    *,
    frame_root: Path,
    cache_dir: Path,
    device_name: str = "auto",
) -> list[Any]:
    """Attach cached frozen ResNet18 ROI and frame embeddings to every graph."""
    import torch
    from PIL import Image

    cache_dir = cache_dir / RESNET18_BACKBONE
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = {path.name: path for path in frame_root.rglob("*.jpg")}
    embedder: FrozenResNet18Embedder | None = None
    output = [graph.clone() for graph in graphs]
    for index, graph in enumerate(output, start=1):
        frame_name = f"{graph.video_id}_{int(float(graph.timestamp)):04d}.jpg"
        frame_path = frame_paths.get(frame_name)
        if frame_path is None:
            raise FileNotFoundError(f"frame not found for visual embedding: {frame_name}")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "schema": IMPROVED_DATASET_SCHEMA,
                    "backbone": RESNET18_BACKBONE,
                    "frame": [
                        str(frame_path),
                        frame_path.stat().st_size,
                        frame_path.stat().st_mtime_ns,
                    ],
                    "boxes": graph.x[:, :4].tolist(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache_path = cache_dir / str(graph.video_id) / f"{int(float(graph.timestamp)):04d}.pt"
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("fingerprint") == fingerprint:
                roi = cached["roi"]
                frame_embedding = cached["frame"]
                graph.roi_visual_x = roi
                graph.frame_visual_x = frame_embedding.repeat(int(graph.num_nodes), 1)
                continue
        if embedder is None:
            embedder = FrozenResNet18Embedder(device_name)
        with Image.open(frame_path) as opened:
            image = opened.convert("RGB")
            crops = []
            for box in graph.x[:, :4]:
                x1, y1, x2, y2 = [float(value) for value in box]
                pixels = (
                    int(x1 * image.width), int(y1 * image.height),
                    max(int(x2 * image.width), int(x1 * image.width) + 1),
                    max(int(y2 * image.height), int(y1 * image.height) + 1),
                )
                crops.append(image.crop(pixels))
            roi = embedder.encode(crops)
            frame_embedding = embedder.encode([image])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": IMPROVED_DATASET_SCHEMA,
            "fingerprint": fingerprint,
            "backbone": RESNET18_BACKBONE,
            "roi": roi,
            "frame": frame_embedding,
        }, cache_path)
        graph.roi_visual_x = roi
        graph.frame_visual_x = frame_embedding.repeat(int(graph.num_nodes), 1)
        if index % 100 == 0:
            print(f"[visual embeddings] {index}/{len(output)} graphs", flush=True)
    return output


def _temporal_cache_fingerprint(video_graphs: Sequence[Any]) -> str:
    """Fingerprint the ordered frozen frame embeddings used by one clip cache."""
    digest = hashlib.sha256()
    digest.update(TEMPORAL_CLIP_SCHEMA.encode("utf-8"))
    digest.update(TEMPORAL_CLIP_ENCODER.encode("utf-8"))
    for graph in video_graphs:
        import torch

        frame = graph.frame_visual_x[0].detach().to(torch.float32).cpu().contiguous()
        digest.update(f"{float(graph.timestamp):.6f}".encode("ascii"))
        digest.update(frame.numpy().tobytes())
    return digest.hexdigest()


def attach_frozen_temporal_clip_features(
    graphs: Sequence[Any],
    *,
    cache_dir: Path,
    maximum_neighbor_gap_seconds: float = 2.1,
) -> list[Any]:
    """Attach one cached short-clip descriptor built from frozen ResNet features.

    AVA source media is intentionally deleted after validated feature extraction,
    so this storage-safe encoder operates on the persisted ImageNet-pretrained
    frame embeddings.  It combines the previous, current, and next annotated
    frames with fixed (non-trainable) temporal pooling and a signed change term.
    This is one 512-dimensional frozen clip embedding, not optical flow and not a
    separately trained action model.
    """
    import torch
    from torch.nn import functional as functional

    if maximum_neighbor_gap_seconds <= 0:
        raise ValueError("maximum temporal neighbor gap must be positive")
    output = [graph.clone() for graph in graphs]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for graph in output:
        if not hasattr(graph, "frame_visual_x"):
            raise ValueError("frozen frame embeddings are missing")
        grouped[str(graph.video_id)].append(graph)
    encoder_cache = cache_dir / TEMPORAL_CLIP_ENCODER
    encoder_cache.mkdir(parents=True, exist_ok=True)
    for video_id, video_graphs in grouped.items():
        video_graphs.sort(key=lambda graph: float(graph.timestamp))
        fingerprint = _temporal_cache_fingerprint(video_graphs)
        cache_path = encoder_cache / f"{video_id}.pt"
        timestamps = [float(graph.timestamp) for graph in video_graphs]
        embeddings = None
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            candidate = cached.get("embeddings")
            if (
                cached.get("schema_version") == TEMPORAL_CLIP_SCHEMA
                and cached.get("fingerprint") == fingerprint
                and cached.get("timestamps") == timestamps
                and isinstance(candidate, torch.Tensor)
                and tuple(candidate.shape) == (len(video_graphs), 512)
                and bool(torch.isfinite(candidate).all())
            ):
                embeddings = candidate.to(torch.float32)
        if embeddings is None:
            frames = torch.stack(
                [
                    graph.frame_visual_x[0].detach().to(torch.float32).cpu()
                    for graph in video_graphs
                ]
            )
            rows = []
            for index, graph in enumerate(video_graphs):
                current = frames[index]
                previous = current
                following = current
                if index > 0 and (
                    float(graph.timestamp) - float(video_graphs[index - 1].timestamp)
                    <= maximum_neighbor_gap_seconds
                ):
                    previous = frames[index - 1]
                if index + 1 < len(video_graphs) and (
                    float(video_graphs[index + 1].timestamp) - float(graph.timestamp)
                    <= maximum_neighbor_gap_seconds
                ):
                    following = frames[index + 1]
                pooled = 0.25 * previous + 0.5 * current + 0.25 * following
                signed_change = 0.25 * (following - previous)
                rows.append(functional.normalize(pooled + signed_change, dim=0))
            embeddings = torch.stack(rows)
            torch.save(
                {
                    "schema_version": TEMPORAL_CLIP_SCHEMA,
                    "encoder": TEMPORAL_CLIP_ENCODER,
                    "fingerprint": fingerprint,
                    "video_id": video_id,
                    "timestamps": timestamps,
                    "maximum_neighbor_gap_seconds": maximum_neighbor_gap_seconds,
                    "embeddings": embeddings,
                },
                cache_path,
            )
        for graph, embedding in zip(video_graphs, embeddings, strict=True):
            graph.temporal_visual_x = embedding.repeat(int(graph.num_nodes), 1)
    return output


def materialize_feature_variants(graphs: Sequence[Any]) -> list[Any]:
    import torch

    output = []
    for graph in graphs:
        if not hasattr(graph, "roi_visual_x") or not hasattr(graph, "frame_visual_x"):
            raise ValueError("pretrained visual embeddings are missing")
        selected = graph.clone()
        selected.handcrafted_x = selected.x.clone()
        selected.roi_x = torch.cat((selected.x, selected.roi_visual_x), dim=1)
        selected.visual_x = torch.cat(
            (selected.x, selected.roi_visual_x, selected.frame_visual_x), dim=1
        )
        temporal = getattr(selected, "temporal_visual_x", None)
        if temporal is not None:
            selected.visual_temporal_x = torch.cat(
                (selected.visual_x, temporal), dim=1
            )
        motion = getattr(selected, "motion_x", None)
        if motion is None:
            motion = torch.zeros((int(selected.num_nodes), len(MOTION_FEATURE_NAMES)))
        selected.visual_motion_x = torch.cat((selected.visual_x, motion), dim=1)
        output.append(selected)
    return output


def save_improved_graph_cache(graphs: Sequence[Any], path: Path) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": IMPROVED_DATASET_SCHEMA, "graphs": list(graphs)}, path)
    metadata = {
        "schema_version": IMPROVED_DATASET_SCHEMA,
        "backbone": RESNET18_BACKBONE,
        "graphs": len(graphs),
        "videos": sorted({str(graph.video_id) for graph in graphs}),
        "nodes": sum(int(graph.num_nodes) for graph in graphs),
        "annotated_person_nodes": sum(int(graph.loss_mask.sum()) for graph in graphs),
        "feature_dimensions": {
            name: int(getattr(graphs[0], name).shape[1])
            for name in (
                "handcrafted_x",
                "roi_x",
                "visual_x",
                "visual_motion_x",
                "visual_temporal_x",
            )
            if hasattr(graphs[0], name)
        } if graphs else {},
        "motion_features": list(MOTION_FEATURE_NAMES),
        "temporal_clip_encoder": (
            TEMPORAL_CLIP_ENCODER
            if graphs and hasattr(graphs[0], "visual_temporal_x")
            else None
        ),
        "cache_path": str(path.as_posix()),
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
