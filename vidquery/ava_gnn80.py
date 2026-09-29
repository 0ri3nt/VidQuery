from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ava_labels import AVA_V22_ACTIONS, load_ava_v22_label_map

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

AVA80_LABEL_IDS = tuple(range(1, 81))
AVA80_LABELS = tuple(AVA_V22_ACTIONS[label_id] for label_id in AVA80_LABEL_IDS)
AVA80_MLP_SCHEMA = "ava-person-action-mlp80-v1"
AVA80_GNN_SCHEMA = "ava-person-action-pyg-gat80-v1"
AVA80_DATASET_SCHEMA = "ava-timestamp-graph80-v2"

COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "unknown",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(COCO_CLASS_NAMES)}

VISUAL_GRID_SIZE = 4
VISUAL_FEATURE_NAMES = tuple(
    [
        f"grid_{channel}_{row}_{column}"
        for channel in "rgb"
        for row in range(4)
        for column in range(4)
    ]
    + [f"mean_{channel}" for channel in "rgb"]
    + [f"std_{channel}" for channel in "rgb"]
)
AUDIO_FEATURE_NAMES = (
    "log_audio_segment_count",
    "log_audio_speaker_count",
    "log_audio_duration",
    "log_audio_text_characters",
)
TRANSCRIPT_HASH_BUCKETS = 16
CONTINUOUS_FEATURE_NAMES = (
    "bbox_x1_norm",
    "bbox_y1_norm",
    "bbox_x2_norm",
    "bbox_y2_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "bbox_aspect_ratio",
    "detector_confidence",
    "annotation_only",
    *(f"roi_{name}" for name in VISUAL_FEATURE_NAMES),
    *(f"frame_{name}" for name in VISUAL_FEATURE_NAMES),
    *AUDIO_FEATURE_NAMES,
    *(f"transcript_hash_{index}" for index in range(TRANSCRIPT_HASH_BUCKETS)),
)
EDGE_FEATURE_NAMES = (
    "relative_center_x",
    "relative_center_y",
    "center_distance",
    "bbox_iou",
    "log_area_ratio",
    "direction_x",
    "direction_y",
    "overlap",
)


@dataclass(frozen=True, slots=True)
class AVAPersonAnnotation:
    video_id: str
    timestamp: int
    entity_id: str
    bbox_normalized: tuple[float, float, float, float]
    action_ids: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def multi_hot_action_target(action_ids: Iterable[int]) -> tuple[float, ...]:
    selected = {int(action_id) for action_id in action_ids}
    invalid = selected.difference(AVA80_LABEL_IDS)
    if invalid:
        raise ValueError(f"invalid AVA v2.2 action IDs: {sorted(invalid)}")
    return tuple(float(action_id in selected) for action_id in AVA80_LABEL_IDS)


def load_available_annotations(
    annotation_paths: Sequence[Path], video_ids: set[str] | None = None
) -> dict[tuple[str, int], list[AVAPersonAnnotation]]:
    grouped: dict[
        tuple[str, int, str, tuple[float, float, float, float]], set[int]
    ] = defaultdict(set)
    for annotation_path in annotation_paths:
        with Path(annotation_path).open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) < 7 or (video_ids is not None and row[0] not in video_ids):
                    continue
                bbox = tuple(float(value) for value in row[2:6])
                if len(bbox) != 4:
                    continue
                entity_id = row[7] if len(row) > 7 else ":".join(row[2:6])
                grouped[(row[0], int(row[1]), entity_id, bbox)].add(int(row[6]))
    by_timestamp: dict[tuple[str, int], list[AVAPersonAnnotation]] = defaultdict(list)
    for (video_id, timestamp, entity_id, bbox), action_ids in grouped.items():
        by_timestamp[(video_id, timestamp)].append(
            AVAPersonAnnotation(
                video_id=video_id,
                timestamp=timestamp,
                entity_id=entity_id,
                bbox_normalized=bbox,
                action_ids=tuple(sorted(action_ids)),
            )
        )
    for timestamp_annotations in by_timestamp.values():
        timestamp_annotations.sort(key=lambda item: (item.entity_id, item.bbox_normalized))
    return dict(by_timestamp)


def ava_dataset_coverage(
    annotation_paths: Sequence[Path], available_video_ids: set[str]
) -> dict[str, Any]:
    official_counts: Counter[int] = Counter()
    available_counts: Counter[int] = Counter()
    official_videos: set[str] = set()
    official_people: set[tuple[str, ...]] = set()
    available_people: set[tuple[str, ...]] = set()
    for annotation_path in annotation_paths:
        with Path(annotation_path).open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) < 7:
                    continue
                action_id = int(row[6])
                person_key = tuple(row[:6] + ([row[7]] if len(row) > 7 else []))
                official_counts[action_id] += 1
                official_videos.add(row[0])
                official_people.add(person_key)
                if row[0] in available_video_ids:
                    available_counts[action_id] += 1
                    available_people.add(person_key)

    def rows(counts: Counter[int]) -> list[dict[str, Any]]:
        return [
            {
                "action_id": action_id,
                "canonical_name": AVA_V22_ACTIONS[action_id],
                "official_name": load_ava_v22_label_map()[action_id - 1].official_name,
                "samples": counts[action_id],
            }
            for action_id in AVA80_LABEL_IDS
        ]

    missing = [action_id for action_id in AVA80_LABEL_IDS if not available_counts[action_id]]
    extremely_rare = [
        action_id for action_id in AVA80_LABEL_IDS if 0 < available_counts[action_id] <= 5
    ]
    return {
        "schema_version": "ava-v2.2-coverage-v1",
        "official_annotations": {
            "videos": len(official_videos),
            "annotated_persons": len(official_people),
            "action_rows": sum(official_counts.values()),
            "per_action": rows(official_counts),
        },
        "available_media_subset": {
            "video_ids": sorted(available_video_ids),
            "videos": len(available_video_ids),
            "annotated_persons": len(available_people),
            "action_rows": sum(available_counts.values()),
            "actions_present": 80 - len(missing),
            "missing_action_ids": missing,
            "extremely_rare_action_ids_at_most_5_samples": extremely_rare,
            "per_action": rows(available_counts),
        },
    }


def strict_video_split(video_ids: Iterable[str], seed: int = 2026) -> dict[str, list[str]]:
    unique = sorted(set(video_ids))
    if len(unique) < 5:
        raise ValueError("AVA 80-label training requires at least five independent videos")
    ordered = sorted(
        unique,
        key=lambda video_id: hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest(),
    )
    split = {
        "train": sorted(ordered[:-2]),
        "validation": [ordered[-2]],
        "test": [ordered[-1]],
    }
    sets = [set(values) for values in split.values()]
    if any(left.intersection(right) for i, left in enumerate(sets) for right in sets[i + 1 :]):
        raise RuntimeError("video-level split leakage detected")
    return split


def _image_descriptor(image: Any) -> tuple[float, ...]:
    import numpy as np
    from PIL import Image

    if image is None:
        return (0.0,) * len(VISUAL_FEATURE_NAMES)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    if image.width < 1 or image.height < 1:
        return (0.0,) * len(VISUAL_FEATURE_NAMES)
    array = np.asarray(image, dtype=np.float32) / 255.0
    pooled = np.asarray(
        image.resize((VISUAL_GRID_SIZE, VISUAL_GRID_SIZE), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    channel_first = pooled.transpose(2, 0, 1).reshape(-1)
    values = [*channel_first, *array.mean(axis=(0, 1)), *array.std(axis=(0, 1))]
    return tuple(float(value) for value in values)


def _aligned_context(audio_segments: Sequence[dict[str, Any]]) -> tuple[float, ...]:
    segments = [segment for segment in audio_segments if isinstance(segment, dict)]
    speakers = {str(segment.get("speaker", "")) for segment in segments if segment.get("speaker")}
    duration = sum(
        max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
        for segment in segments
    )
    text = " ".join(str(segment.get("text", "")) for segment in segments).strip().lower()
    audio = (
        math.log1p(len(segments)),
        math.log1p(len(speakers)),
        math.log1p(duration),
        math.log1p(len(text)),
    )
    buckets = [0.0] * TRANSCRIPT_HASH_BUCKETS
    sanitized = "".join(character if character.isalnum() else " " for character in text)
    tokens = sanitized.split()
    for token in tokens:
        index = int(hashlib.sha1(token.encode()).hexdigest()[:8], 16) % TRANSCRIPT_HASH_BUCKETS
        buckets[index] += 1.0
    scale = max(1.0, math.sqrt(sum(value * value for value in buckets)))
    return (*audio, *(value / scale for value in buckets))


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _edge_features(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    relative_x = right_center[0] - left_center[0]
    relative_y = right_center[1] - left_center[1]
    distance = math.hypot(relative_x, relative_y)
    iou = _bbox_iou(left, right)
    left_area = max(1e-6, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-6, (right[2] - right[0]) * (right[3] - right[1]))
    return (
        relative_x,
        relative_y,
        distance,
        iou,
        max(-4.0, min(4.0, math.log(right_area / left_area))),
        float(relative_x > 0) - float(relative_x < 0),
        float(relative_y > 0) - float(relative_y < 0),
        float(iou > 0),
    )


def _should_connect(left_person: bool, right_person: bool, distance: float) -> bool:
    if left_person and right_person:
        return True
    if left_person or right_person:
        return distance <= 0.65
    return distance <= 0.30


def build_pyg_timestamp_graph(
    *,
    video_id: str,
    timestamp: float,
    nodes: Sequence[dict[str, Any]],
    audio_segments: Sequence[dict[str, Any]],
    width: int,
    height: int,
    frame_image: Any = None,
    require_targets: bool = True,
) -> Any:
    import numpy as np
    import torch
    from PIL import Image
    from torch_geometric.data import Data

    safe_width = max(1.0, float(width))
    safe_height = max(1.0, float(height))
    if frame_image is not None and not isinstance(frame_image, Image.Image):
        frame_image = Image.fromarray(frame_image)
    frame_visual = _image_descriptor(frame_image)
    aligned = _aligned_context(audio_segments)
    continuous: list[tuple[float, ...]] = []
    class_indices: list[int] = []
    normalized_boxes: list[tuple[float, float, float, float]] = []
    labels: list[tuple[float, ...]] = []
    person_mask: list[bool] = []
    loss_mask: list[bool] = []
    node_ids: list[str] = []
    class_names: list[str] = []

    for index, node in enumerate(nodes):
        bbox = node.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        x1, x2 = sorted((max(0.0, x1), min(safe_width, x2)))
        y1, y2 = sorted((max(0.0, y1), min(safe_height, y2)))
        normalized = (x1 / safe_width, y1 / safe_height, x2 / safe_width, y2 / safe_height)
        box_width = max(0.0, normalized[2] - normalized[0])
        box_height = max(0.0, normalized[3] - normalized[1])
        if frame_image is not None and x2 > x1 and y2 > y1:
            crop = frame_image.crop((int(x1), int(y1), int(x2), int(y2)))
            roi_visual = _image_descriptor(crop)
        else:
            roi_visual = (0.0,) * len(VISUAL_FEATURE_NAMES)
        class_name = str(node.get("class_name", "unknown")).strip().lower()
        if class_name not in CLASS_TO_INDEX:
            class_name = "unknown"
        action_ids = tuple(int(value) for value in node.get("action_ids", []))
        annotated = bool(node.get("annotated", False))
        is_person = class_name == "person"
        features = (
            *normalized,
            box_width,
            box_height,
            box_width * box_height,
            min(8.0, box_width / max(box_height, 1e-6)),
            float(node.get("confidence", 0.0)),
            float(node.get("annotation_only", False)),
            *roi_visual,
            *frame_visual,
            *aligned,
        )
        if len(features) != len(CONTINUOUS_FEATURE_NAMES):
            raise RuntimeError("AVA80 continuous feature schema length mismatch")
        continuous.append(tuple(float(value) for value in features))
        class_indices.append(CLASS_TO_INDEX[class_name])
        normalized_boxes.append(normalized)
        labels.append(multi_hot_action_target(action_ids))
        person_mask.append(is_person)
        loss_mask.append(is_person and annotated)
        node_ids.append(str(node.get("node_id", index)))
        class_names.append(class_name)
    if not continuous or (require_targets and not any(loss_mask)) or not any(person_mask):
        return None

    edges: list[tuple[int, int]] = []
    edge_attributes: list[tuple[float, ...]] = []
    for left_index in range(len(continuous)):
        for right_index in range(left_index + 1, len(continuous)):
            features = _edge_features(normalized_boxes[left_index], normalized_boxes[right_index])
            if not _should_connect(
                person_mask[left_index], person_mask[right_index], float(features[2])
            ):
                continue
            edges.extend(((left_index, right_index), (right_index, left_index)))
            edge_attributes.extend(
                (
                    features,
                    _edge_features(
                        normalized_boxes[right_index], normalized_boxes[left_index]
                    ),
                )
            )
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_attributes, dtype=torch.float32)
        if edge_attributes
        else torch.empty((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32)
    )
    graph = Data(
        x=torch.tensor(np.asarray(continuous), dtype=torch.float32),
        class_ids=torch.tensor(class_indices, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(np.asarray(labels), dtype=torch.float32),
        person_mask=torch.tensor(person_mask, dtype=torch.bool),
        loss_mask=torch.tensor(loss_mask, dtype=torch.bool),
    )
    graph.video_id = video_id
    graph.timestamp = float(timestamp)
    graph.node_ids = node_ids
    graph.class_names = class_names
    return graph


def build_ava80_graph_dataset(
    *,
    fused_root: Path,
    frame_root: Path,
    annotation_paths: Sequence[Path],
    feature_cache_dir: Path,
    force_rebuild: bool = False,
) -> list[Any]:
    import torch
    from PIL import Image

    fused_paths = sorted(Path(fused_root).rglob("*.fused.json"))
    available_video_ids = {
        str(json.loads(path.read_text(encoding="utf-8")).get("video_id", path.stem))
        for path in fused_paths
    }
    annotations = load_available_annotations(annotation_paths, available_video_ids)
    fingerprint_payload = {
        "schema": AVA80_DATASET_SCHEMA,
        "fused": [
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in fused_paths
        ],
        "annotations": [
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in annotation_paths
        ],
        "continuous_features": list(CONTINUOUS_FEATURE_NAMES),
        "edge_features": list(EDGE_FEATURE_NAMES),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = feature_cache_dir / "ava80_graph_dataset.pt"
    cache_metadata_path = feature_cache_dir / "ava80_graph_dataset.metadata.json"
    if not force_rebuild and cache_path.is_file() and cache_metadata_path.is_file():
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") == AVA80_DATASET_SCHEMA:
                return list(payload["graphs"])

    frame_paths = {path.name: path for path in Path(frame_root).rglob("*.jpg")}
    graphs: list[Any] = []
    for fused_path in fused_paths:
        payload = json.loads(fused_path.read_text(encoding="utf-8"))
        video_id = str(payload.get("video_id", fused_path.name.removesuffix(".fused.json")))
        for frame in payload.get("frames", []):
            timestamp = int(float(frame.get("timestamp", 0)))
            frame_file = str(frame.get("frame_file", ""))
            image_path = frame_paths.get(frame_file)
            if image_path is None:
                continue
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                source_nodes = frame.get("nodes", [])
                graph_nodes: list[dict[str, Any]] = []
                used_entities: set[str] = set()
                timestamp_annotations = annotations.get((video_id, timestamp), [])
                annotation_by_entity = {
                    item.entity_id: item for item in timestamp_annotations
                }
                for fallback_id, node in enumerate(
                    source_nodes if isinstance(source_nodes, list) else []
                ):
                    if not isinstance(node, dict):
                        continue
                    matched = node.get("matched_annotation")
                    annotation: AVAPersonAnnotation | None = None
                    if isinstance(matched, dict):
                        entity_id = str(matched.get("entity_id", ""))
                        if entity_id not in used_entities:
                            annotation = annotation_by_entity.get(entity_id)
                            used_entities.add(entity_id)
                    action_ids = (
                        annotation.action_ids
                        if annotation is not None
                        else ()
                    )
                    graph_nodes.append(
                        {
                            "node_id": f"det:{node.get('node_id', fallback_id)}",
                            "class_name": node.get("class_name", "unknown"),
                            "confidence": node.get("confidence", 0.0),
                            "bbox": node.get("bbox", [0, 0, 0, 0]),
                            "annotated": bool(action_ids),
                            "annotation_only": False,
                            "action_ids": action_ids,
                        }
                    )
                for annotation in timestamp_annotations:
                    if annotation.entity_id in used_entities:
                        continue
                    x1, y1, x2, y2 = annotation.bbox_normalized
                    graph_nodes.append(
                        {
                            "node_id": f"ava:{annotation.entity_id}",
                            "class_name": "person",
                            "confidence": 0.0,
                            "bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
                            "annotated": True,
                            "annotation_only": True,
                            "action_ids": annotation.action_ids,
                        }
                    )
                audio_segments = frame.get("audio_segments", [])
                graph = build_pyg_timestamp_graph(
                    video_id=video_id,
                    timestamp=float(timestamp),
                    nodes=graph_nodes,
                    audio_segments=audio_segments if isinstance(audio_segments, list) else [],
                    width=width,
                    height=height,
                    frame_image=image,
                )
                if graph is not None:
                    graphs.append(graph)
    graphs.sort(key=lambda graph: (str(graph.video_id), float(graph.timestamp)))
    torch.save(
        {"schema_version": AVA80_DATASET_SCHEMA, "graphs": graphs},
        cache_path,
    )
    cache_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": AVA80_DATASET_SCHEMA,
                "fingerprint": fingerprint,
                "graphs": len(graphs),
                "annotated_person_nodes": sum(int(graph.loss_mask.sum()) for graph in graphs),
                "nodes": sum(int(graph.num_nodes) for graph in graphs),
                "directed_edges": sum(int(graph.edge_index.shape[1]) for graph in graphs),
                "video_ids": sorted({str(graph.video_id) for graph in graphs}),
                "cache_path": str(cache_path.as_posix()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return graphs


class AVA80MLP:
    def __new__(
        cls,
        continuous_features: int,
        class_count: int,
        output_labels: int = 80,
        class_embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.class_embedding = nn.Embedding(class_count, class_embedding_dim)
                self.network = nn.Sequential(
                    nn.Linear(continuous_features + class_embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_labels),
                )

            def forward(self, features: Any, class_index: Any) -> Any:
                embedded = self.class_embedding(class_index)
                return self.network(torch.cat((features, embedded), dim=-1))

        return _Model()


class AVA80GAT:
    def __new__(
        cls,
        continuous_features: int,
        edge_features: int,
        class_count: int,
        output_labels: int = 80,
        class_embedding_dim: int = 16,
        hidden_dim: int = 128,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        import torch
        from torch import nn
        from torch_geometric.nn import GATv2Conv

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.class_embedding = nn.Embedding(class_count, class_embedding_dim)
                self.node_projection = nn.Linear(
                    continuous_features + class_embedding_dim, hidden_dim
                )
                self.gat1 = GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_features,
                    dropout=dropout,
                )
                self.gat2 = GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_features,
                    dropout=dropout,
                )
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_dim, output_labels)

            def forward(
                self, features: Any, class_index: Any, edge_index: Any, edge_attr: Any
            ) -> Any:
                embedded = self.class_embedding(class_index)
                hidden = torch.relu(
                    self.node_projection(torch.cat((features, embedded), dim=-1))
                )
                first = self.dropout(
                    torch.relu(self.gat1(hidden, edge_index, edge_attr=edge_attr))
                )
                hidden = self.norm1(hidden + first)
                second = self.dropout(
                    torch.relu(self.gat2(hidden, edge_index, edge_attr=edge_attr))
                )
                hidden = self.norm2(hidden + second)
                return self.classifier(hidden)

        return _Model()


def masked_multilabel_loss(logits: Any, targets: Any, mask: Any, loss_function: Any) -> Any:
    if int(mask.sum()) == 0:
        raise ValueError("batch contains no annotated person nodes")
    return loss_function(logits[mask], targets[mask])


def _calibrate_thresholds(targets: Any, probabilities: Any) -> Any:
    import numpy as np

    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    thresholds = np.ones(80, dtype=np.float32)
    for index in range(80):
        truth = targets[:, index] == 1
        if not truth.any():
            continue
        best = (0.0, 0.5)
        for threshold in np.arange(0.05, 0.951, 0.025):
            predicted = probabilities[:, index] >= threshold
            tp = int(np.logical_and(truth, predicted).sum())
            fp = int(np.logical_and(~truth, predicted).sum())
            fn = int(np.logical_and(truth, ~predicted).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            candidate = (f1, -abs(float(threshold) - 0.5))
            current = (best[0], -abs(best[1] - 0.5))
            if candidate > current:
                best = (f1, float(threshold))
        thresholds[index] = best[1]
    return thresholds


def _average_precision(truth: Any, scores: Any) -> float | None:
    import numpy as np

    truth = np.asarray(truth, dtype=bool)
    if not truth.any():
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = truth[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / truth.sum())


def decode_ava_predictions(probabilities: Any, thresholds: Any) -> Any:
    """Apply per-action thresholds and AVA category cardinality constraints."""

    import numpy as np

    scores = np.asarray(probabilities, dtype=np.float32)
    squeeze = scores.ndim == 1
    if squeeze:
        scores = scores[None, :]
    selected = scores >= np.asarray(thresholds, dtype=np.float32)[None, :]
    # AVA annotates one pose/movement plus a small number of concurrent
    # person-object and person-person actions. Keep the strongest validated
    # candidates in each official label group to avoid impossible label floods.
    for row in range(len(scores)):
        for start, end, maximum in ((0, 14, 1), (14, 63, 2), (63, 80, 2)):
            candidates = np.flatnonzero(selected[row, start:end]) + start
            if len(candidates) <= maximum:
                continue
            keep = candidates[np.argsort(-scores[row, candidates], kind="stable")[:maximum]]
            selected[row, candidates] = False
            selected[row, keep] = True
    return selected[0] if squeeze else selected


def classification_metrics(targets: Any, probabilities: Any, thresholds: Any) -> dict[str, Any]:
    import numpy as np

    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predicted = decode_ava_predictions(probabilities, thresholds)
    truth = targets == 1
    per_class: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    for index, label in enumerate(AVA80_LABELS):
        tp = int(np.logical_and(truth[:, index], predicted[:, index]).sum())
        fp = int(np.logical_and(~truth[:, index], predicted[:, index]).sum())
        fn = int(np.logical_and(truth[:, index], ~predicted[:, index]).sum())
        support = int(truth[:, index].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        average_precision = _average_precision(truth[:, index], probabilities[:, index])
        per_class.append(
            {
                "action_id": index + 1,
                "label": label,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
                "average_precision": (
                    round(average_precision, 6) if average_precision is not None else None
                ),
                "threshold": round(float(thresholds[index]), 4),
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    supported = [row for row in per_class if row["support"] > 0]
    return {
        "macro_f1": round(sum(float(row["f1"]) for row in per_class) / 80, 6),
        "macro_f1_supported_classes": round(
            sum(float(row["f1"]) for row in supported) / max(1, len(supported)), 6
        ),
        "micro_f1": round(micro_f1, 6),
        "mean_average_precision": round(
            sum(float(row["average_precision"]) for row in supported) / max(1, len(supported)),
            6,
        ),
        "supported_classes": len(supported),
        "per_class": per_class,
    }


def _seed_training(seed: int) -> tuple[Any, Any, Any]:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np, torch, torch.nn


def _training_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for GNN training but is unavailable")
    return torch.device(requested)


def _split_graphs(graphs: Sequence[Any], split: dict[str, list[str]]) -> dict[str, list[Any]]:
    selected = {
        name: [graph for graph in graphs if str(graph.video_id) in video_ids]
        for name, video_ids in split.items()
    }
    if any(not values for values in selected.values()):
        raise ValueError("one or more video-level graph splits are empty")
    return selected


def _normalization(graphs: Sequence[Any], torch: Any) -> tuple[Any, Any]:
    values = torch.cat([graph.x for graph in graphs], dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std[std < 1e-6] = 1.0
    return mean, std


def _targets_for_graphs(graphs: Sequence[Any], torch: Any) -> Any:
    return torch.cat([graph.y[graph.loss_mask] for graph in graphs], dim=0)


def _positive_weights(targets: Any, torch: Any) -> Any:
    positives = targets.sum(dim=0)
    negatives = len(targets) - positives
    return torch.clamp(negatives / torch.clamp(positives, min=1), max=30.0)


def _state_copy(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _hardware(torch: Any, device: Any) -> dict[str, str]:
    return {
        "device": str(device),
        "name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine()
        ),
    }


def _class_frequency(graphs: Sequence[Any]) -> list[dict[str, Any]]:
    import torch

    targets = _targets_for_graphs(graphs, torch)
    return [
        {
            "action_id": index + 1,
            "label": AVA80_LABELS[index],
            "positive": int(targets[:, index].sum()),
            "negative": int(len(targets) - targets[:, index].sum()),
        }
        for index in range(80)
    ]


def _measure_latency(callable_: Any, repeats: int, torch: Any, device: Any) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000 / max(1, repeats)


def train_ava80_mlp(
    *,
    graphs: Sequence[Any],
    split: dict[str, list[str]],
    checkpoint_path: Path,
    metadata_path: Path,
    seed: int = 2026,
    epochs: int = 100,
    batch_size: int = 128,
    patience: int = 15,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
) -> dict[str, Any]:
    np, torch, nn = _seed_training(seed)
    from torch.utils.data import DataLoader, TensorDataset

    started = time.perf_counter()
    device = _training_device(torch, device_name)
    split_graphs = _split_graphs(graphs, split)
    feature_mean, feature_std = _normalization(split_graphs["train"], torch)

    def person_tensors(selected: Sequence[Any]) -> tuple[Any, Any, Any]:
        features = torch.cat([graph.x[graph.loss_mask] for graph in selected], dim=0)
        classes = torch.cat(
            [graph.class_ids[graph.loss_mask] for graph in selected], dim=0
        )
        targets = _targets_for_graphs(selected, torch)
        return (features - feature_mean) / feature_std, classes, targets

    train_x, train_classes, train_y = person_tensors(split_graphs["train"])
    validation_x, validation_classes, validation_y = person_tensors(
        split_graphs["validation"]
    )
    test_x, test_classes, test_y = person_tensors(split_graphs["test"])
    positive_weights = _positive_weights(train_y, torch)
    model: Any = AVA80MLP(len(CONTINUOUS_FEATURE_NAMES), len(COCO_CLASS_NAMES))
    model = model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_classes, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for features, classes, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(
                model(features.to(device), classes.to(device)), targets.to(device)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(
                    model(validation_x.to(device), validation_classes.to(device)),
                    validation_y.to(device),
                ).cpu()
            )
        training_loss = sum(losses) / len(losses)
        history.append(
            {
                "epoch": epoch,
                "training_loss": round(training_loss, 8),
                "validation_loss": round(validation_loss, 8),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = _state_copy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("80-label MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(
            model(validation_x.to(device), validation_classes.to(device))
        ).cpu().numpy()
        test_probabilities = torch.sigmoid(
            model(test_x.to(device), test_classes.to(device))
        ).cpu().numpy()
    thresholds = _calibrate_thresholds(validation_y.numpy(), validation_probabilities)
    validation_metrics = classification_metrics(
        validation_y.numpy(), validation_probabilities, thresholds
    )
    test_metrics = classification_metrics(test_y.numpy(), test_probabilities, thresholds)
    with torch.no_grad():
        latency_ms = _measure_latency(
            lambda: model(test_x.to(device), test_classes.to(device)), 20, torch, device
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": AVA80_MLP_SCHEMA,
            "state_dict": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": list(AVA80_LABELS),
            "label_ids": list(AVA80_LABEL_IDS),
            "thresholds": torch.from_numpy(thresholds),
            "continuous_feature_names": list(CONTINUOUS_FEATURE_NAMES),
            "class_names": list(COCO_CLASS_NAMES),
        },
        checkpoint_path,
    )
    metadata = {
        "schema_version": AVA80_MLP_SCHEMA,
        "task": "80-label multilabel person-centric AVA v2.2 action classification",
        "model_type": "MLP baseline using the same person node features as the GNN",
        "split": split,
        "sample_counts": {
            name: int(_targets_for_graphs(selected, torch).shape[0])
            for name, selected in split_graphs.items()
        },
        "class_frequency": {
            name: _class_frequency(selected) for name, selected in split_graphs.items()
        },
        "feature_schema": {
            "continuous": list(CONTINUOUS_FEATURE_NAMES),
            "class_names": list(COCO_CLASS_NAMES),
            "class_embedding_dimension": 16,
        },
        "architecture": {"hidden_dimensions": [128, 128], "dropout": 0.2, "outputs": 80},
        "loss": "BCEWithLogitsLoss",
        "class_imbalance": {
            "method": "training-split positive weights clipped at 30",
            "positive_weights": [round(float(value), 6) for value in positive_weights],
        },
        "postprocessing": (
            "per-class validation thresholds, then top-1 movement, top-2 object actions, "
            "top-2 person interactions per person"
        ),
        "thresholds": {
            label: round(float(thresholds[index]), 4)
            for index, label in enumerate(AVA80_LABELS)
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "inference_latency": {
            "milliseconds_per_held_out_person_batch": round(latency_ms, 4),
            "held_out_persons_per_batch": int(len(test_y)),
        },
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "history": history,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "hardware": _hardware(torch, device),
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation": {"status": "validated", "held_out_video_ids": split["test"]},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def train_ava80_gnn(
    *,
    graphs: Sequence[Any],
    split: dict[str, list[str]],
    checkpoint_path: Path,
    metadata_path: Path,
    seed: int = 2026,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
    model_version: str = "ava80-gat-v1",
) -> dict[str, Any]:
    np, torch, nn = _seed_training(seed)
    from torch_geometric.loader import DataLoader

    started = time.perf_counter()
    device = _training_device(torch, device_name)
    split_graphs = _split_graphs(graphs, split)
    feature_mean, feature_std = _normalization(split_graphs["train"], torch)
    train_targets = _targets_for_graphs(split_graphs["train"], torch)
    validation_targets = _targets_for_graphs(split_graphs["validation"], torch)
    test_targets = _targets_for_graphs(split_graphs["test"], torch)
    positive_weights = _positive_weights(train_targets, torch)
    model: Any = AVA80GAT(
        len(CONTINUOUS_FEATURE_NAMES), len(EDGE_FEATURE_NAMES), len(COCO_CLASS_NAMES)
    )
    model = model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    mean = feature_mean.to(device)
    std = feature_std.to(device)

    def forward(batch: Any, *, remove_edges: bool = False) -> Any:
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        if remove_edges:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty(
                (0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32, device=device
            )
        return model(
            (batch.x - mean) / std,
            batch.class_ids,
            edge_index,
            edge_attr,
        )

    def loader_for(name: str, *, shuffle: bool = False) -> Any:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            split_graphs[name],
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def probabilities(name: str, *, remove_edges: bool = False) -> Any:
        output: list[Any] = []
        model.eval()
        with torch.no_grad():
            for raw_batch in loader_for(name):
                batch = raw_batch.to(device)
                predicted = torch.sigmoid(forward(batch, remove_edges=remove_edges))
                output.append(predicted[batch.loss_mask].cpu())
        return torch.cat(output).numpy()

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in loader_for("train", shuffle=True):
            batch = raw_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_multilabel_loss(forward(batch), batch.y, batch.loss_mask, loss_function)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for raw_batch in loader_for("validation"):
                batch = raw_batch.to(device)
                validation_losses.append(
                    float(
                        masked_multilabel_loss(
                            forward(batch), batch.y, batch.loss_mask, loss_function
                        ).cpu()
                    )
                )
        training_loss = sum(losses) / len(losses)
        validation_loss = sum(validation_losses) / len(validation_losses)
        history.append(
            {
                "epoch": epoch,
                "training_loss": round(training_loss, 8),
                "validation_loss": round(validation_loss, 8),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = _state_copy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("80-label GNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities = probabilities("validation")
    test_probabilities = probabilities("test")
    thresholds = _calibrate_thresholds(validation_targets.numpy(), validation_probabilities)
    validation_metrics = classification_metrics(
        validation_targets.numpy(), validation_probabilities, thresholds
    )
    test_metrics = classification_metrics(test_targets.numpy(), test_probabilities, thresholds)
    no_edge_metrics = classification_metrics(
        test_targets.numpy(), probabilities("test", remove_edges=True), thresholds
    )
    test_batches = list(loader_for("test"))
    latency_samples: list[float] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in test_batches:
            batch = raw_batch.to(device)
            latency_samples.append(
                _measure_latency(lambda selected=batch: forward(selected), 5, torch, device)
                / max(1, int(batch.num_graphs))
            )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": AVA80_GNN_SCHEMA,
            "model_version": model_version,
            "state_dict": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "labels": list(AVA80_LABELS),
            "label_ids": list(AVA80_LABEL_IDS),
            "thresholds": torch.from_numpy(thresholds),
            "continuous_feature_names": list(CONTINUOUS_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "class_names": list(COCO_CLASS_NAMES),
        },
        checkpoint_path,
    )
    metadata = {
        "schema_version": AVA80_GNN_SCHEMA,
        "model_version": model_version,
        "task": "80-label multilabel person-centric AVA v2.2 action classification",
        "model_type": "PyTorch Geometric two-layer edge-aware GATv2",
        "semantic_scope": (
            "predicts AVA actions on person nodes; it does not predict explicit object targets"
        ),
        "split": split,
        "graph_counts": {name: len(selected) for name, selected in split_graphs.items()},
        "sample_counts": {
            name: int(_targets_for_graphs(selected, torch).shape[0])
            for name, selected in split_graphs.items()
        },
        "class_frequency": {
            name: _class_frequency(selected) for name, selected in split_graphs.items()
        },
        "feature_schema": {
            "continuous": list(CONTINUOUS_FEATURE_NAMES),
            "edge": list(EDGE_FEATURE_NAMES),
            "class_names": list(COCO_CLASS_NAMES),
            "class_embedding_dimension": 16,
            "object_nodes_masked_from_loss": True,
        },
        "architecture": {
            "node_projection": 128,
            "layers": ["GATv2Conv(128, heads=2)", "GATv2Conv(128, heads=2)"],
            "edge_features_consumed": True,
            "residual_connections": True,
            "dropout": 0.2,
            "outputs": 80,
        },
        "loss": "BCEWithLogitsLoss over annotated person nodes only",
        "class_imbalance": {
            "method": "training-split positive weights clipped at 30",
            "positive_weights": [round(float(value), 6) for value in positive_weights],
        },
        "postprocessing": (
            "per-class validation thresholds, then top-1 movement, top-2 object actions, "
            "top-2 person interactions per person"
        ),
        "thresholds": {
            label: round(float(thresholds[index]), 4)
            for index, label in enumerate(AVA80_LABELS)
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "no_edge_ablation_test_metrics": no_edge_metrics,
        "inference_latency": {
            "mean_milliseconds_per_graph": round(
                sum(latency_samples) / max(1, len(latency_samples)), 4
            ),
            "batch_size": batch_size,
        },
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "history": history,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "hardware": _hardware(torch, device),
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation": {"status": "validated", "held_out_video_ids": split["test"]},
        "limitations": [
            "Only five local AVA videos are available for feature extraction and training.",
            "Classes absent from validation use threshold 1.0 and cannot be claimed as learned.",
            "The model predicts person-centric action labels, never explicit target identities.",
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def train_ava80_suite(
    *,
    fused_root: Path,
    frame_root: Path,
    annotation_paths: Sequence[Path],
    feature_cache_dir: Path,
    output_dir: Path,
    existing_six_label_report: Path | None = None,
    seed: int = 2026,
    epochs: int = 100,
    patience: int = 15,
    device_name: str = "auto",
    model_version: str = "ava80-gat-v1",
    force_rebuild_cache: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    available_video_ids = {
        str(json.loads(path.read_text(encoding="utf-8")).get("video_id", path.stem))
        for path in Path(fused_root).rglob("*.fused.json")
    }
    coverage = ava_dataset_coverage(annotation_paths, available_video_ids)
    (output_dir / "class_frequency.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    label_map = {
        "source": "https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt",
        "version": "2.2",
        "labels": [asdict(label) for label in load_ava_v22_label_map()],
    }
    (output_dir / "ava_v2.2_label_map.json").write_text(
        json.dumps(label_map, indent=2) + "\n", encoding="utf-8"
    )
    graphs = build_ava80_graph_dataset(
        fused_root=fused_root,
        frame_root=frame_root,
        annotation_paths=annotation_paths,
        feature_cache_dir=feature_cache_dir,
        force_rebuild=force_rebuild_cache,
    )
    split = strict_video_split(str(graph.video_id) for graph in graphs)
    split_manifest = {
        "schema_version": "ava-video-split-v1",
        "seed": seed,
        "policy": "SHA-256 seeded ordering; last video validation, final video test",
        **split,
        "leakage_check": "passed",
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    feature_schema = {
        "schema_version": AVA80_DATASET_SCHEMA,
        "node_classes": list(COCO_CLASS_NAMES),
        "continuous_node_features": list(CONTINUOUS_FEATURE_NAMES),
        "roi_visual_features": [
            name for name in CONTINUOUS_FEATURE_NAMES if name.startswith("roi_")
        ],
        "whole_frame_visual_features": [
            name for name in CONTINUOUS_FEATURE_NAMES if name.startswith("frame_")
        ],
        "aligned_audio_transcript_features": [
            name
            for name in CONTINUOUS_FEATURE_NAMES
            if name.startswith("log_audio_") or name.startswith("transcript_hash_")
        ],
        "edge_features": list(EDGE_FEATURE_NAMES),
        "targets": "80-dimensional AVA v2.2 multi-hot vector per annotated person node",
        "loss_mask": "annotated person nodes only; object and unannotated person nodes excluded",
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2) + "\n", encoding="utf-8"
    )

    mlp_checkpoint = output_dir / "ava80_mlp.pt"
    mlp_metadata_path = output_dir / "ava80_mlp.metadata.json"
    mlp_metadata = train_ava80_mlp(
        graphs=graphs,
        split=split,
        checkpoint_path=mlp_checkpoint,
        metadata_path=mlp_metadata_path,
        seed=seed,
        epochs=epochs,
        patience=patience,
        device_name=device_name,
    )
    gnn_checkpoint = output_dir / "ava80_gat.pt"
    gnn_metadata_path = output_dir / "ava80_gat.metadata.json"
    gnn_metadata = train_ava80_gnn(
        graphs=graphs,
        split=split,
        checkpoint_path=gnn_checkpoint,
        metadata_path=gnn_metadata_path,
        seed=seed,
        epochs=epochs,
        patience=patience,
        device_name=device_name,
        model_version=model_version,
    )

    import torch

    checkpoint_payload = torch.load(gnn_checkpoint, map_location="cpu", weights_only=True)
    normalization = {
        "continuous_feature_names": list(CONTINUOUS_FEATURE_NAMES),
        "mean": [round(float(value), 8) for value in checkpoint_payload["feature_mean"]],
        "standard_deviation": [
            round(float(value), 8) for value in checkpoint_payload["feature_std"]
        ],
        "fit_scope": "training-video nodes only",
    }
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "selection_scope": "validation video only",
                "absent_validation_classes": "threshold 1.0 (disabled)",
                "mlp80": mlp_metadata["thresholds"],
                "gnn80": gnn_metadata["thresholds"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(
            {"mlp80": mlp_metadata["history"], "gnn80": gnn_metadata["history"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_version": model_version,
                "seed": seed,
                "epochs": epochs,
                "patience": patience,
                "device": device_name,
                "mlp80": mlp_metadata["architecture"],
                "gnn80": gnn_metadata["architecture"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    existing_six: dict[str, Any] | None = None
    if existing_six_label_report is not None and existing_six_label_report.is_file():
        report = json.loads(existing_six_label_report.read_text(encoding="utf-8"))
        existing_six = {
            "labels": report.get("labels"),
            "test_metrics": report.get("test_metrics"),
            "note": "six-label metric scope is not directly comparable to the 80-label task",
        }
    comparison = {
        "existing_six_label_mlp": existing_six,
        "new_80_label_mlp": mlp_metadata["test_metrics"],
        "new_80_label_gnn": gnn_metadata["test_metrics"],
        "gnn_no_edge_ablation": gnn_metadata["no_edge_ablation_test_metrics"],
        "held_out_video_ids": split["test"],
    }
    result = {
        "schema_version": "ava80-training-suite-v1",
        "coverage": coverage,
        "split": split_manifest,
        "graphs": len(graphs),
        "annotated_person_nodes": sum(int(graph.loss_mask.sum()) for graph in graphs),
        "mlp80": mlp_metadata,
        "gnn80": gnn_metadata,
        "comparison": comparison,
        "artifacts": {
            "output_dir": str(output_dir.as_posix()),
            "feature_cache_dir": str(feature_cache_dir.as_posix()),
            "mlp_checkpoint": str(mlp_checkpoint.as_posix()),
            "gnn_checkpoint": str(gnn_checkpoint.as_posix()),
        },
    }
    (output_dir / "held_out_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def ava80_gnn_status(
    checkpoint_path: Path,
    metadata_path: Path,
    expected_model_version: str | None = None,
) -> str:
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        return "unavailable_no_valid_checkpoint"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        schema_version = metadata.get("schema_version")
        if schema_version not in {AVA80_GNN_SCHEMA, "ava80-gat-ablation-v2"}:
            return "unavailable_schema_mismatch"
        if expected_model_version and metadata.get("model_version") != expected_model_version:
            return "unavailable_model_version_mismatch"
        if schema_version == AVA80_GNN_SCHEMA:
            if metadata.get("validation", {}).get("status") != "validated":
                return "unavailable_not_validated"
            if len(metadata.get("thresholds", {})) != 80:
                return "unavailable_threshold_schema_mismatch"
        elif not isinstance(metadata.get("validation_metrics"), dict):
            return "unavailable_not_validated"
        if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
            return "unavailable_checksum_mismatch"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "unavailable_invalid_metadata"
    return "available_validated"


class AVA80GNNPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        metadata_path: Path,
        *,
        device: str = "cpu",
        expected_model_version: str | None = None,
    ) -> None:
        import torch

        status = ava80_gnn_status(checkpoint_path, metadata_path, expected_model_version)
        if status != "available_validated":
            raise ValueError(f"AVA80 GNN checkpoint cannot be activated: {status}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        schema_version = payload.get("schema_version")
        if schema_version not in {AVA80_GNN_SCHEMA, "ava80-gat-ablation-v2"}:
            raise ValueError("AVA80 GNN checkpoint schema mismatch")
        if payload.get("labels") != list(AVA80_LABELS):
            raise ValueError("AVA80 GNN label map mismatch")
        if expected_model_version and payload.get("model_version") != expected_model_version:
            raise ValueError("AVA80 GNN model version mismatch")
        self.device = torch.device(device)
        self.feature_attribute = "x"
        self._visual_embedder: Any | None = None
        if schema_version == AVA80_GNN_SCHEMA:
            if payload.get("continuous_feature_names") != list(CONTINUOUS_FEATURE_NAMES):
                raise ValueError("AVA80 GNN node feature schema mismatch")
            if payload.get("edge_feature_names") != list(EDGE_FEATURE_NAMES):
                raise ValueError("AVA80 GNN edge feature schema mismatch")
            if payload.get("class_names") != list(COCO_CLASS_NAMES):
                raise ValueError("AVA80 GNN class vocabulary mismatch")
            built_model: Any = AVA80GAT(
                len(CONTINUOUS_FEATURE_NAMES), len(EDGE_FEATURE_NAMES), len(COCO_CLASS_NAMES)
            )
        else:
            from .ava_ablation import StaticActionModel

            self.feature_attribute = str(payload.get("feature_attribute", ""))
            if self.feature_attribute not in {
                "handcrafted_x",
                "roi_x",
                "visual_x",
                "visual_temporal_x",
            }:
                raise ValueError("improved AVA80 feature attribute is not inference-compatible")
            feature_dimension = int(payload.get("feature_dimension", 0))
            if feature_dimension <= 0:
                raise ValueError("improved AVA80 feature dimension is invalid")
            built_model = StaticActionModel(
                feature_dimension,
                use_edges=bool(payload.get("use_edges")),
                mlp_only=bool(payload.get("mlp_only")),
            )
        self.model = built_model.to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.feature_mean = payload["feature_mean"].to(self.device)
        self.feature_std = payload["feature_std"].to(self.device)
        self.thresholds = payload["thresholds"].to(self.device)

    def _features(self, graph: Any, frame_image: Any = None) -> Any:
        import torch

        if self.feature_attribute == "x":
            return graph.x
        if hasattr(graph, self.feature_attribute):
            return getattr(graph, self.feature_attribute)
        if self.feature_attribute == "visual_temporal_x":
            raise ValueError(
                "visual_temporal_x requires a validated cached short-clip embedding"
            )
        if frame_image is None:
            raise ValueError(
                f"{self.feature_attribute} requires a source frame or cached visual embeddings"
            )
        if self._visual_embedder is None:
            from .ava_improved import FrozenResNet18Embedder

            self._visual_embedder = FrozenResNet18Embedder(str(self.device))
        from PIL import Image

        if not isinstance(frame_image, Image.Image):
            frame_image = Image.fromarray(frame_image)
        frame_image = frame_image.convert("RGB")
        crops = []
        for box in graph.x[:, :4]:
            x1, y1, x2, y2 = [float(value) for value in box]
            crops.append(frame_image.crop((
                int(x1 * frame_image.width),
                int(y1 * frame_image.height),
                max(int(x2 * frame_image.width), int(x1 * frame_image.width) + 1),
                max(int(y2 * frame_image.height), int(y1 * frame_image.height) + 1),
            )))
        roi = self._visual_embedder.encode(crops)
        if self.feature_attribute == "roi_x":
            return torch.cat((graph.x.cpu(), roi), dim=1)
        frame = self._visual_embedder.encode([frame_image]).repeat(int(graph.num_nodes), 1)
        return torch.cat((graph.x.cpu(), roi, frame), dim=1)

    def predict_graph(
        self, graph: Any, *, frame_image: Any = None
    ) -> dict[str, list[dict[str, float | str]]]:
        import torch

        features = self._features(graph, frame_image).to(self.device)
        graph = graph.to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.model(
                    (features - self.feature_mean) / self.feature_std,
                    graph.class_ids,
                    graph.edge_index,
                    graph.edge_attr,
                )
            )
        decoded = decode_ava_predictions(
            probabilities.detach().cpu().numpy(), self.thresholds.detach().cpu().numpy()
        )
        output: dict[str, list[dict[str, float | str]]] = {}
        person_indices = torch.nonzero(graph.person_mask, as_tuple=False).flatten().tolist()
        for node_index in person_indices:
            node_predictions: list[dict[str, float | str]] = [
                {"label": label, "confidence": float(probabilities[node_index, index])}
                for index, label in enumerate(AVA80_LABELS)
                if decoded[node_index, index]
            ]
            output[str(graph.node_ids[node_index])] = node_predictions
        return output


def build_ava80_action_index(
    predictor: AVA80GNNPredictor,
    *,
    graph_cache_path: Path,
    segment_duration: float = 5.0,
) -> dict[tuple[str, float], dict[str, float]]:
    import torch

    if segment_duration <= 0:
        raise ValueError("segment_duration must be positive")
    if not graph_cache_path.is_file():
        raise ValueError(f"AVA80 graph cache is missing: {graph_cache_path}")
    payload = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") not in {
        AVA80_DATASET_SCHEMA,
        "ava80-pretrained-visual-motion-v1",
    }:
        raise ValueError("AVA80 graph cache schema mismatch")
    index: dict[tuple[str, float], dict[str, float]] = {}
    for graph in payload["graphs"]:
        segment_start = math.floor(float(graph.timestamp) / segment_duration) * segment_duration
        action_scores = index.setdefault((str(graph.video_id), segment_start), {})
        for person_predictions in predictor.predict_graph(graph).values():
            for prediction in person_predictions:
                label = str(prediction["label"])
                action_scores[label] = max(
                    action_scores.get(label, 0.0), float(prediction["confidence"])
                )
    return index


TARGET_RESOLVER_ACTIONS = {
    "carry/hold": "object",
    "point to": "object",
    "write": "object",
    "talk to": "person",
    "listen to": "person",
    "watch person": "person",
    "give/serve": "person",
    "grab person": "person",
    "hand shake": "person",
    "hug": "person",
    "kick person": "person",
    "kiss": "person",
    "lift person": "person",
    "push person": "person",
    "sing to": "person",
    "take from person": "person",
}

TARGET_RESOLVER_PREDICATES = {
    "carry/hold": "carry_hold",
    "point to": "point_to",
    "write": "write_on",
    "talk to": "talk_to",
    "listen to": "listen_to",
    "watch person": "watch_person",
    "give/serve": "give_serve_to",
    "grab person": "grab_person",
    "hand shake": "hand_shake",
    "hug": "hug",
    "kick person": "kick_person",
    "kiss": "kiss",
    "lift person": "lift_person",
    "push person": "push_person",
    "sing to": "sing_to",
    "take from person": "take_from_person",
}


def resolve_action_targets(
    *,
    nodes: Sequence[dict[str, Any]],
    predictions: dict[str, list[dict[str, float | str]]],
) -> list[dict[str, Any]]:
    by_id = {str(node.get("node_id")): node for node in nodes}
    resolved: list[dict[str, Any]] = []
    for source_id, node_predictions in predictions.items():
        source = by_id.get(source_id)
        if source is None:
            continue
        source_bbox = source.get("bbox", [0, 0, 0, 0])
        source_center = (
            (float(source_bbox[0]) + float(source_bbox[2])) / 2,
            (float(source_bbox[1]) + float(source_bbox[3])) / 2,
        )
        for prediction in node_predictions:
            action = str(prediction["label"])
            target_kind = TARGET_RESOLVER_ACTIONS.get(action)
            if target_kind is None:
                continue
            candidates: list[tuple[float, str]] = []
            for target_id, target in by_id.items():
                if target_id == source_id:
                    continue
                is_person = str(target.get("class_name", "")).lower() == "person"
                if (target_kind == "person") != is_person:
                    continue
                target_bbox = target.get("bbox", [0, 0, 0, 0])
                target_center = (
                    (float(target_bbox[0]) + float(target_bbox[2])) / 2,
                    (float(target_bbox[1]) + float(target_bbox[3])) / 2,
                )
                candidates.append((math.dist(source_center, target_center), target_id))
            if not candidates:
                continue
            _, target_id = min(candidates)
            resolved.append(
                {
                    "source_id": source_id,
                    "predicate": TARGET_RESOLVER_PREDICATES[action],
                    "target_id": target_id,
                    "confidence": float(prediction["confidence"]),
                    "source_method": "gnn_action_plus_target_resolver",
                    "action": action,
                }
            )
    return resolved
