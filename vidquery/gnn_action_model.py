from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_model import (
    ACTION_IDS,
    ACTION_LABELS,
    CONTEXT_CLASSES,
    _audio_features,
    _calibrate_thresholds,
    _classification_metrics,
    _video_dimensions,
)

GNN_MODEL_SCHEMA_VERSION = "ava-person-action-graphsage-v1"
NODE_CLASS_NAMES = ("person", *CONTEXT_CLASSES, "other")
NODE_FEATURE_NAMES = (
    "bbox_x1_norm",
    "bbox_y1_norm",
    "bbox_x2_norm",
    "bbox_y2_norm",
    "center_x_norm",
    "center_y_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "bbox_aspect_ratio",
    "detector_confidence",
    *(f"class_is_{name.replace(' ', '_')}" for name in NODE_CLASS_NAMES),
    "log_audio_segment_count",
    "log_audio_speaker_count",
    "log_audio_duration",
    "log_audio_text_characters",
)


@dataclass(frozen=True, slots=True)
class GraphActionSample:
    video_id: str
    timestamp: float
    node_ids: tuple[int, ...]
    node_features: tuple[tuple[float, ...], ...]
    edge_pairs: tuple[tuple[int, int], ...]
    person_indices: tuple[int, ...]
    person_node_ids: tuple[int, ...]
    targets: tuple[tuple[float, ...], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _node_feature(
    node: dict[str, Any],
    audio: list[float],
    *,
    width: int,
    height: int,
) -> tuple[float, ...] | None:
    raw_bbox = node.get("bbox", [0, 0, 0, 0])
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    safe_width = max(1.0, float(width))
    safe_height = max(1.0, float(height))
    x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    box_width = max(0.0, x2 - x1) / safe_width
    box_height = max(0.0, y2 - y1) / safe_height
    center = node.get("center")
    if isinstance(center, list) and len(center) == 2:
        center_x = float(center[0]) / safe_width
        center_y = float(center[1]) / safe_height
    else:
        center_x = (x1 + x2) / (2 * safe_width)
        center_y = (y1 + y2) / (2 * safe_height)
    label = str(node.get("class_name", "other")).lower()
    canonical_class = label if label in NODE_CLASS_NAMES[:-1] else "other"
    class_features = [float(canonical_class == name) for name in NODE_CLASS_NAMES]
    values = (
        x1 / safe_width,
        y1 / safe_height,
        x2 / safe_width,
        y2 / safe_height,
        center_x,
        center_y,
        box_width,
        box_height,
        box_width * box_height,
        min(4.0, box_width / max(box_height, 1e-6)),
        float(node.get("confidence", 0.0)),
        *class_features,
        *audio,
    )
    if len(values) != len(NODE_FEATURE_NAMES):
        raise RuntimeError("GNN node feature schema length mismatch")
    return tuple(values)


def build_frame_graph(
    *,
    video_id: str,
    timestamp: float,
    nodes: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]],
    width: int,
    height: int,
    require_targets: bool,
    near_threshold: float = 0.45,
) -> GraphActionSample | None:
    audio = _audio_features(audio_segments)
    node_ids: list[int] = []
    node_features: list[tuple[float, ...]] = []
    normalized_centers: list[tuple[float, float]] = []
    source_nodes: list[dict[str, Any]] = []
    for fallback_id, node in enumerate(nodes):
        feature = _node_feature(node, audio, width=width, height=height)
        if feature is None:
            continue
        node_ids.append(int(node.get("node_id", fallback_id)))
        node_features.append(feature)
        normalized_centers.append((feature[4], feature[5]))
        source_nodes.append(node)
    if not node_features:
        return None

    edge_pairs: list[tuple[int, int]] = []
    for source_index, source_center in enumerate(normalized_centers):
        for target_index, target_center in enumerate(normalized_centers):
            if source_index == target_index:
                continue
            distance = math.dist(source_center, target_center)
            if distance <= near_threshold:
                edge_pairs.append((source_index, target_index))

    person_indices: list[int] = []
    person_node_ids: list[int] = []
    targets: list[tuple[float, ...]] = []
    for index, node in enumerate(source_nodes):
        if str(node.get("class_name", "")).lower() != "person":
            continue
        annotation = node.get("matched_annotation")
        if require_targets and not isinstance(annotation, dict):
            continue
        action_ids = (
            {int(value) for value in annotation.get("action_ids", [])}
            if isinstance(annotation, dict)
            else set()
        )
        if require_targets and not action_ids:
            continue
        person_indices.append(index)
        person_node_ids.append(node_ids[index])
        targets.append(tuple(float(action_id in action_ids) for action_id in ACTION_IDS))
    if not person_indices:
        return None
    return GraphActionSample(
        video_id=video_id,
        timestamp=timestamp,
        node_ids=tuple(node_ids),
        node_features=tuple(node_features),
        edge_pairs=tuple(edge_pairs),
        person_indices=tuple(person_indices),
        person_node_ids=tuple(person_node_ids),
        targets=tuple(targets),
    )


def build_graph_action_samples(fused_root: Path, video_root: Path) -> list[GraphActionSample]:
    samples: list[GraphActionSample] = []
    for fused_path in sorted(Path(fused_root).rglob("*.fused.json")):
        payload = json.loads(fused_path.read_text(encoding="utf-8"))
        video_id = str(payload.get("video_id", fused_path.name.removesuffix(".fused.json")))
        width, height = _video_dimensions(Path(video_root) / f"{video_id}.mp4")
        for frame in payload.get("frames", []):
            nodes = frame.get("nodes", [])
            audio_segments = frame.get("audio_segments", [])
            graph = build_frame_graph(
                video_id=video_id,
                timestamp=float(frame.get("timestamp", 0.0)),
                nodes=nodes if isinstance(nodes, list) else [],
                audio_segments=audio_segments if isinstance(audio_segments, list) else [],
                width=width,
                height=height,
                require_targets=True,
            )
            if graph is not None:
                samples.append(graph)
    return sorted(samples, key=lambda item: (item.video_id, item.timestamp))


def deterministic_graph_video_split(samples: list[GraphActionSample]) -> dict[str, list[str]]:
    video_ids = sorted({sample.video_id for sample in samples})
    if len(video_ids) < 5:
        raise ValueError("GraphSAGE action training requires at least five independent videos")
    return {
        "train": video_ids[:-2],
        "validation": [video_ids[-2]],
        "test": [video_ids[-1]],
    }


class GraphSAGEPersonActionModel:
    """Native PyTorch two-layer mean-aggregation GraphSAGE action head."""

    def __new__(cls, input_features: int, output_labels: int):
        import torch
        from torch import nn

        class _MeanSAGELayer(nn.Module):
            def __init__(self, input_size: int, output_size: int) -> None:
                super().__init__()
                self.self_linear = nn.Linear(input_size, output_size)
                self.neighbor_linear = nn.Linear(input_size, output_size, bias=False)

            def forward(self, features: Any, edge_index: Any) -> Any:
                neighbors = torch.zeros_like(features)
                degrees = torch.zeros(
                    (features.shape[0], 1), dtype=features.dtype, device=features.device
                )
                if edge_index.numel():
                    source, target = edge_index
                    neighbors.index_add_(0, target, features[source])
                    degrees.index_add_(
                        0,
                        target,
                        torch.ones(
                            (target.shape[0], 1),
                            dtype=features.dtype,
                            device=features.device,
                        ),
                    )
                neighbor_mean = neighbors / degrees.clamp_min(1.0)
                return torch.relu(
                    self.self_linear(features) + self.neighbor_linear(neighbor_mean)
                )

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sage1 = _MeanSAGELayer(input_features, 64)
                self.sage2 = _MeanSAGELayer(64, 32)
                self.dropout = nn.Dropout(0.15)
                self.classifier = nn.Linear(32, output_labels)

            def forward(self, features: Any, edge_index: Any, person_indices: Any) -> Any:
                hidden = self.dropout(self.sage1(features, edge_index))
                hidden = self.dropout(self.sage2(hidden, edge_index))
                return self.classifier(hidden[person_indices])

        return _Model()


def _class_frequency(targets: Any) -> dict[str, dict[str, int]]:
    return {
        label: {
            "positive": int(targets[:, index].sum()),
            "negative": int(len(targets) - targets[:, index].sum()),
        }
        for index, label in enumerate(ACTION_LABELS)
    }


def _graph_stats(samples: list[GraphActionSample]) -> dict[str, float | int]:
    return {
        "graphs": len(samples),
        "person_targets": sum(len(item.person_indices) for item in samples),
        "nodes": sum(len(item.node_features) for item in samples),
        "directed_edges": sum(len(item.edge_pairs) for item in samples),
        "graphs_with_edges": sum(bool(item.edge_pairs) for item in samples),
        "mean_nodes_per_graph": round(
            sum(len(item.node_features) for item in samples) / max(1, len(samples)), 4
        ),
    }


def train_graphsage_action_model(
    *,
    fused_root: Path,
    video_root: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    seed: int = 2026,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn

    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else (
            "cpu" if device_name == "auto" else device_name
        )
    )

    samples = build_graph_action_samples(fused_root, video_root)
    split = deterministic_graph_video_split(samples)
    split_samples = {
        name: [sample for sample in samples if sample.video_id in video_ids]
        for name, video_ids in split.items()
    }
    if any(not selected for selected in split_samples.values()):
        raise ValueError("one or more GraphSAGE dataset splits are empty")

    training_nodes = np.concatenate(
        [np.asarray(sample.node_features, dtype=np.float32) for sample in split_samples["train"]]
    )
    feature_mean = training_nodes.mean(axis=0)
    feature_std = training_nodes.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    def targets_for(selected: list[GraphActionSample]) -> Any:
        return np.asarray(
            [target for sample in selected for target in sample.targets], dtype=np.float32
        )

    train_targets_array = targets_for(split_samples["train"])
    validation_targets_array = targets_for(split_samples["validation"])
    test_targets_array = targets_for(split_samples["test"])
    positives = torch.from_numpy(train_targets_array).sum(dim=0)
    negatives = len(train_targets_array) - positives
    positive_weights = torch.clamp(negatives / torch.clamp(positives, min=1), max=20.0)

    model: Any = GraphSAGEPersonActionModel(
        len(NODE_FEATURE_NAMES), len(ACTION_LABELS)
    )
    model = model.to(device)
    mean_tensor = torch.from_numpy(feature_mean).to(device)
    std_tensor = torch.from_numpy(feature_std).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def collate(batch: list[GraphActionSample], *, remove_edges: bool = False):
        feature_parts = []
        edge_parts = []
        person_parts = []
        target_parts = []
        offset = 0
        for graph in batch:
            features = torch.tensor(graph.node_features, dtype=torch.float32)
            feature_parts.append(features)
            if graph.edge_pairs and not remove_edges:
                edges = torch.tensor(graph.edge_pairs, dtype=torch.long).t().contiguous()
                edge_parts.append(edges + offset)
            person_parts.append(torch.tensor(graph.person_indices, dtype=torch.long) + offset)
            target_parts.append(torch.tensor(graph.targets, dtype=torch.float32))
            offset += len(graph.node_features)
        features = torch.cat(feature_parts).to(device)
        edge_index = (
            torch.cat(edge_parts, dim=1).to(device)
            if edge_parts
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )
        return (
            (features - mean_tensor) / std_tensor,
            edge_index,
            torch.cat(person_parts).to(device),
            torch.cat(target_parts).to(device),
        )

    def probabilities(selected: list[GraphActionSample], *, remove_edges: bool = False):
        output = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(selected), batch_size):
                features, edges, people, _ = collate(
                    selected[start : start + batch_size], remove_edges=remove_edges
                )
                output.append(torch.sigmoid(model(features, edges, people)).cpu())
        return torch.cat(output).numpy()

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(split_samples["train"])))
        random.Random(seed + epoch).shuffle(order)
        batch_losses = []
        for start in range(0, len(order), batch_size):
            batch = [split_samples["train"][index] for index in order[start : start + batch_size]]
            features, edges, people, targets = collate(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features, edges, people), targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            selected = split_samples["validation"]
            for start in range(0, len(selected), batch_size):
                features, edges, people, targets = collate(
                    selected[start : start + batch_size]
                )
                validation_losses.append(
                    float(loss_function(model(features, edges, people), targets).cpu())
                )
        validation_loss = sum(validation_losses) / len(validation_losses)
        training_loss = sum(batch_losses) / len(batch_losses)
        history.append(
            {
                "epoch": epoch,
                "training_loss": round(training_loss, 8),
                "validation_loss": round(validation_loss, 8),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("GraphSAGE training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    validation_probabilities = probabilities(split_samples["validation"])
    test_probabilities = probabilities(split_samples["test"])
    thresholds = _calibrate_thresholds(validation_targets_array, validation_probabilities)
    validation_metrics = _classification_metrics(
        validation_targets_array, validation_probabilities, thresholds
    )
    test_metrics = _classification_metrics(test_targets_array, test_probabilities, thresholds)
    no_edge_metrics = _classification_metrics(
        test_targets_array,
        probabilities(split_samples["test"], remove_edges=True),
        thresholds,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": GNN_MODEL_SCHEMA_VERSION,
            "state_dict": best_state,
            "feature_mean": torch.from_numpy(feature_mean),
            "feature_std": torch.from_numpy(feature_std),
            "labels": list(ACTION_LABELS),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "thresholds": torch.from_numpy(thresholds),
        },
        checkpoint_path,
    )
    duration = time.perf_counter() - started
    hardware = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    metadata: dict[str, Any] = {
        "schema_version": GNN_MODEL_SCHEMA_VERSION,
        "task": "multilabel person-centric AVA action classification with scene-graph context",
        "model_type": "two-layer mean-aggregation GraphSAGE",
        "architecture": {
            "input_features": len(NODE_FEATURE_NAMES),
            "hidden_dimensions": [64, 32],
            "output_labels": len(ACTION_LABELS),
            "dropout": 0.15,
            "edge_policy": "bidirectional normalized-center distance <= 0.45",
        },
        "labels": list(ACTION_LABELS),
        "official_ava_action_ids": list(ACTION_IDS),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "node_class_names": list(NODE_CLASS_NAMES),
        "split": split,
        "sample_counts": {
            name: sum(len(item.person_indices) for item in selected)
            for name, selected in split_samples.items()
        },
        "graph_counts": {name: len(selected) for name, selected in split_samples.items()},
        "graph_statistics": {
            name: _graph_stats(selected) for name, selected in split_samples.items()
        },
        "class_frequency": {
            "train": _class_frequency(train_targets_array),
            "validation": _class_frequency(validation_targets_array),
            "test": _class_frequency(test_targets_array),
        },
        "loss": "BCEWithLogitsLoss",
        "class_imbalance": {
            "method": "training-split positive class weights clipped at 20",
            "positive_weights": [round(float(value), 6) for value in positive_weights],
        },
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size_graphs": batch_size,
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": patience,
        "history": history,
        "thresholds": {
            label: round(float(thresholds[index]), 4)
            for index, label in enumerate(ACTION_LABELS)
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_without_edges": no_edge_metrics,
        "hardware": {"device": str(device), "name": hardware},
        "training_duration_seconds": round(duration, 3),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation": {
            "status": "validated",
            "schema_checked": True,
            "held_out_test_video_count": len(split["test"]),
            "message_passing_ablation_reported": True,
        },
        "limitations": [
            "Only five AVA videos are available; validation and test each contain one video.",
            "The task predicts person-centric AVA actions, not arbitrary object-object predicates.",
            "Edges are deterministic spatial candidates; AVA action labels supervise person nodes.",
            "Metrics do not establish generalization beyond the checked five-video subset.",
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def gnn_action_model_status(checkpoint_path: Path, metadata_path: Path) -> str:
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        return "unavailable_no_valid_checkpoint"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != GNN_MODEL_SCHEMA_VERSION:
            return "unavailable_schema_mismatch"
        if metadata.get("node_feature_names") != list(NODE_FEATURE_NAMES):
            return "unavailable_schema_mismatch"
        if metadata.get("labels") != list(ACTION_LABELS):
            return "unavailable_schema_mismatch"
        if metadata.get("validation", {}).get("status") != "validated":
            return "unavailable_not_validated"
        if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
            return "unavailable_checksum_mismatch"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "unavailable_invalid_metadata"
    return "available_validated"


class GraphSAGEActionPredictor:
    def __init__(self, checkpoint_path: Path, metadata_path: Path, device: str = "cpu"):
        import torch

        status = gnn_action_model_status(checkpoint_path, metadata_path)
        if status != "available_validated":
            raise ValueError(f"GraphSAGE checkpoint cannot be activated: {status}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload.get("schema_version") != GNN_MODEL_SCHEMA_VERSION:
            raise ValueError("GraphSAGE checkpoint schema mismatch")
        if payload.get("node_feature_names") != list(NODE_FEATURE_NAMES):
            raise ValueError("GraphSAGE node feature schema mismatch")
        if payload.get("labels") != list(ACTION_LABELS):
            raise ValueError("GraphSAGE label schema mismatch")
        self.device = torch.device(device)
        model: Any = GraphSAGEPersonActionModel(
            len(NODE_FEATURE_NAMES), len(ACTION_LABELS)
        )
        self.model = model.to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.feature_mean = payload["feature_mean"].to(self.device)
        self.feature_std = payload["feature_std"].to(self.device)
        self.thresholds = payload["thresholds"].to(self.device)

    def predict_graph(
        self,
        *,
        video_id: str,
        timestamp: float,
        nodes: list[dict[str, Any]],
        audio_segments: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> dict[int, list[dict[str, float | str]]]:
        graph = build_frame_graph(
            video_id=video_id,
            timestamp=timestamp,
            nodes=nodes,
            audio_segments=audio_segments,
            width=width,
            height=height,
            require_targets=False,
        )
        if graph is None:
            return {}
        return self.predict_sample(graph)

    def predict_sample(
        self, graph: GraphActionSample
    ) -> dict[int, list[dict[str, float | str]]]:
        import torch

        features = torch.tensor(
            graph.node_features, dtype=torch.float32, device=self.device
        )
        normalized = (features - self.feature_mean) / self.feature_std
        edge_index = (
            torch.tensor(graph.edge_pairs, dtype=torch.long, device=self.device)
            .t()
            .contiguous()
            if graph.edge_pairs
            else torch.empty((2, 0), dtype=torch.long, device=self.device)
        )
        person_indices = torch.tensor(
            graph.person_indices, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.model(normalized, edge_index, person_indices)
            )
        return {
            node_id: [
                {"label": label, "confidence": round(float(probabilities[row, index]), 6)}
                for index, label in enumerate(ACTION_LABELS)
                if probabilities[row, index] >= self.thresholds[index]
            ]
            for row, node_id in enumerate(graph.person_node_ids)
        }


def build_gnn_action_index(
    predictor: GraphSAGEActionPredictor,
    *,
    fused_root: Path,
    video_root: Path,
    segment_duration: float = 5.0,
) -> dict[tuple[str, float], dict[str, float]]:
    if segment_duration <= 0:
        raise ValueError("segment_duration must be positive")
    index: dict[tuple[str, float], dict[str, float]] = {}
    for sample in build_graph_action_samples(fused_root, video_root):
        segment_start = math.floor(sample.timestamp / segment_duration) * segment_duration
        action_scores = index.setdefault((sample.video_id, segment_start), {})
        for person_predictions in predictor.predict_sample(sample).values():
            for prediction in person_predictions:
                label = str(prediction["label"])
                action_scores[label] = max(
                    action_scores.get(label, 0.0), float(prediction["confidence"])
                )
    return index
