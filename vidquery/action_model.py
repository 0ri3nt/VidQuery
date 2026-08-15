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

from .ava_labels import AVA_V22_ACTIONS

MODEL_SCHEMA_VERSION = "ava-person-action-mlp-v1"
ACTION_IDS = (11, 12, 17, 74, 79, 80)
ACTION_LABELS = tuple(AVA_V22_ACTIONS[action_id] for action_id in ACTION_IDS)
CONTEXT_CLASSES = (
    "chair",
    "couch",
    "bed",
    "bottle",
    "cup",
    "cell phone",
    "laptop",
    "tv",
    "dining table",
    "book",
    "suitcase",
    "backpack",
    "refrigerator",
)
FEATURE_NAMES = (
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
    "log_person_count",
    "log_other_object_count",
    *(f"context_has_{name.replace(' ', '_')}" for name in CONTEXT_CLASSES),
    "log_audio_segment_count",
    "log_audio_speaker_count",
    "log_audio_duration",
    "log_audio_text_characters",
)


@dataclass(frozen=True, slots=True)
class PersonActionSample:
    video_id: str
    timestamp: float
    node_id: int
    features: tuple[float, ...]
    targets: tuple[float, ...]


def _audio_features(audio_segments: list[dict[str, Any]]) -> list[float]:
    speakers = {
        str(segment.get("speaker", "unknown")).strip().lower()
        for segment in audio_segments
    }
    duration = sum(
        max(
            0.0,
            float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)),
        )
        for segment in audio_segments
    )
    text_characters = sum(len(str(segment.get("text", "")).strip()) for segment in audio_segments)
    return [
        math.log1p(len(audio_segments)),
        math.log1p(len(speakers)),
        math.log1p(duration),
        math.log1p(text_characters),
    ]


def extract_person_feature_rows(
    nodes: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> list[tuple[int, tuple[float, ...]]]:
    """Build the versioned non-pixel feature schema used in training and inference."""
    safe_width = max(1.0, float(width))
    safe_height = max(1.0, float(height))
    labels = [str(node.get("class_name", "unknown")).lower() for node in nodes]
    person_count = sum(label == "person" for label in labels)
    other_count = sum(label != "person" for label in labels)
    context = [float(name in labels) for name in CONTEXT_CLASSES]
    audio = _audio_features(audio_segments)
    rows: list[tuple[int, tuple[float, ...]]] = []
    for fallback_id, node in enumerate(nodes):
        if str(node.get("class_name", "")).lower() != "person":
            continue
        raw_bbox = [float(value) for value in node.get("bbox", [0, 0, 0, 0])]
        if len(raw_bbox) != 4:
            continue
        x1, y1, x2, y2 = raw_bbox
        width_norm = max(0.0, x2 - x1) / safe_width
        height_norm = max(0.0, y2 - y1) / safe_height
        center = node.get("center")
        if isinstance(center, list) and len(center) == 2:
            center_x = float(center[0]) / safe_width
            center_y = float(center[1]) / safe_height
        else:
            center_x = (x1 + x2) / (2 * safe_width)
            center_y = (y1 + y2) / (2 * safe_height)
        aspect_ratio = min(4.0, width_norm / max(height_norm, 1e-6))
        features = (
            x1 / safe_width,
            y1 / safe_height,
            x2 / safe_width,
            y2 / safe_height,
            center_x,
            center_y,
            width_norm,
            height_norm,
            width_norm * height_norm,
            aspect_ratio,
            float(node.get("confidence", 0.0)),
            math.log1p(person_count),
            math.log1p(other_count),
            *context,
            *audio,
        )
        if len(features) != len(FEATURE_NAMES):
            raise RuntimeError("person-action feature schema length mismatch")
        rows.append((int(node.get("node_id", fallback_id)), tuple(features)))
    return rows


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - base dependency in normal installs
        raise RuntimeError("opencv-python-headless is required to read video dimensions") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"could not read dimensions from {video_path}")
    return width, height


def build_person_action_samples(
    fused_root: Path,
    video_root: Path,
) -> list[PersonActionSample]:
    samples: list[PersonActionSample] = []
    for fused_path in sorted(Path(fused_root).rglob("*.fused.json")):
        payload = json.loads(fused_path.read_text(encoding="utf-8"))
        video_id = str(payload.get("video_id", fused_path.name.removesuffix(".fused.json")))
        width, height = _video_dimensions(Path(video_root) / f"{video_id}.mp4")
        for frame in payload.get("frames", []):
            nodes = frame.get("nodes", [])
            audio_segments = frame.get("audio_segments", [])
            rows = dict(
                extract_person_feature_rows(
                    nodes,
                    audio_segments if isinstance(audio_segments, list) else [],
                    width=width,
                    height=height,
                )
            )
            timestamp = float(frame.get("timestamp", 0.0))
            for fallback_id, node in enumerate(nodes):
                node_id = int(node.get("node_id", fallback_id))
                features = rows.get(node_id)
                annotation = node.get("matched_annotation")
                if features is None or not isinstance(annotation, dict):
                    continue
                action_ids = {int(value) for value in annotation.get("action_ids", [])}
                if not action_ids:
                    continue
                targets = tuple(float(action_id in action_ids) for action_id in ACTION_IDS)
                samples.append(
                    PersonActionSample(
                        video_id=video_id,
                        timestamp=timestamp,
                        node_id=node_id,
                        features=features,
                        targets=targets,
                    )
                )
    return sorted(samples, key=lambda item: (item.video_id, item.timestamp, item.node_id))


def deterministic_video_split(samples: list[PersonActionSample]) -> dict[str, list[str]]:
    video_ids = sorted({sample.video_id for sample in samples})
    if len(video_ids) < 5:
        raise ValueError("person-action training requires at least five independent videos")
    return {
        "train": video_ids[:-2],
        "validation": [video_ids[-2]],
        "test": [video_ids[-1]],
    }


def _class_frequency(samples: list[PersonActionSample]) -> dict[str, dict[str, int]]:
    return {
        label: {
            "positive": int(sum(sample.targets[index] for sample in samples)),
            "negative": int(sum(not sample.targets[index] for sample in samples)),
        }
        for index, label in enumerate(ACTION_LABELS)
    }


def _classification_metrics(
    targets: Any,
    probabilities: Any,
    thresholds: Any,
) -> dict[str, Any]:
    import numpy as np

    truth = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(probabilities) >= np.asarray(thresholds)
    per_class: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    for index, label in enumerate(ACTION_LABELS):
        expected = truth[:, index] == 1
        predicted = predictions[:, index]
        tp = int(np.logical_and(expected, predicted).sum())
        fp = int(np.logical_and(~expected, predicted).sum())
        fn = int(np.logical_and(expected, ~predicted).sum())
        support = int(expected.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
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
    return {
        "macro_f1": round(
            sum(float(values["f1"]) for values in per_class.values()) / len(per_class),
            6,
        ),
        "micro_f1": round(micro_f1, 6),
        "per_class": per_class,
    }


def _calibrate_thresholds(targets: Any, probabilities: Any) -> Any:
    import numpy as np

    thresholds = np.full(len(ACTION_LABELS), 0.5, dtype=np.float32)
    for index in range(len(ACTION_LABELS)):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.arange(0.1, 0.91, 0.05):
            truth = np.asarray(targets)[:, index] == 1
            predicted = np.asarray(probabilities)[:, index] >= threshold
            tp = int(np.logical_and(truth, predicted).sum())
            fp = int(np.logical_and(~truth, predicted).sum())
            fn = int(np.logical_and(truth, ~predicted).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            if f1 > best_f1 or (
                f1 == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[index] = best_threshold
    return thresholds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_person_action_model(
    *,
    fused_root: Path,
    video_root: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    seed: int = 2026,
    epochs: int = 100,
    batch_size: int = 128,
    patience: int = 15,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else (
            "cpu" if device_name == "auto" else device_name
        )
    )

    samples = build_person_action_samples(fused_root, video_root)
    split = deterministic_video_split(samples)
    split_samples = {
        name: [sample for sample in samples if sample.video_id in video_ids]
        for name, video_ids in split.items()
    }
    if any(not values for values in split_samples.values()):
        raise ValueError("one or more deterministic dataset splits are empty")

    train_x = np.asarray([sample.features for sample in split_samples["train"]], dtype=np.float32)
    feature_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    def tensors(name: str) -> tuple[Any, Any]:
        selected = split_samples[name]
        features = np.asarray([sample.features for sample in selected], dtype=np.float32)
        targets = np.asarray([sample.targets for sample in selected], dtype=np.float32)
        return (
            torch.from_numpy((features - feature_mean) / feature_std),
            torch.from_numpy(targets),
        )

    train_features, train_targets = tensors("train")
    validation_features, validation_targets = tensors("validation")
    test_features, test_targets = tensors("test")
    positives = train_targets.sum(dim=0)
    negatives = len(train_targets) - positives
    positive_weights = torch.clamp(negatives / torch.clamp(positives, min=1), max=20.0)

    model: Any = PersonActionMLP(len(FEATURE_NAMES), len(ACTION_LABELS))
    model = model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
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
        batch_losses: list[float] = []
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(
                    model(validation_features.to(device)),
                    validation_targets.to(device),
                ).cpu()
            )
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
        raise RuntimeError("training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(
            model(validation_features.to(device))
        ).cpu().numpy()
        test_probabilities = torch.sigmoid(model(test_features.to(device))).cpu().numpy()
    thresholds = _calibrate_thresholds(validation_targets.numpy(), validation_probabilities)
    validation_metrics = _classification_metrics(
        validation_targets.numpy(), validation_probabilities, thresholds
    )
    test_metrics = _classification_metrics(test_targets.numpy(), test_probabilities, thresholds)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state_dict": best_state,
            "feature_mean": torch.from_numpy(feature_mean),
            "feature_std": torch.from_numpy(feature_std),
            "labels": list(ACTION_LABELS),
            "feature_names": list(FEATURE_NAMES),
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
        "schema_version": MODEL_SCHEMA_VERSION,
        "task": "multilabel person-centric AVA action classification",
        "model_type": "two-hidden-layer multilayer perceptron",
        "architecture": {
            "input_features": len(FEATURE_NAMES),
            "hidden_dimensions": [64, 32],
            "output_labels": len(ACTION_LABELS),
            "dropout": 0.15,
        },
        "labels": list(ACTION_LABELS),
        "official_ava_action_ids": list(ACTION_IDS),
        "feature_names": list(FEATURE_NAMES),
        "split": split,
        "sample_counts": {name: len(values) for name, values in split_samples.items()},
        "class_frequency": {
            name: _class_frequency(values) for name, values in split_samples.items()
        },
        "loss": "BCEWithLogitsLoss",
        "class_imbalance": {
            "method": "training-split positive class weights clipped at 20",
            "positive_weights": [round(float(value), 6) for value in positive_weights],
        },
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
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
        "hardware": {"device": str(device), "name": hardware},
        "training_duration_seconds": round(duration, 3),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation": {
            "status": "validated",
            "schema_checked": True,
            "held_out_test_video_count": len(split["test"]),
        },
        "limitations": [
            "Only five AVA videos are available; the held-out test split contains one video.",
            "Features contain box geometry, detector confidence, object context, "
            "and audio summaries; no pixels or motion clips are used.",
            "Metrics do not establish generalization beyond the checked five-video subset.",
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def action_model_status(checkpoint_path: Path, metadata_path: Path) -> str:
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        return "unavailable_no_valid_checkpoint"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
            return "unavailable_schema_mismatch"
        if metadata.get("feature_names") != list(FEATURE_NAMES):
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


class PersonActionMLP:  # runtime base is replaced below when Torch is imported
    def __new__(cls, input_features: int, output_labels: int):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_features, 64),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(32, output_labels),
                )

            def forward(self, features: Any) -> Any:
                return self.network(features)

        return _Model()


class PersonActionPredictor:
    def __init__(self, checkpoint_path: Path, metadata_path: Path, device: str = "cpu"):
        import torch

        status = action_model_status(checkpoint_path, metadata_path)
        if status != "available_validated":
            raise ValueError(f"person-action checkpoint cannot be activated: {status}")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("person-action checkpoint schema mismatch")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("person-action feature schema mismatch")
        if payload.get("labels") != list(ACTION_LABELS):
            raise ValueError("person-action label schema mismatch")
        self.device = torch.device(device)
        built_model: Any = PersonActionMLP(len(FEATURE_NAMES), len(ACTION_LABELS))
        self.model = built_model.to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.feature_mean = payload["feature_mean"].to(self.device)
        self.feature_std = payload["feature_std"].to(self.device)
        self.thresholds = payload["thresholds"].to(self.device)

    def predict(self, features: list[float]) -> list[dict[str, float | str]]:
        import torch

        if len(features) != len(FEATURE_NAMES):
            raise ValueError("person-action inference feature schema mismatch")
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        normalized = (tensor - self.feature_mean) / self.feature_std
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(normalized.unsqueeze(0)))[0]
        return [
            {"label": label, "confidence": round(float(probabilities[index]), 6)}
            for index, label in enumerate(ACTION_LABELS)
            if probabilities[index] >= self.thresholds[index]
        ]


def build_learned_action_index(
    predictor: PersonActionPredictor,
    *,
    fused_root: Path,
    video_root: Path,
    segment_duration: float = 5.0,
) -> dict[tuple[str, float], dict[str, float]]:
    """Project person-level predictions into canonical timestamp windows."""
    if segment_duration <= 0:
        raise ValueError("segment_duration must be positive")
    index: dict[tuple[str, float], dict[str, float]] = {}
    for sample in build_person_action_samples(fused_root, video_root):
        segment_start = math.floor(sample.timestamp / segment_duration) * segment_duration
        action_scores = index.setdefault((sample.video_id, segment_start), {})
        for prediction in predictor.predict(list(sample.features)):
            label = str(prediction["label"])
            action_scores[label] = max(
                action_scores.get(label, 0.0), float(prediction["confidence"])
            )
    return index
