from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .ava_gnn80 import (
    AVA80_LABELS,
    COCO_CLASS_NAMES,
    EDGE_FEATURE_NAMES,
    _calibrate_thresholds,
    _positive_weights,
    _seed_training,
    _state_copy,
    _targets_for_graphs,
    _training_device,
    classification_metrics,
    masked_multilabel_loss,
)

STATIC_SCHEMA = "ava80-gat-ablation-v2"
TEMPORAL_SCHEMA = "ava80-temporal-gat-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StaticActionModel:
    def __new__(
        cls,
        input_dimension: int,
        *,
        use_edges: bool,
        mlp_only: bool = False,
        hidden_dimension: int = 128,
    ) -> Any:
        import torch
        from torch import nn
        from torch_geometric.nn import GATv2Conv

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_edges = use_edges
                self.mlp_only = mlp_only
                self.class_embedding = nn.Embedding(len(COCO_CLASS_NAMES), 16)
                self.projection = nn.Linear(input_dimension + 16, hidden_dimension)
                if mlp_only:
                    self.mlp_block = nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dimension, hidden_dimension),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                    )
                self.gat1 = GATv2Conv(
                    hidden_dimension,
                    hidden_dimension // 2,
                    heads=2,
                    edge_dim=len(EDGE_FEATURE_NAMES),
                    dropout=0.2,
                )
                self.gat2 = GATv2Conv(
                    hidden_dimension,
                    hidden_dimension // 2,
                    heads=2,
                    edge_dim=len(EDGE_FEATURE_NAMES),
                    dropout=0.2,
                )
                self.norm1 = nn.LayerNorm(hidden_dimension)
                self.norm2 = nn.LayerNorm(hidden_dimension)
                self.dropout = nn.Dropout(0.2)
                self.classifier = nn.Linear(hidden_dimension, 80)

            def encode(
                self, features: Any, class_ids: Any, edge_index: Any, edge_attr: Any
            ) -> Any:
                embedded = self.class_embedding(class_ids)
                hidden = torch.relu(self.projection(torch.cat((features, embedded), dim=1)))
                if self.mlp_only:
                    return self.mlp_block(hidden)
                if not self.use_edges:
                    edge_index = torch.empty(
                        (2, 0), dtype=torch.long, device=features.device
                    )
                    edge_attr = torch.empty(
                        (0, len(EDGE_FEATURE_NAMES)),
                        dtype=features.dtype,
                        device=features.device,
                    )
                first = self.dropout(torch.relu(self.gat1(hidden, edge_index, edge_attr=edge_attr)))
                hidden = self.norm1(hidden + first)
                second = self.dropout(
                    torch.relu(self.gat2(hidden, edge_index, edge_attr=edge_attr))
                )
                return self.norm2(hidden + second)

            def forward(
                self, features: Any, class_ids: Any, edge_index: Any, edge_attr: Any
            ) -> Any:
                return self.classifier(self.encode(features, class_ids, edge_index, edge_attr))

        return _Model()


class FocalMultilabelLoss:
    def __new__(cls, positive_weights: Any, gamma: float = 2.0) -> Any:
        import torch
        from torch import nn

        class _Loss(nn.Module):
            def forward(self, logits: Any, targets: Any) -> Any:
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=positive_weights.to(logits.device),
                    reduction="none",
                )
                probabilities = torch.sigmoid(logits)
                correct_probability = targets * probabilities + (1 - targets) * (1 - probabilities)
                return (((1 - correct_probability) ** gamma) * raw).mean()

        return _Loss()


def _loss_function(mode: str, targets: Any, torch: Any, nn: Any) -> Any:
    positive_weights = _positive_weights(targets, torch)
    if mode == "bce_positive_weights":
        return nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    if mode == "focal":
        return FocalMultilabelLoss(positive_weights)
    if mode == "class_balanced":
        positives = targets.sum(dim=0)
        beta = 0.999
        effective = (1 - beta**torch.clamp(positives, min=1)) / (1 - beta)
        weights = (1 / effective)
        weights = weights / weights.mean()
        return nn.BCEWithLogitsLoss(pos_weight=torch.clamp(weights, max=30.0))
    raise ValueError(f"unknown multilabel loss mode: {mode}")


def _select_graph(graph: Any, feature_attribute: str) -> Any:
    from torch_geometric.data import Data

    selected = Data(
        x=getattr(graph, feature_attribute),
        class_ids=graph.class_ids,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        y=graph.y,
        person_mask=graph.person_mask,
        loss_mask=graph.loss_mask,
    )
    selected.video_id = str(graph.video_id)
    selected.timestamp = float(graph.timestamp)
    selected.node_ids = list(graph.node_ids)
    selected.track_ids = list(getattr(graph, "track_ids", [""] * int(graph.num_nodes)))
    return selected


def _split_graphs(
    graphs: Sequence[Any], split: dict[str, Any], feature_attribute: str
) -> dict[str, list[Any]]:
    return {
        name: [
            _select_graph(graph, feature_attribute)
            for graph in graphs
            if str(graph.video_id) in set(split[name])
        ]
        for name in ("train", "validation", "test")
    }


def _normalization(graphs: Sequence[Any], torch: Any) -> tuple[Any, Any]:
    values = torch.cat([graph.x for graph in graphs], dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std[std < 1e-6] = 1.0
    return mean, std


def rare_class_graph_sampling_weights(
    graphs: Sequence[Any], torch: Any, *, maximum_weight: float = 8.0
) -> tuple[Any, dict[str, Any]]:
    """Return deterministic graph weights that favor rare positive AVA classes."""
    if not graphs:
        raise ValueError("rare-class sampling requires at least one graph")
    graph_targets = torch.stack(
        [
            (graph.y[graph.loss_mask].sum(dim=0) > 0).to(torch.float32)
            for graph in graphs
        ]
    )
    class_support = torch.cat(
        [graph.y[graph.loss_mask] for graph in graphs], dim=0
    ).sum(dim=0)
    inverse_support = torch.zeros_like(class_support)
    supported = class_support > 0
    inverse_support[supported] = torch.rsqrt(class_support[supported])
    raw = (graph_targets * inverse_support.unsqueeze(0)).amax(dim=1)
    raw[raw <= 0] = 1.0
    weights = raw / raw.mean().clamp(min=1e-8)
    weights = torch.clamp(weights, min=0.25, max=maximum_weight)
    weights = weights / weights.mean().clamp(min=1e-8)
    effective_size = float(weights.sum() ** 2 / (weights.square().sum()))
    return weights.to(torch.double), {
        "strategy": "inverse_sqrt_person_support_graph_sampler",
        "replacement": True,
        "graphs": len(graphs),
        "minimum_weight": round(float(weights.min()), 6),
        "maximum_weight": round(float(weights.max()), 6),
        "mean_weight": round(float(weights.mean()), 6),
        "effective_sample_size": round(effective_size, 3),
        "supported_training_classes": int(supported.sum()),
    }


def validate_static_checkpoint(
    checkpoint_path: Path,
    *,
    expected_feature_attribute: str | None = None,
    expected_feature_dimension: int | None = None,
) -> dict[str, Any]:
    """Validate the self-describing static AVA checkpoint before evaluation."""
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != STATIC_SCHEMA:
        raise ValueError("static AVA checkpoint schema mismatch")
    if payload.get("labels") != list(AVA80_LABELS):
        raise ValueError("static AVA checkpoint label map mismatch")
    attribute = str(payload.get("feature_attribute", ""))
    dimension = int(payload.get("feature_dimension", 0))
    if expected_feature_attribute and attribute != expected_feature_attribute:
        raise ValueError("static AVA checkpoint feature attribute mismatch")
    if expected_feature_dimension and dimension != expected_feature_dimension:
        raise ValueError("static AVA checkpoint feature dimension mismatch")
    for name in ("feature_mean", "feature_std"):
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (dimension,):
            raise ValueError(f"static AVA checkpoint {name} schema mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"static AVA checkpoint {name} is non-finite")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, torch.Tensor) or tuple(thresholds.shape) != (80,):
        raise ValueError("static AVA checkpoint threshold schema mismatch")
    if not bool(torch.isfinite(thresholds).all()):
        raise ValueError("static AVA checkpoint thresholds are non-finite")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("static AVA checkpoint state dict is missing")
    return {
        "valid": True,
        "feature_attribute": attribute,
        "feature_dimension": dimension,
        "model_version": str(payload.get("model_version", "")),
        "sha256": _sha256(checkpoint_path),
    }


def train_static_ablation(
    *,
    graphs: Sequence[Any],
    split: dict[str, Any],
    feature_attribute: str,
    use_edges: bool,
    mlp_only: bool,
    loss_mode: str,
    output_dir: Path,
    model_version: str,
    feature_schema: dict[str, Any],
    seed: int = 2026,
    epochs: int = 60,
    patience: int = 10,
    batch_size: int = 32,
    device_name: str = "auto",
    rare_class_sampling: bool = False,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    np, torch, nn = _seed_training(seed)
    from torch_geometric.loader import DataLoader

    started = time.perf_counter()
    device = _training_device(torch, device_name)
    selected = _split_graphs(graphs, split, feature_attribute)
    if any(not values for values in selected.values()):
        raise ValueError("one or more ablation splits contain no graphs")
    mean, std = _normalization(selected["train"], torch)
    train_targets = _targets_for_graphs(selected["train"], torch)
    model_factory: Any = StaticActionModel(
        int(selected["train"][0].x.shape[1]), use_edges=use_edges, mlp_only=mlp_only
    )
    model = model_factory.to(device)
    loss_function = _loss_function(loss_mode, train_targets, torch, nn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    device_mean, device_std = mean.to(device), std.to(device)
    sampling_weights = None
    training_sampler = None
    sampling_metadata: dict[str, Any] = {"strategy": "uniform_shuffle"}
    if rare_class_sampling:
        sampling_weights, sampling_metadata = rare_class_graph_sampling_weights(
            selected["train"], torch
        )
        training_sampler = torch.utils.data.WeightedRandomSampler(
            sampling_weights,
            num_samples=len(selected["train"]),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def loader(name: str, shuffle: bool = False) -> Any:
        sampler = training_sampler if name == "train" else None
        return DataLoader(
            selected[name],
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            generator=torch.Generator().manual_seed(seed),
        )

    def forward(batch: Any) -> Any:
        return model(
            (batch.x - device_mean) / device_std,
            batch.class_ids,
            batch.edge_index,
            batch.edge_attr,
        )

    def probabilities(name: str) -> tuple[Any, Any]:
        predictions = []
        targets = []
        model.eval()
        with torch.inference_mode():
            for raw_batch in loader(name):
                batch = raw_batch.to(device)
                predictions.append(torch.sigmoid(forward(batch))[batch.loss_mask].cpu())
                targets.append(batch.y[batch.loss_mask].cpu())
        return torch.cat(targets).numpy(), torch.cat(predictions).numpy()

    best_score = (-1.0, -1.0)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        for raw_batch in loader("train", shuffle=True):
            batch = raw_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_multilabel_loss(
                forward(batch), batch.y, batch.loss_mask, loss_function
            )
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for raw_batch in loader("validation"):
                batch = raw_batch.to(device)
                validation_losses.append(float(masked_multilabel_loss(
                    forward(batch), batch.y, batch.loss_mask, loss_function
                ).cpu()))
        train_loss = sum(training_losses) / len(training_losses)
        validation_loss = sum(validation_losses) / len(validation_losses)
        epoch_targets, epoch_probabilities = probabilities("validation")
        epoch_thresholds = _calibrate_thresholds(epoch_targets, epoch_probabilities)
        epoch_metrics = classification_metrics(
            epoch_targets, epoch_probabilities, epoch_thresholds
        )
        score = (
            float(epoch_metrics["macro_f1_supported_classes"]),
            float(epoch_metrics["mean_average_precision"]),
        )
        history.append({
            "epoch": epoch,
            "training_loss": round(train_loss, 8),
            "validation_loss": round(validation_loss, 8),
            "validation_macro_f1_supported_classes": score[0],
            "validation_mean_average_precision": score[1],
        })
        if score > best_score:
            best_score = score
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = _state_copy(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("static ablation did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_targets, validation_probabilities = probabilities("validation")
    thresholds = _calibrate_thresholds(validation_targets, validation_probabilities)
    validation_metrics = classification_metrics(
        validation_targets, validation_probabilities, thresholds
    )
    test_metrics = None
    if evaluate_test:
        test_targets, test_probabilities = probabilities("test")
        test_metrics = classification_metrics(test_targets, test_probabilities, thresholds)
    model.eval()
    sample = next(iter(loader("validation"))).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(10):
            forward(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = (time.perf_counter() - latency_started) * 1000 / (10 * int(sample.num_graphs))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{model_version}.pt"
    metadata_path = output_dir / f"{model_version}.metadata.json"
    torch.save({
        "schema_version": STATIC_SCHEMA,
        "model_version": model_version,
        "state_dict": best_state,
        "feature_mean": mean,
        "feature_std": std,
        "thresholds": torch.from_numpy(thresholds),
        "feature_attribute": feature_attribute,
        "feature_dimension": int(mean.shape[0]),
        "feature_schema": feature_schema,
        "use_edges": use_edges,
        "mlp_only": mlp_only,
        "labels": list(AVA80_LABELS),
    }, checkpoint)
    metadata = {
        "schema_version": STATIC_SCHEMA,
        "model_version": model_version,
        "task": "person-centric AVA 80-label multilabel action classification",
        "feature_attribute": feature_attribute,
        "feature_schema": feature_schema,
        "architecture": "MLP" if mlp_only else "two-layer GATv2",
        "graph_edges_consumed": use_edges and not mlp_only,
        "loss_mode": loss_mode,
        "sampling": sampling_metadata,
        "checkpoint_selection": {
            "primary": "validation_macro_f1_supported_classes",
            "tie_break": "validation_mean_average_precision",
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_validation_loss, 8),
        },
        "split": {name: list(split[name]) for name in ("train", "validation", "test")},
        "graph_counts": {name: len(values) for name, values in selected.items()},
        "sample_counts": {
            name: int(_targets_for_graphs(values, torch).shape[0])
            for name, values in selected.items()
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_evaluation_status": (
            "evaluated" if evaluate_test else "withheld_until_final_selection"
        ),
        "thresholds_disabled_for_insufficient_validation": [
            index + 1 for index, value in enumerate(validation_targets.sum(axis=0)) if value == 0
        ],
        "inference_latency_mean_milliseconds_per_graph": round(latency, 4),
        "history": history,
        "seed": seed,
        "hardware": {
            "device": str(device),
            "name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
        },
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint.as_posix()),
        "checkpoint_sha256": _sha256(checkpoint),
        "semantic_scope": "predicts person actions only; target resolution remains separate",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def evaluate_static_checkpoint(
    *,
    checkpoint_path: Path,
    graphs: Sequence[Any],
    split: dict[str, Any],
    split_name: str = "test",
    batch_size: int = 32,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate a frozen selected checkpoint without changing its thresholds."""
    np, torch, _ = _seed_training(int(split.get("seed", 2026)))
    del np
    from torch_geometric.loader import DataLoader

    validation = validate_static_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if split_name not in {"validation", "test"}:
        raise ValueError("static AVA evaluation split must be validation or test")
    selected = _split_graphs(graphs, split, validation["feature_attribute"])[split_name]
    if not selected:
        raise ValueError(f"static AVA {split_name} split contains no graphs")
    device = _training_device(torch, device_name)
    model_factory: Any = StaticActionModel(
        validation["feature_dimension"],
        use_edges=bool(payload.get("use_edges")),
        mlp_only=bool(payload.get("mlp_only")),
    )
    model = model_factory.to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = payload["feature_mean"].to(device)
    std = payload["feature_std"].to(device)
    targets = []
    probabilities = []
    started = time.perf_counter()
    graph_count = 0
    with torch.inference_mode():
        for raw_batch in DataLoader(selected, batch_size=batch_size, shuffle=False):
            batch = raw_batch.to(device)
            logits = model(
                (batch.x - mean) / std,
                batch.class_ids,
                batch.edge_index,
                batch.edge_attr,
            )
            probabilities.append(torch.sigmoid(logits)[batch.loss_mask].cpu())
            targets.append(batch.y[batch.loss_mask].cpu())
            graph_count += int(batch.num_graphs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(
        torch.cat(targets).numpy(),
        torch.cat(probabilities).numpy(),
        payload["thresholds"].numpy(),
    )
    return {
        "split": split_name,
        "metrics": metrics,
        "graphs": graph_count,
        "supervised_persons": int(sum(item.shape[0] for item in targets)),
        "inference_latency_mean_milliseconds_per_graph": round(
            elapsed * 1000 / max(1, graph_count), 4
        ),
        "checkpoint_validation": validation,
    }


class TemporalActionModel:
    def __new__(cls, input_dimension: int = 128, hidden_dimension: int = 128) -> Any:
        from torch import nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_dimension, hidden_dimension, batch_first=True)
                self.classifier = nn.Linear(hidden_dimension, 80)

            def forward(self, sequence: Any) -> Any:
                encoded, _ = self.gru(sequence)
                return self.classifier(encoded[:, 1])

        return _Model()


def _static_embeddings(
    graphs: Sequence[Any], checkpoint: Path, device_name: str
) -> tuple[list[dict[str, Any]], Any, Any, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = _training_device(torch, device_name)
    model_factory: Any = StaticActionModel(
        int(payload["feature_dimension"]),
        use_edges=bool(payload["use_edges"]),
        mlp_only=bool(payload["mlp_only"]),
    )
    model = model_factory.to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = payload["feature_mean"].to(device)
    std = payload["feature_std"].to(device)
    rows = []
    with torch.inference_mode():
        for original in graphs:
            graph = _select_graph(original, str(payload["feature_attribute"])).to(device)
            embeddings = model.encode(
                (graph.x - mean) / std,
                graph.class_ids,
                graph.edge_index,
                graph.edge_attr,
            ).cpu()
            logits = model.classifier(embeddings.to(device)).cpu()
            for index in range(int(graph.num_nodes)):
                if not bool(graph.loss_mask[index]):
                    continue
                rows.append({
                    "video_id": str(graph.video_id),
                    "timestamp": float(graph.timestamp),
                    "track_id": graph.track_ids[index],
                    "embedding": embeddings[index],
                    "logits": logits[index],
                    "target": graph.y[index].cpu(),
                })
    return rows, payload, model, device


def train_temporal_ablation(
    *,
    graphs: Sequence[Any],
    split: dict[str, Any],
    static_checkpoint: Path,
    output_dir: Path,
    model_version: str = "ava80-temporal-gat-v1",
    loss_mode: str = "bce_positive_weights",
    seed: int = 2026,
    epochs: int = 60,
    patience: int = 10,
    device_name: str = "auto",
) -> dict[str, Any]:
    np, torch, nn = _seed_training(seed)
    from torch.utils.data import DataLoader, TensorDataset

    started = time.perf_counter()
    rows, static_payload, _, device = _static_embeddings(graphs, static_checkpoint, device_name)
    by_track: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[(row["video_id"], row["track_id"])].append(row)
    samples = []
    for track_rows in by_track.values():
        track_rows.sort(key=lambda row: row["timestamp"])
        for index, row in enumerate(track_rows):
            previous = track_rows[index - 1] if index else row
            following = track_rows[index + 1] if index + 1 < len(track_rows) else row
            samples.append({
                **row,
                "sequence": torch.stack((
                    previous["embedding"], row["embedding"], following["embedding"]
                )),
            })

    def tensors(name: str) -> tuple[Any, Any, list[dict[str, Any]]]:
        selected = [row for row in samples if row["video_id"] in set(split[name])]
        return (
            torch.stack([row["sequence"] for row in selected]),
            torch.stack([row["target"] for row in selected]),
            selected,
        )

    split_tensors = {name: tensors(name) for name in ("train", "validation", "test")}
    model_factory: Any = TemporalActionModel()
    model = model_factory.to(device)
    train_targets = split_tensors["train"][1]
    loss_function = _loss_function(loss_mode, train_targets, torch, nn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loader(name: str, shuffle: bool = False) -> Any:
        features, targets, _ = split_tensors[name]
        return DataLoader(
            TensorDataset(features, targets),
            batch_size=128,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
        )

    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, targets in loader("train", True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for features, targets in loader("validation"):
                validation_losses.append(float(loss_function(
                    model(features.to(device)), targets.to(device)
                ).cpu()))
        training_loss = sum(losses) / len(losses)
        validation_loss = sum(validation_losses) / len(validation_losses)
        history.append({
            "epoch": epoch,
            "training_loss": round(training_loss, 8),
            "validation_loss": round(validation_loss, 8),
        })
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = _state_copy(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("temporal model did not produce a checkpoint")
    model.load_state_dict(best_state)

    def probabilities(name: str) -> Any:
        output = []
        model.eval()
        with torch.inference_mode():
            for features, _ in loader(name):
                output.append(torch.sigmoid(model(features.to(device))).cpu())
        return torch.cat(output).numpy()

    validation_targets = split_tensors["validation"][1].numpy()
    validation_probabilities = probabilities("validation")
    test_targets = split_tensors["test"][1].numpy()
    test_probabilities = probabilities("test")
    thresholds = _calibrate_thresholds(validation_targets, validation_probabilities)
    static_test_probabilities = torch.sigmoid(torch.stack([
        row["logits"] for row in split_tensors["test"][2]
    ])).numpy()
    smoothed = []
    test_rows = split_tensors["test"][2]
    for row in test_rows:
        neighbors = [
            other for other in test_rows
            if other["video_id"] == row["video_id"]
            and other["track_id"] == row["track_id"]
            and abs(other["timestamp"] - row["timestamp"]) <= 1.1
        ]
        smoothed.append(torch.stack([
            torch.sigmoid(other["logits"]) for other in neighbors
        ]).mean(dim=0))
    smoothed_probabilities = torch.stack(smoothed).numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{model_version}.pt"
    torch.save({
        "schema_version": TEMPORAL_SCHEMA,
        "model_version": model_version,
        "state_dict": best_state,
        "static_checkpoint": str(static_checkpoint.as_posix()),
        "static_checkpoint_sha256": _sha256(static_checkpoint),
        "thresholds": torch.from_numpy(thresholds),
        "labels": list(AVA80_LABELS),
    }, checkpoint)
    metadata = {
        "schema_version": TEMPORAL_SCHEMA,
        "model_version": model_version,
        "architecture": "static GATv2 person embedding at t-1/t/t+1 -> GRU -> 80 logits",
        "loss_mode": loss_mode,
        "split": {name: list(split[name]) for name in ("train", "validation", "test")},
        "validation_metrics": classification_metrics(
            validation_targets, validation_probabilities, thresholds
        ),
        "test_metrics": classification_metrics(test_targets, test_probabilities, thresholds),
        "static_test_metrics": classification_metrics(
            test_targets, static_test_probabilities, thresholds
        ),
        "deterministic_smoothing_test_metrics": classification_metrics(
            test_targets, smoothed_probabilities, thresholds
        ),
        "sample_counts": {
            name: len(values[2]) for name, values in split_tensors.items()
        },
        "history": history,
        "seed": seed,
        "hardware": {"device": str(device)},
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint.as_posix()),
        "checkpoint_sha256": _sha256(checkpoint),
        "semantic_scope": "person-centric AVA actions; no explicit action target prediction",
    }
    (output_dir / f"{model_version}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
