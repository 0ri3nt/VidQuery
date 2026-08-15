"""VidOR explicit subject-predicate-object graph dataset and learned baselines."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

VIDOR_RELATION_SCHEMA = "vidor-relation-graphs-v1"
VIDOR_RELATION_CHECKPOINT_SCHEMA = "vidor-relation-model-v1"

VIDOR_OBJECTS = (
    "adult", "aircraft", "antelope", "baby", "baby_seat", "baby_walker",
    "backpack", "ball/sports_ball", "bat", "bear", "bench", "bicycle", "bird",
    "bottle", "bread", "bus/truck", "cake", "camel", "camera", "car", "cat",
    "cattle/cow", "cellphone", "chair", "chicken", "child", "crab", "crocodile",
    "cup", "dish", "dog", "duck", "electric_fan", "elephant", "faucet", "fish",
    "frisbee", "fruits", "guitar", "hamster/rat", "handbag", "horse", "kangaroo",
    "laptop", "leopard", "lion", "microwave", "motorcycle", "oven", "panda",
    "penguin", "piano", "pig", "rabbit", "racket", "refrigerator", "scooter",
    "screen/monitor", "sheep/goat", "sink", "skateboard", "ski", "snake",
    "snowboard", "sofa", "squirrel", "stingray", "stool", "stop_sign", "suitcase",
    "surfboard", "table", "tiger", "toilet", "toy", "traffic_light", "train",
    "turtle", "vegetables", "watercraft",
)

VIDOR_PREDICATES = (
    "above", "away", "behind", "beneath", "bite", "caress", "carry", "chase",
    "clean", "close", "cut", "drive", "feed", "get_off", "get_on", "grab", "hit",
    "hold", "hold_hand_of", "hug", "in_front_of", "inside", "kick", "kiss", "knock",
    "lean_on", "lick", "lift", "next_to", "open", "pat", "play(instrument)",
    "point_to", "press", "pull", "push", "release", "ride", "shake_hand_with",
    "shout_at", "smell", "speak_to", "squeeze", "throw", "touch", "towards", "use",
    "watch", "wave", "wave_hand_to",
)

PERSON_CLASSES = frozenset({"adult", "child", "baby"})
SYMMETRIC_PREDICATES = frozenset({"next_to", "kiss", "hug", "shake_hand_with"})
SPATIAL_PREDICATES = frozenset(
    {"above", "away", "behind", "beneath", "in_front_of", "inside", "next_to", "towards"}
)
NODE_FEATURE_NAMES = (
    "x1", "y1", "x2", "y2", "area", "aspect_ratio", "annotation_confidence",
    "is_person",
)
PAIR_FEATURE_NAMES = (
    "relative_x", "relative_y", "center_distance", "iou", "log_size_ratio",
    "subject_left_of_object", "subject_above_object", "overlap",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> tuple[Any, Any]:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np, torch


def validate_vidor_label_map(annotation_zip: Path) -> dict[str, Any]:
    predicate_support: Counter[str] = Counter()
    object_support: Counter[str] = Counter()
    relation_count = 0
    with zipfile.ZipFile(annotation_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("VidOR annotation archive failed ZIP integrity validation")
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for name in names:
            item = json.loads(archive.read(name))
            object_support.update(row["category"] for row in item["subject/objects"])
            predicate_support.update(
                row["predicate"] for row in item["relation_instances"]
            )
            relation_count += len(item["relation_instances"])
    if set(predicate_support) != set(VIDOR_PREDICATES):
        raise ValueError("VidOR predicate vocabulary does not match the versioned map")
    if set(object_support) != set(VIDOR_OBJECTS):
        raise ValueError("VidOR object vocabulary does not match the versioned map")
    return {
        "schema_version": "vidor-training-coverage-v1",
        "archive": annotation_zip.as_posix(),
        "archive_sha256": _sha256(annotation_zip),
        "video_count": len(names),
        "relation_instances": relation_count,
        "predicate_count": len(predicate_support),
        "object_count": len(object_support),
        "predicate_support": dict(sorted(predicate_support.items())),
        "object_support": dict(sorted(object_support.items())),
    }


def _annotation_summaries(
    annotation_zip: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    summaries: list[dict[str, Any]] = []
    global_support: Counter[str] = Counter()
    with zipfile.ZipFile(annotation_zip) as archive:
        for name in sorted(item for item in archive.namelist() if item.endswith(".json")):
            payload = json.loads(archive.read(name))
            support = Counter(row["predicate"] for row in payload["relation_instances"])
            global_support.update(support)
            summaries.append(
                {
                    "member": name,
                    "video_id": str(payload["video_id"]),
                    "support": dict(support),
                    "relation_instances": len(payload["relation_instances"]),
                }
            )
    return summaries, global_support


def select_relation_videos(
    summaries: Sequence[dict[str, Any]],
    global_support: Counter[str],
    target_videos: int,
) -> list[dict[str, Any]]:
    if target_videos < 3:
        raise ValueError("relationship dataset requires at least three videos")
    if target_videos > len(summaries):
        raise ValueError("target exceeds the available VidOR videos")

    def score(item: dict[str, Any]) -> tuple[float, int, str]:
        support = item["support"]
        rare_coverage = sum(
            min(int(count), 25) / math.sqrt(max(1, global_support[predicate]))
            for predicate, count in support.items()
        )
        return (rare_coverage + len(support) * 0.02, len(support), item["video_id"])

    return sorted(summaries, key=score, reverse=True)[:target_videos]


def strict_video_split(video_ids: Sequence[str], seed: int = 2026) -> dict[str, list[str]]:
    ordered = sorted(
        set(video_ids),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * 0.1))
    test_count = max(1, round(len(ordered) * 0.1))
    split = {
        "validation": ordered[:validation_count],
        "test": ordered[validation_count : validation_count + test_count],
        "train": ordered[validation_count + test_count :],
    }
    sets = [set(split[name]) for name in ("train", "validation", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("VidOR video split leakage")
    return split


def _bbox_features(
    bbox: dict[str, float], width: float, height: float, generated: int, is_person: bool
) -> list[float]:
    x1 = max(0.0, min(1.0, float(bbox["xmin"]) / width))
    y1 = max(0.0, min(1.0, float(bbox["ymin"]) / height))
    x2 = max(x1, min(1.0, float(bbox["xmax"]) / width))
    y2 = max(y1, min(1.0, float(bbox["ymax"]) / height))
    box_width, box_height = x2 - x1, y2 - y1
    return [
        x1, y1, x2, y2, box_width * box_height,
        box_width / max(box_height, 1e-6),
        1.0 if int(generated) == 0 else 0.5,
        float(is_person),
    ]


def _pair_features(source: Sequence[float], target: Sequence[float]) -> list[float]:
    sx1, sy1, sx2, sy2 = source[:4]
    tx1, ty1, tx2, ty2 = target[:4]
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    ix1, iy1, ix2, iy2 = max(sx1, tx1), max(sy1, ty1), min(sx2, tx2), min(sy2, ty2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = source[4] + target[4] - intersection
    iou = intersection / max(union, 1e-6)
    return [
        tcx - scx,
        tcy - scy,
        math.hypot(tcx - scx, tcy - scy),
        iou,
        math.log(max(source[4], 1e-6) / max(target[4], 1e-6)),
        float(scx < tcx),
        float(scy < tcy),
        float(intersection > 0),
    ]


def _candidate_pairs(features: Sequence[Sequence[float]], nearest_k: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for source in range(len(features)):
        distances = sorted(
            (
                (_pair_features(features[source], features[target])[2], target)
                for target in range(len(features))
                if target != source
            ),
            key=lambda item: (item[0], item[1]),
        )
        for distance, target in distances[:nearest_k]:
            if distance <= 0.85:
                candidates.append((source, target))
    return candidates


def _selected_frames(
    relations: Sequence[dict[str, Any]],
    predicate_support: Counter[str],
    max_frames: int,
) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for relation in relations:
        begin, end = int(relation["begin_fid"]), int(relation["end_fid"])
        if end <= begin:
            continue
        candidates = {begin, max(begin, end - 1), (begin + end - 1) // 2}
        weight = 1.0 / math.sqrt(max(1, predicate_support[relation["predicate"]]))
        for frame_id in candidates:
            scores[frame_id] += weight
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [frame_id for frame_id, _ in ranked[:max_frames]]


def build_relation_graphs(
    annotation_zip: Path,
    selected: Sequence[dict[str, Any]],
    predicate_support: Counter[str],
    *,
    max_frames_per_video: int = 6,
    nearest_k: int = 8,
) -> tuple[list[Any], dict[str, Any]]:
    import torch
    from torch_geometric.data import Data

    object_to_id = {name: index for index, name in enumerate(VIDOR_OBJECTS)}
    predicate_to_id = {name: index for index, name in enumerate(VIDOR_PREDICATES)}
    selected_members = {item["member"] for item in selected}
    graphs: list[Any] = []
    feasible_positive_instances = 0
    proposed_positive_instances = 0
    total_candidate_edges = 0
    total_positive_edges = 0
    with zipfile.ZipFile(annotation_zip) as archive:
        for member in sorted(selected_members):
            payload = json.loads(archive.read(member))
            relations = payload["relation_instances"]
            categories = {
                int(item["tid"]): str(item["category"])
                for item in payload["subject/objects"]
            }
            for frame_id in _selected_frames(relations, predicate_support, max_frames_per_video):
                if frame_id >= len(payload["trajectories"]):
                    continue
                observations = sorted(
                    payload["trajectories"][frame_id],
                    key=lambda row: int(row["tid"]),
                )
                tids = [int(row["tid"]) for row in observations if int(row["tid"]) in categories]
                observations = [row for row in observations if int(row["tid"]) in categories]
                if len(tids) < 2:
                    continue
                node_features = [
                    _bbox_features(
                        row["bbox"],
                        float(payload["width"]),
                        float(payload["height"]),
                        int(row.get("generated", 0)),
                        categories[int(row["tid"])] in PERSON_CLASSES,
                    )
                    for row in observations
                ]
                candidates = _candidate_pairs(node_features, nearest_k)
                if not candidates:
                    continue
                tid_to_index = {tid: index for index, tid in enumerate(tids)}
                active: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
                active_instances: list[tuple[int, int, str]] = []
                for relation in relations:
                    if int(relation["begin_fid"]) <= frame_id < int(relation["end_fid"]):
                        source_tid = int(relation["subject_tid"])
                        target_tid = int(relation["object_tid"])
                        if source_tid in tid_to_index and target_tid in tid_to_index:
                            pair = (tid_to_index[source_tid], tid_to_index[target_tid])
                            predicate = str(relation["predicate"])
                            active[pair].add(predicate)
                            active_instances.append((*pair, predicate))
                candidate_set = set(candidates)
                feasible_positive_instances += len(active_instances)
                proposed_positive_instances += sum(
                    (source, target) in candidate_set
                    for source, target, _ in active_instances
                )
                edge_y = torch.zeros((len(candidates), len(VIDOR_PREDICATES)), dtype=torch.float32)
                for edge_index, pair in enumerate(candidates):
                    for predicate in active.get(pair, set()):
                        edge_y[edge_index, predicate_to_id[predicate]] = 1.0
                edge_index_tensor = torch.tensor(candidates, dtype=torch.long).T.contiguous()
                graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    class_ids=torch.tensor(
                        [object_to_id[categories[tid]] for tid in tids], dtype=torch.long
                    ),
                    edge_index=edge_index_tensor,
                    edge_attr=torch.tensor(
                        [_pair_features(node_features[s], node_features[t]) for s, t in candidates],
                        dtype=torch.float32,
                    ),
                    edge_y=edge_y,
                )
                graph.video_id = str(payload["video_id"])
                graph.frame_id = frame_id
                graph.fps = float(payload["fps"])
                graph.timestamp = frame_id / max(float(payload["fps"]), 1e-6)
                graph.track_ids = [str(tid) for tid in tids]
                graph.class_names = [categories[tid] for tid in tids]
                graphs.append(graph)
                total_candidate_edges += len(candidates)
                total_positive_edges += int((edge_y.sum(dim=1) > 0).sum())
    if not graphs:
        raise ValueError("VidOR graph construction produced no graphs")
    return graphs, {
        "graphs": len(graphs),
        "candidate_edges": total_candidate_edges,
        "positive_candidate_edges": total_positive_edges,
        "feasible_positive_instances": feasible_positive_instances,
        "proposed_positive_instances": proposed_positive_instances,
        "candidate_edge_recall": (
            proposed_positive_instances / feasible_positive_instances
            if feasible_positive_instances
            else 0.0
        ),
    }


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for relation training but is unavailable")
    return torch.device(requested)


class RelationModel:
    def __new__(cls, *, use_graph: bool) -> Any:
        from torch import nn
        from torch_geometric.nn import GATv2Conv

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_graph = use_graph
                self.class_embedding = nn.Embedding(len(VIDOR_OBJECTS), 32)
                self.node_projection = nn.Sequential(
                    nn.Linear(len(NODE_FEATURE_NAMES) + 32, 128), nn.ReLU(), nn.Dropout(0.2)
                )
                if use_graph:
                    self.gat1 = GATv2Conv(128, 64, heads=2, edge_dim=len(PAIR_FEATURE_NAMES))
                    self.gat2 = GATv2Conv(128, 128, heads=1, edge_dim=len(PAIR_FEATURE_NAMES))
                    self.norm1 = nn.LayerNorm(128)
                    self.norm2 = nn.LayerNorm(128)
                self.classifier = nn.Sequential(
                    nn.Linear(128 * 2 + len(PAIR_FEATURE_NAMES), 192),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(192, len(VIDOR_PREDICATES)),
                )

            def encode(self, x: Any, class_ids: Any, edge_index: Any, edge_attr: Any) -> Any:
                hidden = self.node_projection(
                    __import__("torch").cat((x, self.class_embedding(class_ids)), dim=1)
                )
                if not self.use_graph:
                    return hidden
                first = __import__("torch").relu(
                    self.gat1(hidden, edge_index, edge_attr=edge_attr)
                )
                hidden = self.norm1(hidden + first)
                second = __import__("torch").relu(
                    self.gat2(hidden, edge_index, edge_attr=edge_attr)
                )
                return self.norm2(hidden + second)

            def forward(
                self, x: Any, class_ids: Any, edge_index: Any, edge_attr: Any
            ) -> Any:
                hidden = self.encode(x, class_ids, edge_index, edge_attr)
                source, target = edge_index
                return self.classifier(
                    __import__("torch").cat(
                        (hidden[source], hidden[target], edge_attr), dim=1
                    )
                )

        return _Model()


def _thresholds(targets: Any, probabilities: Any, np: Any) -> Any:
    values = np.ones(len(VIDOR_PREDICATES), dtype=np.float32)
    for index in range(len(VIDOR_PREDICATES)):
        truth = targets[:, index].astype(bool)
        if not truth.any():
            continue
        best = (0.0, 0.5)
        for threshold in np.arange(0.05, 0.86, 0.05):
            predicted = probabilities[:, index] >= threshold
            tp = int((predicted & truth).sum())
            fp = int((predicted & ~truth).sum())
            fn = int((~predicted & truth).sum())
            score = 2 * tp / max(1, 2 * tp + fp + fn)
            if score > best[0]:
                best = (score, float(threshold))
        values[index] = best[1]
    return values


def _metrics(targets: Any, probabilities: Any, thresholds: Any, np: Any) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score  # type: ignore[import-not-found]

    predictions = probabilities >= thresholds
    supported = targets.sum(axis=0) > 0
    rows = []
    f1_values = []
    average_precisions = []
    for index, predicate in enumerate(VIDOR_PREDICATES):
        truth = targets[:, index].astype(bool)
        predicted = predictions[:, index]
        tp = int((truth & predicted).sum())
        fp = int((~truth & predicted).sum())
        fn = int((truth & ~predicted).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        support = int(truth.sum())
        ap = float(average_precision_score(truth, probabilities[:, index])) if support else None
        if support:
            f1_values.append(f1)
            if ap is not None:
                average_precisions.append(ap)
        rows.append(
            {
                "predicate": predicate,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
                "average_precision": round(ap, 6) if ap is not None else None,
                "threshold": round(float(thresholds[index]), 4),
            }
        )
    truth_flat = targets.astype(bool).ravel()
    predicted_flat = predictions.ravel()
    tp = int((truth_flat & predicted_flat).sum())
    fp = int((~truth_flat & predicted_flat).sum())
    fn = int((truth_flat & ~predicted_flat).sum())
    return {
        "macro_f1_supported_predicates": round(float(np.mean(f1_values)), 6),
        "micro_f1": round(2 * tp / max(1, 2 * tp + fp + fn), 6),
        "mean_average_precision_supported_predicates": round(
            float(np.mean(average_precisions)), 6
        ),
        "supported_predicates": int(supported.sum()),
        "candidate_edges": int(targets.shape[0]),
        "positive_labels": int(targets.sum()),
        "per_predicate": rows,
    }


def _graph_split(graphs: Sequence[Any], split: dict[str, list[str]]) -> dict[str, list[Any]]:
    return {
        name: [graph for graph in graphs if str(graph.video_id) in set(split[name])]
        for name in ("train", "validation", "test")
    }


def _collect_targets(graphs: Sequence[Any], torch: Any) -> Any:
    return torch.cat([graph.edge_y for graph in graphs], dim=0)


def _geometry_probabilities(graphs: Sequence[Any], torch: Any) -> tuple[Any, Any]:
    probabilities = []
    targets = []
    predicate_id = {name: index for index, name in enumerate(VIDOR_PREDICATES)}
    for graph in graphs:
        output = torch.zeros_like(graph.edge_y)
        dx, dy = graph.edge_attr[:, 0], graph.edge_attr[:, 1]
        distance, iou = graph.edge_attr[:, 2], graph.edge_attr[:, 3]
        output[:, predicate_id["next_to"]] = (distance < 0.25).float() * 0.9
        output[:, predicate_id["above"]] = ((dy > 0.12) & (dx.abs() < 0.35)).float() * 0.9
        output[:, predicate_id["beneath"]] = ((dy < -0.12) & (dx.abs() < 0.35)).float() * 0.9
        output[:, predicate_id["inside"]] = (iou > 0.7).float() * 0.9
        probabilities.append(output)
        targets.append(graph.edge_y)
    return torch.cat(targets).numpy(), torch.cat(probabilities).numpy()


def _class_pair_prior(
    train: Sequence[Any], selected: Sequence[Any], torch: Any
) -> tuple[Any, Any]:
    positives: defaultdict[tuple[int, int], Any] = defaultdict(
        lambda: torch.zeros(len(VIDOR_PREDICATES))
    )
    totals: Counter[tuple[int, int]] = Counter()
    for graph in train:
        for edge, (source, target) in enumerate(graph.edge_index.T.tolist()):
            key = (int(graph.class_ids[source]), int(graph.class_ids[target]))
            positives[key] += graph.edge_y[edge]
            totals[key] += 1
    output = []
    targets = []
    for graph in selected:
        rows = []
        for source, target in graph.edge_index.T.tolist():
            key = (int(graph.class_ids[source]), int(graph.class_ids[target]))
            rows.append((positives[key] + 1.0) / (totals[key] + 2.0))
        output.append(torch.stack(rows))
        targets.append(graph.edge_y)
    return torch.cat(targets).numpy(), torch.cat(output).numpy()


def train_relation_model(
    *,
    graphs: Sequence[Any],
    split: dict[str, list[str]],
    output_dir: Path,
    model_version: str,
    use_graph: bool,
    epochs: int = 40,
    patience: int = 7,
    batch_size: int = 16,
    seed: int = 2026,
    device_name: str = "auto",
) -> dict[str, Any]:
    np, torch = _seed(seed)
    from torch import nn
    from torch_geometric.loader import DataLoader

    started = time.perf_counter()
    device = _device(torch, device_name)
    selected = _graph_split(graphs, split)
    if any(not rows for rows in selected.values()):
        raise ValueError("one or more relationship splits contain no graphs")
    train_targets = _collect_targets(selected["train"], torch)
    positives = train_targets.sum(dim=0)
    negatives = train_targets.shape[0] - positives
    positive_weights = torch.clamp(negatives / torch.clamp(positives, min=1), max=30.0)
    model_factory: Any = RelationModel(use_graph=use_graph)
    model = model_factory.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loader(name: str, shuffle: bool = False) -> Any:
        return DataLoader(
            selected[name], batch_size=batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
        )

    def forward(batch: Any) -> Any:
        return model(batch.x, batch.class_ids, batch.edge_index, batch.edge_attr)

    def probabilities(name: str) -> tuple[Any, Any]:
        predictions, targets = [], []
        model.eval()
        with torch.inference_mode():
            for raw in loader(name):
                batch = raw.to(device)
                predictions.append(torch.sigmoid(forward(batch)).cpu())
                targets.append(batch.edge_y.cpu())
        return torch.cat(targets).numpy(), torch.cat(predictions).numpy()

    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        for raw in loader("train", True):
            batch = raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(forward(batch), batch.edge_y)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for raw in loader("validation"):
                batch = raw.to(device)
                validation_losses.append(float(loss_function(forward(batch), batch.edge_y).cpu()))
        train_loss = sum(training_losses) / len(training_losses)
        validation_loss = sum(validation_losses) / len(validation_losses)
        history.append(
            {"epoch": epoch, "training_loss": round(train_loss, 8),
             "validation_loss": round(validation_loss, 8)}
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("relation model did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_targets, validation_probabilities = probabilities("validation")
    thresholds = _thresholds(validation_targets, validation_probabilities, np)
    test_targets, test_probabilities = probabilities("test")
    sample = next(iter(loader("test"))).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):
            forward(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = (time.perf_counter() - latency_started) * 1000 / (20 * int(sample.num_graphs))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{model_version}.pt"
    torch.save(
        {
            "schema_version": VIDOR_RELATION_CHECKPOINT_SCHEMA,
            "model_version": model_version,
            "state_dict": best_state,
            "thresholds": torch.from_numpy(thresholds),
            "object_classes": list(VIDOR_OBJECTS),
            "predicates": list(VIDOR_PREDICATES),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "use_graph": use_graph,
        },
        checkpoint,
    )
    metadata = {
        "schema_version": VIDOR_RELATION_CHECKPOINT_SCHEMA,
        "model_version": model_version,
        "task": "explicit directed VidOR subject-predicate-object edge classification",
        "architecture": "two-layer GATv2 edge classifier" if use_graph else "pairwise MLP",
        "feature_scope": "annotation class, bounding-box, and pair geometry; no pixel features",
        "background_policy": (
            "candidate edges with no active predicate are all-zero multilabel negatives"
        ),
        "split": split,
        "graph_counts": {name: len(rows) for name, rows in selected.items()},
        "edge_counts": {
            name: int(_collect_targets(rows, torch).shape[0])
            for name, rows in selected.items()
        },
        "validation_metrics": _metrics(
            validation_targets, validation_probabilities, thresholds, np
        ),
        "test_metrics": _metrics(test_targets, test_probabilities, thresholds, np),
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
        "inference_latency_mean_milliseconds_per_graph": round(latency, 4),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (output_dir / f"{model_version}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def build_and_train_vidor(
    *,
    annotation_zip: Path,
    output_dir: Path,
    target_videos: int = 500,
    max_frames_per_video: int = 6,
    nearest_k: int = 8,
    epochs: int = 40,
    patience: int = 7,
    batch_size: int = 16,
    seed: int = 2026,
    device_name: str = "auto",
) -> dict[str, Any]:
    np, torch = _seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = validate_vidor_label_map(annotation_zip)
    summaries, global_support = _annotation_summaries(annotation_zip)
    selected = select_relation_videos(summaries, global_support, target_videos)
    split = strict_video_split([item["video_id"] for item in selected], seed)
    graphs, graph_statistics = build_relation_graphs(
        annotation_zip,
        selected,
        global_support,
        max_frames_per_video=max_frames_per_video,
        nearest_k=nearest_k,
    )
    cache = output_dir / "vidor_relation_graphs.pt"
    torch.save(
        {
            "schema_version": VIDOR_RELATION_SCHEMA,
            "graphs": graphs,
            "object_classes": list(VIDOR_OBJECTS),
            "predicates": list(VIDOR_PREDICATES),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
        },
        cache,
    )
    reloaded = torch.load(cache, map_location="cpu", weights_only=False)
    valid_cache = (
        reloaded.get("schema_version") == VIDOR_RELATION_SCHEMA
        and len(reloaded["graphs"]) == len(graphs)
    )
    if not valid_cache:
        raise ValueError("VidOR graph cache validation failed")
    manifest = {
        "schema_version": "vidor-relation-split-v1",
        "seed": seed,
        "selection": "rare-predicate-weighted deterministic selection",
        "target_videos": target_videos,
        "selected_videos": selected,
        **split,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    vocabulary = {
        "schema_version": "vidquery-relation-vocabulary-v1",
        "source_dataset": "VidOR",
        "predicates": [
            {
                "predicate_id": index,
                "canonical_name": name,
                "aliases": [],
                "symmetric": name in SYMMETRIC_PREDICATES,
                "directional": name not in SYMMETRIC_PREDICATES,
                "kind": "spatial" if name in SPATIAL_PREDICATES else "interaction",
            }
            for index, name in enumerate(VIDOR_PREDICATES)
        ],
    }
    (output_dir / "relationship_vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    train_graphs = _graph_split(graphs, split)
    geometry_targets, geometry_probabilities = _geometry_probabilities(
        train_graphs["test"], torch
    )
    geometry_thresholds = np.full(len(VIDOR_PREDICATES), 0.5, dtype=np.float32)
    prior_val_targets, prior_val_probabilities = _class_pair_prior(
        train_graphs["train"], train_graphs["validation"], torch
    )
    prior_thresholds = _thresholds(prior_val_targets, prior_val_probabilities, np)
    prior_targets, prior_probabilities = _class_pair_prior(
        train_graphs["train"], train_graphs["test"], torch
    )
    baselines = {
        "geometry_rules": _metrics(
            geometry_targets, geometry_probabilities, geometry_thresholds, np
        ),
        "class_pair_prior": _metrics(
            prior_targets, prior_probabilities, prior_thresholds, np
        ),
    }
    mlp = train_relation_model(
        graphs=graphs,
        split=split,
        output_dir=output_dir,
        model_version="vidor-relation-mlp-v1",
        use_graph=False,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device_name=device_name,
    )
    gnn = train_relation_model(
        graphs=graphs,
        split=split,
        output_dir=output_dir,
        model_version="vidor-relation-gat-v1",
        use_graph=True,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device_name=device_name,
    )
    report = {
        "schema_version": "vidor-relation-training-report-v1",
        "selection_policy": "validation metrics only; test metrics never choose checkpoints",
        "source": coverage,
        "dataset": {
            **graph_statistics,
            "selected_videos": target_videos,
            "split_video_counts": {name: len(split[name]) for name in split},
            "cache": cache.as_posix(),
            "cache_sha256": _sha256(cache),
        },
        "baselines": baselines,
        "pairwise_mlp": mlp,
        "relationship_gnn": gnn,
        "validation_comparison": {
            "pairwise_mlp": mlp["validation_metrics"],
            "relationship_gnn": gnn["validation_metrics"],
        },
    }
    report_path = output_dir / "held_out_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


YOLO_TO_VIDOR = {
    "person": "adult",
    "sports ball": "ball/sports_ball",
    "cow": "cattle/cow",
    "cell phone": "cellphone",
    "dining table": "table",
    "tv": "screen/monitor",
    "monitor": "screen/monitor",
    "potted plant": "toy",
    "boat": "watercraft",
    "truck": "bus/truck",
    "bus": "bus/truck",
    "mouse": "hamster/rat",
    "sheep": "sheep/goat",
    "goat": "sheep/goat",
}


def mapped_vidor_class(class_label: str) -> str | None:
    normalized = class_label.strip().lower().replace("_", " ")
    direct = normalized.replace(" ", "_")
    if direct in VIDOR_OBJECTS:
        return direct
    return YOLO_TO_VIDOR.get(normalized)


def relation_gnn_status(
    checkpoint: Path, metadata: Path, expected_model_version: str | None = None
) -> str:
    if not checkpoint.is_file() or not metadata.is_file():
        return "unavailable_missing_checkpoint"
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        details = json.loads(metadata.read_text(encoding="utf-8"))
        version = payload.get("model_version")
        if payload.get("schema_version") != VIDOR_RELATION_CHECKPOINT_SCHEMA:
            return "unavailable_incompatible_schema"
        if details.get("schema_version") != VIDOR_RELATION_CHECKPOINT_SCHEMA:
            return "unavailable_incompatible_metadata"
        if version != details.get("model_version"):
            return "unavailable_model_version_mismatch"
        if expected_model_version and version != expected_model_version:
            return "unavailable_model_version_mismatch"
        if payload.get("object_classes") != list(VIDOR_OBJECTS):
            return "unavailable_object_map_mismatch"
        if payload.get("predicates") != list(VIDOR_PREDICATES):
            return "unavailable_predicate_map_mismatch"
        if payload.get("node_feature_names") != list(NODE_FEATURE_NAMES):
            return "unavailable_feature_schema_mismatch"
        if payload.get("pair_feature_names") != list(PAIR_FEATURE_NAMES):
            return "unavailable_feature_schema_mismatch"
        if details.get("checkpoint_sha256") != _sha256(checkpoint):
            return "unavailable_checkpoint_hash_mismatch"
    except Exception:
        return "unavailable_load_error"
    return "available_validated"


class VidORRelationPredictor:
    def __init__(
        self,
        checkpoint: Path,
        metadata: Path,
        *,
        device: str = "cpu",
        expected_model_version: str | None = None,
    ) -> None:
        import torch

        status = relation_gnn_status(checkpoint, metadata, expected_model_version)
        if status != "available_validated":
            raise ValueError(f"VidOR relationship GNN checkpoint is invalid: {status}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model_version = str(payload["model_version"])
        self.device = torch.device(device)
        model_factory: Any = RelationModel(use_graph=bool(payload["use_graph"]))
        self.model = model_factory.to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.thresholds = payload["thresholds"].to(self.device)

    def predict_detections(
        self, detections: Sequence[Any], *, nearest_k: int = 8, max_predicates: int = 3
    ) -> list[dict[str, Any]]:
        import torch

        mapped = [
            (detection, mapped_vidor_class(str(detection.class_label)))
            for detection in detections
        ]
        mapped = [(detection, label) for detection, label in mapped if label is not None]
        if len(mapped) < 2:
            return []
        features = []
        for detection, label in mapped:
            box = detection.normalized_bbox
            width, height = box.x2 - box.x1, box.y2 - box.y1
            features.append(
                [
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                    width * height,
                    width / max(height, 1e-6),
                    float(detection.confidence),
                    float(label in PERSON_CLASSES),
                ]
            )
        candidates = _candidate_pairs(features, nearest_k)
        if not candidates:
            return []
        class_to_id = {name: index for index, name in enumerate(VIDOR_OBJECTS)}
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        class_ids = torch.tensor(
            [class_to_id[str(label)] for _, label in mapped],
            dtype=torch.long,
            device=self.device,
        )
        edge_index = torch.tensor(candidates, dtype=torch.long, device=self.device).T
        edge_attr = torch.tensor(
            [_pair_features(features[source], features[target]) for source, target in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            probabilities = torch.sigmoid(
                self.model(x, class_ids, edge_index, edge_attr)
            )
        predictions: list[dict[str, Any]] = []
        for edge, (source, target) in enumerate(candidates):
            eligible = torch.nonzero(
                probabilities[edge] >= self.thresholds, as_tuple=False
            ).flatten()
            ranked = sorted(
                eligible.tolist(),
                key=lambda index: float(probabilities[edge, index]),
                reverse=True,
            )[:max_predicates]
            source_detection, source_class = mapped[source]
            target_detection, target_class = mapped[target]
            for predicate_id in ranked:
                predictions.append(
                    {
                        "source_id": str(source_detection.detection_id),
                        "source_class": source_class,
                        "predicate": VIDOR_PREDICATES[predicate_id],
                        "target_id": str(target_detection.detection_id),
                        "target_class": target_class,
                        "confidence": float(probabilities[edge, predicate_id]),
                        "threshold": float(self.thresholds[predicate_id]),
                        "model_version": self.model_version,
                    }
                )
        return predictions
