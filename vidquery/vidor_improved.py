"""Focused VidOR pair-visual experiment with controlled negative sampling.

This module is deliberately separate from :mod:`vidquery.vidor_relation` so the
validated v1 checkpoints remain loadable.  It requires real decoded VidOR
frames; missing pixels are an error rather than silently substituting zeros.
"""

from __future__ import annotations

import copy
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

from .vidor_relation import (
    NODE_FEATURE_NAMES,
    PERSON_CLASSES,
    VIDOR_OBJECTS,
    VIDOR_PREDICATES,
    _bbox_features,
    _candidate_pairs,
    _device,
    _graph_split,
    _metrics,
    _pair_features,
    _seed,
    _selected_frames,
    _sha256,
    _thresholds,
)

VIDOR_IMPROVED_GRAPH_SCHEMA = "vidor-relation-graphs-v2-pair-visual"
VIDOR_IMPROVED_CHECKPOINT_SCHEMA = "vidor-relation-model-v2-pair-visual"
VISUAL_FEATURE_DIM = 512
MOTION_FEATURE_NAMES = (
    "center_delta_x",
    "center_delta_y",
    "log_area_ratio_to_previous",
    "previous_frame_visible",
)
PAIR_MOTION_FEATURE_NAMES = tuple(
    f"{role}_{name}" for role in ("subject", "object") for name in MOTION_FEATURE_NAMES
)
NODE_INPUT_FEATURE_NAMES = (
    *NODE_FEATURE_NAMES,
    *tuple(f"class_embedding_{index}" for index in range(32)),
    *tuple(f"visual_{index}" for index in range(VISUAL_FEATURE_DIM)),
    *MOTION_FEATURE_NAMES,
)
IMPROVED_PAIR_FEATURE_NAMES = (
    *tuple(f"subject_{name}" for name in NODE_INPUT_FEATURE_NAMES),
    *tuple(f"object_{name}" for name in NODE_INPUT_FEATURE_NAMES),
    *tuple(
        f"geometry_{name}"
        for name in (
            "relative_x",
            "relative_y",
            "center_distance",
            "iou",
            "log_size_ratio",
            "subject_left_of_object",
            "subject_above_object",
            "overlap",
        )
    ),
    *tuple(f"union_visual_{index}" for index in range(VISUAL_FEATURE_DIM)),
)

POSITIVE_EDGE = 0
HARD_NEGATIVE_EDGE = 1
RANDOM_NEGATIVE_EDGE = 2


def union_box(
    subject: Sequence[float], object_: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return the normalized box enclosing both ordered pair members."""

    return (
        min(float(subject[0]), float(object_[0])),
        min(float(subject[1]), float(object_[1])),
        max(float(subject[2]), float(object_[2])),
        max(float(subject[3]), float(object_[3])),
    )


def _pixel_crop(image: Any, box: Sequence[float]) -> Any:
    width, height = image.size
    x1 = max(0, min(width - 1, int(math.floor(float(box[0]) * width))))
    y1 = max(0, min(height - 1, int(math.floor(float(box[1]) * height))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(box[2]) * width))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(box[3]) * height))))
    return image.crop((x1, y1, x2, y2))


class FrozenResNetPairEncoder:
    """Frozen ImageNet ResNet-18 pooling used once during graph caching."""

    feature_dim = VISUAL_FEATURE_DIM
    encoder_name = "torchvision-resnet18-imagenet1k-v1-frozen"

    def __init__(self, *, device: str = "auto") -> None:
        import torch
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        self.device = _device(torch, device)
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=self.weights)
        model.fc = nn.Identity()
        self.model = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = self.weights.transforms()

    def encode(
        self,
        image: Any,
        node_boxes: Sequence[Sequence[float]],
        pair_boxes: Sequence[Sequence[float]],
    ) -> tuple[Any, Any]:
        import torch

        crops = [_pixel_crop(image, box) for box in (*node_boxes, *pair_boxes)]
        if not crops:
            empty = torch.empty((0, self.feature_dim), dtype=torch.float32)
            return empty, empty
        batch = torch.stack([self.transform(crop.convert("RGB")) for crop in crops])
        with torch.inference_mode():
            encoded = self.model(batch.to(self.device)).float().cpu()
        node_count = len(node_boxes)
        return encoded[:node_count].contiguous(), encoded[node_count:].contiguous()


def _frame_path(frame_root: Path, payload: dict[str, Any], frame_id: int) -> Path:
    video_path = Path(str(payload["video_path"]))
    group, video_id = video_path.parent.name, video_path.stem
    directory = frame_root / group / video_id
    candidates = (
        directory / f"frame_{frame_id + 1:06d}.jpg",
        directory / f"frame_{frame_id:06d}.jpg",
        directory / f"{frame_id + 1:06d}.jpg",
        directory / f"{frame_id:06d}.jpg",
        directory / f"{frame_id + 1:05d}.jpg",
        directory / f"{frame_id:05d}.jpg",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"VidOR frame pixels are required for {group}/{video_id} frame {frame_id}; "
        f"expected an extracted JPEG under {directory}"
    )


def _normalized_box(row: dict[str, Any], width: float, height: float) -> list[float]:
    return _bbox_features(row["bbox"], width, height, int(row.get("generated", 0)), False)[:4]


def _motion_features(
    payload: dict[str, Any], observations: Sequence[dict[str, Any]], frame_id: int
) -> list[list[float]]:
    width, height = float(payload["width"]), float(payload["height"])
    previous = {}
    if frame_id > 0:
        previous = {int(row["tid"]): row for row in payload["trajectories"][frame_id - 1]}
    output: list[list[float]] = []
    for row in observations:
        current_box = _normalized_box(row, width, height)
        prior = previous.get(int(row["tid"]))
        if prior is None:
            output.append([0.0, 0.0, 0.0, 0.0])
            continue
        prior_box = _normalized_box(prior, width, height)
        current_center = (
            (current_box[0] + current_box[2]) / 2,
            (current_box[1] + current_box[3]) / 2,
        )
        prior_center = (
            (prior_box[0] + prior_box[2]) / 2,
            (prior_box[1] + prior_box[3]) / 2,
        )
        current_area = max(
            (current_box[2] - current_box[0]) * (current_box[3] - current_box[1]),
            1e-6,
        )
        prior_area = max(
            (prior_box[2] - prior_box[0]) * (prior_box[3] - prior_box[1]),
            1e-6,
        )
        output.append(
            [
                current_center[0] - prior_center[0],
                current_center[1] - prior_center[1],
                math.log(current_area / prior_area),
                1.0,
            ]
        )
    return output


def controlled_candidate_pairs(
    features: Sequence[Sequence[float]],
    class_names: Sequence[str],
    positive_pairs: set[tuple[int, int]],
    *,
    hard_negative_ratio: float = 2.0,
    random_negative_ratio: float = 1.0,
    hard_distance: float = 0.45,
    seed: int = 2026,
) -> tuple[list[tuple[int, int]], list[int], dict[str, int]]:
    """Keep every positive, then add nearby plausible and background negatives."""

    all_pairs = [
        (source, target)
        for source in range(len(features))
        for target in range(len(features))
        if source != target
    ]
    available = [pair for pair in all_pairs if pair not in positive_pairs]

    def plausible(pair: tuple[int, int]) -> bool:
        source, target = pair
        geometry = _pair_features(features[source], features[target])
        involves_person = (
            class_names[source] in PERSON_CLASSES or class_names[target] in PERSON_CLASSES
        )
        return involves_person and (geometry[2] <= hard_distance or bool(geometry[7]))

    hard_pool = sorted(
        (pair for pair in available if plausible(pair)),
        key=lambda pair: (_pair_features(features[pair[0]], features[pair[1]])[2], pair),
    )
    budget_base = max(1, len(positive_pairs))
    hard_budget = min(len(hard_pool), max(1, round(budget_base * hard_negative_ratio)))
    hard = hard_pool[:hard_budget]
    hard_set = set(hard)
    background_pool = [pair for pair in available if pair not in hard_set]
    generator = random.Random(seed)
    generator.shuffle(background_pool)
    random_budget = min(len(background_pool), max(1, round(budget_base * random_negative_ratio)))
    background = sorted(background_pool[:random_budget])
    positives = sorted(positive_pairs)
    pairs = positives + hard + background
    edge_types = (
        [POSITIVE_EDGE] * len(positives)
        + [HARD_NEGATIVE_EDGE] * len(hard)
        + [RANDOM_NEGATIVE_EDGE] * len(background)
    )
    return (
        pairs,
        edge_types,
        {
            "positive": len(positives),
            "hard_nearby_negative": len(hard),
            "random_background_negative": len(background),
            "all_possible_directed_pairs": len(all_pairs),
        },
    )


def build_improved_relation_graphs(
    annotation_zip: Path,
    selected: Sequence[dict[str, Any]],
    predicate_support: Counter[str],
    *,
    frame_root: Path,
    visual_encoder: Any | None = None,
    max_frames_per_video: int = 6,
    hard_negative_ratio: float = 2.0,
    random_negative_ratio: float = 1.0,
    seed: int = 2026,
) -> tuple[list[Any], dict[str, Any]]:
    """Build pair-visual graphs, failing if any selected frame image is missing."""

    import torch
    from PIL import Image
    from torch_geometric.data import Data

    encoder = visual_encoder or FrozenResNetPairEncoder(device="auto")
    if int(encoder.feature_dim) != VISUAL_FEATURE_DIM:
        raise ValueError(
            f"VidOR visual encoder produced {encoder.feature_dim} dimensions; "
            f"expected {VISUAL_FEATURE_DIM}"
        )
    object_to_id = {name: index for index, name in enumerate(VIDOR_OBJECTS)}
    predicate_to_id = {name: index for index, name in enumerate(VIDOR_PREDICATES)}
    selected_members = {item["member"] for item in selected}
    graphs: list[Any] = []
    counts: Counter[str] = Counter()
    feasible_positive_instances = 0
    proposed_positive_instances = 0
    with zipfile.ZipFile(annotation_zip) as archive:
        for member in sorted(selected_members):
            payload = json.loads(archive.read(member))
            relations = payload["relation_instances"]
            categories = {
                int(item["tid"]): str(item["category"]) for item in payload["subject/objects"]
            }
            frame_ids = _selected_frames(relations, predicate_support, max_frames_per_video)
            for frame_id in frame_ids:
                if frame_id >= len(payload["trajectories"]):
                    continue
                observations = sorted(
                    (
                        row
                        for row in payload["trajectories"][frame_id]
                        if int(row["tid"]) in categories
                    ),
                    key=lambda row: int(row["tid"]),
                )
                if len(observations) < 2:
                    continue
                tids = [int(row["tid"]) for row in observations]
                class_names = [categories[tid] for tid in tids]
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
                positive_pairs = set(active)
                graph_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}:{payload['video_id']}:{frame_id}".encode()).digest()[
                        :8
                    ],
                    "big",
                )
                candidates, edge_types, sampling = controlled_candidate_pairs(
                    node_features,
                    class_names,
                    positive_pairs,
                    hard_negative_ratio=hard_negative_ratio,
                    random_negative_ratio=random_negative_ratio,
                    seed=graph_seed,
                )
                if not candidates:
                    continue
                node_boxes = [feature[:4] for feature in node_features]
                pair_boxes = [
                    union_box(node_boxes[source], node_boxes[target])
                    for source, target in candidates
                ]
                frame_path = _frame_path(frame_root, payload, frame_id)
                with Image.open(frame_path) as image:
                    node_visual, union_visual = encoder.encode(
                        image.convert("RGB"), node_boxes, pair_boxes
                    )
                expected_node_shape = (len(node_features), VISUAL_FEATURE_DIM)
                expected_union_shape = (len(candidates), VISUAL_FEATURE_DIM)
                if tuple(node_visual.shape) != expected_node_shape:
                    raise ValueError(
                        f"node visual cache shape {tuple(node_visual.shape)} != "
                        f"{expected_node_shape}"
                    )
                if tuple(union_visual.shape) != expected_union_shape:
                    raise ValueError(
                        f"union visual cache shape {tuple(union_visual.shape)} != "
                        f"{expected_union_shape}"
                    )
                edge_y = torch.zeros((len(candidates), len(VIDOR_PREDICATES)), dtype=torch.float32)
                for edge, pair in enumerate(candidates):
                    for predicate in active.get(pair, set()):
                        edge_y[edge, predicate_to_id[predicate]] = 1.0
                graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    class_ids=torch.tensor(
                        [object_to_id[name] for name in class_names], dtype=torch.long
                    ),
                    node_visual=node_visual.float().contiguous(),
                    node_motion=torch.tensor(
                        _motion_features(payload, observations, frame_id),
                        dtype=torch.float32,
                    ),
                    edge_index=torch.tensor(candidates, dtype=torch.long).T.contiguous(),
                    edge_attr=torch.tensor(
                        [
                            _pair_features(node_features[source], node_features[target])
                            for source, target in candidates
                        ],
                        dtype=torch.float32,
                    ),
                    union_visual=union_visual.float().contiguous(),
                    edge_y=edge_y,
                    edge_sample_type=torch.tensor(edge_types, dtype=torch.long),
                )
                graph.video_id = str(payload["video_id"])
                graph.frame_id = frame_id
                graph.fps = float(payload["fps"])
                graph.timestamp = frame_id / max(float(payload["fps"]), 1e-6)
                graph.track_ids = [str(tid) for tid in tids]
                graph.class_names = class_names
                graph.frame_path = frame_path.as_posix()
                graphs.append(graph)
                feasible_positive_instances += len(active_instances)
                proposed_positive_instances += len(active_instances)
                counts.update(sampling)
    if not graphs:
        raise ValueError("VidOR improved graph construction produced no graphs")
    return graphs, {
        "graphs": len(graphs),
        "candidate_edges": sum(int(graph.edge_y.shape[0]) for graph in graphs),
        "positive_candidate_edges": counts["positive"],
        "feasible_positive_instances": feasible_positive_instances,
        "proposed_positive_instances": proposed_positive_instances,
        "candidate_edge_recall": (
            proposed_positive_instances / feasible_positive_instances
            if feasible_positive_instances
            else 0.0
        ),
        "negative_sampling": {
            "policy": (
                "all positives + nearest plausible hard negatives + "
                "seeded random/background negatives"
            ),
            "hard_negative_ratio": hard_negative_ratio,
            "random_negative_ratio": random_negative_ratio,
            "positive_edges": counts["positive"],
            "hard_nearby_negative_edges": counts["hard_nearby_negative"],
            "random_background_negative_edges": counts["random_background_negative"],
            "all_possible_directed_pairs": counts["all_possible_directed_pairs"],
        },
        "visual_cache": {
            "encoder": str(encoder.encoder_name),
            "feature_dim": VISUAL_FEATURE_DIM,
            "subject_roi": True,
            "object_roi": True,
            "union_box": True,
            "computed_once_and_cached": True,
        },
    }


def raw_pair_inputs(
    x: Any,
    class_embedding: Any,
    class_ids: Any,
    node_visual: Any,
    node_motion: Any,
    edge_index: Any,
    edge_attr: Any,
    union_visual: Any,
) -> Any:
    """Assemble the exact shared pair input used by both fair baselines."""

    import torch

    source, target = edge_index
    node_inputs = torch.cat((x, class_embedding(class_ids), node_visual, node_motion), dim=1)
    return torch.cat(
        (
            node_inputs[source],
            node_inputs[target],
            edge_attr,
            union_visual,
        ),
        dim=1,
    )


class ImprovedRelationModel:
    """Matched pair MLP/GATv2; graph mode only adds message-passing context."""

    def __new__(cls, *, use_graph: bool) -> Any:
        import torch
        from torch import nn
        from torch_geometric.nn import GATv2Conv

        node_input_dim = (
            len(NODE_FEATURE_NAMES) + 32 + VISUAL_FEATURE_DIM + len(MOTION_FEATURE_NAMES)
        )
        pair_input_dim = node_input_dim * 2 + 8 + VISUAL_FEATURE_DIM

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.use_graph = use_graph
                self.class_embedding = nn.Embedding(len(VIDOR_OBJECTS), 32)
                self.pair_encoder = nn.Sequential(
                    nn.Linear(pair_input_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                )
                if use_graph:
                    self.node_projection = nn.Sequential(nn.Linear(node_input_dim, 192), nn.ReLU())
                    self.edge_projection = nn.Sequential(
                        nn.Linear(8 + VISUAL_FEATURE_DIM, 64), nn.ReLU()
                    )
                    self.gat1 = GATv2Conv(192, 96, heads=2, edge_dim=64)
                    self.gat2 = GATv2Conv(192, 192, heads=1, edge_dim=64)
                    self.norm1 = nn.LayerNorm(192)
                    self.norm2 = nn.LayerNorm(192)
                    classifier_dim = 256 + 192 * 2
                else:
                    classifier_dim = 256
                self.classifier = nn.Sequential(
                    nn.Linear(classifier_dim, 192),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(192, len(VIDOR_PREDICATES)),
                )

            def pair_inputs(
                self,
                x: Any,
                class_ids: Any,
                node_visual: Any,
                node_motion: Any,
                edge_index: Any,
                edge_attr: Any,
                union_visual: Any,
            ) -> Any:
                return raw_pair_inputs(
                    x,
                    self.class_embedding,
                    class_ids,
                    node_visual,
                    node_motion,
                    edge_index,
                    edge_attr,
                    union_visual,
                )

            def forward(
                self,
                x: Any,
                class_ids: Any,
                node_visual: Any,
                node_motion: Any,
                edge_index: Any,
                edge_attr: Any,
                union_visual: Any,
            ) -> Any:
                pair_raw = self.pair_inputs(
                    x,
                    class_ids,
                    node_visual,
                    node_motion,
                    edge_index,
                    edge_attr,
                    union_visual,
                )
                pair_hidden = self.pair_encoder(pair_raw)
                if not self.use_graph:
                    return self.classifier(pair_hidden)
                node_raw = torch.cat(
                    (x, self.class_embedding(class_ids), node_visual, node_motion), dim=1
                )
                hidden = self.node_projection(node_raw)
                message_edge = self.edge_projection(torch.cat((edge_attr, union_visual), dim=1))
                first = torch.relu(self.gat1(hidden, edge_index, edge_attr=message_edge))
                hidden = self.norm1(hidden + first)
                second = torch.relu(self.gat2(hidden, edge_index, edge_attr=message_edge))
                hidden = self.norm2(hidden + second)
                source, target = edge_index
                return self.classifier(
                    torch.cat((pair_hidden, hidden[source], hidden[target]), dim=1)
                )

        return _Model()


def validate_improved_graph_cache(payload: dict[str, Any]) -> tuple[bool, str]:
    if payload.get("schema_version") != VIDOR_IMPROVED_GRAPH_SCHEMA:
        return False, "incompatible_schema"
    if payload.get("node_feature_names") != list(NODE_FEATURE_NAMES):
        return False, "node_feature_schema_mismatch"
    if payload.get("motion_feature_names") != list(MOTION_FEATURE_NAMES):
        return False, "motion_feature_schema_mismatch"
    if payload.get("visual_feature_dim") != VISUAL_FEATURE_DIM:
        return False, "visual_feature_schema_mismatch"
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        return False, "missing_graphs"
    for graph in graphs:
        if graph.node_visual.shape[1] != VISUAL_FEATURE_DIM:
            return False, "node_visual_shape_mismatch"
        if graph.union_visual.shape != (graph.edge_y.shape[0], VISUAL_FEATURE_DIM):
            return False, "union_visual_shape_mismatch"
    return True, "valid"


def _collect_targets(graphs: Sequence[Any], torch: Any) -> Any:
    return torch.cat([graph.edge_y for graph in graphs], dim=0)


def train_improved_relation_model(
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
    """Train with validation macro-F1/mAP selection and touch test only afterward."""

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
    model: Any = ImprovedRelationModel(use_graph=use_graph)
    model = model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    def loader(name: str, shuffle: bool = False) -> Any:
        return DataLoader(
            selected[name],
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
        )

    def forward(batch: Any) -> Any:
        return model(
            batch.x,
            batch.class_ids,
            batch.node_visual,
            batch.node_motion,
            batch.edge_index,
            batch.edge_attr,
            batch.union_visual,
        )

    def probabilities(name: str) -> tuple[Any, Any]:
        predictions, targets = [], []
        model.eval()
        with torch.inference_mode():
            for raw in loader(name):
                batch = raw.to(device)
                predictions.append(torch.sigmoid(forward(batch)).cpu())
                targets.append(batch.edge_y.cpu())
        return torch.cat(targets).numpy(), torch.cat(predictions).numpy()

    best_key = (-1.0, -1.0)
    best_state = None
    best_thresholds = None
    stale = 0
    history: list[dict[str, Any]] = []
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
        validation_targets, validation_probabilities = probabilities("validation")
        epoch_thresholds = _thresholds(validation_targets, validation_probabilities, np)
        epoch_metrics = _metrics(validation_targets, validation_probabilities, epoch_thresholds, np)
        selection_key = (
            float(epoch_metrics["macro_f1_supported_predicates"]),
            float(epoch_metrics["mean_average_precision_supported_predicates"]),
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": round(sum(training_losses) / len(training_losses), 8),
                "validation_macro_f1_supported_predicates": selection_key[0],
                "validation_mean_average_precision_supported_predicates": selection_key[1],
            }
        )
        if selection_key > best_key:
            best_key = selection_key
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            best_thresholds = epoch_thresholds.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_thresholds is None:
        raise RuntimeError("improved relation model did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_targets, validation_probabilities = probabilities("validation")
    validation_metrics = _metrics(validation_targets, validation_probabilities, best_thresholds, np)
    # Test is evaluated exactly once, after validation has frozen state and thresholds.
    test_targets, test_probabilities = probabilities("test")
    test_metrics = _metrics(test_targets, test_probabilities, best_thresholds, np)
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
            "schema_version": VIDOR_IMPROVED_CHECKPOINT_SCHEMA,
            "model_version": model_version,
            "state_dict": best_state,
            "thresholds": torch.from_numpy(best_thresholds),
            "object_classes": list(VIDOR_OBJECTS),
            "predicates": list(VIDOR_PREDICATES),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "motion_feature_names": list(MOTION_FEATURE_NAMES),
            "pair_input_feature_names": list(IMPROVED_PAIR_FEATURE_NAMES),
            "visual_feature_dim": VISUAL_FEATURE_DIM,
            "use_graph": use_graph,
        },
        checkpoint,
    )
    metadata = {
        "schema_version": VIDOR_IMPROVED_CHECKPOINT_SCHEMA,
        "model_version": model_version,
        "task": "explicit directed VidOR subject-predicate-object edge classification",
        "architecture": (
            "matched pair inputs plus two-layer GATv2 context"
            if use_graph
            else "pairwise MLP on matched pair inputs"
        ),
        "feature_scope": (
            "annotation/detector class, boxes, motion, subject ROI, object ROI, "
            "and union-box frozen ResNet embeddings"
        ),
        "fair_pair_input_schema": list(IMPROVED_PAIR_FEATURE_NAMES),
        "graph_only_advantage": "GATv2 neighborhood message passing/context",
        "selection_policy": (
            "validation supported-predicate macro F1; validation mAP tie-break; "
            "test evaluated after selection"
        ),
        "split": split,
        "graph_counts": {name: len(rows) for name, rows in selected.items()},
        "edge_counts": {
            name: int(_collect_targets(rows, torch).shape[0]) for name, rows in selected.items()
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "selected_epoch": int(
            max(
                history,
                key=lambda row: (
                    row["validation_macro_f1_supported_predicates"],
                    row["validation_mean_average_precision_supported_predicates"],
                ),
            )["epoch"]
        ),
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


def improved_relation_checkpoint_status(
    checkpoint: Path,
    metadata: Path,
    expected_model_version: str | None = None,
) -> str:
    if not checkpoint.is_file() or not metadata.is_file():
        return "unavailable_missing_checkpoint"
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        details = json.loads(metadata.read_text(encoding="utf-8"))
        version = payload.get("model_version")
        if payload.get("schema_version") != VIDOR_IMPROVED_CHECKPOINT_SCHEMA:
            return "unavailable_incompatible_schema"
        if details.get("schema_version") != VIDOR_IMPROVED_CHECKPOINT_SCHEMA:
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
        if payload.get("motion_feature_names") != list(MOTION_FEATURE_NAMES):
            return "unavailable_feature_schema_mismatch"
        if payload.get("pair_input_feature_names") != list(IMPROVED_PAIR_FEATURE_NAMES):
            return "unavailable_feature_schema_mismatch"
        if payload.get("visual_feature_dim") != VISUAL_FEATURE_DIM:
            return "unavailable_feature_schema_mismatch"
        if details.get("checkpoint_sha256") != _sha256(checkpoint):
            return "unavailable_checkpoint_hash_mismatch"
    except Exception:
        return "unavailable_load_error"
    return "available_validated"


def configured_relation_checkpoint_status(
    checkpoint: Path,
    metadata: Path,
    expected_model_version: str | None = None,
) -> str:
    """Validate either the pair-visual v2 artifact or a preserved legacy artifact.

    The configured model version is the authoritative selector.  This avoids
    treating an invalid v2 artifact as a legacy checkpoint and producing a
    misleading compatibility error.
    """

    if expected_model_version and "pair-visual" in expected_model_version:
        return improved_relation_checkpoint_status(
            checkpoint, metadata, expected_model_version
        )
    try:
        details = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        details = {}
    if details.get("schema_version") == VIDOR_IMPROVED_CHECKPOINT_SCHEMA:
        return improved_relation_checkpoint_status(
            checkpoint, metadata, expected_model_version
        )
    from .vidor_relation import relation_gnn_status

    return relation_gnn_status(checkpoint, metadata, expected_model_version)


def _runtime_motion_features(
    detections: Sequence[Any], previous_detections: Sequence[Any] | None
) -> list[list[float]]:
    previous_by_track = {
        str(item.track_id): item
        for item in (previous_detections or [])
        if getattr(item, "track_id", None)
    }
    output: list[list[float]] = []
    for detection in detections:
        track_id = getattr(detection, "track_id", None)
        prior = previous_by_track.get(str(track_id)) if track_id else None
        if prior is None:
            output.append([0.0, 0.0, 0.0, 0.0])
            continue
        current_box = detection.normalized_bbox
        prior_box = prior.normalized_bbox
        current_area = max(
            (current_box.x2 - current_box.x1) * (current_box.y2 - current_box.y1),
            1e-6,
        )
        prior_area = max(
            (prior_box.x2 - prior_box.x1) * (prior_box.y2 - prior_box.y1),
            1e-6,
        )
        output.append(
            [
                float(detection.centroid_normalized[0]) - float(prior.centroid_normalized[0]),
                float(detection.centroid_normalized[1]) - float(prior.centroid_normalized[1]),
                math.log(current_area / prior_area),
                1.0,
            ]
        )
    return output


class ImprovedVidORRelationPredictor:
    """Strict frame-grounded inference adapter for the pair-visual v2 model."""

    def __init__(
        self,
        checkpoint: Path,
        metadata: Path,
        *,
        device: str = "cpu",
        expected_model_version: str | None = None,
        visual_encoder: Any | None = None,
    ) -> None:
        import torch

        status = improved_relation_checkpoint_status(checkpoint, metadata, expected_model_version)
        if status != "available_validated":
            raise ValueError(f"VidOR pair-visual relationship GNN checkpoint is invalid: {status}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not bool(payload.get("use_graph")):
            raise ValueError("VidOR pair-visual relationship predictor requires a GNN checkpoint")
        self.model_version = str(payload["model_version"])
        self.device = torch.device(device)
        self.visual_encoder = visual_encoder or FrozenResNetPairEncoder(device=device)
        if int(self.visual_encoder.feature_dim) != VISUAL_FEATURE_DIM:
            raise ValueError("VidOR pair-visual inference encoder feature schema is incompatible")
        model: Any = ImprovedRelationModel(use_graph=True)
        self.model = model.to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.thresholds = payload["thresholds"].to(self.device)

    def predict_detections(
        self,
        detections: Sequence[Any],
        *,
        frame_path: Path | str,
        previous_detections: Sequence[Any] | None = None,
        nearest_k: int = 8,
        max_predicates: int = 3,
    ) -> list[dict[str, Any]]:
        """Predict ordered explicit relations from detections and their real frame pixels."""

        import torch
        from PIL import Image

        from .vidor_relation import mapped_vidor_class

        resolved_frame = Path(frame_path)
        if not resolved_frame.is_file():
            raise FileNotFoundError(
                f"VidOR pair-visual inference requires a readable frame: {resolved_frame}"
            )
        mapped = [
            (detection, mapped_vidor_class(str(detection.class_label))) for detection in detections
        ]
        mapped = [(detection, label) for detection, label in mapped if label is not None]
        if len(mapped) < 2:
            return []
        features: list[list[float]] = []
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
        node_boxes = [feature[:4] for feature in features]
        pair_boxes = [
            union_box(node_boxes[source], node_boxes[target]) for source, target in candidates
        ]
        with Image.open(resolved_frame) as image:
            node_visual, union_visual = self.visual_encoder.encode(
                image.convert("RGB"), node_boxes, pair_boxes
            )
        if tuple(node_visual.shape) != (len(mapped), VISUAL_FEATURE_DIM):
            raise ValueError("VidOR pair-visual node embedding shape is incompatible")
        if tuple(union_visual.shape) != (len(candidates), VISUAL_FEATURE_DIM):
            raise ValueError("VidOR pair-visual union embedding shape is incompatible")
        class_to_id = {name: index for index, name in enumerate(VIDOR_OBJECTS)}
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        class_ids = torch.tensor(
            [class_to_id[str(label)] for _, label in mapped],
            dtype=torch.long,
            device=self.device,
        )
        edge_index = torch.tensor(candidates, dtype=torch.long, device=self.device).T.contiguous()
        edge_attr = torch.tensor(
            [_pair_features(features[source], features[target]) for source, target in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        mapped_detections = [detection for detection, _ in mapped]
        node_motion = torch.tensor(
            _runtime_motion_features(mapped_detections, previous_detections),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            probabilities = torch.sigmoid(
                self.model(
                    x,
                    class_ids,
                    node_visual.to(self.device),
                    node_motion,
                    edge_index,
                    edge_attr,
                    union_visual.to(self.device),
                )
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
                        "visual_provenance": ("frozen_resnet18_subject_object_union_roi"),
                    }
                )
        return predictions


def build_and_train_improved_vidor(
    *,
    annotation_zip: Path,
    frame_root: Path,
    output_dir: Path,
    selected: Sequence[dict[str, Any]],
    split: dict[str, list[str]],
    predicate_support: Counter[str],
    max_frames_per_video: int = 6,
    hard_negative_ratio: float = 2.0,
    random_negative_ratio: float = 1.0,
    epochs: int = 40,
    patience: int = 7,
    batch_size: int = 16,
    seed: int = 2026,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Cache visuals once, train matched models, and select using validation only."""

    _, torch = _seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs, statistics = build_improved_relation_graphs(
        annotation_zip,
        selected,
        predicate_support,
        frame_root=frame_root,
        max_frames_per_video=max_frames_per_video,
        hard_negative_ratio=hard_negative_ratio,
        random_negative_ratio=random_negative_ratio,
        seed=seed,
    )
    cache = output_dir / "vidor_relation_pair_visual_graphs.pt"
    torch.save(
        {
            "schema_version": VIDOR_IMPROVED_GRAPH_SCHEMA,
            "graphs": graphs,
            "object_classes": list(VIDOR_OBJECTS),
            "predicates": list(VIDOR_PREDICATES),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "motion_feature_names": list(MOTION_FEATURE_NAMES),
            "visual_feature_dim": VISUAL_FEATURE_DIM,
            "statistics": statistics,
        },
        cache,
    )
    reloaded = torch.load(cache, map_location="cpu", weights_only=False)
    valid, reason = validate_improved_graph_cache(reloaded)
    if not valid:
        raise ValueError(f"VidOR pair-visual graph cache validation failed: {reason}")
    mlp = train_improved_relation_model(
        graphs=graphs,
        split=split,
        output_dir=output_dir,
        model_version="vidor-relation-pair-visual-mlp-v2",
        use_graph=False,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device_name=device_name,
    )
    gnn = train_improved_relation_model(
        graphs=graphs,
        split=split,
        output_dir=output_dir,
        model_version="vidor-relation-pair-visual-gat-v2",
        use_graph=True,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device_name=device_name,
    )
    mlp_key = (
        mlp["validation_metrics"]["macro_f1_supported_predicates"],
        mlp["validation_metrics"]["mean_average_precision_supported_predicates"],
    )
    gnn_key = (
        gnn["validation_metrics"]["macro_f1_supported_predicates"],
        gnn["validation_metrics"]["mean_average_precision_supported_predicates"],
    )
    winner = "relationship_gnn" if gnn_key > mlp_key else "pairwise_mlp"
    report = {
        "schema_version": "vidor-relation-pair-visual-report-v2",
        "selection_policy": (
            "validation supported-predicate macro F1, then validation mAP; test never selects"
        ),
        "dataset": {
            **statistics,
            "selected_videos": len(selected),
            "split_video_counts": {name: len(split[name]) for name in split},
            "cache": cache.as_posix(),
            "cache_sha256": _sha256(cache),
        },
        "pairwise_mlp": mlp,
        "relationship_gnn": gnn,
        "validation_winner": winner,
        "activation": (
            "eligible_after_product inference supplies the same frame-derived visual schema"
        ),
    }
    (output_dir / "held_out_metrics_pair_visual.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
