"""Slim, versioned per-video AVA feature shards.

The legacy per-video cache (``ava80-incremental-video-cache-v1``) pickles PyG
graphs carrying seven overlapping feature tensors, including the whole-frame
ResNet embedding copied onto every node, plus a master cache that duplicates
every video.  At full AVA scale that exceeds this laptop's disk and RAM.

A shard stores, per video, exactly what the active training/inference path
needs, once:

* ``handcrafted`` - the 138 hand-crafted node features (fp32, small);
* ``roi``         - one 512-d frozen ResNet-18 ROI embedding per node (fp16);
* ``frame``       - one 512-d whole-frame embedding per *graph* (fp16);
* graph structure (class ids, edges, edge features, offsets), labels (uint8),
  masks, the ``annotation_only`` provenance flag (kept as metadata, never as a
  feature), node ids, and timestamps.

``visual_x`` and ``visual_noleak_x`` are reconstructed on load, so one shard
serves both schemas.  Writes are atomic; a JSON sidecar records the content
hash and provenance, and a shard counts as complete only when the sidecar
validates against the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .ava_gnn80 import AVA80_LABELS, CONTINUOUS_FEATURE_NAMES, leaking_column_indices

SHARD_SCHEMA = "ava80-slim-shard-v1"
ROI_DIMENSION = 512
FRAME_DIMENSION = 512
HANDCRAFTED_DIMENSION = len(CONTINUOUS_FEATURE_NAMES)


class ShardError(ValueError):
    """A shard is missing, incomplete, corrupt, or from another schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_paths(shard_dir: Path, video_id: str) -> tuple[Path, Path]:
    return shard_dir / f"{video_id}.shard.pt", shard_dir / f"{video_id}.shard.json"


def write_shard(
    graphs: Sequence[Any],
    shard_dir: Path,
    *,
    video_id: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise one video's graphs into a slim shard and its sidecar."""

    import torch

    graphs = sorted(graphs, key=lambda graph: float(graph.timestamp))
    if not graphs:
        raise ShardError(f"no graphs to shard for {video_id}")
    handcrafted: list[Any] = []
    roi: list[Any] = []
    frame: list[Any] = []
    class_ids: list[Any] = []
    person: list[Any] = []
    loss: list[Any] = []
    labels: list[Any] = []
    flags: list[Any] = []
    edges: list[Any] = []
    edge_attr: list[Any] = []
    node_ids: list[str] = []
    node_ptr, edge_ptr, timestamps = [0], [0], []
    annotation_column = leaking_column_indices()[1]
    for graph in graphs:
        if str(graph.video_id) != video_id:
            raise ShardError(f"graph from {graph.video_id} in shard for {video_id}")
        base = graph.x.float()
        if int(base.shape[1]) != HANDCRAFTED_DIMENSION:
            raise ShardError("hand-crafted feature dimension mismatch")
        handcrafted.append(base)
        roi.append(graph.roi_visual_x.to(torch.float16))
        frame_rows = graph.frame_visual_x
        if not bool(torch.equal(frame_rows, frame_rows[:1].expand_as(frame_rows))):
            raise ShardError("frame embedding differs across nodes of one graph")
        frame.append(frame_rows[:1].to(torch.float16))
        class_ids.append(graph.class_ids.long())
        person.append(graph.person_mask.bool())
        loss.append(graph.loss_mask.bool())
        labels.append(graph.y.to(torch.uint8))
        flags.append(base[:, annotation_column] > 0.5)
        edges.append(graph.edge_index.long())
        edge_attr.append(graph.edge_attr.float())
        node_ids.extend(str(value) for value in graph.node_ids)
        node_ptr.append(node_ptr[-1] + int(graph.num_nodes))
        edge_ptr.append(edge_ptr[-1] + int(graph.edge_index.shape[1]))
        timestamps.append(float(graph.timestamp))
    payload: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "video_id": video_id,
        "labels": list(AVA80_LABELS),
        "handcrafted_feature_names": list(CONTINUOUS_FEATURE_NAMES),
        "handcrafted": torch.cat(handcrafted),
        "roi": torch.cat(roi),
        "frame": torch.cat(frame),
        "class_ids": torch.cat(class_ids),
        "person_mask": torch.cat(person),
        "loss_mask": torch.cat(loss),
        "y": torch.cat(labels),
        "annotation_only": torch.cat(flags),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(edge_attr),
        "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
        "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
        "timestamps": torch.tensor(timestamps, dtype=torch.float64),
        "node_ids": node_ids,
    }
    if int(payload["roi"].shape[1]) != ROI_DIMENSION or int(
        payload["frame"].shape[1]
    ) != FRAME_DIMENSION:
        raise ShardError("visual embedding dimension mismatch")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard, sidecar = shard_paths(shard_dir, video_id)
    temporary = shard.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, shard)
    supervised = int(payload["loss_mask"].sum())
    metadata = {
        "schema_version": SHARD_SCHEMA,
        "video_id": video_id,
        "graphs": len(graphs),
        "nodes": int(payload["class_ids"].shape[0]),
        "edges": int(payload["edge_index"].shape[1]),
        "supervised_persons": supervised,
        "annotation_only_persons": int((payload["annotation_only"] & payload["loss_mask"]).sum()),
        "action_rows": int(payload["y"][payload["loss_mask"]].sum()),
        "bytes": shard.stat().st_size,
        "sha256": _sha256(shard),
        "provenance": provenance or {},
    }
    temporary_sidecar = sidecar.with_suffix(".json.tmp")
    temporary_sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_sidecar, sidecar)
    validate_shard(shard_dir, video_id)
    return metadata


def validate_shard(shard_dir: Path, video_id: str) -> dict[str, Any]:
    """Raise ``ShardError`` unless the shard and its sidecar agree and are complete."""

    shard, sidecar = shard_paths(shard_dir, video_id)
    if not shard.is_file() or not sidecar.is_file():
        raise ShardError(f"shard for {video_id} is incomplete")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShardError(f"shard sidecar for {video_id} is unreadable") from exc
    if metadata.get("schema_version") != SHARD_SCHEMA:
        raise ShardError(
            f"shard {video_id} has schema {metadata.get('schema_version')!r}, "
            f"expected {SHARD_SCHEMA!r}"
        )
    if metadata.get("video_id") != video_id:
        raise ShardError(f"shard sidecar video mismatch for {video_id}")
    if shard.stat().st_size != metadata.get("bytes") or _sha256(shard) != metadata.get("sha256"):
        raise ShardError(f"shard {video_id} does not match its recorded hash")
    return metadata


def completed_shard_video_ids(shard_dir: Path) -> set[str]:
    """Video ids whose shard validates; partial or corrupt shards are ignored."""

    completed: set[str] = set()
    for sidecar in sorted(shard_dir.glob("*.shard.json")):
        video_id = sidecar.name.removesuffix(".shard.json")
        try:
            validate_shard(shard_dir, video_id)
        except ShardError:
            continue
        completed.add(video_id)
    return completed


def load_shard(shard_dir: Path, video_id: str, *, dtype: Any = None) -> list[Any]:
    """Rebuild PyG graphs (``x``, ``visual_x``, ``visual_noleak_x``, labels, masks)."""

    import torch
    from torch_geometric.data import Data

    validate_shard(shard_dir, video_id)
    shard, _ = shard_paths(shard_dir, video_id)
    payload = torch.load(shard, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SHARD_SCHEMA:
        raise ShardError(f"shard {video_id} payload schema mismatch")
    if payload.get("labels") != list(AVA80_LABELS):
        raise ShardError(f"shard {video_id} label order differs from the active label map")
    if payload.get("handcrafted_feature_names") != list(CONTINUOUS_FEATURE_NAMES):
        raise ShardError(f"shard {video_id} feature schema differs from the active schema")
    dtype = dtype or torch.float32
    drop = set(leaking_column_indices())
    keep = torch.tensor([index for index in range(HANDCRAFTED_DIMENSION) if index not in drop])
    graphs: list[Any] = []
    node_ptr, edge_ptr = payload["node_ptr"].tolist(), payload["edge_ptr"].tolist()
    for index, timestamp in enumerate(payload["timestamps"].tolist()):
        start, end = node_ptr[index], node_ptr[index + 1]
        edge_start, edge_end = edge_ptr[index], edge_ptr[index + 1]
        base = payload["handcrafted"][start:end]
        roi = payload["roi"][start:end].float()
        frame = payload["frame"][index : index + 1].float().expand(end - start, -1)
        visual = torch.cat((base, roi, frame), dim=1)
        graph = Data(
            x=base,
            class_ids=payload["class_ids"][start:end],
            edge_index=payload["edge_index"][:, edge_start:edge_end],
            edge_attr=payload["edge_attr"][edge_start:edge_end],
            y=payload["y"][start:end].float(),
            person_mask=payload["person_mask"][start:end],
            loss_mask=payload["loss_mask"][start:end],
        )
        graph.visual_x = visual.to(dtype)
        graph.visual_noleak_x = torch.cat(
            (base.index_select(1, keep), roi, frame), dim=1
        ).to(dtype)
        graph.annotation_only_flag = payload["annotation_only"][start:end]
        graph.video_id = video_id
        graph.timestamp = float(timestamp)
        graph.node_ids = payload["node_ids"][start:end]
        graphs.append(graph)
    return graphs


class ShardGraphDataset:
    """Index over many shards that keeps only a few decoded shards in memory.

    Use with ``ShardBlockSampler`` so batches are drawn shard by shard; random
    access across shards would reload them repeatedly.
    """

    def __init__(
        self,
        shard_dir: Path,
        video_ids: Sequence[str],
        *,
        feature_attribute: str = "visual_noleak_x",
        cache_shards: int = 2,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.feature_attribute = feature_attribute
        self.cache_shards = max(1, cache_shards)
        self._cache: OrderedDict[str, list[Any]] = OrderedDict()
        self.index: list[tuple[str, int]] = []
        self.blocks: dict[str, list[int]] = {}
        for video_id in video_ids:
            metadata = validate_shard(self.shard_dir, video_id)
            first = len(self.index)
            self.index.extend((video_id, local) for local in range(int(metadata["graphs"])))
            self.blocks[video_id] = list(range(first, len(self.index)))
        self.loads = 0

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, video_id: str) -> list[Any]:
        if video_id in self._cache:
            self._cache.move_to_end(video_id)
            return self._cache[video_id]
        graphs = load_shard(self.shard_dir, video_id)
        self.loads += 1
        self._cache[video_id] = graphs
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return graphs

    def __getitem__(self, position: int) -> Any:
        from torch_geometric.data import Data

        video_id, local = self.index[position]
        graph = self._shard(video_id)[local]
        selected = Data(
            x=getattr(graph, self.feature_attribute),
            class_ids=graph.class_ids,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            person_mask=graph.person_mask,
            loss_mask=graph.loss_mask,
        )
        selected.video_id = graph.video_id
        selected.timestamp = graph.timestamp
        return selected

    def iter_graphs(self) -> Iterator[Any]:
        for video_id in self.blocks:
            yield from self._shard(video_id)


class ShardBlockSampler:
    """Seeded order: shuffle shards, then graphs within each shard."""

    def __init__(self, dataset: ShardGraphDataset, *, seed: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(self.dataset.blocks)
        if self.shuffle:
            rng.shuffle(order)
        for video_id in order:
            block = list(self.dataset.blocks[video_id])
            if self.shuffle:
                rng.shuffle(block)
            yield from block

    def __len__(self) -> int:
        return len(self.dataset)


def convert_graph_cache_to_shards(
    graphs: Sequence[Any],
    shard_dir: Path,
    *,
    provenance: dict[str, Any] | None = None,
    skip_completed: bool = True,
) -> dict[str, Any]:
    """Write one shard per video from an in-memory legacy cache (resumable)."""

    by_video: dict[str, list[Any]] = {}
    for graph in graphs:
        by_video.setdefault(str(graph.video_id), []).append(graph)
    done = completed_shard_video_ids(shard_dir) if skip_completed else set()
    written: dict[str, Any] = {}
    for video_id in sorted(by_video):
        if video_id in done:
            continue
        written[video_id] = write_shard(
            by_video[video_id], shard_dir, video_id=video_id, provenance=provenance
        )
    return {"written": written, "skipped_completed": sorted(done & set(by_video))}
