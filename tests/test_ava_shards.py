from __future__ import annotations

import json

import pytest
import torch
from torch_geometric.data import Data

from vidquery import ava_expansion
from vidquery.ava_gnn80 import leaking_column_indices
from vidquery.ava_shards import (
    SHARD_SCHEMA,
    ShardBlockSampler,
    ShardError,
    ShardGraphDataset,
    completed_shard_video_ids,
    convert_graph_cache_to_shards,
    load_shard,
    shard_paths,
    validate_shard,
    write_shard,
)


def _graph(video_id: str, timestamp: float, nodes: int = 3, seed: int = 0) -> Data:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((nodes, 138), generator=generator)
    annotation_column = leaking_column_indices()[1]
    x[:, annotation_column] = 0.0
    x[0, annotation_column] = 1.0  # first node is a ground-truth-only box
    frame = torch.rand((1, 512), generator=generator).repeat(nodes, 1)
    y = torch.zeros((nodes, 80))
    y[0, 11] = 1.0
    y[1, 78] = 1.0
    graph = Data(
        x=x,
        class_ids=torch.zeros(nodes, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.rand((2, 8), generator=generator),
        y=y,
        person_mask=torch.ones(nodes, dtype=torch.bool),
        loss_mask=torch.tensor([True, True] + [False] * (nodes - 2)),
    )
    graph.roi_visual_x = torch.rand((nodes, 512), generator=generator)
    graph.frame_visual_x = frame
    graph.video_id = video_id
    graph.timestamp = timestamp
    graph.node_ids = [f"n{index}" for index in range(nodes)]
    return graph


def test_round_trip_preserves_structure_labels_and_both_feature_schemas(tmp_path):
    graphs = [_graph("vid", 905.0, seed=2), _graph("vid", 902.0, seed=1)]
    metadata = write_shard(graphs, tmp_path, video_id="vid", provenance={"source": "test"})
    assert metadata["graphs"] == 2 and metadata["supervised_persons"] == 4
    assert metadata["annotation_only_persons"] == 2
    loaded = load_shard(tmp_path, "vid")
    original = sorted(graphs, key=lambda graph: graph.timestamp)
    assert [graph.timestamp for graph in loaded] == [902.0, 905.0]
    for before, after in zip(original, loaded, strict=True):
        expected = torch.cat((before.x, before.roi_visual_x, before.frame_visual_x), dim=1)
        assert after.visual_x.shape == (3, 1162)
        assert torch.allclose(after.visual_x, expected, atol=1e-3)
        assert after.visual_noleak_x.shape == (3, 1160)
        assert torch.equal(after.y, before.y)
        assert torch.equal(after.edge_index, before.edge_index)
        assert after.annotation_only_flag.tolist() == [True, False, False]
        assert after.node_ids == before.node_ids
    drop = leaking_column_indices()
    keep = [index for index in range(1162) if index not in drop]
    assert torch.equal(loaded[0].visual_noleak_x, loaded[0].visual_x[:, keep])


def test_shards_are_smaller_than_pickled_legacy_graphs(tmp_path):
    graphs = [_graph("vid", float(902 + index), nodes=6, seed=index) for index in range(10)]
    for graph in graphs:
        graph.visual_x = torch.cat((graph.x, graph.roi_visual_x, graph.frame_visual_x), dim=1)
        graph.roi_x = torch.cat((graph.x, graph.roi_visual_x), dim=1)
        graph.handcrafted_x = graph.x.clone()
        graph.visual_motion_x = torch.cat((graph.visual_x, torch.zeros((6, 10))), dim=1)
    legacy = tmp_path / "legacy.pt"
    torch.save({"graphs": graphs}, legacy)
    metadata = write_shard(graphs, tmp_path / "shards", video_id="vid")
    assert metadata["bytes"] * 3 < legacy.stat().st_size


def test_incomplete_corrupt_and_foreign_schema_shards_are_rejected(tmp_path):
    write_shard([_graph("ok", 902.0)], tmp_path, video_id="ok")
    write_shard([_graph("bad", 902.0)], tmp_path, video_id="bad")
    write_shard([_graph("old", 902.0)], tmp_path, video_id="old")
    shard, sidecar = shard_paths(tmp_path, "bad")
    shard.write_bytes(shard.read_bytes()[:-10])
    old_sidecar = shard_paths(tmp_path, "old")[1]
    payload = json.loads(old_sidecar.read_text(encoding="utf-8"))
    payload["schema_version"] = "ava80-slim-shard-v0"
    old_sidecar.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "partial.shard.pt.tmp").write_bytes(b"partial")
    assert completed_shard_video_ids(tmp_path) == {"ok"}
    with pytest.raises(ShardError, match="recorded hash"):
        validate_shard(tmp_path, "bad")
    with pytest.raises(ShardError, match="expected 'ava80-slim-shard-v1'"):
        load_shard(tmp_path, "old")
    shard_paths(tmp_path, "ok")[1].unlink()
    with pytest.raises(ShardError, match="incomplete"):
        load_shard(tmp_path, "ok")
    assert SHARD_SCHEMA == "ava80-slim-shard-v1"


def test_inconsistent_inputs_are_refused(tmp_path):
    mixed = _graph("other", 902.0)
    with pytest.raises(ShardError, match="graph from other"):
        write_shard([mixed], tmp_path, video_id="vid")
    varying = _graph("vid", 902.0)
    varying.frame_visual_x[1] += 1.0
    with pytest.raises(ShardError, match="frame embedding differs"):
        write_shard([varying], tmp_path, video_id="vid")
    with pytest.raises(ShardError, match="no graphs"):
        write_shard([], tmp_path, video_id="vid")


def test_label_order_change_is_detected(tmp_path):
    write_shard([_graph("vid", 902.0)], tmp_path, video_id="vid")
    shard, sidecar = shard_paths(tmp_path, "vid")
    payload = torch.load(shard, weights_only=False)
    payload["labels"] = list(reversed(payload["labels"]))
    torch.save(payload, shard)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    from vidquery.ava_shards import _sha256

    metadata["bytes"] = shard.stat().st_size
    metadata["sha256"] = _sha256(shard)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ShardError, match="label order"):
        load_shard(tmp_path, "vid")


def test_dataset_streams_shard_by_shard_with_bounded_memory(tmp_path):
    for video in ("a", "b", "c"):
        write_shard(
            [_graph(video, float(902 + index), seed=index) for index in range(4)],
            tmp_path,
            video_id=video,
        )
    dataset = ShardGraphDataset(tmp_path, ["a", "b", "c"], cache_shards=1)
    sampler = ShardBlockSampler(dataset, seed=7)
    first = list(sampler)
    second = list(ShardBlockSampler(dataset, seed=7))
    assert first == second and sorted(first) == list(range(12))
    videos = [dataset.index[position][0] for position in first]
    # Contiguous blocks: each shard is visited once per epoch.
    assert sum(1 for left, right in zip(videos, videos[1:], strict=False) if left != right) == 2
    graphs = [dataset[position] for position in first]
    assert dataset.loads == 3
    assert graphs[0].x.shape[1] == 1160


def test_conversion_is_resumable(tmp_path):
    graphs = [_graph("a", 902.0), _graph("b", 902.0), _graph("a", 903.0)]
    first = convert_graph_cache_to_shards(graphs, tmp_path)
    assert sorted(first["written"]) == ["a", "b"]
    second = convert_graph_cache_to_shards(graphs, tmp_path)
    assert second["written"] == {} and second["skipped_completed"] == ["a", "b"]


# --------------------------------------------------------------------------
# Network backoff in the AVA expansion


def test_wait_for_network_backs_off_exponentially_and_gives_up():
    delays: list[float] = []
    result = ava_expansion.wait_for_network(
        attempts=4, base_delay=15, sleep=delays.append, resolver=lambda host: False
    )
    assert result["available"] is False
    assert delays == [15, 30, 60]
    answers = iter([False, True])
    recovered = ava_expansion.wait_for_network(
        attempts=4, base_delay=1, sleep=delays.append, resolver=lambda host: next(answers)
    )
    assert recovered == {"available": True, "attempts": 2, "waited_seconds": 1}


def test_dns_failures_are_recognised():
    assert ava_expansion.is_dns_failure(
        {"error": "x", "curl_error": "curl: (6) Could not resolve host: s3.amazonaws.com"}
    )
    assert not ava_expansion.is_dns_failure({"error": "HTTP Error 404: Not Found"})


def test_expansion_stops_cleanly_when_network_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(ava_expansion, "load_video_coverage", lambda paths: {})
    monkeypatch.setattr(ava_expansion, "network_available", lambda host="": False)
    monkeypatch.setattr(
        ava_expansion,
        "fetch_ava_inventory",
        lambda: pytest.fail("inventory must not be fetched while DNS is down"),
    )
    state = ava_expansion.expand_ava_dataset(
        target_videos=1,
        feature_root=tmp_path / "features",
        temporary_root=tmp_path / "temp",
        master_cache=tmp_path / "master.pt",
        selection_manifest=tmp_path / "selection.json",
        state_path=tmp_path / "state.json",
        shard_dir=tmp_path / "shards",
        legacy_cache=False,
        rebuild_master=False,
        network_attempts=2,
        network_base_delay=0,
        sleep=lambda seconds: None,
    )
    assert state["network_unavailable"]["available"] is False
    assert state["completed_video_ids"] == []
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["network_unavailable"]


def test_shard_only_mode_requires_a_shard_directory(tmp_path):
    with pytest.raises(ValueError, match="requires a shard directory"):
        ava_expansion.expand_ava_dataset(target_videos=1, legacy_cache=False)
    with pytest.raises(ValueError, match="master cache"):
        ava_expansion.expand_ava_dataset(
            target_videos=1, legacy_cache=False, shard_dir=tmp_path, rebuild_master=True
        )
