import shutil
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from vidquery.ava_coverage import VideoCoverage
from vidquery.ava_expansion import (
    _classify_previous_download_failures,
    _permanent_download_failure,
    _safe_unlink,
    _save_video_feature_cache,
    _storage_safe_batch,
    expand_ava_dataset,
    select_incremental_videos,
    validate_video_feature_cache,
)
from vidquery.ava_improved import IMPROVED_DATASET_SCHEMA, save_improved_graph_cache


def _coverage(video_id: str, actions: dict[int, int], split: str = "train"):
    return VideoCoverage(video_id, Counter(actions), split)


def _graph(video_id: str, *, finite: bool = True) -> Data:
    value = 0.0 if finite else float("nan")
    graph = Data(
        x=torch.full((2, 138), value),
        handcrafted_x=torch.full((2, 138), value),
        roi_x=torch.full((2, 650), value),
        visual_x=torch.full((2, 1162), value),
        visual_motion_x=torch.full((2, 1172), value),
        visual_temporal_x=torch.full((2, 1674), value),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 8)),
        y=torch.tensor([[1.0, *([0.0] * 79)], [0.0] * 80]),
        loss_mask=torch.tensor([True, False]),
        person_mask=torch.tensor([True, False]),
        class_ids=torch.tensor([0, 1]),
    )
    graph.video_id = video_id
    graph.timestamp = 1.0
    graph.node_ids = ["person", "object"]
    graph.track_ids = ["track", ""]
    return graph


def test_incremental_selection_prioritizes_rare_support_and_is_deterministic():
    coverage = {
        "current": _coverage("current", {1: 100, 2: 1}),
        "rare": _coverage("rare", {2: 20, 3: 10}, "validation"),
        "common": _coverage("common", {1: 1000}),
        "mixed": _coverage("mixed", {1: 2, 4: 2}),
    }

    first = select_incremental_videos(coverage, ["current"], target_videos=3)
    second = select_incremental_videos(coverage, ["current"], target_videos=3)

    assert first["selected_video_ids"] == second["selected_video_ids"]
    assert first["selected_video_ids"][0] == "rare"
    assert "common" not in first["selected_video_ids"]
    assert first["actions"][1]["projected_support"] == 21


def test_per_video_cache_round_trip_and_non_finite_rejection(tmp_path: Path):
    report = _save_video_feature_cache(
        video_id="video-a",
        graphs=[_graph("video-a")],
        cache_dir=tmp_path,
        provenance={"source_media_sha256": "abc"},
    )
    cache_path = tmp_path / "video-a.pt"
    assert report["validation"]["valid"] is True
    assert validate_video_feature_cache(cache_path, "video-a")["action_rows"] == 1
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert cached["graphs"][0].visual_temporal_x.shape[1] == 1674

    invalid = tmp_path / "invalid.pt"
    torch.save(
        {
            "schema_version": "ava80-incremental-video-cache-v1",
            "video_id": "invalid",
            "graphs": [_graph("invalid", finite=False)],
        },
        invalid,
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_video_feature_cache(invalid, "invalid")


def test_cleanup_is_scoped_to_temporary_root(tmp_path: Path):
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    inside = temporary_root / "clip.mp4"
    inside.write_bytes(b"media")
    _safe_unlink(inside, temporary_root)
    assert not inside.exists()

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError, match="outside temporary root"):
        _safe_unlink(outside, temporary_root)
    assert outside.exists()


def test_storage_safe_batch_reserves_master_rebuild_and_free_space(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "videos"
    master = tmp_path / "master.pt"
    master.write_bytes(b"x" * 100)
    inventory = {
        "large": {"filename": "large.mp4", "bytes": 1024**3},
        "small": {"filename": "small.mp4", "bytes": 10 * 1024**2},
    }
    monkeypatch.setattr(
        "vidquery.ava_expansion.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            total=2 * 1024**3, used=1100 * 1024**2, free=948 * 1024**2
        ),
    )

    batch, report = _storage_safe_batch(
        ["large", "small"],
        inventory,
        destination_dir=destination,
        master_cache=master,
        batch_size=2,
        min_free_bytes=500 * 1024**2,
    )

    assert batch == ["small"]
    assert report["master_rebuild_reserve_bytes"] == 100
    assert report["skipped_candidates"][0]["video_id"] == "large"


def test_download_failure_classification_does_not_blacklist_network_outage() -> None:
    batches = [
        {
            "downloads": {
                "dns": {"status": "failed", "error": "Could not resolve host"},
                "missing": {"status": "failed", "error": "HTTP 404 Not Found"},
                "done": {"status": "failed", "error": "HTTP 404 Not Found"},
            }
        }
    ]

    permanent, transient = _classify_previous_download_failures(batches, {"done"})

    assert permanent == {"missing"}
    assert transient == {"dns"}
    assert _permanent_download_failure({"error": "not in S3 inventory"}) is True
    assert _permanent_download_failure({"error": "connection reset"}) is False


def test_expansion_resume_uses_validated_master_without_network(tmp_path: Path, monkeypatch):
    master = tmp_path / "master.pt"
    save_improved_graph_cache([_graph("already")], master)
    assert torch.load(master, weights_only=False)["schema_version"] == IMPROVED_DATASET_SCHEMA
    annotations = [tmp_path / "train.csv", tmp_path / "val.csv"]
    annotations[0].write_text("already,1,0,0,1,1,1,1\n", encoding="utf-8")
    annotations[1].write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "vidquery.ava_expansion.fetch_ava_inventory",
        lambda: pytest.fail("resume should not access the network"),
    )

    result = expand_ava_dataset(
        target_videos=1,
        annotation_paths=annotations,
        temporary_root=tmp_path / "temporary",
        feature_root=tmp_path / "features",
        master_cache=master,
        selection_manifest=tmp_path / "selection.json",
        state_path=tmp_path / "state.json",
    )

    assert result["target_reached"] is True
    assert result["completed_video_ids"] == ["already"]
