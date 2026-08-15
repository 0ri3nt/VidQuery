from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from vidquery.ava_scaling import (
    build_fixed_holdout_scaling_splits,
    graph_corpus_statistics,
)


def _graph(video_id: str, action_id: int = 1) -> Data:
    target = torch.zeros((1, 80))
    target[0, action_id - 1] = 1
    graph = Data(
        y=target,
        loss_mask=torch.tensor([True]),
        x=torch.zeros((1, 1)),
    )
    graph.video_id = video_id
    return graph


def test_scaling_splits_keep_fixed_video_holdout_without_leakage() -> None:
    graphs = [_graph(video_id) for video_id in ("a", "b", "c", "d", "e")]
    baseline = {"train": ["a"], "validation": ["b"], "test": ["c"]}

    splits = build_fixed_holdout_scaling_splits(
        graphs, baseline, [3, 5], expansion_order=["e", "d"]
    )

    assert splits[3]["train"] == ["a"]
    assert splits[5]["train"] == ["a", "e", "d"]
    assert splits[3]["validation"] == splits[5]["validation"] == ["b"]
    assert splits[3]["test"] == splits[5]["test"] == ["c"]
    assert not set(splits[5]["train"]) & {"b", "c"}


def test_scaling_split_rejects_missing_or_leaking_baseline_videos() -> None:
    graphs = [_graph("a"), _graph("b")]
    with pytest.raises(ValueError, match="missing"):
        build_fixed_holdout_scaling_splits(
            graphs,
            {"train": ["a"], "validation": ["b"], "test": ["missing"]},
            [3],
        )
    with pytest.raises(ValueError, match="leakage"):
        build_fixed_holdout_scaling_splits(
            graphs,
            {"train": ["a"], "validation": ["b"], "test": ["a"]},
            [2],
        )


def test_scaling_statistics_count_supervised_people_and_actions(tmp_path: Path) -> None:
    del tmp_path
    graphs = [_graph("train", 1), _graph("train", 2), _graph("val", 1), _graph("test", 3)]
    statistics = graph_corpus_statistics(
        graphs,
        {"train": ["train"], "validation": ["val"], "test": ["test"]},
    )

    assert statistics["train"]["timestamp_graphs"] == 2
    assert statistics["train"]["supervised_persons"] == 2
    assert statistics["train"]["action_rows"] == 2
    assert statistics["train"]["action_coverage"] == 2
    assert statistics["supported_test_classes"] == 1
