from __future__ import annotations

import copy
from collections import Counter

import pytest
from test_vidor_relation import _DeterministicVisualEncoder, _write_annotation_zip

from vidquery.vidor_improved import (
    _save_improved_graph_cache,
    build_and_train_improved_vidor,
    build_improved_relation_graphs,
    train_improved_relation_model,
)
from vidquery.vidor_relation import (
    extend_relation_selection,
    frame_selection_support,
    select_relation_videos,
)


def _summaries(count: int):
    rows = []
    for index in range(count):
        support = {"next_to": 1 + index % 3, "in_front_of": index % 5, "hold": 1 + index % 7}
        rows.append({"member": f"training/0000/v{index:03d}.json", "video_id": f"v{index:03d}",
                     "support": {key: value for key, value in support.items() if value},
                     "relation_instances": sum(support.values())})
    global_support: Counter[str] = Counter()
    for row in rows:
        global_support.update(row["support"])
    return rows, global_support


def _base_manifest(rows, global_support, count):
    selected = select_relation_videos(rows, global_support, count)
    ids = [row["video_id"] for row in selected]
    return {"seed": 2026, "selected_videos": selected, "validation": ids[:2],
            "test": ids[2:4], "train": ids[4:]}


def test_extended_selection_keeps_validation_test_and_freezes_frame_counts():
    rows, global_support = _summaries(40)
    base = _base_manifest(rows, global_support, 12)
    extended = extend_relation_selection(rows, global_support, base, 30)
    assert extended["validation"] == base["validation"]
    assert extended["test"] == base["test"]
    assert extended["train"][: len(base["train"])] == base["train"]
    assert len(extended["train"]) == len(base["train"]) + 18
    assert extended["selected_videos"][:12] == base["selected_videos"]
    assert frame_selection_support(extended) == frame_selection_support(base)
    summed: Counter[str] = Counter()
    for row in extended["selected_videos"]:
        summed.update(row["support"])
    assert summed != frame_selection_support(extended)
    with pytest.raises(ValueError, match="must exceed"):
        extend_relation_selection(rows, global_support, base, 12)


def test_extension_refuses_a_base_that_is_not_the_ranking_prefix():
    rows, global_support = _summaries(40)
    base = _base_manifest(rows, global_support, 12)
    worst = select_relation_videos(rows, global_support, 40)[-1]
    base["selected_videos"] = [*base["selected_videos"][:-1], worst]
    with pytest.raises(ValueError, match="not contained"):
        extend_relation_selection(rows, global_support, base, 20)


def _graph_splits(tmp_path):
    from PIL import Image

    annotation_zip = tmp_path / "annotations.zip"
    _write_annotation_zip(annotation_zip)
    frame = tmp_path / "frames" / "0000" / "synthetic-video" / "frame_000001.jpg"
    frame.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), (20, 40, 60)).save(frame)
    graphs, statistics = build_improved_relation_graphs(
        annotation_zip,
        [{"member": "training/0000/synthetic-video.json"}],
        Counter({"next_to": 1}),
        frame_root=tmp_path / "frames",
        visual_encoder=_DeterministicVisualEncoder(),
        max_frames_per_video=1,
    )
    copies = []
    for name in ("a", "b", "c"):
        graph = copy.deepcopy(graphs[0])
        graph.video_id = name
        copies.append(graph)
    return copies, statistics, {"train": ["a"], "validation": ["b"], "test": ["c"]}


def test_withheld_test_and_warm_start_are_recorded(tmp_path):
    graphs, _, split = _graph_splits(tmp_path)
    first = train_improved_relation_model(
        graphs=graphs, split=split, output_dir=tmp_path / "m", model_version="vidor-t-gat-a",
        use_graph=True, epochs=1, patience=1, device_name="cpu", evaluate_test=False,
    )
    assert first["test_metrics"] is None
    assert first["test_evaluation_status"] == "withheld_until_final_selection"
    warm = train_improved_relation_model(
        graphs=graphs, split=split, output_dir=tmp_path / "m", model_version="vidor-t-gat-b",
        use_graph=True, epochs=1, patience=1, device_name="cpu",
        init_checkpoint=tmp_path / "m" / "vidor-t-gat-a.pt",
    )
    assert warm["warm_start"]["model_version"] == "vidor-t-gat-a"
    assert warm["test_metrics"] is not None
    with pytest.raises(ValueError, match="architecture mismatch"):
        train_improved_relation_model(
            graphs=graphs, split=split, output_dir=tmp_path / "m", model_version="vidor-t-mlp",
            use_graph=False, epochs=1, device_name="cpu",
            init_checkpoint=tmp_path / "m" / "vidor-t-gat-a.pt",
        )


def test_reused_graph_cache_must_match_frame_counts_and_selection(tmp_path):
    import torch

    graphs, statistics, split = _graph_splits(tmp_path)
    cache = tmp_path / "cache.pt"
    _save_improved_graph_cache(torch, cache, graphs, statistics, "not-the-current-counts")
    selected = [{"video_id": name} for name in ("a", "b", "c")]
    arguments = dict(annotation_zip=tmp_path / "annotations.zip", frame_root=tmp_path / "frames",
                     output_dir=tmp_path / "out", selected=selected, split=split,
                     predicate_support=Counter({"next_to": 1}), epochs=1, device_name="cpu",
                     graph_cache=cache, evaluate_test=False)
    with pytest.raises(ValueError, match="different frame-selection counts"):
        build_and_train_improved_vidor(**arguments)
