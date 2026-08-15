"""Strict fixed-holdout manifests and measured statistics for AVA data scaling."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def ordered_video_ids(graphs: Sequence[Any]) -> list[str]:
    return sorted({str(graph.video_id) for graph in graphs})


def build_fixed_holdout_scaling_splits(
    graphs: Sequence[Any],
    baseline_split: dict[str, Any],
    corpus_sizes: Iterable[int],
    *,
    expansion_order: Iterable[str] = (),
) -> dict[int, dict[str, list[str]]]:
    """Keep the original validation/test videos fixed as training data grows."""

    available = set(ordered_video_ids(graphs))
    baseline = {
        name: [str(video_id) for video_id in baseline_split[name]]
        for name in ("train", "validation", "test")
    }
    baseline_all = set().union(*(set(values) for values in baseline.values()))
    if sum(len(values) for values in baseline.values()) != len(baseline_all):
        raise ValueError("baseline split contains video leakage")
    missing = baseline_all.difference(available)
    if missing:
        raise ValueError(f"baseline split videos missing from graph cache: {sorted(missing)}")
    held_out = set(baseline["validation"]) | set(baseline["test"])
    ordered_extras: list[str] = []
    for video_id in [*expansion_order, *sorted(available)]:
        candidate = str(video_id)
        if (
            candidate in available
            and candidate not in baseline_all
            and candidate not in ordered_extras
        ):
            ordered_extras.append(candidate)

    manifests: dict[int, dict[str, list[str]]] = {}
    baseline_size = len(baseline_all)
    for requested_size in sorted(set(int(value) for value in corpus_sizes)):
        if requested_size < baseline_size or requested_size > len(available):
            continue
        extra_count = requested_size - baseline_size
        train = [*baseline["train"], *ordered_extras[:extra_count]]
        manifest = {
            "train": train,
            "validation": list(baseline["validation"]),
            "test": list(baseline["test"]),
        }
        train_set = set(train)
        if train_set & held_out:
            raise ValueError("generated scaling split leaks held-out videos into training")
        if sum(len(values) for values in manifest.values()) != len(
            set().union(*(set(values) for values in manifest.values()))
        ):
            raise ValueError("generated scaling split contains duplicate video IDs")
        manifests[requested_size] = manifest
    return manifests


def graph_corpus_statistics(
    graphs: Sequence[Any], split: dict[str, list[str]]
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        video_ids = set(split[name])
        selected = [graph for graph in graphs if str(graph.video_id) in video_ids]
        support = torch.zeros(80, dtype=torch.long)
        supervised_persons = 0
        for graph in selected:
            mask = graph.loss_mask.bool()
            supervised_persons += int(mask.sum())
            if bool(mask.any()):
                support += graph.y[mask].sum(dim=0).to(dtype=torch.long)
        result[name] = {
            "videos": len(video_ids),
            "timestamp_graphs": len(selected),
            "supervised_persons": supervised_persons,
            "action_rows": int(support.sum()),
            "action_coverage": int((support > 0).sum()),
            "per_action_support": support.tolist(),
        }
    result["supported_test_classes"] = result["test"]["action_coverage"]
    return result
