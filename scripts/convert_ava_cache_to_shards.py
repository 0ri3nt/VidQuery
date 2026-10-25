"""Convert the legacy improved AVA graph cache into slim per-video shards.

Resumable: videos whose shard already validates are skipped.  The legacy
cache is only read, never modified or deleted.  ``--parity-checkpoint``
compares a checkpoint's probabilities on the original graphs and on graphs
rebuilt from shards (fp16 embeddings), so storage savings are not bought with
silent numerical drift.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from vidquery.ava_ablation import StaticActionModel
from vidquery.ava_gnn80 import decode_ava_predictions
from vidquery.ava_shards import convert_graph_cache_to_shards, load_shard


def _probabilities(payload: dict, graphs: list, attribute: str) -> torch.Tensor:
    model: Any = StaticActionModel(
        int(payload["feature_dimension"]),
        use_edges=bool(payload.get("use_edges")),
        mlp_only=bool(payload.get("mlp_only")),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    outputs = []
    with torch.inference_mode():
        for graph in graphs:
            features = getattr(graph, attribute).float()
            logits = model(
                (features - payload["feature_mean"]) / payload["feature_std"],
                graph.class_ids,
                graph.edge_index,
                graph.edge_attr,
            )
            outputs.append(torch.sigmoid(logits)[graph.loss_mask.bool()])
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=Path("data/app/models/ava80_improved/ava80_improved_graphs.pt"),
    )
    parser.add_argument("--shard-dir", type=Path, default=Path("data/app/features/ava80_shards"))
    parser.add_argument("--parity-checkpoint", type=Path, default=None)
    parser.add_argument("--parity-videos", nargs="*", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"{args.report} exists; choose a new versioned file name")

    started = time.perf_counter()
    legacy = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    graphs = list(legacy["graphs"])
    load_seconds = time.perf_counter() - started
    converted = convert_graph_cache_to_shards(
        graphs,
        args.shard_dir,
        provenance={"converted_from": args.graph_cache.as_posix(), "legacy_schema": legacy.get(
            "schema_version"
        )},
    )
    shard_bytes = sum(path.stat().st_size for path in args.shard_dir.glob("*.shard.pt"))
    videos = sorted({str(graph.video_id) for graph in graphs})
    report: dict = {
        "legacy_cache": args.graph_cache.as_posix(),
        "legacy_cache_bytes": args.graph_cache.stat().st_size,
        "legacy_cache_load_seconds": round(load_seconds, 1),
        "videos": len(videos),
        "graphs": len(graphs),
        "shard_dir": args.shard_dir.as_posix(),
        "shard_bytes_total": shard_bytes,
        "shard_bytes_per_video": round(shard_bytes / max(1, len(videos))),
        "written": sorted(converted["written"]),
        "skipped_completed": converted["skipped_completed"],
    }
    started = time.perf_counter()
    for video_id in videos:
        load_shard(args.shard_dir, video_id)
    report["all_shards_load_seconds"] = round(time.perf_counter() - started, 1)

    if args.parity_checkpoint is not None:
        payload = torch.load(args.parity_checkpoint, map_location="cpu", weights_only=True)
        attribute = str(payload["feature_attribute"])
        original = [graph for graph in graphs if str(graph.video_id) in set(args.parity_videos)]
        if attribute == "visual_noleak_x":
            from vidquery.ava_gnn80 import drop_leaking_columns

            for graph in original:
                graph.visual_noleak_x = drop_leaking_columns(graph.visual_x)
        rebuilt = [
            graph
            for video_id in args.parity_videos
            for graph in load_shard(args.shard_dir, video_id)
        ]
        original.sort(key=lambda graph: (str(graph.video_id), float(graph.timestamp)))
        rebuilt.sort(key=lambda graph: (str(graph.video_id), float(graph.timestamp)))
        before = _probabilities(payload, original, attribute)
        after = _probabilities(payload, rebuilt, attribute)
        thresholds = payload["thresholds"].numpy()
        decoded_before = decode_ava_predictions(before.numpy(), thresholds)
        decoded_after = decode_ava_predictions(after.numpy(), thresholds)
        report["parity"] = {
            "checkpoint": args.parity_checkpoint.as_posix(),
            "videos": args.parity_videos,
            "persons": int(before.shape[0]),
            "max_abs_probability_difference": float((before - after).abs().max()),
            "mean_abs_probability_difference": float((before - after).abs().mean()),
            "decoded_label_agreement": float((decoded_before == decoded_after).mean()),
            "persons_with_identical_decoded_labels": float(
                (decoded_before == decoded_after).all(axis=1).mean()
            ),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "written"}, indent=2))


if __name__ == "__main__":
    main()
