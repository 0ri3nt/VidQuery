"""Score VidOR relation checkpoints on the same test graphs and compare them.

Test graphs come from ``--graphs`` (default: the scale-500 cache, whose 50
test videos every earlier comparison used). Test is scored only after the
``--decision`` file exists: the validation-only model choices, written
beforehand. Output: point metrics with ``vidor_relation._metrics``, per-video
metrics, and paired video-level bootstrap CIs (all models share each
resample).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "macro_f1_supported_predicates",
    "mean_average_precision_supported_predicates",
    "micro_f1",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predict(path: Path, graphs: list[Any], device: Any) -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    from vidquery.vidor_improved import ImprovedRelationModel

    payload = torch.load(path, map_location="cpu", weights_only=True)
    factory: Any = ImprovedRelationModel(use_graph=bool(payload["use_graph"]))
    model = factory.to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    probabilities, targets = [], []
    with torch.inference_mode():
        for raw in DataLoader(graphs, batch_size=16, shuffle=False):
            batch = raw.to(device)
            logits = model(batch.x, batch.class_ids, batch.node_visual, batch.node_motion,
                           batch.edge_index, batch.edge_attr, batch.union_visual)
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append(batch.edge_y.cpu())
    return {
        "model_version": str(payload.get("model_version", path.stem)),
        "targets": torch.cat(targets).numpy(),
        "probabilities": torch.cat(probabilities).numpy(),
        "thresholds": payload["thresholds"].numpy(),
    }


def _scalar(targets: Any, probabilities: Any, thresholds: Any) -> dict[str, float]:
    from vidquery.vidor_relation import _metrics

    metrics = _metrics(targets, probabilities, thresholds, np)
    return {key: float(metrics[key]) for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        type=Path,
        default=Path("data/app/models/vidor_relation_scale500/vidor_relation_pair_visual_graphs.pt"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/app/models/vidor_relation/split_manifest.json"),
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--pair", action="append", default=[], help="A:B reports B - A")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new versioned file name")
    if not args.decision.is_file():
        raise SystemExit("write the validation-only decision file before scoring test")
    import torch

    from vidquery.vidor_relation import _graph_split

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    test_ids = [str(video) for video in split["test"]]
    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)["graphs"]
    test_graphs = _graph_split(graphs, {"train": [], "validation": [], "test": test_ids})["test"]
    edge_videos = np.asarray(
        [str(graph.video_id) for graph in test_graphs for _ in range(int(graph.edge_y.shape[0]))]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    predictions = {name: _predict(Path(path), test_graphs, device)
                   for name, path in checkpoints.items()}
    reference = next(iter(predictions.values()))["targets"]
    for name, item in predictions.items():
        if not np.array_equal(item["targets"], reference):
            raise SystemExit(f"{name} was not scored on the same test edges")
    point = {name: _scalar(item["targets"], item["probabilities"], item["thresholds"])
             for name, item in predictions.items()}
    videos = sorted(set(edge_videos))
    rows_by_video = {video: np.flatnonzero(edge_videos == video) for video in videos}
    per_video = {}
    for video, rows in rows_by_video.items():
        if not reference[rows].any():
            continue
        per_video[video] = {
            "edges": int(len(rows)),
            **{name: _scalar(item["targets"][rows], item["probabilities"][rows],
                             item["thresholds"])
               for name, item in predictions.items()},
        }
    rng = np.random.default_rng(args.seed)
    draws: dict[str, list[dict[str, float]]] = {name: [] for name in predictions}
    for _ in range(args.samples):
        rows = np.concatenate([rows_by_video[video]
                               for video in rng.choice(videos, size=len(videos), replace=True)])
        for name, item in predictions.items():
            draws[name].append(_scalar(item["targets"][rows], item["probabilities"][rows],
                                       item["thresholds"]))
    paired = {}
    for a, b in (tuple(item.split(":", 1)) for item in args.pair):
        entry = {}
        for key in METRICS:
            differences = np.asarray([right[key] - left[key]
                                      for left, right in zip(draws[a], draws[b], strict=True)])
            low, high = np.percentile(differences, [2.5, 97.5])
            entry[key] = {
                "difference": round(point[b][key] - point[a][key], 6),
                "ci95": [round(float(low), 6), round(float(high), 6)],
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        paired[f"{b} - {a}"] = entry
    report = {
        "schema_version": "vidor-checkpoint-comparison-v1",
        "graphs": args.graphs.as_posix(),
        "graphs_sha256": _sha256(args.graphs),
        "split_manifest": args.split_manifest.as_posix(),
        "decision_file": args.decision.as_posix(),
        "decision_sha256": _sha256(args.decision),
        "test_videos": len(videos),
        "test_graphs": len(test_graphs),
        "test_edges": int(len(reference)),
        "checkpoints": {name: {"path": Path(path).as_posix(), "sha256": _sha256(Path(path)),
                               "model_version": predictions[name]["model_version"]}
                        for name, path in checkpoints.items()},
        "point": {name: {key: round(value, 6) for key, value in values.items()}
                  for name, values in point.items()},
        "per_video": per_video,
        "paired_video_bootstrap": paired,
        "resampling": {"unit": "test video", "samples": args.samples, "seed": args.seed},
        "created_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"point": report["point"], "paired": paired}, indent=1))


if __name__ == "__main__":
    main()
