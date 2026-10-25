"""Train VidOR relation models on detector boxes and compare under deployment conditions.

The active ``vidor-relation-pair-visual-gat-v2`` was trained on annotation
boxes and classes, with node confidence 1.0 (manual) or 0.5 (interpolated).
At inference it sees YOLO boxes, YOLO classes mapped to VidOR, and detector
confidences.  This script:

1. runs the pipeline's YOLO (same weights, confidence, and IoU settings) on
   the staged frames and matches detections one-to-one to annotations;
2. trains matched MLP and GATv2 models on those detector-matched graphs
   (new versions, new directory; validation selects, test is read once);
3. evaluates the active checkpoints on the same detector-matched test graphs
   and on their original annotation-box test graphs.

Relations whose endpoints YOLO missed cannot be recovered by any relation
model; the reported recall ceiling makes that loss explicit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vidquery.config import get_settings
from vidquery.eval_registry import assert_not_trained_on
from vidquery.vidor_improved import (
    ImprovedRelationModel,
    build_and_train_improved_vidor,
)
from vidquery.vidor_relation import VIDOR_PREDICATES, _graph_split, _metrics

ACTIVE_DIR = Path("data/app/models/vidor_relation_pair_visual")


class YoloFrameDetector:
    """frame path -> [(normalized box, COCO label, confidence)] with pipeline settings."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        self.settings = get_settings()
        weights = Path(self.settings.yolo_model)
        if not weights.is_file():
            weights = Path("data/app/models/yolo") / weights.name
        self.weights = weights
        self.model = YOLO(str(weights))
        self.frames = 0

    def __call__(self, frame_path: Path) -> list[tuple[list[float], str, float]]:
        result = self.model.predict(
            source=str(frame_path),
            conf=self.settings.detection_confidence,
            iou=self.settings.detection_iou,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        names = result.names
        self.frames += 1
        output = []
        boxes: Any = result.boxes if result.boxes is not None else []
        for box in boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            output.append(
                (
                    [x1 / width, y1 / height, x2 / width, y2 / height],
                    str(names[int(box.cls.item())]),
                    float(box.conf.item()),
                )
            )
        return output


def _evaluate_checkpoint(path: Path, graphs: list[Any], device: torch.device) -> dict[str, Any]:
    from torch_geometric.loader import DataLoader

    payload = torch.load(path, map_location="cpu", weights_only=True)
    factory: Any = ImprovedRelationModel(use_graph=bool(payload["use_graph"]))
    model = factory.to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for raw in DataLoader(graphs, batch_size=16, shuffle=False):
            batch = raw.to(device)
            logits = model(
                batch.x,
                batch.class_ids,
                batch.node_visual,
                batch.node_motion,
                batch.edge_index,
                batch.edge_attr,
                batch.union_visual,
            )
            predictions.append(torch.sigmoid(logits).cpu())
            targets.append(batch.edge_y.cpu())
    target = torch.cat(targets).numpy()
    probability = torch.cat(predictions).numpy()
    thresholds = payload["thresholds"].numpy()
    metrics = _metrics(target, probability, thresholds, np)
    metrics["predicted_relations_per_edge"] = round(
        float((probability >= thresholds[None, :]).sum(axis=1).mean()), 4
    )
    return metrics


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not isinstance(value, list | dict)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/source/vidor/training-annotation.zip")
    )
    parser.add_argument(
        "--frame-root", type=Path, default=Path("data/source/vidor/selected-frames")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/source/vidor/selected-frames/split_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/app/models/vidor_relation_detector_v3")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/vidor-detector-matched-comparison.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for path in (args.output_dir, args.report):
        if path.exists():
            raise SystemExit(f"{path} exists; choose a new versioned path")
    if args.output_dir.resolve() == ACTIVE_DIR.resolve():
        raise SystemExit("refusing to write into the active relation model directory")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = json.loads(Path(manifest["source_split_manifest"]).read_text(encoding="utf-8"))
    support: Counter[str] = Counter()
    for row in source["selected_videos"]:
        support.update(row["support"])
    split = {
        name: [str(value) for value in manifest[name]]
        for name in ("train", "validation", "test")
    }
    assert_not_trained_on(split["train"], context="VidOR detector-matched training split")
    detector = YoloFrameDetector()
    report = build_and_train_improved_vidor(
        annotation_zip=args.annotations,
        frame_root=args.frame_root,
        output_dir=args.output_dir,
        selected=manifest["selected_videos"],
        split=split,
        predicate_support=support,
        seed=args.seed,
        detector=detector,
        version_tag="v3-detector",
    )
    detector_graphs = torch.load(
        args.output_dir / "vidor_relation_pair_visual_graphs.pt", map_location="cpu",
        weights_only=False,
    )["graphs"]
    annotation_graphs = torch.load(
        ACTIVE_DIR / "vidor_relation_pair_visual_graphs.pt", map_location="cpu",
        weights_only=False,
    )["graphs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_test = _graph_split(detector_graphs, split)["test"]
    annotation_test = _graph_split(annotation_graphs, split)["test"]
    winner = (
        "gat" if report["validation_winner"] == "relationship_gnn" else "mlp"
    )
    comparison: dict[str, Any] = {}
    name = "vidor-relation-pair-visual-{}-{}.pt"
    for label, path in (
        ("active_gat_v2_annotation_trained", ACTIVE_DIR / name.format("gat", "v2")),
        ("active_mlp_v2_annotation_trained", ACTIVE_DIR / name.format("mlp", "v2")),
        ("gat_v3_detector_trained", args.output_dir / name.format("gat", "v3-detector")),
        ("mlp_v3_detector_trained", args.output_dir / name.format("mlp", "v3-detector")),
    ):
        on_detector = _evaluate_checkpoint(path, detector_test, device)
        comparison[label] = {
            "checkpoint": path.as_posix(),
            "test_on_detector_boxes": _summary(on_detector),
            "per_predicate_on_detector_boxes": on_detector.get("per_predicate"),
        }
        if "active" in label:
            comparison[label]["test_on_annotation_boxes"] = _summary(
                _evaluate_checkpoint(path, annotation_test, device)
            )
    result = {
        "schema_version": "vidor-detector-matched-comparison-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "detector": {
            "weights": detector.weights.as_posix(),
            "confidence": detector.settings.detection_confidence,
            "iou": detector.settings.detection_iou,
            "frames_processed": detector.frames,
        },
        "split_video_counts": {name: len(values) for name, values in split.items()},
        "detector_matching": report["dataset"].get("detector_matching"),
        "detector_dataset": {
            key: value for key, value in report["dataset"].items() if key != "detector_matching"
        },
        "validation_winner_v3": winner,
        "validation_v3": {
            "mlp": _summary(report["pairwise_mlp"]["validation_metrics"]),
            "gat": _summary(report["relationship_gnn"]["validation_metrics"]),
        },
        "predicates": list(VIDOR_PREDICATES),
        "test": comparison,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("test", "predicates")}
    print(json.dumps(summary, indent=2))
    for label, values in comparison.items():
        print(label, values["test_on_detector_boxes"], values.get("test_on_annotation_boxes", ""))


if __name__ == "__main__":
    main()
