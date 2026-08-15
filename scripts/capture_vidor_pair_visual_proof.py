from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vidquery.domain import BoundingBox, Detection
from vidquery.vidor_improved import (
    ImprovedVidORRelationPredictor,
    improved_relation_checkpoint_status,
)
from vidquery.vidor_relation import VIDOR_PREDICATES


def _detections(graph: Any, width: int, height: int) -> list[Detection]:
    output = []
    for index, values in enumerate(graph.x.tolist()):
        x1, y1, x2, y2 = values[:4]
        output.append(
            Detection(
                detection_id=f"vidor-{graph.video_id}-{graph.frame_id}-{index}",
                frame_id=f"{graph.video_id}:{graph.frame_id}",
                video_id=str(graph.video_id),
                timestamp=float(graph.timestamp),
                class_id=int(graph.class_ids[index]),
                class_label=str(graph.class_names[index]),
                confidence=float(values[6]),
                pixel_bbox=BoundingBox(
                    x1=x1 * width, y1=y1 * height, x2=x2 * width, y2=y2 * height
                ),
                normalized_bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                centroid=((x1 + x2) * width / 2, (y1 + y2) * height / 2),
                centroid_normalized=((x1 + x2) / 2, (y1 + y2) / 2),
                track_id=str(graph.track_ids[index]),
            )
        )
    return output


def main() -> None:
    checkpoint = Path(
        "data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt"
    )
    metadata = checkpoint.with_suffix(".metadata.json")
    status = improved_relation_checkpoint_status(
        checkpoint, metadata, "vidor-relation-pair-visual-gat-v2"
    )
    if status != "available_validated":
        raise RuntimeError(f"pair-visual checkpoint is not usable: {status}")
    cache = torch.load(
        "data/app/models/vidor_relation_pair_visual/vidor_relation_pair_visual_graphs.pt",
        map_location="cpu",
        weights_only=False,
    )
    split = json.loads(
        Path("data/source/vidor/selected-frames/split_manifest.json").read_text(encoding="utf-8")
    )
    test_ids = set(split["test"])
    predictor = ImprovedVidORRelationPredictor(
        checkpoint,
        metadata,
        device="cuda" if torch.cuda.is_available() else "cpu",
        expected_model_version="vidor-relation-pair-visual-gat-v2",
    )
    selected_graph = None
    predictions: list[dict[str, Any]] = []
    detections: list[Detection] = []
    for graph in cache["graphs"]:
        if str(graph.video_id) not in test_ids:
            continue
        frame_path = Path(str(graph.frame_path))
        with Image.open(frame_path) as image:
            detections = _detections(graph, image.width, image.height)
        predictions = predictor.predict_detections(
            detections, frame_path=frame_path, max_predicates=3
        )
        selected_graph = graph
        if predictions:
            break
    if selected_graph is None:
        raise RuntimeError("no held-out VidOR graph was available for inference proof")
    truth = []
    for edge, labels in enumerate(selected_graph.edge_y):
        source, target = selected_graph.edge_index[:, edge].tolist()
        for predicate_id in torch.nonzero(labels, as_tuple=False).flatten().tolist():
            truth.append(
                {
                    "source_id": detections[source].detection_id,
                    "source_class": detections[source].class_label,
                    "predicate": VIDOR_PREDICATES[predicate_id],
                    "target_id": detections[target].detection_id,
                    "target_class": detections[target].class_label,
                }
            )
    proof = {
        "schema_version": "vidor-pair-visual-inference-proof-v1",
        "checkpoint_status": status,
        "model_version": predictor.model_version,
        "split": "test",
        "video_id": str(selected_graph.video_id),
        "frame_id": int(selected_graph.frame_id),
        "timestamp": float(selected_graph.timestamp),
        "frame_path": str(selected_graph.frame_path),
        "frame_pixels_used": True,
        "visual_provenance": "frozen_resnet18_subject_object_union_roi",
        "detection_count": len(detections),
        "annotated_relationships": truth,
        "prediction_count": len(predictions),
        "predictions": predictions[:20],
        "claim": (
            "inference execution proof only; aggregate held-out metrics remain the "
            "evidence for model quality"
        ),
    }
    output = Path("evaluation/results/vidor-pair-visual-inference-proof.json")
    output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
