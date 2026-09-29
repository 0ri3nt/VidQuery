from __future__ import annotations

import json
from pathlib import Path

import torch

from vidquery.ava_ablation import train_static_ablation
from vidquery.ava_improved import MOTION_FEATURE_NAMES, RESNET18_BACKBONE


def main() -> None:
    model_root = Path("data/app/models/ava80_improved")
    output_dir = model_root / "ablations"
    payload = torch.load(
        model_root / "ava80_improved_graphs.pt", map_location="cpu", weights_only=False
    )
    graphs = list(payload["graphs"])
    split = json.loads((model_root / "split_manifest.json").read_text(encoding="utf-8"))
    dimensions = {
        name: int(getattr(graphs[0], name).shape[1])
        for name in ("handcrafted_x", "roi_x", "visual_x", "visual_motion_x")
    }
    schema = {
        "dataset": payload["schema_version"],
        "backbone": RESNET18_BACKBONE,
        "dimensions": dimensions,
        "motion_features": list(MOTION_FEATURE_NAMES),
        "active_feature_attribute": "handcrafted_x",
    }
    specifications = [
        ("ava80-mlp-v2-balanced", False, True),
        ("ava80-gat-v2-no-edges", False, False),
    ]
    results = {}
    for version, use_edges, mlp_only in specifications:
        print(f"[structural baseline] {version}", flush=True)
        results[version] = train_static_ablation(
            graphs=graphs,
            split=split,
            feature_attribute="handcrafted_x",
            use_edges=use_edges,
            mlp_only=mlp_only,
            loss_mode="bce_positive_weights",
            output_dir=output_dir,
            model_version=version,
            feature_schema=schema,
            epochs=60,
            patience=10,
            device_name="auto",
        )
    report_path = output_dir / "ablation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name, metadata in results.items():
        report["ablations"][name] = {
            "validation_metrics": metadata["validation_metrics"],
            "test_metrics": metadata["test_metrics"],
            "loss_mode": metadata["loss_mode"],
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: {
            "validation": metadata["validation_metrics"],
            "test": metadata["test_metrics"],
        }
        for name, metadata in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
