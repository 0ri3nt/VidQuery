"""Record validation-only choices for the AVA-100 comparison before any test scoring."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

C = {
    "legacy_gat_24": "data/app/models/ava80_noleak/ava80-gat-v7-noleak-lr0.0003",
    "legacy_mlp_24": "data/app/models/ava80_noleak/ava80-mlp-v7-noleak-lr0.0003",
    "stream_gat_24": "data/app/models/ava80_shard_stream_check/ava80-gat-v7-noleak-shardstream-lr0.0003",
    "stream_mlp_24": "data/app/models/ava80_scale100_baselines/ava80-mlp-v7-noleak-shardstream-s30-lr0.0003",
    "stream_gat_94": "data/app/models/ava80_scale100/ava80-gat-v8-noleak-scale100-lr0.0003",
    "stream_mlp_94": "data/app/models/ava80_scale100/ava80-mlp-v8-noleak-scale100-lr0.0003",
    "active_v5_leaky_reference": "data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame",
}
out = Path("evaluation/results/ava80-scale100-2026-10-26/validation-decision.json")
assert not out.exists()
candidates = {}
for name, stem in C.items():
    meta = json.loads(Path(stem + ".metadata.json").read_text())
    pt = Path(stem + ".pt")
    v = meta["validation_metrics"]
    candidates[name] = {
        "checkpoint": pt.as_posix(), "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
        "train_videos": len(meta["split"]["train"]), "validation_videos": meta["split"]["validation"],
        "architecture": meta.get("architecture"), "feature_attribute": meta.get("feature_attribute"),
        "learning_rate": meta.get("optimizer", {}).get("learning_rate"),
        "best_epoch": meta.get("checkpoint_selection", {}).get("best_epoch"),
        "test_status_at_decision": meta.get("test_evaluation_status") or "evaluated earlier on the 3 base test videos (reference only)",
        "validation": {k: v[k] for k in ("macro_f1_supported_classes", "mean_average_precision", "micro_f1")},
    }
key = lambda n: (candidates[n]["validation"]["macro_f1_supported_classes"], candidates[n]["validation"]["mean_average_precision"])
choice = max(("stream_gat_94", "stream_mlp_94"), key=key)
decision = {
    "schema_version": "ava80-validation-decision-v1",
    "written_before_test": True,
    "validation_set": candidates["stream_gat_94"]["validation_videos"],
    "candidates": candidates,
    "selection_rule": "highest validation macro F1 (supported classes, all persons), tie-break validation mAP",
    "scale100_validation_choice": choice,
    "pre_registered_comparisons": [
        ["stream_gat_24", "stream_gat_94", "data scaling, GAT, same trainer"],
        ["stream_mlp_24", "stream_mlp_94", "data scaling, MLP, same trainer"],
        ["legacy_gat_24", "stream_gat_94", "vs existing leakage-free baseline"],
        ["legacy_gat_24", "stream_mlp_94", "vs existing leakage-free baseline"],
        ["stream_gat_94", "stream_mlp_94", "GAT vs MLP at 94 training videos"],
        ["legacy_mlp_24", "stream_mlp_94", "vs existing leakage-free MLP baseline"],
    ],
    "primary_metric": "test macro F1 (supported classes, all supervised persons); mAP and micro F1 secondary; detector-only persons reported",
    "significance_rule": "claim only if the paired 95% CI (keyframe-within-video bootstrap) excludes 0; the video-level bootstrap and per-video results are reported as a conservative check",
    "promotion": "none; production defaults unchanged pending review",
    "created_at": datetime.now(timezone.utc).isoformat(),
}
out.write_text(json.dumps(decision, indent=1) + "\n")
print(json.dumps({"choice": choice, "validation": {n: c["validation"] for n, c in candidates.items()}}, indent=1))
