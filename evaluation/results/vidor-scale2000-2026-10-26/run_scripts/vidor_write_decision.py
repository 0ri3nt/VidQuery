"""Record validation-only choices for the VidOR-2000 comparison before any test scoring."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

C = {
    "gat_v2": "data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2",
    "mlp_v2": "data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-mlp-v2",
    "gat_v3_scale500": "data/app/models/vidor_relation_scale500/vidor-relation-pair-visual-gat-v3-scale500",
    "mlp_v3_scale500": "data/app/models/vidor_relation_scale500/vidor-relation-pair-visual-mlp-v3-scale500",
    "gat_v4_scale2000": "data/app/models/vidor_relation_scale2000/vidor-relation-pair-visual-gat-v4-scale2000",
    "mlp_v4_scale2000": "data/app/models/vidor_relation_scale2000/vidor-relation-pair-visual-mlp-v4-scale2000",
    "gat_v4_scale2000_warm": "data/app/models/vidor_relation_scale2000_warm/vidor-relation-pair-visual-gat-v4-scale2000-warm",
    "mlp_v4_scale2000_warm": "data/app/models/vidor_relation_scale2000_warm/vidor-relation-pair-visual-mlp-v4-scale2000-warm",
}
out = Path("evaluation/results/vidor-scale2000-2026-10-26/validation-decision.json")
assert not out.exists()
cands = {}
for name, stem in C.items():
    m = json.loads(Path(stem + ".metadata.json").read_text()); pt = Path(stem + ".pt")
    cands[name] = {
        "checkpoint": pt.as_posix(), "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
        "train_videos": len(m["split"]["train"]), "validation_videos": len(m["split"]["validation"]),
        "use_graph": name.startswith("gat"), "warm_start": m.get("warm_start"),
        "selected_epoch": m.get("selected_epoch"),
        "test_status_at_decision": m.get("test_evaluation_status", "evaluated during its own earlier run (baseline)"),
        "validation": {k: m["validation_metrics"][k] for k in ("macro_f1_supported_predicates", "mean_average_precision_supported_predicates", "micro_f1")},
    }
key = lambda n: (cands[n]["validation"]["macro_f1_supported_predicates"], cands[n]["validation"]["mean_average_precision_supported_predicates"])
new = ["gat_v4_scale2000", "mlp_v4_scale2000", "gat_v4_scale2000_warm", "mlp_v4_scale2000_warm"]
decision = {
    "schema_version": "vidor-validation-decision-v1",
    "written_before_test": True,
    "validation_set": "the fixed 50 validation videos of the scale-500 split (graphs verified byte-identical in the 2,000 cache); v2 metrics come from its own 6-video validation split and are not comparable",
    "candidates": cands,
    "selection_rule": "validation supported-predicate macro F1, tie-break validation mAP",
    "overall_validation_choice": max(new, key=key),
    "runtime_eligible_choice": max([n for n in new if n.startswith("gat")], key=key),
    "runtime_note": "ImprovedVidORRelationPredictor requires a GNN checkpoint, so only GAT models can be activated",
    "pre_registered_comparisons": [
        ["gat_v2", "gat_v4_scale2000"], ["gat_v2", "gat_v4_scale2000_warm"],
        ["gat_v3_scale500", "gat_v4_scale2000"], ["gat_v3_scale500", "gat_v4_scale2000_warm"],
        ["mlp_v3_scale500", "mlp_v4_scale2000"], ["mlp_v3_scale500", "mlp_v4_scale2000_warm"],
        ["gat_v4_scale2000", "gat_v4_scale2000_warm"], ["mlp_v4_scale2000", "mlp_v4_scale2000_warm"],
        ["gat_v4_scale2000", "mlp_v4_scale2000"], ["gat_v2", "gat_v3_scale500"],
    ],
    "test_graphs": "scale-500 cache, 50 test videos (the set every earlier VidOR comparison used)",
    "significance_rule": "claim only if the paired 95% video-bootstrap CI excludes 0",
    "promotion": "none; production defaults unchanged pending review",
    "created_at": datetime.now(timezone.utc).isoformat(),
}
out.write_text(json.dumps(decision, indent=1) + "\n")
print(json.dumps({k: decision[k] for k in ("overall_validation_choice", "runtime_eligible_choice")}))
