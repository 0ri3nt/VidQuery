# GNN status

Evidence date: 13 August 2026.

## Verified conclusion

VidQuery now has two genuinely trained PyTorch Geometric models:

1. `ava80-gat-v5-scale30-roi-frame`, an 80-label person-centric AVA v2.2
   action classifier; and
2. `vidor-relation-pair-visual-gat-v2`, a 50-predicate explicit subject-object relation
   classifier trained from VidOR annotations.

Both have versioned checkpoints, strict held-out video splits, validation-only
thresholds, metadata/checksum validation, held-out metrics, deterministic
inference tests, canonical-segment integration, SQLite persistence, Neo4j
serialization, API search, and frontend provenance. Random weights are never
loaded.

## AVA action GNN

The active action checkpoint projects handcrafted node features plus frozen
ImageNet ResNet-18 ROI and whole-frame features, applies two edge-aware GATv2
layers with residual connections, layer normalization, and dropout, then emits
80 logits for person nodes. Object nodes participate in message passing but are
masked from the action loss.

The corpus has 30 official AVA videos, 23,200 timestamp graphs, 68,565 nodes,
and 43,747 supervised persons. The video-disjoint split is 24/3/3; training has
positive support for all 80 actions, validation supports 57, and test supports
65. Exact IDs and counts are saved in
`data/app/models/ava80_scaling/30/split_manifest.json`.

| Variant | validation supported macro F1 | validation mAP | test supported macro F1 | test micro F1 | test mAP |
|---|---:|---:|---:|---:|---:|
| GAT + frozen ROI (v4) | 0.109693 | 0.094486 | 0.067558 | 0.418315 | 0.071208 |
| **GAT + frozen ROI/frame (v5, active)** | **0.125380** | **0.108653** | 0.072086 | 0.462054 | **0.086745** |
| GAT + ROI/frame/motion, BCE | 0.115852 | 0.096692 | **0.079769** | **0.466916** | 0.083150 |
| GAT + ROI/frame/motion, focal | 0.098335 | 0.086547 | 0.075038 | 0.385523 | 0.078182 |

The ROI/frame model is active because it won the declared validation rule.
Selecting the motion model for its higher test macro F1 would be test leakage.
Detailed results are in
`data/app/models/ava80_scaling/30/ablations/ablation_report.json`.

The action GNN predicts person actions, not targets. A separate compatible-
entity resolver may create `person -> action -> target` evidence with provenance
`gnn_action_plus_target_resolver`; that target must not be described as a GNN
prediction.

### Final AVA scale/temporal decision

Storage-safe expansion reached 37 validated videos (28,775 graphs and 55,113
supervised person nodes) before the final training cutoff, short of the requested
75-video target. The scale-37 ROI+frame candidate achieved validation supported
macro F1 0.125064 and validation mAP 0.093991; its validation-selected held-out
metrics are supported macro F1 0.076767, micro F1 0.454177, and mAP 0.079676.
The otherwise matched frozen-temporal candidate achieved validation supported
macro F1 0.121453 and mAP 0.097977, so its test split was not evaluated.

Neither new candidate exceeded the deployed scale-30 validation result
(supported macro F1 0.125380, mAP 0.108653). Therefore the active checkpoint
remains `ava80-gat-v5-scale30-roi-frame`; no weaker checkpoint was activated.
See `data/app/models/ava80_final/comparison.json` for the complete measured
comparison and checksums.

## VidOR relationship GNN

The final pair-visual experiment uses 67 official VidOR videos from archive
part 1 with a strict 57/6/4 split. It contains 402 timestamp graphs and 11,981
candidate edges over 80 object classes and 50 predicates: 7,900 positives,
1,441 nearby hard negatives, and 2,640 random/background negatives. Supervised
candidate-edge recall is 1.0.

| Model | validation supported macro F1 | test supported macro F1 | test micro F1 | test mAP |
|---|---:|---:|---:|---:|
| Old pair MLP v1 | **0.259568** | 0.161331 | 0.326389 | 0.189743 |
| Old GATv2 v1 | 0.240843 | 0.178522 | 0.329635 | 0.203428 |
| Pair-visual MLP v2 | 0.270589 | **0.182475** | 0.319328 | **0.210712** |
| **Pair-visual GATv2 v2 (active)** | **0.308495** | 0.178194 | **0.371113** | 0.191364 |

The pair-visual GNN beats its matched MLP on both declared validation metrics
and was selected before test evaluation. Both receive identical frozen
ResNet-18 subject, object, and union-box embeddings plus class, geometry, and
motion features; message passing is the graph model's only extra signal. The
new test split has only four videos, so old-vs-new figures are descriptive and
test superiority must not be claimed.

The active v2 runtime requires real extracted-frame pixels, refuses zero visual
fallbacks, and stores canonicalized predicates with `relationship_gnn`
provenance. Complete evidence is in `docs/VIDOR_RELATION_FINAL_PASS.md` and
`evaluation/results/vidor-relation-final-comparison.json`; old v1 artifacts are
preserved under `data/app/models/vidor_relation/`.

## Temporal evidence

Fresh video processing assigns deterministic class-aware short-range track IDs
using adjacent-frame IoU and centroid proximity. Consecutive relationship
observations for a tracked tuple are merged into intervals with an aggregated
confidence and observation count. Each interval is rebound to concrete
detections in every overlapping canonical segment, preserving exact tuple
matching and Neo4j entity links. This is local tracking, not identity
recognition or cross-shot re-identification.

## Activation

The active `.env` selects both validated checkpoints. Health must report
`gnn_person_action_classifier=available_validated_enabled` and
`gnn_relationship_prediction=available_validated_enabled`. Missing, checksum-
mismatched, schema-incompatible, or version-incompatible artifacts are rejected
with an explicit unavailable state.

Held-out action API/Neo4j evidence is recorded in
`evaluation/results/ava80-scale30-api-unseen-proof.json` and
`evaluation/results/ava80-scale30-neo4j-proof.json`.
