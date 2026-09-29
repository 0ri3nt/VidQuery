# AVA v2.2 80-label GNN

## Verified status

VidQuery contains a genuinely trained PyTorch Geometric GNN for person-centric
AVA action classification. It produces 80 action logits for each person node.
It does not predict an explicit action target or arbitrary object-object
predicate. A separate deterministic compatible-entity resolver may turn a
predicted action into an explicit edge; those edges carry the distinct
provenance `gnn_action_plus_target_resolver`.

The official AVA v2.2 label map is loaded and validated by
`vidquery.ava_labels`. Every official ID, display name, and label type is
preserved; compact canonical names form the search vocabulary.

## Data coverage and split

The official train/validation annotations inspected by the coverage tool contain
299 videos, 426,346 timestamped annotated-person boxes, 1,105,692 person-action
rows, and all 80 action IDs. The executable local subset was expanded to 30
official AVA videos. Its graph cache has 23,200 timestamp graphs, 68,565 nodes,
and 43,747 supervised person nodes. Training retains positive support for all
80 actions.

The deterministic video-disjoint split is:

| Split | Videos | positive classes |
|---|---:|---:|
| train | 24 | 80 |
| validation | 3 | 57 |
| test | 3 | 65 |

No timestamp from a video occurs in multiple splits. Exact video IDs and
per-action supports are saved in
`data/app/models/ava80_scaling/30/split_manifest.json`.

## Graph and feature contract

One graph is built per annotated AVA timestamp. Nodes include annotated or
detected persons and detected objects. Every node includes the original
138-dimensional handcrafted schema: class embedding, normalized box geometry,
area/aspect ratio, YOLO confidence, pooled ROI/frame descriptors, and aligned
audio/transcript summaries.

The improved cache adds frozen torchvision ResNet-18 ImageNet1K-v1 features:

- `roi_x`: handcrafted features plus 512-dimensional object-crop embedding;
- `visual_x`: ROI features plus 512-dimensional whole-frame context;
- `visual_motion_x`: visual features plus ten local track/motion values.

Tracking uses an AVA entity ID when available and otherwise adjacent-frame IoU.
Motion includes previous/next centre displacement, log area ratio, track age,
and track duration. It is not face recognition, re-identification, optical flow,
or a long-term identity tracker. Frozen embeddings are cached once per
timestamp under `data/app/models/ava80_improved/visual_embedding_cache`.

Edges include all person-person pairs, nearby person-object pairs, and nearby
object-object pairs. Both directions store relative x/y, centre distance, IoU,
log size ratio, direction, and overlap. Every node has an 80-dimensional
multi-hot target, but only annotated person nodes are selected by `loss_mask`.

## Models and training

The comparison includes:

- an 80-label MLP using the same handcrafted person features;
- GAT without graph edges;
- two-layer residual edge-aware GATv2 with handcrafted features;
- frozen ROI, ROI+frame, and ROI+frame+motion GAT variants;
- BCE positive weighting, focal loss, and class-balanced loss;
- a temporal model that feeds previous/current/next GNN person embeddings into
  a GRU;
- deterministic adjacent-timestamp probability smoothing.

All trainable variants use person-node multilabel loss, deterministic seed 2026,
validation-based early stopping, and per-class thresholds selected on
validation data. A class absent or insufficient in validation is disabled with
threshold 1.0. Test metrics are computed only after selection.

## Held-out results

| Variant | validation supported macro F1 | test supported macro F1 | test micro F1 | test mAP |
|---|---:|---:|---:|---:|
| GAT + frozen ROI (v4) | 0.109693 | 0.067558 | 0.418315 | 0.071208 |
| **GAT + frozen ROI/frame (v5)** | **0.125380** | 0.072086 | 0.462054 | **0.086745** |
| GAT + visual/motion, BCE | 0.115852 | **0.079769** | **0.466916** | 0.083150 |
| GAT + visual/motion, focal | 0.098335 | 0.075038 | 0.385523 | 0.078182 |

`ava80-gat-v5-scale30-roi-frame` is active because it has the highest validation
supported-class macro F1 (0.125380) and validation mAP (0.108653) in the fixed
30-video experiment. The visual/motion BCE model has better test macro/micro F1,
but choosing it after looking at test would be leakage.

This is mixed evidence. Visual ROI features clearly improve validation results
over handcrafted variants, but these experiments do not establish general GNN,
edge, motion, or temporal superiority. The dataset is small, imbalanced, and
domain-specific.

## Artifacts

Core files:

- `data/app/models/ava80_improved/ava80_improved_graphs.pt`;
- `data/app/models/ava80_scaling/30/split_manifest.json`;
- `data/app/models/ava80_scaling/30/ablations/ablation_report.json`;
- `data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.pt`;
- `data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.metadata.json`.

Every `.pt` checkpoint has a sibling `.bundle.json` with its label map,
thresholds, feature schema, normalization or static-checkpoint reference,
backbone, split, class coverage, configuration, training history, metrics,
checksums, and git/data revision. Inference rejects missing, unvalidated,
checksum-mismatched, model-version-mismatched, label-mismatched,
feature-mismatched, or threshold-mismatched artifacts. Random weights are never
accepted.

## Reproduction

```powershell
python scripts/select_ava_coverage.py --target-per-class 5 --output evaluation/results/ava-expansion-selection-target5.json
python scripts/download_ava_selection.py --selection evaluation/results/ava-expansion-selection-target5.json
python scripts/process_ava_selection.py --selection evaluation/results/ava-expansion-selection-target5.json
python scripts/build_ava_improved_features.py
python scripts/train_ava_improved_ablations.py
python scripts/retrain_ava_structural_baselines.py
python scripts/finalize_ava_improved_artifacts.py
```

Activate the validation-selected model with:

```dotenv
ENABLE_GNN_ACTIONS=true
GNN_CHECKPOINT=data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.pt
GNN_METADATA=data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.metadata.json
GNN_DEVICE=auto
GNN_MODEL_VERSION=ava80-gat-v5-scale30-roi-frame
GNN_FEATURE_CACHE_DIR=data/app/models/ava80_improved
```

The held-out live API proof is
`evaluation/results/ava80-scale30-api-unseen-proof.json`. The live Neo4j
idempotency, action-node, resolver-edge, and timestamp proof is
`evaluation/results/ava80-scale30-neo4j-proof.json`.

## Final scale/temporal pass (37 videos)

The storage-safe expansion targeted 75 videos but stopped at a validated atomic
boundary with 37 usable videos when final training began. The corpus contains
28,775 timestamp graphs, 85,447 nodes, and 55,113 supervised person nodes. The
fixed scale-30 held-out videos remain unchanged (31/3/3 train/validation/test),
so no video crosses splits. Temporary raw media and frames were deleted only
after per-video cache validation and atomic master-cache rebuilding.

The single new temporal feature is a cached 512-dimensional frozen short-clip
descriptor built from previous/current/next ImageNet-ResNet18 frame embeddings,
fixed temporal pooling, and a signed change term. No video model was fine-tuned.
Both new candidates retain BCE positive weights, validation-only per-class
thresholds, and deterministic rare-class-aware graph sampling.

| Model | Validation supported macro F1 | Validation mAP | Held-out supported macro F1 | Held-out micro F1 | Held-out mAP |
|---|---:|---:|---:|---:|---:|
| Existing 30-video ROI+frame | **0.125380** | **0.108653** | 0.072086 | **0.462054** | **0.086745** |
| 37-video ROI+frame + rare sampling | 0.125064 | 0.093991 | **0.076767** | 0.454177 | 0.079676 |
| 37-video ROI+frame+temporal + rare sampling | 0.121453 | 0.097977 | withheld | withheld | withheld |

The 37-video ROI+frame model won the new A/B comparison by validation supported
macro F1, and only its test split was then evaluated. It did not exceed the
existing model's validation result, so deployment retains
`ava80-gat-v5-scale30-roi-frame`. The temporal candidate was not selected and
its test labels remain withheld.

Machine-readable evidence is in
`data/app/models/ava80_final/comparison.json`,
`data/app/models/ava80_final/split_manifest.json`, and
`evaluation/results/ava-expansion75-state-final.json`.

```powershell
python -m vidquery expand-ava-dataset --target-videos 75 --batch-size 3 --workers 3 --target-support 25 --min-free-gb 4
python -m scripts.train_ava_final_temporal --maximum-videos 37 --skip-graph-cache --epochs 30 --patience 6 --batch-size 32 --device auto
```
