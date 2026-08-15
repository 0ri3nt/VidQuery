# VidOR relationship final pass

This is the final focused relationship-model experiment. Development stops here;
the experiment does not add another dataset or architecture family.

## Outcome

The matched pair-visual GATv2 won by the required validation rule. Its validation
supported-predicate macro F1 is **0.308495**, versus **0.270589** for the matched
pair MLP. Validation mAP is **0.287223** versus **0.222036**. Test was evaluated
only after validation selected each checkpoint.

| Model | Videos (train/val/test) | Val macro F1 | Test macro F1 | Test micro F1 | Test mAP |
|---|---:|---:|---:|---:|---:|
| old pair MLP v1 | 500 (400/50/50) | 0.259568 | 0.161331 | 0.326389 | 0.189743 |
| old GATv2 v1 | 500 (400/50/50) | 0.240843 | 0.178522 | 0.329635 | 0.203428 |
| pair-visual MLP v2 | 67 (57/6/4) | 0.270589 | 0.182475 | 0.319328 | 0.210712 |
| pair-visual GATv2 v2 | 67 (57/6/4) | **0.308495** | 0.178194 | **0.371113** | 0.191364 |

The new MLP/GNN comparison is fair: both receive identical subject ROI, object
ROI, union-box, class, geometry, and motion inputs. The GNN's only additional
signal is graph message passing. Old-vs-new values are descriptive, not a clean
feature ablation, because the visual run covers the 67 manifest videos present
in official video archive part 1 rather than all 500 annotation-only videos.

## Data and sampling evidence

- Official VidOR source: `shangxd/vidor`, `training-video-part1.zip`.
- 67 videos, inherited disjoint split: 57 train / 6 validation / 4 test.
- 402 decoded annotated frames; temporary MP4s were deleted immediately.
- 402 graphs and 11,981 candidate edges.
- 7,900 positive edges, 1,441 nearby plausible hard negatives, and 2,640
  seeded random/background negatives.
- Candidate-edge recall during supervised graph construction: 1.0.
- Frozen ImageNet ResNet-18 produces cached 512-dimensional subject, object,
  and union-box embeddings. They are not recomputed during epochs.

## Selection and checkpoints

Primary selection is validation supported-predicate macro F1; validation mAP is
the tie-break. Test metrics never select a model.

- Selected GNN: `data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt`
  (3,531,614 bytes, SHA-256
  `0da6be0b16d704083e785aa468ff7f34836bd071378bce6b8531a700fce3fa39`).
- Matched MLP: `data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-mlp-v2.pt`
  (1,971,300 bytes).
- Old v1 checkpoints remain unchanged under `data/app/models/vidor_relation/`.

The runtime adapter is `ImprovedVidORRelationPredictor`. Its inference method is:

```python
predict_detections(
    detections,
    *,
    frame_path: Path | str,
    previous_detections=None,
    nearest_k=8,
    max_predicates=3,
)
```

It requires readable frame pixels and computes the same frozen subject/object/
union features. An optional previous-frame detection list supplies track-aligned
motion; absent tracks receive the documented zero-motion/missing flag. Missing
frames and mismatched checkpoints fail explicitly.

The production processing service now selects this adapter from the configured
model version, passes the matching extracted frame plus prior timestamp
detections, normalizes predicted predicates into the canonical vocabulary, and
stores the existing `relationship_gnn` provenance in canonical segments,
SQLite, and Neo4j. It skips timestamps without pixels and records a warning;
it never fabricates zero visual embeddings. The selected v2 paths are the
defaults in `.env.example`, while explicitly configured v1 artifacts remain
loadable by the legacy adapter.

## Exact reproduction

```powershell
python -m scripts.stage_vidor_pair_visual_frames --part 1 --maximum-selected-videos 500 --max-frames-per-video 6
python -m scripts.train_vidor_pair_visual --frame-root data/source/vidor/selected-frames --output-dir data/app/models/vidor_relation_pair_visual --epochs 40 --patience 7 --batch-size 16 --device cuda
python -m scripts.capture_vidor_pair_visual_proof
python -m pytest -q tests/test_vidor_relation.py
python -m compileall -q vidquery tests scripts
```

Machine-readable evidence is in
`evaluation/results/vidor-relation-final-comparison.json`; full per-predicate
metrics and training histories are in
`data/app/models/vidor_relation_pair_visual/held_out_metrics_pair_visual.json`.
The real-frame held-out inference execution is saved at
`evaluation/results/vidor-pair-visual-inference-proof.json`.

## Honest limitations

- The new test split has only four videos and 17 supported predicates; it is a
  real held-out test but materially smaller than the old experiment.
- Training boxes are annotations, while live boxes come from YOLO.
- Supervised construction retains all labelled positive pairs, so its 1.0
  candidate recall is not a claim about detector/runtime candidate recall.
- Predicates absent from validation retain threshold 1.0 and are not activated.
- The selected checkpoint predicts explicit ordered VidOR predicates, not AVA
  action targets and not unrestricted natural-language relationships.
