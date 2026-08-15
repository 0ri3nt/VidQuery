# VidOR explicit relationship GNN

## Task boundary

This model predicts a predicate for an explicit ordered entity pair at a video
timestamp. It is separate from the AVA person-action model and from deterministic
bounding-box geometry. Predictions use provenance `relationship_gnn` and model
version `vidor-relation-gat-v1`.

The official VidOR training annotations contain 7,000 JSON video records,
267,210 relation intervals, 80 object classes, and 50 predicates. The downloaded
archive is `data/source/vidor/training-annotation.zip`; its verified official MD5
is `85F39FDD81A780BBB9B975CCA8F219A2`, and ZIP integrity passes.

## Graphs and split

The reproducible storage-safe experiment selects 500 videos and uses a strict
400 training / 50 validation / 50 test video split. No video contributes frames
to more than one split.

- 3,000 timestamp graphs;
- 89,322 directed candidate edges;
- 94.3764% candidate recall over feasible positive relation instances;
- node inputs: object class embedding, normalized box, area/aspect ratio,
  confidence, and person flag;
- edge inputs: relative position, centre distance, overlap/IoU, and size ratio.

The class and boxes come from VidOR annotations during training. Product
inference maps supported YOLO labels into the same vocabulary. There are no ROI
pixel, transcript, or audio features in this relation checkpoint.

## Models and held-out metrics

| Model | validation supported macro F1 | test supported macro F1 | test micro F1 | test supported mAP |
|---|---:|---:|---:|---:|
| **Pairwise MLP** | **0.259568** | 0.161331 | 0.326389 | 0.189743 |
| GATv2 relation GNN | 0.240843 | **0.178522** | **0.329635** | **0.203428** |

The MLP wins the declared validation criterion. The GNN has higher held-out test
numbers, but that observation did not change model selection. The deployed GNN
is therefore evidence that a real graph model is integrated, not evidence that
graph message passing is superior to the matched MLP.

## Reproduction and artifacts

```powershell
python scripts/train_vidor_relationship_gnn.py
python scripts/capture_vidor_relation_proof.py
```

Key artifacts:

- `data/app/models/vidor_relation/held_out_metrics.json`;
- `data/app/models/vidor_relation/vidor-relation-mlp-v1.pt`;
- `data/app/models/vidor_relation/vidor-relation-gat-v1.pt`;
- `data/app/models/vidor_relation/split_manifest.json`;
- `data/app/models/vidor_relation/relationship_vocabulary.json`;
- `evaluation/results/vidor-relation-live-proof.json`.

Inference refuses a missing, unvalidated, checksum-mismatched, schema-mismatched,
or model-version-mismatched checkpoint. Search continues to use static,
parameterized tuple plans; the model never produces executable Cypher.
