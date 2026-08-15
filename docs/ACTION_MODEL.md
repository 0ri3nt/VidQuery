# Trained person-action model

## Task and evidence boundary

The learned task is multilabel, person-centric atomic action classification. It
does not predict object-object relationships and is not a GNN. Targets come from
official AVA v2.2 numeric action IDs. The importer now treats those IDs as
authoritative because historical fused `action_labels` strings were generated
with an incorrect local mapping.

The reduced label set is `sit` (11), `stand` (12), `carry/hold` (17),
`listen to` (74), `talk to` (79), and `watch person` (80). These are the six
semantically valid labels with usable support across every available video.
Classes such as point and write were not included because their cross-video
support is inadequate for this five-video split.

## Dataset and split

Splitting is deterministic and video-disjoint:

| Split | Videos | Person samples |
|---|---|---:|
| Train | `-FaXLcSFjUI`, `-IELREHX_js`, `-OyDO1g74vc` | 2,711 |
| Validation | `-XpUuIgyUHE` | 1,129 |
| Test | `0f39OWEqJ24` | 994 |

Positive label counts are multilabel and therefore do not sum to sample count:

| Split | sit | stand | carry/hold | listen to | talk to | watch person |
|---|---:|---:|---:|---:|---:|---:|
| Train | 523 | 1,439 | 985 | 571 | 894 | 1,043 |
| Validation | 124 | 525 | 804 | 283 | 363 | 503 |
| Test | 277 | 583 | 286 | 149 | 319 | 361 |

## Feature and model schema

The 30-dimensional feature vector contains normalized person box coordinates,
center, width, height, area, aspect ratio, detector confidence, log person/object
counts, a 13-class same-frame object-context vector, and four log-scaled aligned
audio summary features. It contains no RGB pixels, optical flow, or video clip
encoder. `vidquery.action_model.FEATURE_NAMES` is the versioned source of truth.

The network is a 30→64→32→6 MLP with ReLU and 0.15 dropout. Training uses
`BCEWithLogitsLoss`, training-split positive class weights clipped at 20,
AdamW, learning rate 0.001, batch size 128, seed 2026, and validation-loss early
stopping with patience 15. Per-class thresholds are chosen on the validation
video by a fixed 0.05 grid and never adjusted on the test video.

Training stopped after 21 epochs; the best validation loss was 0.84769517 at
epoch 6. The first train/validation losses were 0.91089657/0.92570299. The final
executed epoch losses were 0.73222480/0.94642758; the saved checkpoint contains
the epoch-6 state.

## Held-out test result

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| sit | 0.394558 | 0.209386 | 0.273585 | 277 |
| stand | 0.611502 | 0.893654 | 0.726132 | 583 |
| carry/hold | 0.288986 | 0.926573 | 0.440565 | 286 |
| listen to | 0.273810 | 0.308725 | 0.290221 | 149 |
| talk to | 0.512064 | 0.598746 | 0.552023 | 319 |
| watch person | 0.460352 | 0.578947 | 0.512883 | 361 |

- Macro F1: **0.465902**
- Micro F1: **0.528039**
- Hardware: CPU, Intel64 Family 6 Model 154
- Measured training duration: 3.557 seconds
- Checkpoint SHA-256:
  `013d1f5f7b84d9587ff12760265e32c58bfe815dbe49586128fcd8443ac34a70`

The complete validation metrics, thresholds, class weights, and per-epoch loss
history are in `evaluation/results/action-model-test.json`.

## Reproduce, load, and activate

```powershell
python -m vidquery.cli train-action-model --seed 2026 --device cpu
```

This saves the ignored binary checkpoint at
`data/app/models/action_classifier/action_model.pt`, its runtime metadata beside
it, and the checked machine-readable report under `evaluation/results/`.
`PersonActionPredictor` refuses activation unless the checkpoint exists, the
feature and label schemas match, metadata says validated, and the SHA-256 digest
matches. Unit tests execute a complete validated checkpoint load and inference.

Fresh-video inference remains off by default. To opt in only after reviewing the
test report:

```dotenv
ENABLE_LEARNED_ACTIONS=true
ACTION_MODEL_CHECKPOINT=data/app/models/action_classifier/action_model.pt
ACTION_MODEL_METADATA=data/app/models/action_classifier/action_model.metadata.json
```

Heuristic geometry and learned action modes remain separate. Processing metadata
records `learned_action_model`, `learned_action_predictions_used`, action source,
and confidence. API health reports a validated checkpoint as active or inactive.

## Claim boundary

Safe claim: a lightweight person-action model was genuinely trained with
video-disjoint validation/test splits and integrated behind strict activation
gates. Unsafe claim: state-of-the-art action recognition, learned graph relation
prediction, or generalization beyond this five-video AVA-derived subset.
