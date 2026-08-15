# Evaluation

## Reproducible smoke evaluation

The repository includes `evaluation/queries.json`, five evidence-grounded queries across transcript, visual, action, relational, and multimodal categories. The expected intervals were verified against the existing fused AVA artifact for source video `-FaXLcSFjUI`. The relational smoke query uses an exact stored `person left_of chair` edge; the former `person near chair` label was removed because the interval merely co-located those classes while its `near` edges connected other endpoint pairs. Import and run:

```powershell
python -m vidquery import-ava --limit 1
python -m vidquery evaluate --dataset evaluation/queries.json --output evaluation/results/latest.json
```

The original pre-fix run is retained as `evaluation/results/ava-smoke.json`. The current exact-tuple run is `evaluation/results/ava-smoke-exact.json` (one indexed video, 154 canonical segments, cutoff 5):

| Method | P@1 | P@5 | Recall@5 | MRR | timestamp tolerance | mean temporal IoU | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transcript-only lexical | 0.40 | 0.08 | 0.40 | 0.40 | 0.40 | 0.40 | 13.75 ms |
| Structured multimodal hybrid | 1.00 | 0.32 | 1.00 | 1.00 | 1.00 | 1.00 | 18.64 ms |

These are local single-run timings and a regression smoke test, not headline metrics or a general performance claim. Regenerate the machine-readable result after importing only the named smoke artifact; latency varies by host. The small set contains one or two relevant labelled intervals per query. P@5 can count multiple returned windows within the tolerance around those labels, while recall deduplicates each labelled target.

`evaluation/results/relation-tuple-regression.json` is a deliberately harsher run of the same one-video labels against the two-video local index. It includes distractor-video effects and is retained as failure-analysis evidence, not as an independent multi-video benchmark.

## Metric definitions

- Precision@k: returned windows matching a labelled source/time target divided by k.
- Recall@5: distinct labelled targets found in the first five divided by total labelled targets.
- MRR: reciprocal rank of the first matching window.
- Timestamp tolerance rate: whether the first match starts within the query's declared tolerance.
- Temporal IoU: intersection-over-union between the first matched and labelled intervals.
- Latency: in-process wall-clock ranking time; it excludes HTTP, media processing, and playback.

A match requires the stable source-video filename stem and either interval overlap or start-time tolerance. Database UUIDs are deliberately excluded from labels so results remain reproducible after re-import.

## Baseline and ablation interpretation

The transcript-only method sees transcript tokens and cannot intentionally
satisfy visual/action/relationship queries. The hybrid adds allowlisted
structured evidence and a deterministic hashing-vector lexical score. This
retrieval table verifies that the modality paths affect ranking; it does not
compare the learned action models because the smoke corpus is not the frozen
AVA action-classification test split. Learned-model results are reported below.

## AVA80 learned-model evaluation

The learned action experiment is separate from the retrieval smoke test. It
uses 17 official AVA videos and a deterministic video-level split of 11 train,
three validation, and three test videos. There is no frame leakage. Training has
positive examples for all 80 AVA actions; validation supports 57 actions and
test supports 65. The frozen test set was not used for model selection.

| Variant | validation supported macro F1 | test supported macro F1 | test micro F1 | test mAP |
|---|---:|---:|---:|---:|
| 80-label MLP, handcrafted | 0.069865 | 0.057890 | 0.399430 | 0.059321 |
| GAT, no graph edges | 0.062326 | 0.053429 | 0.376516 | 0.052420 |
| GAT, handcrafted | 0.065337 | 0.050924 | 0.364483 | 0.053808 |
| **GAT + frozen ResNet-18 ROI, selected** | **0.108907** | 0.060309 | 0.408184 | 0.064652 |
| GAT + frozen ROI/frame | 0.099612 | 0.062869 | 0.427799 | 0.067587 |
| GAT + ROI/frame/motion, BCE | 0.105661 | **0.065643** | **0.457517** | **0.069662** |
| GAT + ROI/frame/motion, focal | 0.107828 | 0.063879 | 0.393098 | 0.067218 |
| temporal GAT + GRU | 0.102127 | 0.061961 | 0.397833 | 0.064333 |

The selected model is the validation winner, not the hindsight test winner.
The class-balanced-loss run collapsed to zero thresholded F1. The experiments
show that frozen ROI features helped this split, while graph edges, additional
frame/motion features, and temporal recurrence did not establish consistent
superiority. All-class macro F1 is lower than supported-class macro F1 because
unsupported validation/test labels are scored as zero.

Exact per-class precision, recall, F1, support, AP, thresholds, training losses,
hardware, durations, and checkpoint hashes are in
`data/app/models/ava80_improved/ablations/ablation_report.json` and the sibling
checkpoint metadata/bundle files. The old five-video and intermediate
17-video/single-test-video runs remain in that report for provenance, but are
not directly comparable to the balanced three-video held-out evaluation.

The selected checkpoint was exercised on the unseen test video
`hbYvDvJrpNk` through the live VidQuery API. Its machine-readable search proof
is `evaluation/results/ava80-improved-api-unseen-proof.json`. A separate live
Neo4j proof, including idempotent re-ingestion and parameterized action and
resolver-edge retrieval, is
`evaluation/results/ava80-improved-neo4j-proof.json`.

## Extending the benchmark

Add queries with independent annotations and preserve provenance. The next useful set should contain at least:

- 20+ source videos with video-level train/development/test separation;
- positive and no-match queries, paraphrases, negation, and distractor entities;
- relation tuples whose source/target classes are annotated explicitly;
- short actions below the sample interval and overlapping speakers;
- per-category sample counts, multiple relevance windows, and annotator agreement;
- cold/warm latency, indexing throughput, memory/GPU use, and model-load time.

Report confidence intervals and failure examples, not only aggregate scores. A learned semantic or GNN method must be compared with transcript lexical, structured heuristic, and simple feature baselines on the same frozen split.
