# Final capstone technical evidence

Evidence date: 12 August 2026. This document separates implemented behavior from heuristics, source annotations, learned output, and unimplemented claims.

## 1. Final architecture

```text
MP4 upload/register
  -> validation + SHA-256 deduplication
  -> frames + audio
  -> YOLO / Whisper / required Pyannote / EasyOCR
  -> class-aware short-range entity tracking
  -> timestamp scene graphs
       -> optional validated 80-label PyG GNN person-action inference
       -> optional target resolver with separate provenance
       -> validated VidOR GATv2 explicit relationship inference
  -> geometry + learned relation intervals + validated action evidence
  -> canonical 5-second segments
  -> SQLite (authoritative video/job/evidence store)
       -> local exact-tuple lexical + MiniLM hybrid retrieval
       -> media thumbnails and byte-range playback
  -> optional Neo4j mirror
       -> static parameterized graph candidate retrieval
       -> SQLite hydration of the public result contract
  -> FastAPI + packaged browser UI
       -> optional Groq synthesis over top-k returned evidence only
       -> validated evidence IDs converted to exact playback citations
```

SQLite is intentionally not removed or demoted. It owns upload hashes, job state, warnings, media paths, canonical JSON, default retrieval, and result hydration. Neo4j is the optional explicit graph traversal layer.

## 2. Actual implemented features

- Bounded MP4 validation, SHA-256 deduplication, persistent processing state, and safe retry.
- Frame/audio extraction with YOLO, Whisper, compulsory Pyannote for audio, and
  real EasyOCR adapters in the active configuration.
- Class-aware adjacent-frame entity tracking and confidence-aggregated
  relationship intervals with observation counts.
- Timestamp-aligned canonical segments with detections, transcripts, speakers, actions, relationships, provenance, thumbnails, and processing metadata.
- Exact relationship tuple matching over source ID/class, normalized predicate, target ID/class, directionality, confidence, source method, and timestamp.
- Predicate aliases (`beside`/`next to` → `near`, `talking to` → `talk_to`, `writing on` → `write_on`), symmetric reversal for `near`/`overlaps_with`, and strict direction for directional predicates.
- SQLite hybrid search and explicit Neo4j search using source-owned parameterized Cypher only.
- FastAPI, installed static frontend, video selection, evidence chips, tuple explanations, thumbnails, and HTTP byte ranges.
- Optional Groq grounded-answer synthesis over existing ranked results, with
  strict JSON validation, exact timestamp citations, and failure isolation.
- Real Groq end-to-end proof on 13 August 2026: `openai/gpt-oss-20b` returned a
  supported answer citing `E1`/`E2`; VidQuery resolved them to `final_demo.mp4`
  at 00:00–00:05 and 00:15–00:20 while preserving the raw ranked results.
- A 30-video AVA80 action experiment, a 500-video VidOR relation experiment,
  controlled demo/OCR assets, and machine-readable live proof.

## 3. Heuristic features

- Same-frame geometry labels (`near`, `overlaps_with`, left/right, above/below, containment) and their evidence strengths.
- Deterministic structured-query parsing and weighted hybrid ranking. Cached
  MiniLM embeddings provide the active semantic score; signed hashing vectors
  remain the reproducible baseline.
- Transcript-to-speaker overlap assignment and canonical fixed-window alignment.

Heuristic confidence and result score are not calibrated correctness probabilities.

## 4. Annotation-derived features

- AVA v2.2 person-centric action IDs, mapped from the official numeric label map.
- Imported AVA boxes, source ASR, and diarization artifacts.
- Controlled-demo boxes, speakers, transcripts, and relationships, all labelled `manual_annotation`.

AVA labels are ground truth annotations, not GNN predictions. The controlled demo overlay is ground truth, not automatic inference.

## 5. Learned features

The original six-label and five-video AVA80 checkpoints remain available as
historical evidence. The active experiment uses 30 official AVA videos, a
strict 24/3/3 video split, 23,200 timestamp graphs, and 43,747 supervised person
nodes. Training has positive examples for all 80 classes; validation supports
57 and test supports 65.

The validation-selected model is `ava80-gat-v5-scale30-roi-frame`: a two-layer
edge-aware GATv2 using the handcrafted schema plus frozen ImageNet ResNet-18
object-crop and whole-frame context. Its three-video held-out supported-class
macro F1 is 0.072086, micro F1 is 0.462054, and mAP is 0.086745. The motion
variant has higher test F1 but lower validation metrics, so selecting it would
be leakage.

The ablation suite also trained no-edge, frame-context, local motion, focal,
class-balanced, and temporal-GRU variants. The visual/motion BCE run had better
test scores but a lower validation selection score, so it was not activated.
Class-balanced loss collapsed to zero thresholded F1. The outcome is mixed and
does not establish general GNN superiority.

The separate `vidor-relation-gat-v1` model was trained on 500 VidOR videos with
a strict 400/50/50 split, 3,000 graphs, 89,322 candidate edges, 80 entity
classes, and 50 predicates. Its held-out supported-predicate macro F1 is
0.178522, micro F1 is 0.329635, and mAP is 0.203428. The matched pairwise MLP
wins validation macro F1 (0.259568 versus 0.240843), so this result also does
not establish GNN superiority.

Three earlier comparison artifacts are preserved:

1. The earlier six-label person-action MLP: macro F1 0.465902 and micro F1
   0.528039 over only six labels.
2. A new 80-label MLP using the same person features as the GNN: held-out macro
   F1 0.031794, micro F1 0.243129, and supported-class mAP 0.114653.
3. A trained 80-label PyTorch Geometric GNN with an input projection, two
   edge-aware GATv2 layers, residual connections, dropout, and a person-node
   classifier: held-out macro F1 0.035135, micro F1 0.299546, and
   supported-class mAP 0.088641.

The improved task uses class embeddings, geometry, detector confidence,
audio/transcript summaries, frozen ROI/frame embeddings, ten adjacent-track
motion values, and eight edge features. It compares BCE positive weighting,
focal, and class-balanced objectives; thresholds remain validation-only. Object
nodes are masked from the action loss, normalization and checksum validation are
saved, and random inference weights are rejected.

The AVA GNN predicts actions on people, not explicit target identities. A separate
resolver can select a compatible entity and labels the resulting edge
`gnn_action_plus_target_resolver`. The VidOR GNN does predict explicit ordered
entity-pair predicates with provenance `relationship_gnn`, using class and box
geometry rather than relation ROI pixels. Complete evidence is documented in
`docs/AVA80_GNN.md`, `docs/VIDOR_RELATION_GNN.md`, and `docs/GNN_STATUS.md`.

## 6. Components not implemented

- No generative or unrestricted Cypher execution.
- No heterogeneous transformer, trained video-clip backbone, optical flow,
  face/identity recognition, cross-shot re-identification, or long-term
  identity tracking.
- No authentication, authorization, multi-tenancy, retention/deletion API, malware scanning, or production queue/object store.
- No independent annotator study, confidence intervals, statistical
  significance, or broad-domain benchmark. The existing five-video evaluation
  is not yet a new independent human re-annotation.
- No completed visual Chrome/Edge browser pass or UI screenshots in this session because no controllable browser was available.

## 7. Test results

Final verified commands/results:

| Check | Result |
|---|---|
| Python test suite | 113 passed; one upstream PyG deprecation warning |
| Live Neo4j integration | Passed against the healthy Docker service |
| Ruff (`vidquery tests scripts`) | Passed |
| mypy (`vidquery scripts`) | No issues in 52 source files |
| Python compileall | Passed |
| Go compatibility package | `go test ./...` passed; no Go test files |
| HTTP/UI-surface QA | 24/24 on the current host build; packaged backend and Neo4j healthy after container restart |

The rebuilt CPU container imports Torch 2.5.1, Torchaudio 2.5.1, PyG,
Ultralytics, Whisper, Pyannote, SentenceTransformers, and EasyOCR. Compulsory
diarization reports configured through the existing Hugging Face token. Model
caches and validated `.pt` checkpoints are deliberately not baked into the
image, so their container health entries remain explicitly unavailable until
mounted; random weights are never loaded.

## 8. Neo4j verification

Actual Neo4j 5.26 Community ran in Docker. Schema application is repeatable.
The improved-model verification ingested the held-out video `hbYvDvJrpNk`
twice; both runs retained 179 segments, 3,568 entity occurrences, 55 action
nodes, and 33,303 relationship-evidence nodes. The live database has 13
constraints and 22 indexes. Identical counts prove idempotent re-ingestion for
this record; GNN-only `write` actions and resolver-derived relationships were
retrieved with timestamps and provenance.

The fresh controlled-demo traversal returned learned, temporally smoothed exact
tuples through both SQLite and Neo4j. The current mirror contains 403 relation
evidence nodes for five segments, including 70 relationship-GNN instances and
190 smoothed instances. The OCR fixture has four OCR nodes. Earlier integration records also
prove symmetric reversal, directional rejection, duplicate prevention, tuple
retrieval, and timestamp retrieval. Evidence:
`docs/NEO4J_VERIFICATION.md`, `database/verification_queries.cypher`,
`evaluation/results/neo4j-live-final.json`, and
`evaluation/results/ava80-scale30-neo4j-proof.json`, and
`evaluation/results/full-fresh-video-e2e-proof.json`.

## 9. Independent evaluation dataset

`evaluation/queries_independent.json` has 62 queries over five videos: 52
positives and ten negatives across transcript, visual entity, spatial tuple,
AVA action, speaker, multimodal, OCR, and negative categories. The executable
audit finds zero overlap with AVA action-model training or validation videos;
its only AVA video is in the strict held-out test split. Labels combine manually
selected controlled-fixture intervals, actual Pyannote/EasyOCR output, and
official held-out AVA evidence. This is independent of model training but still
not a multi-annotator study.

## 10. Evaluation metrics

Positive-query overall results (P@5 denominator is five):

| Method | P@1 | P@5 | R@5 | MRR | timestamp acc. | temporal IoU | latency ms | negative rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transcript lexical | 0.4231 | 0.0885 | 0.3462 | 0.4231 | 0.4231 | 0.4231 | 98.4527 | 0.9000 |
| Transcript semantic | 0.5000 | 0.1885 | 0.4692 | 0.5000 | 0.5000 | 0.5000 | 9.0403 | 0.3000 |
| Visual entity | 0.3462 | 0.3500 | 0.3617 | 0.4109 | 0.5192 | 0.5192 | 93.1440 | 0.9000 |
| Hybrid, tuple ablated | 0.9615 | 0.5692 | 0.7706 | 0.9615 | 0.9615 | 0.9423 | 105.1575 | 0.9000 |
| Exact local hybrid | 0.9615 | 0.5769 | 0.7738 | 0.9615 | 0.9615 | 0.9423 | 106.7754 | 0.9000 |
| Exact relationship-aware graph | 0.8654 | 0.5462 | 0.6776 | 0.8654 | 0.8654 | 0.8654 | 128.7383 | 1.0000 |
| Learned v5 enhancement, supported subset | 0.5000 | 0.3714 | 0.2181 | 0.5000 | 0.5000 | 0.5000 | 523.2959 | n/a |

The learned row covers 14/62 eligible held-out queries (22.58%) and is not a
full-corpus comparison. Exact graph and tuple-ablation spatial P@1/MRR both equal
1.0/1.0; exact tuples improve spatial P@5 from 0.95 to 1.0 and R@5 from 0.6553
to 0.6757. Full slices and raw rankings are in
`evaluation/results/independent-final.json` and `docs/EVALUATION_FINAL.md`.

## 11. Screenshots and visual assets

- Controlled original source image: `data/demo/source/meeting_scene.png`.
- Image prompt/provenance: `demo/image_generation_prompt.md`.
- No UI screenshots were captured because the session exposed no controllable browser. This is an explicit evidence gap, not a hidden pass. Required screenshot list is in `docs/UI_QA.md`.

## 12. Demo instructions

Use `docs/FINAL_DEMO_GUIDE.md`. The generated MP4 is placed at `data/demo/final_demo.mp4` and intentionally not committed. Manifest: `demo/final_demo_manifest.json`. Preparation: `scripts/prepare_final_demo.ps1`. Transparent overlay: `scripts/apply_demo_manifest.py`.

## 13. Hardware requirements

- Core app: Python 3.11/3.12, FFmpeg, CPU, local disk; Docker Desktop for the verified stack.
- Controlled demo generation: Windows SAPI desktop voices.
- YOLO/Whisper/action training: PyTorch; CPU works for the small model, while GPU is optional and model-specific.
- Pyannote: Hugging Face access token and substantially more compute/memory than the model-disabled demo.
- Neo4j: local ports 7474/7687 and persistent Docker volume.

## 14. Known limitations

Thirty videos remain small for 80 actions; validation supports 57 actions and
test supports 65. Frozen
image embeddings, local motion, and temporal context are not a trained video
backbone or proof of generalization. AVA/ASR/diarization source evidence is not
an independent human re-annotation. Geometry is camera-sensitive.
Search aliases and lexical matching are limited. Speaker labels are anonymous
clusters. Fixed windows can split events. Operational security is demo-grade.
See `docs/MODEL_LIMITATIONS.md`.

## 15. Transcript-only baseline comparison

On this independent corpus, exact graph retrieval has higher overall P@1/MRR
(0.8654/0.8654) than lexical transcript retrieval (0.4231/0.4231) and frozen
semantic transcript retrieval (0.5000/0.5000), while graph latency is higher.
The local exact hybrid is stronger overall (0.9615/0.9615). This supports a
corpus-specific multimodal-retrieval claim, not universal superiority.

## 16. Safe report/presentation claims

- VidQuery returns timestamped evidence from speech, entities, speakers, actions, and explicit stored relationship tuples.
- Exact tuple matching fixes the co-occurrence false-positive defect and respects symmetric versus directional predicates.
- SQLite remains the reliable system of record; Neo4j is a verified optional graph retrieval mirror.
- An 80-label PyTorch Geometric person-action GNN and a matched 80-label MLP
  baseline were actually trained, checkpointed, held-out tested, and integrated.
- A separate 50-predicate VidOR relationship GNN and matched pairwise MLP were
  actually trained and held-out tested; the MLP wins validation, so no GNN
  superiority claim is made.
- Fresh processing performs short-range entity association and temporal
  relationship smoothing with explicit observation counts.
- Frozen visual ROI features improved the validation-selected GNN over the
  matched handcrafted GNN/MLP variants on this split; other ablations were
  mixed and no broad superiority claim is justified.
- Explicit GNN-assisted target edges are created by a separate resolver and
  carry `gnn_action_plus_target_resolver` provenance.
- The five-video/62-query independent evaluation and all reported metrics are
  reproducible machine-readable artifacts.
- The controlled demo uses original generated imagery and explicit manual ground truth.

## 17. Claims that must not be made

- Do not call AVA labels, geometry rules, manual demo labels, or old GNN feature exports learned GNN predictions.
- Do not claim the GNN directly predicts object/person targets.
- Do not conflate AVA target-resolver edges with VidOR relationship-GNN output.
- Do not claim the VidOR GNN uses relation pixels, beats its MLP baseline on the
  validation criterion, or generalizes beyond its held-out annotation subset.
- Do not claim the action model recognizes arbitrary actions, uses optical flow
  or a trained video backbone, tracks identities, or generalizes broadly.
- Do not claim exact graph retrieval is always faster or universally superior.
- Do not claim independent human re-annotation, statistical significance, production security/readiness, manual cross-browser approval, or UI screenshot evidence.
- Do not claim model-enabled container inference unless a validated checkpoint is explicitly mounted and health reports it active.
