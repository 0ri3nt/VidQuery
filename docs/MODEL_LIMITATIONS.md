# Model and system limitations

## Visual evidence

- The supplied YOLOv8n checkpoint uses its detector label space (normally COCO); it cannot recognize project-specific concepts without training.
- Query aliases such as `police car` normalize to YOLO's generic `car` class.
  This can locate cars at the correct sampled seconds, but it cannot prove that
  a car is a police vehicle. A dedicated emergency-vehicle checkpoint would be
  required for that claim.
- Product video frames are sampled at 1 fps by default. Short events can be
  missed. Fresh inference now assigns class-aware tracks using adjacent-frame
  IoU and centroid proximity and merges repeated tuple evidence into intervals.
  This is short-range association, not persistent identity recognition,
  cross-shot tracking, face recognition, or re-identification.
- Bounding-box confidence is detector confidence. Spatial relationships are deterministic geometry heuristics over same-frame boxes, not learned semantic facts.
- `near`, direction, overlap, containment, and touch thresholds are sensitive to camera perspective, box quality, and scale. Confidence values are heuristic evidence strengths, not calibrated probabilities.
- Structured relationship search now requires an explicit stored edge between
  the requested source and target classes. Symmetric predicates may reverse the
  endpoints; directional predicates may not. This proves stored tuple
  connectivity, not that the underlying detector or geometry heuristic is true.

## Actions

The AVA adapter uses the official v2.2 numeric action IDs and treats them as
annotations, never as predictions. The legacy six-label MLP remains available
as a fallback.

The active optional GNN is a real 80-label PyTorch Geometric person-action
classifier. The expanded executable corpus has 30 videos with a strict 24/3/3
split. Training contains all 80 actions; validation supports 57 and test
supports 65. The validation-selected ROI/frame GAT achieved supported-class
macro F1 0.072086, micro F1 0.462054, and supported-class mAP 0.086745 on the
three held-out videos. These low values are reported deliberately; they do not
support a broad or production-grade 80-action recognition claim. Absence of a
prediction is not proof of absence, and predictions can create false-positive
timestamp matches.

The improved experiments use frozen ImageNet ResNet-18 ROI/frame embeddings,
box/context geometry, detector confidence, local adjacent-box motion features,
and aligned audio/transcript summaries. The selected model uses ROI features;
frame context, motion, focal loss, and a temporal GRU were evaluated but did not
win the validation-only selection rule. There is no optical flow, trained video
clip backbone, face/identity recognition, or long-term tracker. A separate
heuristic target resolver may attach compatible nearby objects or people; those
targets are not GNN predictions. See `AVA80_GNN.md`.

## Speech and speakers

- Whisper quality depends on language, noise, accents, overlap, music, and model size. Segment confidence may be unavailable.
- Pyannote diarization is token-gated and resource-intensive. It is required by
  the active full-pipeline configuration for videos with audio. `SPEAKER_00`
  labels are local cluster names, not real identities.
- Speaker assignment uses maximum temporal overlap between diarization and transcript intervals; overlapping speakers and boundary errors can be misassigned.
- Fixed five-second windows may duplicate a transcript/action across adjacent evidence windows or split a long event.

## Retrieval

The current ranker combines deterministic structured constraints and lexical
evidence with cached `all-MiniLM-L6-v2` sentence embeddings over transcript,
entity, action, relation, and OCR text. A signed hashing-vector encoder remains
the reproducible ablation baseline. MiniLM improves paraphrase matching but is
still weak on negation, coreference, complex composition, domain-specific
language, and unsupported visual concepts. Scores are ranking signals, not
correctness probabilities.

The API supports explicit `sqlite` and `neo4j` retrieval backends. Neo4j uses an
allowlisted, parameterized subject-predicate-object traversal and SQLite hydrates
the canonical result evidence. SQLite remains the metadata/job system of record.

## GNN

A trained person-action GNN and a separate trained VidOR relationship GNN are
available only with validated checkpoints. The action GNN did not consistently
outperform all matched ablations. The active relation model is the pair-visual
GATv2 v2, which uses frozen ResNet-18 subject, object, and union-box features
plus class, geometry, and motion. It beat its matched MLP on validation
supported macro F1 (0.308495 versus 0.270589), but on the four-video test split
the MLP was marginally higher on macro F1 (0.182475 versus 0.178194) while the
GNN was higher on micro F1. This does not establish GNN superiority. Old random
feature exports remain invalid as predictions. See `GNN_STATUS.md` and
`VIDOR_RELATION_FINAL_PASS.md`.

Both GNNs are trained on AVA movie clips and VidOR videos. On phone recordings,
meetings, or screen captures, some classes are never predicted. For example, the
AVA GNN produced no `drink` or `work on computer` predictions on a 72-second
custom phone video. That is domain shift, not a retrieval bug; see
[RETRIEVAL_DIAGNOSTICS.md](RETRIEVAL_DIAGNOSTICS.md).

## Review-2 localization and reliability

- Localized timestamps are only as precise as the stored evidence. At 1 fps,
  visual peaks have about one-second granularity; speech peaks reach word level
  only for videos indexed with `WHISPER_WORD_TIMESTAMPS=true`.
- Reliability priors, precision confidences, and verdict thresholds are
  documented design constants derived from measured results. They are not
  calibrated probabilities.
- A `supported` verdict means the evidence is strong under those priors, not
  that the moment is verified correct.
- Hypothesis generation recognises three ambiguity patterns (action homonym,
  entity-only, unclassified concept). Other ambiguities rely on the planner.
- The challenge benchmark is implemented but has no annotated manifest yet, so
  no measured Review-2 improvement is claimed.

## Evaluation scope

The old five-query, one-video file is retained only as a regression smoke test.
The final 62-query, five-video independent benchmark
([EVALUATION_FINAL.md](EVALUATION_FINAL.md)) was manually specified with zero
overlap with AVA training videos, but it is capstone-scale: no annotator
agreement study, no confidence intervals, and five videos. It labels
relevance on the five-second grid, so it cannot measure sub-segment timestamp
error. That is what the Review-2 challenge benchmark is for.

## Operational and privacy limits

Uploads, transcripts, frames, and speaker clusters remain on local disk until manually removed. The application has no deletion API, retention policy, encryption-at-rest layer, authentication, authorization, malware scanner, rate limiter, or multi-tenant isolation. Do not process sensitive recordings or expose the service publicly without those controls and an appropriate consent/legal basis.
