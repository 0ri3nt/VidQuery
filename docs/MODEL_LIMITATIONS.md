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
outperform all matched ablations. In the relation experiment, the pairwise MLP
won validation supported macro F1 (0.259568 versus 0.240843), while the GNN had
higher held-out test supported macro F1 (0.178522 versus 0.161331). This does not
establish GNN superiority. The relation checkpoint uses object class and box
geometry rather than ROI pixels, audio, or text. Old random feature exports
remain invalid as predictions. See `GNN_STATUS.md` and
`VIDOR_RELATION_GNN.md`.

## Evaluation scope

The old five-query, one-video file is retained only as a regression smoke test.
The existing 35-query/five-video corpus reports all requested categories and
negatives, but its labels derive from source AVA/ASR/diarization artifacts. It
is not an independent human re-annotation study and provides no annotator
agreement, confidence intervals, statistical significance, or broad-domain
generalization evidence. A new independent manual evaluation remains required
before making final comparative claims.

## Operational and privacy limits

Uploads, transcripts, frames, and speaker clusters remain on local disk until manually removed. The application has no deletion API, retention policy, encryption-at-rest layer, authentication, authorization, malware scanner, rate limiter, or multi-tenant isolation. Do not process sensitive recordings or expose the service publicly without those controls and an appropriate consent/legal basis.
