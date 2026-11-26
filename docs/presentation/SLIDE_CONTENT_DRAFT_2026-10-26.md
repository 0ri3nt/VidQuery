# VidQuery — final-year presentation, slide content (draft for review)

Drafted 26 October 2026 from the repository at commit `034359f` and later
result files. Every number below comes from a committed result file; the
source is listed on each slide. **Nothing has been turned into a PowerPoint
yet.**

## Read this first: flags for review

| # | Flag | Affects |
|---|---|---|
| F1 | **Resolved 26 Oct:** the ablations finished (`docs/INTEGRATED_EVALUATION_2026-10-26.md`). With the selected settings, AVA v8 +0.007 [−0.021, 0.033], VidOR v4 0.000 [−0.012, 0.011] and integrated vs the 41.4% run −0.009 [−0.036, 0.017], all not significant. In strict mode AVA v8 raises in-domain rejection by +0.045 [0.020, 0.072]. **Done:** Slide 13 now has these in Table E. | Slide 13 |
| F2 | **Active models changed on 26 Oct (approved):** the local `.env` and `.env.example` now use AVA `gat-v8-noleak-scale100` and VidOR `gat-v4-scale2000` with deterministic Whisper, and the local library was reprocessed. Soft constraints and frame similarity are still **off** by default; they and the 10 s evaluation spans are what produced the Charades numbers. Describe the active models as v8/v4 throughout. | Slides 3–6, 12–15 |
| F3 | **The 10 s "expand:10" span exists only in the evaluator.** The product UI returns the localizer's peak moment and the 5 s segment. The live demo will therefore not reproduce the 0.40 P@1 interval logic exactly. | Slides 7–8 |
| F4 | **The controlled demo video (`final_demo.mp4`) uses manual annotations** for entities, relations, transcripts and speakers. Present it as "retrieval over labelled evidence", never as model output. | Slides 7–8 |
| F5 | `docs/GNN_STATUS.md` still calls VidOR GAT v2 "active". That is outdated: v3-scale500 is active in the local `.env`. Fix the doc before the viva or don't show it. | — |
| F6 | **Not done:** the lecture-video benchmark (speech/OCR/phonetic) has no recordings or labels yet. The live Neo4j integration test is skipped without a running service. mypy reports 13 pre-existing type errors. | Slides 10, 15 |
| F7 | **The live demo server launch with the integrated profile has not been rehearsed yet** (see Slide 8). | Slide 8 |
| F8 | Published Charades-STA scores of trained models are not quoted, because none were verified this session. If you want a literature row, verify the paper numbers first. | Slide 13 |

Suggested deck: 16 slides plus title (title, agenda and thank-you slides not drafted).

---

## Section 1 — Complete pipeline / architecture (2 slides)

### Slide 1 — End-to-end pipeline: from MP4 to evidence-backed answers

**Bullets**
- **Upload:** streamed with a size limit, MP4 signature probe and SHA-256 de-duplication, then registered in SQLite.
- **Extraction:** frames at 1 fps and 16 kHz mono audio feed 7 extractors: YOLOv8n with tracking, EasyOCR, OpenCLIP, Whisper, pyannote, and the VidOR relation GNN and AVA action GNN.
- **Alignment:** everything is aligned into **5-second canonical segments** that keep each observation's own timestamp and provenance.
- **Storage:** SQLite is the system of record; Neo4j is an optional graph mirror.
- **Query → plan:** a natural-language query becomes a validated plan (entities, actions, relation tuples, speaker, spoken or OCR terms).
- **Answer:** two-stage search. Stage 1 retrieves and ranks segments; stage 2 localizes the peak moment, assigns a reliability verdict, and abstains when the evidence is insufficient. An optional grounded answer cites only real result IDs.

**Visual — architecture diagram (left to right, swim lanes)**
1. *Ingestion lane:* `Browser/CLI → FastAPI → validation + SHA-256 → SQLite (UPLOADED)`.
2. *Extraction lane:* `FFmpeg frames (1 fps) / audio (16 kHz)`, fanning out to the boxes `YOLOv8n + IoU/centroid tracks`, `EasyOCR`, `OpenCLIP ViT-B/32`, `Whisper tiny (greedy, T=0)`, `pyannote 3.1`. YOLO feeds `box geometry relations` and the `VidOR pair-visual GATv2`.
3. *Alignment lane:* all arrows meet at `Canonical 5 s segments`, then `AVA 80-label GATv2`, then `MiniLM embeddings`, then `SQLite` (with a dashed arrow to `Neo4j mirror`).
4. *Query lane:* `Query → rule-based parser (+ constrained Groq planner only if ambiguous) → hypotheses → hybrid/reliability ranking → localizer → verdict / abstain → UI (seek to peak) / grounded answer`.
- Colour code: blue = pretrained frozen models, orange = models trained in this project (two GNNs), grey = deterministic code.

**Speaker notes**
VidQuery turns a raw video into a searchable, timestamped evidence index.
Every upload is checked and de-duplicated, then split into frames and audio.
Several models extract complementary evidence: what is said and by whom, what
text is on screen, which objects are present, how they relate, and what
people are doing. All of it is aligned into five-second segments, but each
observation keeps its own timestamp, so we can later point to the exact
moment. At query time we parse the question into a structured plan, retrieve
and rank segments, localize the moment inside the best segment, and decide
how much the evidence can be trusted. If it isn't enough, the system says so
instead of guessing. The two models we trained ourselves are the action and
relation graph networks; everything else is a pretrained component.

**Sources:** `README.md` (What it does, diagram); `docs/ARCHITECTURE.md`;
`vidquery/pipeline.py`; `.env.example` (`FRAME_SAMPLE_RATE=1.0`,
`SEGMENT_DURATION=5.0`).

### Slide 2 — Data model and query flow

**Bullets**
- **Canonical segment:** `{segment_id, video_id, start, end, frame_ids, detections, entities, relationships, actions, transcript (+ word timings), speakers, OCR evidence, embedding, processing_metadata}`.
- **Provenance on every item:** `source_method`, confidence, timestamp, model version and checkpoint SHA-256 (the SHA was added on 26 Oct).
- **Neo4j mirror:** `Video → Segment → EntityOccurrence / TranscriptSegment / Speaker / Action / OCRText / RelationshipEvidence (FROM_ENTITY, TO_ENTITY)`, written with idempotent `MERGE`.
- **Query understanding:** rule-based parser with an 80-label AVA action lexicon. The Groq LLM planner (`openai/gpt-oss-20b`) is consulted only for ambiguous queries, maps onto allowed vocabularies, and never writes executable Cypher.
- **Ambiguity:** for queries like "find drive" (an action, or a spoken word?), each interpretation is retrieved separately; the system then resolves it or asks the user.
- **Trust:** `confidence = Σ(query_weight × raw_score × reliability_prior) / Σ query_weight`, giving `supported` (≥ 0.45), `weak` (≥ 0.15) or `insufficient` (abstain).

**Visual:** two panels.
- **Left:** an ER-style graph of the Neo4j schema (7 node types, edge labels).
- **Right:** a vertical flow of one query, `"person near a laptop" → plan {entities: person, laptop; tuple: (person, near, laptop)} → 3 ranked segments → localized peak 00:02.0 → verdict chip`, with the fusion formula in a callout box.

**Speaker notes**
Everything lives in one canonical segment record. That record is the contract
between indexing and search. Each piece of evidence remembers where it came
from and how reliable that source is. A graph mirror exists for graph-style
queries, but SQLite alone is enough to run the whole system. Queries are
parsed locally first; the language model is a fallback for ambiguous
phrasing and can only choose from known labels. The reliability formula is
why a single weak model output, such as an action-GNN hit, cannot make a
result look certain on its own.

**Sources:** `vidquery/domain.py` (`CanonicalSegment`); `vidquery/neo4j.py`;
`docs/REVIEW2.md` (fusion, thresholds, hypotheses); `vidquery/config.py`
(`QUERY_PLANNER_MODEL`, `RAG_MODEL` default `openai/gpt-oss-20b`).

---

## Section 2 — Implementation details (4 slides)

### Slide 3 — Module 1: Ingestion and multimodal extraction

**Bullets**
- **API:** FastAPI streaming upload, with a state machine `UPLOADED → PREPROCESSING → VISUAL → AUDIO → SCENE_GRAPH → ALIGNING → INDEXING → READY/FAILED`. Heavy models are lazy-imported.
- **Vision:**
  - YOLOv8n detections, with class-aware short-range tracks from adjacent-frame IoU and centroid distance;
  - EasyOCR for on-screen text;
  - OpenCLIP ViT-B/32 for track appearance and per-frame embeddings.
- **Audio:**
  - Whisper `tiny` with word timestamps, **greedy decoding at temperature 0** (new default);
  - pyannote 3.1 diarization, assigned to transcript words.
- **Text:** MiniLM (`all-MiniLM-L6-v2`) segment embeddings for semantic matching.
- **Engineering decisions:**
  - optional components degrade with a warning rather than failing the job;
  - required diarization fails loudly;
  - random GNN weights are never used.
- **Reproducibility fix, measured:** before, 17/40 clips transcribed identically across repeats; after, 40/40. Two full 20-video rebuilds are now identical in all 131 segments.

**Visual:** a table of extractor → model → output → stored field. Small inset: a before/after bar of repeat-identical transcripts (17/40 vs 40/40).

**Speaker notes**
Ingestion is built so that one failing optional model doesn't lose the whole
video; it records a warning instead. A reproducibility problem showed up
during evaluation: Whisper's default retries with random sampling, so
re-indexing the same video changed its transcripts, and everything derived
from them. Switching to deterministic decoding made index rebuilds exactly
repeatable, which is what makes our later comparisons trustworthy.

**Sources:** `docs/ARCHITECTURE.md` (lifecycle, boundaries); `vidquery/models.py`
(`WhisperTranscriber.decoding_options`); `vidquery/tracking.py`;
`evaluation/results/whisper-determinism-2026-10-26/`; `docs/SETUP.md`.

### Slide 4 — Module 2: Learned graph models (AVA actions, VidOR relations)

**Bullets**
- **AVA action GATv2** (80 labels):
  - one graph per keyframe; nodes are persons and objects;
  - node features are box geometry and class, plus **frozen ResNet-18** ROI and whole-frame embeddings;
  - two edge-aware GATv2 layers (residual, LayerNorm, dropout); object nodes are masked out of the loss.
- **Leakage-free schema (v7/v8):** drops the `detector_confidence` and `annotation_only` columns, which revealed ground-truth boxes. Data is streamed from slim per-video shards.
- **Decoding:** per-class thresholds fitted on validation, plus AVA label-group limits (≤ 1 pose, ≤ 2 object, ≤ 2 person actions).
- **VidOR pair-visual GATv2** (80 object classes, 50 predicates):
  - candidate subject→object edges: all positives, 2× nearby hard negatives and 1× seeded random negatives;
  - frozen ResNet-18 subject, object and union-box features;
  - MLP baseline with identical inputs.
- **Runtime safety:** checkpoints load only if metadata, model version and SHA-256 match. Relation inference refuses zero-pixel fallbacks.
- **Scale this round:** AVA 24 → 94 training videos (61 new videos, 44.6 GB); VidOR 400 → 1,900 training videos, with validation and test frames frozen (byte-identical graphs).

**Visual:** two small graph sketches.
- **Left:** a keyframe graph with person and object nodes, GATv2 arrows and an 80-way output head.
- **Right:** a subject–object pair with ROI and union boxes feeding a 50-predicate head.

**Speaker notes**
These are the two models we trained. Both are graph attention networks over
a frozen image backbone, so they are cheap to train on a laptop GPU. For AVA
we found that two input columns indirectly revealed which boxes came from the
ground truth. We removed them and treat that leakage-free model as the
honest baseline. For VidOR every model is compared against an MLP that sees
exactly the same features, so any gain from the graph is real. The pipeline
only loads a checkpoint whose metadata and hash match, which stopped us
deploying a wrong or random model by mistake.

**Sources:** `docs/GNN_STATUS.md` (architecture; note F5); `vidquery/ava_gnn80.py`
(`AVA80GNNPredictor`, `decode_ava_predictions`, `AVA80_NOLEAK_DROPPED`);
`vidquery/vidor_improved.py`; `docs/EXPERIMENTS_2026-10-26.md`.

### Slide 5 — Module 3: Query understanding, retrieval, localization, answerability

**Bullets**
- **Parser:** a local rule-based parser plus an action lexicon covering all 80 AVA labels, with inflections and veto patterns for idioms. Calibration-set label recall went from 0.53 to 1.00.
- **Ranking:** hybrid lexical, semantic (MiniLM), entity, action, OCR, appearance and exact relation-tuple evidence. Reliability-weighted ranking ("Review-2") is optional.
- **Localizer:** finds the peak moment inside the winning segment, with a precision level of word > frame > interval > utterance > segment. When modalities agree, the peak is boosted.
- **Answerability:** the `supported` / `weak` / `insufficient` verdict, plus abstention. Thresholds are fitted on **validation only**.
- **Experimental retrieval features (off by default):**
  - *soft constraints* lower a segment's score for a missing entity or action instead of rejecting it;
  - *frame-level OpenCLIP text similarity* adds a `visual_semantic` score (prior 0.55).
- **Also implemented:** phonetic matching for ASR errors (off by default), and grounded Groq answers that cite only real result IDs.

**Visual:** a pipeline strip for one Charades query, `"person closes the door"` → plan {action: close, entity: person} → candidate segments with score bars → peak → verdict `weak (0.50)`. Callout: "soft constraints + frame similarity are experimental flags".

**Speaker notes**
The parser decides what kind of evidence a query needs. On real household
video the strict "every word must match" design missed many moments, so we
added two experimental options: soft constraints, and a frame-level
image–text similarity signal. Both are off by default and were tuned only on
validation videos. The localizer then picks the exact moment, and the
reliability layer decides whether to answer at all.

**Sources:** `vidquery/action_lexicon.py`; `vidquery/search.py`;
`vidquery/localization.py`; `vidquery/reliability.py`; `vidquery/frame_semantics.py`;
`vidquery/phonetics.py`; `vidquery/rag.py`; `docs/REVIEW2.md`;
`docs/PROGRESS_2026-10-25.md` (parser recall 0.53 → 1.00).

### Slide 6 — Module 4: Storage, API, UI and engineering for reproducibility

**Bullets**
- **Storage:** SQLite holds the video registry, processing state, warnings and canonical segment JSON. Neo4j is an idempotent mirror, disabled locally.
- **REST API:**
  - `POST /api/videos` (upload);
  - `GET /status` (processing state and model provenance);
  - `POST /api/search` (ranking mode, localize, hypothesis ID);
  - byte-range `/stream`, `/thumbnail`, `/health`.
- **Browser UI:** library, search, hypothesis chips, evidence verdicts, seek-to-peak playback, and an Annotate tab for labelling benchmarks.
- **Experimental model profiles:**
  - `evaluation/configs/model_profiles_2026-10-26.json` and `scripts/run_with_profile.py` apply model overrides as environment variables, after validating every checkpoint;
  - the production `.env` is never edited.
- **Reproducibility:** versioned output folders; scripts refuse to overwrite; fixed seeds (2026); SHA-256 for manifests, checkpoints and shards; resumable expansion; logged timelines and RAM/disk use.

**Visual:** a screenshot collage: UI search results with a verdict chip, the hypothesis chips, `/api/health` JSON and the `/docs` OpenAPI page. *(Screenshots still to be captured; see Slide 9.)*

**Speaker notes**
The API stays responsive because heavy models are loaded only when first
needed. A small profile mechanism let us test new model combinations without
touching the production configuration, which matters because our rule is not
to promote any model without review. Every experiment writes to a new
versioned folder and records the checksums of what it used.

**Sources:** `README.md` (API table); `docs/API.md`; `frontend/` (`app.js`, `annotate.js`);
`vidquery/storage.py`; `vidquery/neo4j.py`; `scripts/run_with_profile.py`;
`evaluation/configs/model_profiles_2026-10-26.json`.

---

## Section 3 — Project demonstration

### Slide 7 — Demo overview (what the video will show)

**Bullets**
- Part A is real model output: Charades household videos indexed end to end with the integrated profile (deterministic Whisper, AVA v8, VidOR v4).
- Part B shows retrieval over labelled evidence: the controlled `final_demo.mp4` (manual annotations; flag F4).
- The demo covers upload, indexing states, search, localization, verdicts, abstention, ambiguity handling and provenance.
- It shows one honest failure (a ranking error) and one known limitation (an in-domain negative that isn't rejected).

**Visual:** a timeline bar of the demo video, Part A then Part B, with the chapter markers listed on Slide 8.

**Speaker notes**
We separate real model output from labelled demo data so the audience knows
which results come from our models and which come from annotations.

### Slide 8 — Demo script, example queries and expected outputs

Sequence (target 6–8 min):

| Step | Action | Expected output | Capture |
|---:|---|---|---|
| 1 | Open `http://127.0.0.1:8000`; open `/api/health` | All models listed with `available_*` status | screenshot of `/api/health` |
| 2 | Upload a short Charades clip not yet in the demo database (about 30 s) | States UPLOADED → … → READY in about 10–35 s (measured indexing 8.5–16.7 s per video on average) | screen recording |
| 3 | `GET /api/videos/{id}/status` | `processing_metadata` shows `whisper_decoding: greedy_temperature_0`, `gnn_model_version: ava80-gat-v8-noleak-scale100-lr0.0003`, relation model `…gat-v4-scale2000`, and checkpoint SHA-256s | screenshot |
| 4 | Query **"person closes the door."** (video `GBD1Y`) | Top segment 25–30 s; ground truth 26.2–31.3 s; verdict `weak`, confidence about 0.50 | recording of seek-to-peak |
| 5 | Query **"person sitting on bed."** (`V1WN7`) | Segment 0–5 s; ground truth 0–15.9 s; verdict `supported` (about 0.59) | screenshot |
| 6 | Honest failure: **"person sit on a chair."** (`FPJ9D`) | Returns 5–10 s; ground truth 25.8–33.5 s (a ranking error) | screenshot + narration |
| 7 | Out-of-domain negative: **"A Eurasian sparrowhawk zips through a dense forest, chasing a small bird."** (`FAO7J`) | Abstains, `insufficient` (confidence about 0.01) | screenshot |
| 8 | In-domain negative: **"person runs up the stairs."** (`NO1GJ`) | Returns a `weak` result (about 0.27) and is not rejected, the known limitation | screenshot |
| 9 | Ambiguity: **"find drive"** | Hypothesis chips for the visible action and the spoken mention; clicking a chip pins it *(behaviour documented, not yet rehearsed on this index; flag F7)* | recording |
| 10 | Part B: `final_demo.mp4`. Queries **"Find where SPEAKER_01 discusses architecture."** → 00:15–00:20; **"Find a person near a laptop."** → 00:00–00:05 (exact tuple chip) | Labelled evidence; say so on screen | recording |
| 11 | Optional: `/docs`, `POST /api/search` with `retrieval_backend: "neo4j"` | Graph retrieval (needs Neo4j running) | screenshot |

**Launch commands for Part A** (to be rehearsed). Copy the snapshot first so
the demo never writes into the frozen evaluation index:

```bash
cp data/app/benchmark/charades_integrated_2026-10-26/det-ava8-vidor4/index-snapshot.sqlite3 data/app/benchmark/demo-2026-10-26.sqlite3
python -m scripts.run_with_profile --profile det-ava8-vidor4 \
  --set DATABASE_PATH=data/app/benchmark/demo-2026-10-26.sqlite3 \
  --set GENERATED_DIR=data/app/benchmark/charades_integrated_2026-10-26/det-ava8-vidor4/generated \
  --set UPLOAD_DIR=data/app/benchmark/demo-2026-10-26-uploads \
  --set SOFT_CONSTRAINT_MATCHING=true --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=true \
  -- python -m vidquery serve --host 127.0.0.1 --port 8000
```

**Speaker notes:** narrate each step's expected output before clicking, and say
plainly when a result is wrong or when the data is labelled. Mention that the
UI returns the 5 s segment and peak moment, while the benchmark scores 10 s
windows (flag F3).

**Sources:** `evaluation/results/charades-integrated-2026-10-26/det-ava8-vidor4/test-T1-selected-soft-frame.json`
(the per-query rows for steps 4–8); `docs/FINAL_DEMO_GUIDE.md` (Part B queries);
`docs/REVIEW2.md` (hypotheses).

### Slide 9 — Recommended screenshots and recordings (checklist)

- Health JSON with model readiness, and the status JSON with model versions and SHA-256s.
- The upload progress states, as a short GIF.
- One correct moment: the player seeking to the peak, with the verdict chip visible.
- One failure, and one abstention.
- The hypothesis chips for "find drive".
- A Neo4j Browser view of one segment's subgraph (optional).
- A terminal still of `pytest` showing **410 passed, 1 skipped**.

---

## Section 4 — Test plan and strategy (2 slides)

### Slide 10 — Software testing: unit, integration and reproducibility

**Bullets**
- **Automated tests:** **40 test modules**; latest run **410 passed, 1 skipped** (the live Neo4j test needs a running service). ruff is clean. mypy has 13 pre-existing errors (flag F6).
- **Unit tests:**
  - parser and lexicon (130 lexicon tests);
  - reliability and hypotheses, localization, OCR, tracking, geometry;
  - phonetic matching;
  - Whisper decoding options;
  - AVA and VidOR checkpoint schema checks;
  - weighted bootstrap metrics (checked exactly against repeated rows).
- **Integration tests:**
  - the media pipeline end to end through the API;
  - search and storage, Neo4j integration;
  - shard round-trips, the frozen-test VidOR invariant, and the profile launcher.
- **Reproducibility checks, measured:** deterministic Whisper (40/40 transcripts); two pipeline rebuilds identical (131/131 segments); the new comparison script reproduces earlier paired results exactly.
- **Run-level safeguards:** versioned outputs that refuse to overwrite; recorded hashes of manifests, checkpoints and shards; resumable long jobs with retry; logged runtime and RAM/disk use.

**Visual:** a test pyramid (unit / integration / end-to-end benchmark) with counts, plus a "reproducibility" side bar.

**Speaker notes**
Beyond normal unit tests, many of our tests protect evaluation integrity:
they check that a model can't be trained on test videos, that held-out data
isn't silently changed, and that our statistics code matches the reference
metric exactly.

**Sources:** `tests/` (40 modules); `tests/test_action_lexicon.py`;
`tests/test_ava_scaling_comparison.py`; `tests/test_vidor_scaling.py`;
`tests/test_whisper_decoding.py`; `evaluation/results/whisper-determinism-2026-10-26/`.

### Slide 11 — Evaluation strategy: datasets, splits, leakage prevention, metrics

**Bullets**
- **Datasets and frozen splits** (all video-disjoint):

  | Task | Data | Train / val / test |
  |---|---|---|
  | AVA action classification | AVA v2.2 | 94 / 3 / **6** videos (3 + 3 untouched evaluation-only); 10,459 test persons |
  | VidOR relation classification | VidOR | 1,900 / 50 / **50** videos; 8,725 test candidate edges |
  | Moment retrieval + answerability | Charades-STA + NA-VMR | no training; validation 150 videos (STA train); test **300** videos (STA test): 863 positives, 863 in-domain + 855 out-of-domain negatives |
  | AVA end-to-end action queries | 3 untouched AVA validation videos | 56 queries (41 positive / 15 negative) |

- **Leakage prevention:**
  - an evaluation-only registry that trainers refuse;
  - leakage-free AVA features;
  - test-only shards in a separate folder;
  - validation-only model and threshold choices, written to a decision file **before** test is scored;
  - query manifests pre-registered and committed before indexing;
  - VidOR test frames and graphs verified byte-identical.
- **Metrics:**
  - classification: supported-class macro F1, micro F1, mAP;
  - retrieval: R@1 at tIoU 0.3/0.5/0.7, R@5@0.5, mIoU, and **P@1 (answered and tIoU ≥ 0.5)**;
  - answerability: rejection rate by negative type, rejection F1, AUROC, AURC.
- **Statistics:** paired bootstrap 95% CIs (resampling videos; also keyframes within video for AVA). A gain is claimed only if the CI excludes 0.

**Visual:** the split table, with a padlock icon on the test columns, plus a flow `validation → decision file → test (once)`.

**Speaker notes**
The most important design rule is that test data never influences a
decision. We pick models and thresholds on validation, write those choices
down, and only then score the test set once. Each task uses its own dataset
and metrics, so classification scores and retrieval scores are never mixed.

**Sources:** `evaluation/ava80_scale100_split.json`; `evaluation/eval_only_videos.json`;
`evaluation/charades_sta_navmr_benchmark.json`; `docs/EVALUATION_CHARADES_STA.md`;
`docs/EVALUATION_END_TO_END.md`; `docs/EXPERIMENTS_2026-10-26.md`;
`*/validation-decision.json`.

---

## Section 5 — Results and discussion (2 slides)

### Slide 12 — Model results: classification (not retrieval)

**Bullets / table A — AVA action classification** (6 test videos, 10,459 persons, 70 classes present)

| Model | Train videos | Macro F1 | mAP | Micro F1 |
|---|---:|---:|---:|---:|
| Existing leakage-free baseline (GAT) | 24 | 0.073 | 0.071 | 0.507 |
| GAT, same trainer | 24 | 0.072 | 0.074 | 0.488 |
| **GAT v8 (validation choice)** | 94 | **0.088** | **0.089** | 0.500 |
| MLP | 94 | 0.086 | 0.088 | 0.485 |

- **94 vs 24 videos (GAT):** macro F1 **+0.016 [0.012, 0.021]**, and still significant when resampling whole videos [0.007, 0.027]. mAP +0.015.
- **Against the existing baseline:** macro F1 +0.015 [0.011, 0.020].
- **GAT vs MLP:** not significant.

**Table B — VidOR relation classification** (same 50 test videos)

| Model | Train videos | Macro F1 | mAP | Micro F1 |
|---|---:|---:|---:|---:|
| GAT v2 | 57 | 0.071 | 0.088 | 0.316 |
| GAT v3-scale500 (active locally) | 400 | 0.171 | 0.187 | 0.328 |
| **GAT v4-scale2000** | 1,900 | **0.218** | **0.242** | **0.355** |

- **v4 vs v3:** macro F1 **+0.048 [0.015, 0.073]**, mAP +0.056 [0.021, 0.085], micro F1 +0.027 [0.006, 0.047].
- **Warm start vs scratch:** not significant.

**Visual:** a scaling line chart of test macro F1 against training videos.
- VidOR: 57 → 99 → 400 → 1,900 videos = 0.071 → 0.095 → 0.171 → 0.218.
- AVA: 24 → 94 videos = 0.072 → 0.088.

Use error bars only where CIs exist (the last step of each).

**Speaker notes**
These are per-label classification scores, not search results. More training
data helped both models significantly, and the improvement held up under
paired confidence intervals. Absolute AVA quality is still low (macro F1 of
about 0.09), so the system treats action evidence as low-reliability. Graph
message passing didn't beat an MLP with the same inputs.

**Sources:** `evaluation/results/ava80-scale100-2026-10-26/test-comparison.json`;
`evaluation/results/vidor-scale2000-2026-10-26/test-comparison.json`;
`evaluation/results/vidor-scale500-comparison.json`; `docs/EXPERIMENTS_2026-10-26.md`.

### Slide 13 — End-to-end results: retrieval (Charades-STA) and answerability (NA-VMR)

**Table C — Charades-STA moment retrieval** (test: 300 videos, 863 queries; Review-2, 10 s evaluation spans)

| System | P@1 [95% CI] | R@1@0.5 | R@1@0.7 | R@5@0.5 |
|---|---:|---:|---:|---:|
| Random 5 s segment (ignores the video) | — | 0.116 | 0.054 | — |
| Fixed window chosen on validation (ignores the video) | — | 0.269 | 0.092 | — |
| Default settings (code defaults) | 0.094 | 0.099 | 0.038 | 0.284 |
| Selected settings, original index | 0.414 [0.381, 0.445] | 0.416 | 0.187 | 0.870 |
| Selected settings, deterministic index (AVA v5, VidOR v2) | 0.403 [0.371, 0.436] | 0.407 | 0.185 | 0.869 |
| **Selected settings, integrated (AVA v8, VidOR v4)** | **0.404 [0.371, 0.437]** | 0.408 | 0.181 | 0.866 |

- **Selected vs default settings:** **+0.318 [0.275, 0.362]**.
- **Selected vs fixed window:** +0.147 [0.094, 0.201].

**Table E — What each component changed** (same test set; full index rebuild per
row; paired video bootstrap, 2,000 resamples; `*` = 95% CI excludes 0)

| Component | **Retrieval:** Charades P@1, selected settings | **Strict-mode rejection** of in-domain negatives | Strict-mode P@1 |
|---|---:|---:|---:|
| AVA v8 vs v5 (with VidOR v2) | +0.007 [−0.021, 0.033] | **+0.045 [0.020, 0.072]\*** (55.3% → 59.8%) | −0.009 [−0.034, 0.016] |
| AVA v8 vs v5 (with VidOR v4) | +0.001 [−0.026, 0.027] | **+0.046 [0.022, 0.073]\*** | −0.002 [−0.028, 0.023] |
| VidOR v4 vs v2 (with AVA v5) | 0.000 [−0.012, 0.011] | +0.004 [−0.002, 0.010] | −0.007 [−0.020, 0.005] |
| VidOR v4 vs v2 (with AVA v8) | −0.006 [−0.017, 0.005] | +0.005 [−0.001, 0.012] | 0.000 [−0.011, 0.010] |
| Deterministic vs fallback Whisper (same models) | +0.005 [−0.015, 0.025] | 0.000 [−0.009, 0.008] | −0.004 [−0.017, 0.013] |
| **Integrated (v8 + v4 + deterministic) vs the 41.4% run** | **−0.009 [−0.036, 0.017]** (0.404 vs 0.414) | **+0.051 [0.026, 0.077]\*** | −0.017 [−0.048, 0.012] |

Read the three measurements separately:

- **Retrieval (Charades-STA P@1): no component improves it.** Every
  P@1 and R@1 difference is not significant. 0.404 vs 0.414 is −0.009
  [−0.036, 0.017], smaller than the difference between two rebuilds of the
  old index (−0.015). **Do not claim a retrieval improvement from the new models.**
- **Strict-mode rejection (NA-VMR in-domain negatives): AVA v8 helps.**
  - This is measured with code-default thresholds and no soft constraints.
  - AVA v8 rejects 4.5 points more plausible-but-absent queries (significant), with no significant P@1 cost.
  - The share of positive queries answered is essentially unchanged (51.2% vs 51.1%), so it discriminates better rather than rejecting more of everything.
  - Strict-mode P@1 is only 0.13–0.15. In the selected settings of Table C, in-domain rejection stays at 0.8% for every index, because a missing action no longer vetoes a segment.
- **Classification is a different measurement (Slide 12).** AVA +0.016 and VidOR +0.048 macro F1 are gains on the models' own classification test sets. They do not carry over to Charades retrieval; only 12 of 2,581 queries are even parsed as relationship searches.
- **Reproducibility (deterministic Whisper): repeatable, not more accurate.**
  - With the old temperature fallback, about 43% of transcripts changed between two rebuilds.
  - With greedy decoding, two rebuilds of 20 test videos were identical in all 131 segments.
  - At full scale, 7 transcripts and 5 detection sets still differ out of 1,935 test segments (≤ 0.4%), likely GPU floating-point nondeterminism.
- Validation (150 videos) showed one small significant drop, R@5@0.5 −0.026 [−0.050, −0.003]. It does not appear on test (−0.004, n.s.).

**Table D — NA-VMR answerability** (1,718 negatives, selected settings)

| | Out-of-domain (855) | In-domain (863) |
|---|---:|---:|
| Rejected (integrated, default thresholds) | 99.3% (100% with the validation-fitted P@1-preserving threshold) | 0.8% |
| AUROC, positives vs negatives (original index) | 0.9995 | 0.60 |

- Integrated rejection F1 0.663; overall AUROC 0.794.

**AVA end-to-end action queries** (56 queries): P@1 rose from 0.122 to **0.268 [0.15, 0.41]** with the action-lexicon parser.

**Speaker notes**
These are end-to-end search scores: the system indexes raw video by itself,
then answers human-written questions. The biggest gain came from retrieval
design, not from new models: soft constraints, frame-level image–text
similarity and wider spans more than quadrupled P@1 over the defaults, and
beat a strong position-only baseline. Better action and relation models
didn't change Charades retrieval: we rebuilt the whole index for each
combination, and no P@1 difference is significant. Their gains are in
classification, on the previous slide. The one end-to-end effect we can
measure is in the strict mode: AVA v8 rejects significantly more
plausible-but-absent questions without costing accuracy. Deterministic Whisper
didn't change accuracy either, but it made rebuilds repeatable, which is what
made these comparisons trustworthy. The system reliably rejects questions about
things that can't be in these indoor videos; in the demo settings it still
cannot tell a plausible question that didn't happen from one that did.

**Sources:** `evaluation/results/charades-2026-10-25/test/*.json`;
`…/test-v3relation/comparison-v2-v2rerun-v3.json`;
`evaluation/results/charades-integrated-2026-10-26/comparison-test-T1-selected-soft-frame.json`
(retrieval) and `comparison-test-T0-default-strict-noframe.json` (strict mode);
`…/det-*/test-T0-default-strict-noframe.json` (rejection rates);
`…/comparison-validation-T1.json`; `…/index-evidence/`;
`evaluation/results/whisper-determinism-2026-10-26/`;
`docs/INTEGRATED_EVALUATION_2026-10-26.md`; `docs/EVALUATION_CHARADES_STA.md`;
`docs/EVALUATION_END_TO_END.md`.

> Optional small footnote, if you show older figures: the Review-1 benchmark
> (P@1 0.96 on 62 queries) used *labelled* evidence and is **not**
> end-to-end (`docs/EVALUATION_FINAL.md`). Keep it off the results slides, or
> label it clearly.

**Limitations (put on Slide 13, or move to Slide 14):**
- in-domain negatives are not detected;
- the ranking bottleneck: single-frame CLIP can't see verbs or temporal order;
- the Charades numbers use experimental flags and evaluation-only 10 s spans;
- AVA has only 6 test videos;
- AVA absolute quality is low;
- YOLOv8n matches only 27.5% of VidOR relation instances;
- validation overestimates test by 2× for AVA and by 0.05–0.07 macro F1 for VidOR;
- there is no lecture benchmark.

---

## Section 6 — Conclusion and future work

### Slide 14 — Conclusion

**Bullets**
- Delivered a working evidence-backed video search system: ingestion, 7 extractors, 2 trained GNNs, canonical index, two-stage search, localization, verdicts and abstention, API and UI.
- Built **leakage-safe, reproducible evaluation** on three task types (action classification, relation classification, moment retrieval with answerability), each with paired CIs.
- **Measured gains:**
  - retrieval design gave **+0.318 P@1** on Charades-STA;
  - more training data gave **+0.016** AVA macro F1 and **+0.048** VidOR macro F1, both significant;
  - deterministic indexing removed rebuild noise.
- **Honest negatives:**
  - new models don't improve Charades retrieval;
  - in-domain negatives are not rejected;
  - absolute action-recognition quality is low.

**Speaker notes:** summarize what works, what was measured and what didn't
improve, and stress that every claim has a test-set CI.

### Slide 15 — Future work (realistic next steps)

**Bullets**
1. **Promotion review:** consider activating VidOR `gat-v4-scale2000` (a three-line `.env` change), and re-run the AVA end-to-end benchmark before activating AVA v8.
2. **Ranking:** add a clip-level video–language encoder or a moment-retrieval head to fix verb and temporal confusions, and to verify in-domain negatives.
3. **Data:** the remaining 196 AVA videos (about 102 GB); a streaming or sharded VidOR graph builder before going past 2,000 videos.
4. **Benchmarks:** record and label the lecture benchmark (speech, OCR and phonetic matching); enlarge the AVA test set.
5. **Detector:** YOLOv8m raises the relation recall ceiling from 27.5% to 38.3% (measured), but needs reprocessing and retraining.

**Visual:** a roadmap with three swim lanes (models / data / evaluation), each card tagged "needs approval" or "ready".

**Sources:** `docs/PROGRESS_2026-10-25.md` (approval gates, YOLOv8m recall);
`docs/EXPERIMENTS_2026-10-26.md` (next steps).

---

## Section 7 — References

### Slide 16 — References (datasets, models, methods)

**Datasets**
1. C. Gu, C. Sun, D. A. Ross, C. Vondrick, C. Pantofaru, Y. Li, S. Vijayanarasimhan, G. Toderici, S. Ricco, R. Sukthankar, C. Schmid, J. Malik. *AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions.* CVPR 2018. https://arxiv.org/abs/1705.08421
2. X. Shang, D. Di, J. Xiao, Y. Cao, X. Yang, T.-S. Chua. *Annotating Objects and Relations in User-Generated Videos.* ICMR 2019. Dataset page: https://paperswithcode.com/dataset/vidor (DOI not verified, flag; archives used from the Hugging Face mirror `shangxd/vidor`)
3. G. A. Sigurdsson, G. Varol, X. Wang, A. Farhadi, I. Laptev, A. Gupta. *Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding* (Charades). ECCV 2016. https://arxiv.org/abs/1604.01753
4. J. Gao, C. Sun, Z. Yang, R. Nevatia. *TALL: Temporal Activity Localization via Language Query* (Charades-STA). ICCV 2017. https://arxiv.org/abs/1705.02101
5. K. Flanagan, D. Damen, M. Wray. *Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval* (NA-VMR). WACV 2025, pp. 5336–5345. https://arxiv.org/abs/2502.08544 ; splits: https://github.com/keflanagan/MomentofUntruth

**Models and tools**
6. G. Jocher, A. Chaurasia, J. Qiu. *Ultralytics YOLOv8* (software, v8.0.0), 2023. https://docs.ultralytics.com/models/yolov8/ (Ultralytics publishes no paper)
7. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). ICML 2023. https://arxiv.org/abs/2212.04356
8. H. Bredin et al. *pyannote.audio: neural building blocks for speaker diarization.* ICASSP 2020. https://arxiv.org/abs/1911.01255
9. A. Plaquet, H. Bredin. *Powerset multi-class cross entropy loss for neural speaker diarization.* Interspeech 2023. https://www.isca-archive.org/interspeech_2023/plaquet23_interspeech.html ; pipeline card: https://huggingface.co/pyannote/speaker-diarization-3.1
10. JaidedAI. *EasyOCR* (software). https://github.com/JaidedAI/EasyOCR ; detector: Y. Baek, B. Lee, D. Han, S. Yun, H. Lee. *Character Region Awareness for Text Detection* (CRAFT). CVPR 2019. https://arxiv.org/abs/1904.01941
11. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021. https://arxiv.org/abs/2103.00020
12. M. Cherti et al. *Reproducible scaling laws for contrastive language-image learning* (OpenCLIP). CVPR 2023. https://arxiv.org/abs/2212.07143 ; code: https://github.com/mlfoundations/open_clip
13. N. Reimers, I. Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP 2019. https://arxiv.org/abs/1908.10084
14. W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS 2020. https://arxiv.org/abs/2002.10957 ; model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
15. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. https://arxiv.org/abs/1512.03385
16. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. *Graph Attention Networks.* ICLR 2018. https://arxiv.org/abs/1710.10903
17. S. Brody, U. Alon, E. Yahav. *How Attentive are Graph Attention Networks?* (GATv2). ICLR 2022. https://arxiv.org/abs/2105.14491
18. M. Fey, J. E. Lenssen. *Fast Graph Representation Learning with PyTorch Geometric.* ICLR 2019 Workshop (RLGM). https://arxiv.org/abs/1903.02428
19. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. https://arxiv.org/abs/1912.01703
20. OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card.* 2025. https://arxiv.org/abs/2508.10925 (used via the Groq API: https://console.groq.com/docs)

**Methods (evaluation)**
21. Y. Geifman, R. El-Yaniv. *Selective Classification for Deep Neural Networks* (risk–coverage, AURC). NeurIPS 2017. https://arxiv.org/abs/1705.08500
22. B. Efron. *Bootstrap Methods: Another Look at the Jackknife.* Annals of Statistics 7(1):1–26, 1979. https://doi.org/10.1214/aos/1176344552

**Systems documentation:** FastAPI https://fastapi.tiangolo.com ; SQLite https://www.sqlite.org ;
Neo4j https://neo4j.com/docs/ ; FFmpeg https://ffmpeg.org

Verification status:
- **Checked by web search this session:** 5, 6, 9, 12, 20, and the authors and venue of 2.
- **From standard citations, not re-checked this session:** the rest. Spot-check the links before printing.
