# VidQuery implementation audit

Baseline date: 2026-08-08

Baseline commit: `6f7d03c` on `dev_aryaman`

Audit rule: repository behavior and reproducible execution take precedence over report and slide claims.

## Executive finding

VidQuery contains useful AVA preprocessing artifacts and several workable pipeline utilities, but the baseline is not an end-to-end semantic video search product. The checked-out artifact set demonstrates that YOLO, Whisper, Pyannote, heuristic scene-graph construction, timestamp fusion, tensor conversion, and some Neo4j ingestion were run previously. It does not demonstrate a trained relationship-prediction GNN, safe semantic querying, ranked retrieval, real upload handling, persistent job state, a playable result UI, evaluation, or automated tests.

The critical honesty statement is:

> GNN-compatible feature construction exists, but a trained and validated relationship-prediction GNN has not been verified.

The shortest defensible route is to preserve the AVA utilities, introduce a canonical timestamped segment contract, build a persistent local vertical slice that can optionally mirror data into Neo4j, replace unrestricted generated Cypher with a safe structured query plan, and expose that behavior through a typed API and real browser UI. GNN training remains a separate experimental track until labels, training, checkpoints, and evaluation exist.

## Repository structure at audit time

```text
VidQuery/
|-- api/                    # FastAPI prototype: health, directory pipeline, GET search
|-- app/                    # Minimal Streamlit prototype
|-- core/                   # Environment-backed dataclass settings
|-- database/               # Neo4j client, schema, and per-frame ingestion
|-- extraction/             # AVA preprocessing, YOLO, Whisper, Pyannote
|-- modelling/              # Alignment, heuristics, tensor export, untrained MPNN
|-- retrieval/              # Groq prompt and unrestricted Cypher execution
|-- data/ava/               # Large ignored/generated AVA dataset and artifacts
|-- docs/                   # Two project PDFs; required third PDF/image absent
|-- main.py                 # Legacy command dispatcher
|-- download.go             # AVA S3 range downloader
|-- requirements.txt        # Unpinned, monolithic dependency list
`-- wrapper scripts         # extract/detect/visualize/audio entry points
```

Generated-data inventory:

| Area | Files | Approximate size | Observation |
|---|---:|---:|---|
| AVA annotations | 2 | 52.7 MB | Train/validation CSVs are present locally and ignored by Git. |
| Videos | 6 | 1.45 GB | Five full MP4s plus the ignored AVA list file. |
| Extracted audio | 6 | 672 MB | Five full WAVs and one 5-second WAV. |
| Frames | 3,508 | 99.5 MB | Train split only. |
| Detection JSON | 3,508 | 5.7 MB | Train split only. |
| Scene-graph JSON | 3,508 | 8.0 MB | Train split only. |
| Audio outputs | 40 | 5.1 MB | Whisper, diarization, and merged files for five videos. |
| Fused JSON | 5 | 10.4 MB | Frame-level visual/audio fusion. |
| GNN-output JSON | 3,513 | 10.5 MB | Per-frame random-weight edge embeddings plus summaries. |

Only 53 files are tracked by Git. The current worktree already had two deleted planning files and an untracked `docs/` directory before this audit; those pre-existing changes were preserved.

## Source material reviewed

- All 32 Python files, the Go downloader, Cypher schema, environment example, ignore rules, dependency list, AVA README, and package initializers.
- Representative records from every generated format: AVA CSV, detection JSON, scene JSON, Whisper JSON/text, Pyannote JSON/RTTM/CSV/SRT, merged JSON, fused JSON, and GNN JSON.
- `docs/Team137_CapstoneReport.pdf` (56 pages), including its architecture, class, ER, UI, and implementation sections.
- `docs/ESAreview.pdf` (28 pages), including its end-to-end architecture and implementation evidence figures.
- The named `Capstone Phase Two PPT Elucidation.pdf` and a standalone architecture image are not present. The architecture figure embedded in `ESAreview.pdf` was inspected instead.

## Entry points and current execution flow

| Entry point | Behavior |
|---|---|
| `python main.py pipeline` | AVA frame extraction, all cached-frame YOLO traversal, scene-graph traversal, optional audio/fusion, optional tensor export. |
| `python main.py audio` | Extract WAV, transcribe or reuse cached Whisper JSON, optionally rerun diarization, merge, then fuse with scene graphs. |
| `python main.py gnn` | Convert scene nodes to tensors, run a newly randomized MPNN, and serialize edge vectors. |
| `python main.py ingest` | Apply Neo4j schema and ingest fused or frame graph JSON. |
| `python main.py db-check` | Validate configuration and attempt a Neo4j health query. |
| `python main.py api` | Run the prototype FastAPI app. |
| Wrapper scripts | Delegate directly to extraction/audio/visualization module `main` functions. |
| `go run download.go` | Read `ava_train_v2.2.txt` from the current directory and download files into that directory. |

Current data flow:

```text
AVA CSV + full MP4
  -> annotation timestamps
  -> JPEG frames
  -> YOLO detections + optional AVA IoU matches
  -> heuristic/annotation-derived per-frame scene graph

MP4
  -> 16 kHz mono WAV
  -> Whisper segments
  -> Pyannote turns
  -> greatest-overlap transcript/speaker merge

scene frames + merged audio
  -> per-frame fused JSON
  -> sparse tensor construction
  -> randomly initialized MPNN edge vectors
  -> per-frame GNN JSON

fused/frame JSON
  -> Neo4j MERGE operations
  -> unrestricted LLM-generated Cypher or all-frames fallback
```

## Existing data formats

1. AVA annotation row: `video_id,timestamp,x1,y1,x2,y2,action_id,entity_id`; boxes are normalized.
2. Detection record: `video_id`, `frame_id`, integer `timestamp`, `frame_file`, denormalized lists, and `detections[]` containing class, confidence, pixel box, normalized box, center, and optional matched AVA annotation.
3. Scene record: `video_id`, `timestamp`, `frame_file`, `nodes[]`, and `edges[]`. Edge provenance is absent.
4. Whisper record: model-native `text`, `language`, and `segments[]` with start/end and diagnostic fields.
5. Diarization record: source path and `segments[]` with speaker/start/end/duration, plus RTTM/CSV/SRT renderings.
6. Merged audio record: `video_id` and transcript segments with the maximum-overlap speaker.
7. Fused record: video/split counts and `frames[]`; each frame receives overlapping audio segments for `[timestamp, timestamp + 1)`.
8. GNN record: node feature tensor, sparse edge index, and edge feature tensor. No class labels or prediction confidence are present.

There is no canonical segment format shared by ingestion, retrieval, API, evaluation, and UI.

## Existing Neo4j model

Constraints exist for `Video.video_id`, composite `Frame(video_id,timestamp)`, `Scene.scene_key`, `Object.object_key`, `Person.person_key`, `Concept.name`, and `AudioSegment.segment_key`. Indexes exist for timestamps, object class, and concept name.

Ingestion uses `MERGE` for most nodes and relationships, so repeated input is intended to be idempotent. It was not integration-tested against a live database. It also has important defects:

- `Scene` is created but visual/audio evidence is attached to `Frame`, not consistently anchored through `Scene` as the documents claim.
- Speakers are represented as `Person` nodes.
- Edge provenance, relationship confidence, and source method are discarded.
- Relationship types are built into query strings after regex sanitization; values are parameterized but the graph vocabulary is not centrally allowlisted.
- Transcript segments may be duplicated across adjacent one-second frames by design.
- The schema has no canonical `Segment`, embedding/vector index, job/video state, or evaluation representation.

## GNN assessment

`SceneGraphMPNN` has trainable layers, but `gnn_exporter.py` creates a fresh model per video and calls it in evaluation mode without loading a checkpoint. The repository has no target construction, label vocabulary, train/validation/test split for relationships, loss, optimizer, training loop, checkpoint save/load, validation metrics, predicted labels, predicted probabilities, or baseline comparison. The exporter serializes latent edge vectors from random weights.

Decision: keep tensor construction as an experimental feature builder; relabel its outputs honestly; do not expose them as predictions. A future trained model must be implemented and evaluated separately.

## Security and configuration findings

- `.env` is correctly ignored and not tracked. It contains local credentials; values were not copied into documentation or output.
- `.env.example` is too small for the requested application and uses legacy names.
- The retrieval prototype executes LLM output as raw Cypher. This is the highest-risk code path and must be replaced.
- The upload endpoint accepts server-side directory paths rather than an uploaded file and has no MIME, extension, filename, size, or media validation.
- CORS, request limits, streaming ranges, safe thumbnail generation, and API error models are absent.
- Ultralytics is imported eagerly and writes to a user profile during import, making unrelated commands fragile.
- Heavy models may download during load; startup boundaries do not prevent this.
- The Go downloader does not validate HTTP status/range behavior or output location and can retain partial/corrupt files.

## Dependency and runtime findings

- `requirements.txt` is unpinned and combines API, UI, visual, audio, GNN, and database stacks into one fragile environment.
- The checked-in/local virtual environment launcher points to a removed Microsoft Store Python executable. Its site-packages can be inspected only by manually adding them to `PYTHONPATH`.
- The system Python can run OpenCV, Torch, FastAPI, and Streamlit, but several project packages are available only in the broken environment's site-packages.
- No `pyproject.toml`, lock file, formatter/linter/type-check configuration, Python-version declaration, or Docker setup exists.
- CPU mode is not configurable in the detector. No explicit warning is emitted for CPU inference.
- The frame-extraction probe passes the FFmpeg executable to an API that expects FFprobe, then continues after validation failure.

## Feature audit

Status values describe the baseline before the remediation work in this repository.

| Feature | Claimed in project document | Repository file or files | Status | Runnable | Tested | Current input | Current output | Problems | Decision | Required action |
|---|---|---|---|---|---|---|---|---|---|---|
| AVA annotation parsing | Yes | `extraction/preprocessor.py`, `extraction/visual.py` | PARTIALLY_WORKING | Yes | No | AVA CSV | Timestamp/annotation maps | Duplicate parsers and incomplete action maps | REFACTOR | One tested parser and taxonomy |
| Video-list generation | Yes | None; generated `data/ava/videos/ava_train_v2.2.txt` only | MISSING | No | No | AVA CSV | Text list | Generator source absent | REPLACE | Add deterministic utility |
| Video downloading | Yes | `download.go` | PARTIALLY_WORKING | Conditional | No | Text list, S3 | MP4 files | CWD assumptions, status/range/partial-file issues | REPAIR | Validate responses and destination |
| Video validation | Yes | `extraction/preprocessor.py` | BROKEN | No | No | MP4 | Console warning | FFmpeg binary passed as FFprobe; failure ignored | REPLACE | Media probe with explicit failure |
| Frame extraction | Yes | `extraction/preprocessor.py` | PARTIALLY_WORKING | Yes | No | MP4 + AVA times or dense video | JPEG paths | Missing canonical frame IDs/metadata; cache not validated | REFACTOR | Reusable extractor and contract |
| Audio extraction | Yes | `extraction/preprocessor.py` | PARTIALLY_WORKING | Conditional | No | MP4 | Mono 16 kHz WAV | Missing no-audio handling and metadata | REPAIR | Graceful typed extractor |
| YOLOv8 detection | Yes | `extraction/visual.py` | WORKING | Conditional | No | JPEG | Detection JSON | Eager import/model load and possible download | REFACTOR | Lazy replaceable interface |
| Bounding-box normalization | Yes | `extraction/visual.py` | WORKING | Yes | No | Pixel box, image size | Normalized box | No shared validation/clamping | REPAIR | Shared tested utility |
| AVA bounding-box matching | Yes | `extraction/visual.py` | PARTIALLY_WORKING | Yes | No | Detections + annotations | Attached match | Reuses only first box per entity; no one-to-one matching | REFACTOR | Tested candidate matching |
| IoU matching | Yes | Two implementations | PARTIALLY_WORKING | Yes | No | Two boxes | IoU | Duplicate logic and invalid-box behavior undefined | REFACTOR | One tested implementation |
| AVA action-label attachment | Yes | `extraction/preprocessor.py`, `extraction/visual.py` | BROKEN | Yes | No | Action IDs | Labels | Conflicting partial maps yield `action_79` etc. | REPLACE | Canonical 80-class taxonomy |
| Spatial-relation construction | Yes | `modelling/scene_graph_builder.py` | PARTIALLY_WORKING | Yes | No | Pixel boxes | Heuristic edges | Resolution-dependent thresholds; no provenance/confidence | REFACTOR | Normalized rules and source metadata |
| Scene-graph serialization | Yes | `modelling/scene_graph_builder.py` | PARTIALLY_WORKING | Yes | No | Detection JSON | Scene JSON | Non-stable integer IDs and no source method | REFACTOR | Canonical domain models |
| Scene-graph visualization | Yes | `modelling/*visualizer.py` | WORKING | Yes | No | Frame + scene JSON | Image/video overlays | Hardcoded paths/settings | KEEP | Parameterize paths |
| Whisper transcription | Yes | `extraction/audio.py` | PARTIALLY_WORKING | Conditional | No | Audio/media | Whisper JSON/text | Downloads may occur; no chunk/global-time policy | REFACTOR | Optional lazy transcriber |
| Pyannote diarization | Yes | `extraction/audio.py` | PARTIALLY_WORKING | Conditional | No | WAV + HF token | JSON/RTTM/CSV/SRT/PNG | Token implicitly forces rerun; large in-memory audio | REFACTOR | Explicit opt-in and graceful fallback |
| Transcript-diarization alignment | Yes | `extraction/audio.py` | PARTIALLY_WORKING | Yes | No | Two segment lists | Speaker-tagged transcript | Tie/zero-duration semantics and `unknown` casing | REPAIR | Shared interval overlap and `UNKNOWN` |
| Audio-video alignment | Yes | `modelling/alignment.py` | PARTIALLY_WORKING | Yes | No | Per-frame graphs + transcript | Audio on one-second frames | Fixed implicit windows and duplicated evidence | REPLACE | Canonical configurable segments |
| Fused multimodal JSON generation | Yes | `main.py`, `modelling/alignment.py` | PARTIALLY_WORKING | Yes from cache | No | Frames + audio | Fused JSON | Not canonical and hardcoded locations | REPLACE | Canonical segment document |
| GNN-compatible tensor construction | Yes | `modelling/gnn_dataset.py` | PARTIALLY_WORKING | Yes | No | Scene nodes/audio | PyG/dict tensors | Raw pixel scale, sparse candidates only by distance | REFACTOR | Normalize and test features |
| GNN model architecture | Yes | `modelling/gnn_model.py` | PARTIALLY_WORKING | Conditional | No | Node/edge tensors | Edge embeddings | No label head or training contract | REPLACE | Experimental labelled predictor |
| GNN training | Yes | None | MISSING | No | No | None | None | Entire training path absent | REPLACE | Dataset, targets, loss, optimizer, loop |
| GNN validation | Yes | None | MISSING | No | No | None | None | No metrics or comparison | REPLACE | Held-out metrics and baseline |
| GNN inference | Yes | `modelling/gnn_exporter.py` | BROKEN | Yes | No | Scene graphs | Random edge vectors | Fresh random weights; no labels/confidence | REPLACE | Checkpoint-backed labelled inference only |
| GNN checkpoint loading | Yes | None | MISSING | No | No | None | None | No checkpoint integration | REPLACE | Strict compatible loader |
| GNN relationship prediction | Yes | None | MISSING | No | No | None | None | Heuristics/AVA labels are not predictions | REPLACE | Train and evaluate before enabling |
| GNN output serialization | Yes | `modelling/gnn_exporter.py` | BROKEN | Yes | No | Random model output | `.gnn.json` | Misleading name/semantics | REPAIR | Mark as untrained feature export |
| Neo4j constraints | Yes | `database/schema.cypher` | PARTIALLY_WORKING | Unverified | No | Cypher | Constraints | Baseline schema differs from claimed model | REPLACE | Canonical segment schema |
| Neo4j indexes | Yes | `database/schema.cypher` | PARTIALLY_WORKING | Unverified | No | Cypher | Indexes | No full-text/vector search indexes | REFACTOR | Add justified indexes |
| Neo4j ingestion | Yes | `database/ingestion.py` | PARTIALLY_WORKING | Unverified | No | Fused/frame JSON | Graph mutations | Live database unavailable; contract mismatch | REFACTOR | Canonical batch ingestion |
| Neo4j ingestion idempotency | Yes | `database/ingestion.py` | UNVERIFIED | Unverified | No | Same JSON repeatedly | MERGE operations | No integration proof; relationship update semantics weak | REPAIR | Container-backed repeat-ingest test |
| Manual Cypher queries | Yes | Slide screenshot only | UNVERIFIED | No | No | Handwritten query | Browser graph | Query file/test absent | REPLACE | Version safe examples/tests |
| Natural-language query parsing | Yes | `retrieval/query_engine.py` | BROKEN | Conditional | No | User string | Raw Cypher | Not a structured parser; fallback ignores query | REPLACE | Allowlisted deterministic plan |
| Safe Cypher compilation | Yes | None | MISSING | No | No | None | None | Raw LLM output is executed | REPLACE | Static templates + parameters |
| Semantic vector retrieval | Yes | None | MISSING | No | No | None | None | No embeddings or index | REPLACE | Optional embedder and lexical baseline |
| Graph traversal retrieval | Yes | `retrieval/query_engine.py` | PARTIALLY_WORKING | Conditional | No | Raw Cypher | Neo4j rows | Unsafe and not ranked | REPLACE | Safe canonical traversal |
| Hybrid ranking | Yes | None | MISSING | No | No | None | None | No scoring | REPLACE | Transparent modality-weighted ranker |
| FastAPI backend | Yes | `api/routes.py` | PLACEHOLDER | Conditional | No | JSON directory paths/query | Minimal JSON | Wrong routes/contracts and eager heavy imports | REPLACE | Typed `/api` application |
| Video streaming | Yes | None | MISSING | No | No | None | None | No range support | REPLACE | Safe HTTP range endpoint |
| Frontend | Yes | `app/ui.py` | PLACEHOLDER | Conditional | No | Query/server path | Text rows | No upload, cards, playback, real endpoint contract | REPLACE | Real browser UI |
| Evaluation dataset | Yes | None | MISSING | No | No | None | None | No ground truth queries | REPLACE | Machine-readable dataset |
| Evaluation scripts | Yes | None | MISSING | No | No | None | None | No metrics or results | REPLACE | Reproducible evaluator |
| Automated tests | Yes | `test.py` GPU print only | MISSING | No | No | None | None | Pytest finds zero tests | REPLACE | Unit/integration/smoke suite |
| Docker setup | Yes | None | MISSING | No | No | None | None | No images or Compose | REPLACE | Neo4j + backend Compose |
| Reproducible setup | Yes | unpinned `requirements.txt` | BROKEN | No | No | Broken local venv | Manual workaround | Launcher invalid; stack unpinned | REPLACE | `pyproject`, optional groups, docs |

## Component decisions

### KEEP

- The ignored AVA source/artifact corpus as reproducibility evidence.
- The high-level split between extraction, modelling, database, and retrieval.
- Working visualization behavior and small wrapper entry points.
- The concept of lazy optional Whisper/Pyannote and a parameterized Neo4j driver.

### REPAIR / REFACTOR

- AVA utilities, detection contracts, timestamp overlap, box math, scene heuristics, Neo4j client, and local configuration.
- Existing generated formats should remain readable through adapters; new application code must not silently mutate them.

### REPLACE

- Raw LLM-to-Cypher execution.
- Random MPNN output presented as GNN inference.
- Server-side-directory “upload” API and static Streamlit pipeline buttons.
- Per-frame fused data as the application-wide contract.

### REMOVE from production paths

- `test.py` as a test substitute.
- Eager heavyweight imports at package import time.
- Any fallback that returns unrelated frames while labeling them search results.

## Prioritized implementation plan

1. Add canonical typed video, frame, transcript, detection, relationship, and segment models.
2. Add central typed settings, safe media probing, streaming upload storage, and persistent video/job registry.
3. Build configurable frame/audio processing with lazy optional model adapters and explicit degradation.
4. Build one shared interval-overlap and fixed-window segment builder with relationship provenance.
5. Add safe structured query parsing, allowlisted compilation, transparent local ranking, and optional Neo4j retrieval.
6. Replace the API contract and add range streaming, thumbnails, status/retry, and a real UI.
7. Add comprehensive lightweight tests plus heavy-boundary integration tests.
8. Add an evaluation dataset/evaluator without inventing measurements.
9. Add reproducible dependency groups, Docker Compose, developer instructions, and demo documentation.
10. Keep GNN support disabled as a prediction source until a trained checkpoint and held-out evaluation are supplied.
