# Baseline pipeline reproduction

Date: 2026-08-08

Platform: Windows, Python 3.12.3, Go 1.26.1

Hardware probe: NVIDIA GeForce RTX 4060 Laptop GPU available

Repository commit: `6f7d03c`

This record describes the repository before the remediation implementation. Existing generated artifacts were treated as evidence, not as proof that every producing command still works.

## Environment discovery

The local `.venv` launcher is broken because `pyvenv.cfg` references a removed Microsoft Store Python 3.12.10 executable. Its pure-Python and compatible binary packages can still be found by temporarily setting `PYTHONPATH` to `../.venv/Lib/site-packages`. This workaround was used only to reproduce the baseline; it is not an acceptable setup procedure.

Ultralytics also attempted to create settings under the roaming user profile during import. A project-local `YOLO_CONFIG_DIR` was required in the restricted environment. Matplotlib and Go were similarly given project-local cache directories.

## Commands and outcomes

| Stage | Command | Input | Output | Time | Compute mode | Records | Status | Warning/error/fix |
|---|---|---|---|---:|---|---:|---|---|
| Source compilation | `python -m compileall -q .` | Python source | Bytecode cache | 1.9 s | CPU | 32 source files | WORKING | No syntax errors |
| Pytest baseline | `python -m pytest -q` | Repository | Test report | 2.7 s | CPU | 0 tests | BROKEN | Exit 1: no tests ran |
| Unittest discovery | `python -m unittest discover -v` | Repository | Test report | 51.9 s | CPU/GPU import | 0 tests | BROKEN | API/extraction imports trigger Ultralytics profile write; `test.py` only prints GPU details |
| Go baseline | `go test ./...` | `download.go` | Build/test status | 1.1 s | CPU | 0 tests | BROKEN | Default Go cache was outside writable workspace |
| Go with local cache | `$env:GOCACHE=...; go test ./...` | `download.go` | Successful build | 11.4 s | CPU | 0 tests | WORKING | Reports `[no test files]` |
| CLI baseline | `python main.py --help` | Application imports | None | 61.6 s | Import only | N/A | BROKEN | Ultralytics profile write failed |
| CLI with local model cache | `YOLO_CONFIG_DIR=... python main.py --help` | Application imports | Help text | 8.8 s | Import only | 6 subcommands | WORKING | Heavy model stack still imported eagerly |
| Cached frame/detection/graph path | `python main.py pipeline --max_videos 1 --skip_audio --skip_gnn_export` with local caches | Five local MP4s and cached artifacts | Existing frames/detections/graphs | 43.2-44.7 s | Cached; no detector inference verified | 3,508 frames; 6,687 detections | PARTIALLY_WORKING | `max_videos=1` only limits preprocessing; later stages traverse all five videos. FFprobe fails and is ignored. |
| Cached GNN export | `python main.py gnn --max_videos 1` | `data/ava/fused/train` | Existing `.gnn.json` records | 6.0 s | Cached | 562 frames, 447 with nodes, 97 with candidate edges, 370 with audio | PARTIALLY_WORKING | Records are feature/latent exports, not relationship predictions |
| Neo4j connectivity | `python main.py db-check` | Local `.env` connection settings | None | 7.9 s | CPU/network | 0 | BROKEN | Port 7687 closed; routing information unavailable |
| Port confirmation | `Test-NetConnection 127.0.0.1 -Port 7687` | Local host | Boolean | 12.7 s | Network | 1 probe | BROKEN | `TcpTestSucceeded=False` |

## Stage-by-stage evidence

### Video input

Five ignored MP4 files are present. OpenCV reports readable video streams with durations from approximately 2,604 to 5,792 seconds and resolutions from 450x360 to 704x480. The files are full source videos rather than a small committed demo fixture.

Status: `WORKING` as local evidence; upload validation is absent.

### AVA annotations

Both train and validation v2.2 CSV files are present. Rows contain video ID, an absolute source-video timestamp, normalized person box, action ID, and entity ID. The parser found 562 annotated timestamps for the first train video.

Status: `PARTIALLY_WORKING`; parsing runs but is duplicated and action labels are incomplete.

### Frame extraction

The baseline command recognized 562 cached frames for the first video. It attempted a probe and printed:

```text
[!] Could not probe -FaXLcSFjUI: ffprobe error ... Continuing without probe.
```

The cached frame count was accepted without decoding or validating every cached JPEG.

Status: `PARTIALLY_WORKING`.

### Object detection

YOLOv8n loaded from a local 6.5 MB checkpoint. All 3,508 cached detection records were traversed:

| Video | Frames | Detections |
|---|---:|---:|
| `-FaXLcSFjUI` | 562 | 641 |
| `-IELREHX_js` | 756 | 1,613 |
| `-OyDO1g74vc` | 716 | 1,729 |
| `-XpUuIgyUHE` | 710 | 1,423 |
| `0f39OWEqJ24` | 764 | 1,281 |

Because every JSON file already existed, this run did not prove fresh YOLO inference. A representative detection includes class `person`, confidence `0.9277`, pixel and normalized boxes, centroid, and an AVA match with IoU `0.933`.

Status: `PARTIALLY_WORKING` from cache; fresh inference remains to be covered by the new smoke path.

### Scene graphs

All 3,508 cached scene files were found. A representative one-node frame has three self-edges named `action_action_11`, `action_action_79`, and `action_action_80`. This exposes the incomplete action mapping. Other relations are geometry rules or AVA-derived heuristics, but no edge states its source.

Status: `PARTIALLY_WORKING`.

### Whisper transcription

Five Whisper JSON/text pairs exist. A representative segment contains absolute start/end seconds and normal Whisper diagnostic fields.

Status: `UNVERIFIED` for a fresh run; existing artifacts are structurally valid.

### Pyannote diarization

Five sets of JSON, RTTM, CSV, SRT, and timeline PNG outputs exist. A representative turn contains `speaker`, `start`, `end`, and `duration`.

Status: `UNVERIFIED` for a fresh run. The current orchestration reruns diarization whenever an HF token is configured, even without the explicit flag.

### Transcript-speaker merge

Five merged JSON files exist. The implementation assigns the speaker with greatest temporal overlap. No-speaker output uses lowercase `unknown` rather than the canonical `UNKNOWN` requested for degradation.

Status: `PARTIALLY_WORKING` from inspected data and source.

### Multimodal fusion

Five fused train JSON files exist. For `-FaXLcSFjUI`, the file declares 562 frames and 431 audio segments. Audio is attached to every frame interval `[timestamp, timestamp + 1)`, so the same transcript may be repeated across frames.

Status: `PARTIALLY_WORKING`; a canonical segment representation is absent.

### GNN-compatible data

The cached export for the first video reports 562 frames, 447 frames with nodes, 97 with sparse candidate edges, and 370 with audio. A representative record has a ten-dimensional node feature and empty edge tensors.

The source instantiates `SceneGraphMPNN` without checkpoint loading and serializes random-weight edge embeddings. No labels or prediction confidence exist.

Status: feature construction `PARTIALLY_WORKING`; trained GNN inference `MISSING`.

### Neo4j ingestion

Configuration is present locally, but no service is listening on port 7687. The schema and parameterized `MERGE` calls were inspected; no live schema or repeat-ingestion test could be executed.

Status: `UNVERIFIED` implementation, `BROKEN` current runtime.

### Semantic query and UI

The current query engine either executes raw Groq-generated Cypher or, without a key/on any error, returns the first 50 frames. The Streamlit UI displays only video ID and timestamp and cannot stream or seek a stored video.

Status: `PLACEHOLDER`/`BROKEN`; not exercised against Neo4j because the database is unavailable.

## Existing sample contracts

```json
{
  "video_id": "-FaXLcSFjUI",
  "frame_id": "-FaXLcSFjUI_0903",
  "timestamp": 903,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.9277,
      "bbox": [91.75, 4.91, 670.82, 474.35],
      "bbox_norm": [0.1303, 0.0102, 0.9529, 0.9882],
      "center": [381.28, 239.63]
    }
  ]
}
```

```json
{
  "video_id": "-FaXLcSFjUI",
  "timestamp": 902,
  "nodes": [],
  "edges": [],
  "audio_segments": [
    {
      "start": 902.0,
      "end": 905.0,
      "speaker": "SPEAKER_09",
      "text": "..."
    }
  ]
}
```

The new implementation must adapt these formats without rewriting the evidence corpus and must record relationship provenance explicitly.

## Post-remediation verification

The following checks use the canonical `vidquery` application and were run after implementation on the same machine:

| Check | Result |
|---|---|
| Artifact adapter | One AVA video registered as `READY`; 154 canonical five-second segments imported non-destructively in 2.2 s |
| Fresh YOLO boundary | 20 preserved JPEGs processed with local `yolov8n.pt`; 19 `person` detections in 2.9 s after process/model startup |
| Fresh Whisper boundary | Bundled/resolved FFmpeg decoded an eight-second clip; cached Whisper `tiny` emitted five timed segments in 4.8 s (content quality was poor, illustrating the small-model limitation) |
| Fresh canonical vertical slice | Eight-second MP4 with audio -> validation -> frames -> YOLO -> Whisper -> alignment -> SQLite: `READY`, 2 segments, 7 detections, 3 transcript segments, 0 relationships, no warnings, 9.5 s |
| Retrieval smoke | 5 queries/1 video/154 segments: hybrid P@1 and MRR 1.0; transcript-only P@1 and MRR 0.4; see `EVALUATION.md` for scope caveats |
| Python tests | 41 passed in 9.1 s; one third-party Torch Geometric deprecation warning |
| Static analysis | Ruff clean; mypy clean on 16 canonical source files |
| Packaging | `vidquery-0.2.0-py3-none-any.whl` built successfully without dependency resolution |
| Go | `go test ./...` builds successfully; no Go test files |
| Compose | `docker compose config --quiet` succeeds |
| Docker runtime | Image/runtime integration not executed because the local Docker daemon was not running |
| Neo4j runtime | Live ingestion remains unverified because neither a local service nor Docker daemon was available; parameterization/idempotent row construction is covered by unit tests |
| GNN feature export | 562 frames regenerated to a temporary location; records state `untrained_feature_export`, with empty edge features and relationship predictions |

The in-app browser runtime exposed no browser session during visual QA. The HTTP server, root page/static asset delivery, upload/search/status/range behavior, and frontend contract are covered through integration tests; a manual cross-browser visual pass remains recommended before public deployment.
