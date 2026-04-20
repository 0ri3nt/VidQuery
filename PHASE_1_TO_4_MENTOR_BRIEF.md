# VidQuery Mentor Brief (Phase 1 to Phase 4)

## 1) What This Covers
This document summarizes implementation progress up to Phase 4 of VidQuery, with focus on:
- what we built in each phase,
- what outputs we generated,
- why those outputs matter,
- what inferences we can draw from the outputs.

## 2) Project Goal (Up to Phase 4)
Build an end-to-end multimodal video understanding pipeline that:
- extracts visual and audio information from AVA videos,
- fuses both modalities into scene-level timeline artifacts,
- converts these artifacts into graph structure,
- ingests the resulting multimodal graph into Neo4j for semantic retrieval.

## 3) Phase-Wise Summary

## Phase 1: Setup and Infrastructure
### What We Did
- Created project structure and modular packages: `core`, `database`, `extraction`, `modelling`, `retrieval`, `api`, `app`.
- Configured environment via `.env` and `core/config.py`.
- Added dependency stack for CV, ASR, diarization, graph modelling, API, and Neo4j.
- Implemented Neo4j client connectivity layer in `database/neo4j_client.py`.
- Added CLI orchestration entrypoints in `main.py`.

### Output
- Reproducible Python environment and project skeleton.
- Working application config loader and DB connection wrappers.
- `main.py` command interface to run pipeline stages independently.

### Relevance
- This phase makes all later phases repeatable and testable.
- Clean modular boundaries reduce coupling and make debugging faster.

### Inference
- The system is production-oriented in structure (clear module ownership and staged execution), not a one-off notebook prototype.

---

## Phase 2: Ingestion and Extraction Layer
### What We Did
- Implemented frame extraction from videos (OpenCV-based preprocessor).
- Implemented visual detection pipeline with YOLOv8.
- Implemented audio extraction, Whisper transcription, and optional speaker diarization.
- Implemented audio merge/alignment logic to combine transcript segments and speaker turns.

### Output
- Frame artifacts under AVA extraction outputs.
- Detection outputs with per-frame visual entities and bounding boxes.
- Audio transcripts and speaker-segment outputs.
- Structured per-frame/per-segment JSON inputs for modelling.

### Relevance
- Converts raw MP4 input into machine-readable primitives needed for graph construction.
- Adds both object-level visual signal and speaker/text audio signal, enabling multimodal understanding.

### Inference
- We now have timestamp-grounded primitives from both modalities, which is the prerequisite for building reliable scene graphs.

---

## Phase 3: Modelling Layer (Scene Graph + Multimodal Alignment + GNN Export)
### What We Did
- Built sparse scene graph construction pipeline (`build_scene_graphs.py`, modelling modules).
- Added multimodal fusion (`modelling/alignment.py`) to attach audio segments to time-aligned visual frames.
- Updated GNN export path to prefer fused artifacts.
- Extended GNN node features with audio-aware dimensions (feature dim now 10).

### Output (Verified Artifacts)
- Fused train artifacts generated in `data/ava/fused/train`.
- Current fused summary from artifacts:
  - `fused_files`: 5
  - `frames`: 3508
  - `audio_segments`: 2716
  - `visual_nodes`: 6686
  - `visual_edges`: 45017
  - `frames_with_audio`: 2270
- Sample regenerated GNN export (`-FaXLcSFjUI`) confirms multimodal export fields:
  - `node_feature_dim`: 10
  - `audio_segment_count` and `audio_unique_speakers` present per frame
  - video summary contains `frames_with_audio`

### Relevance
- This phase transforms low-level detections/transcripts into structured relational intelligence.
- Fused representation enables downstream reasoning/querying beyond pure object detection.
- Audio-aware graph features improve contextual grounding for temporal events.

### Inference
- Multimodal coverage is substantial: $2270/3508 \approx 64.7\%$ frames have aligned audio context.
- Graph density indicates rich relational context for search/retrieval (high edge volume relative to node count).

---

## Phase 4: Storage and MMKG Integration (Neo4j)
### What We Did
- Implemented schema constraints/indexes in `database/schema.cypher`.
- Implemented robust ingestion flow in `database/ingestion.py` for fused scene graphs.
- Added event-level audio modelling with `AudioSegment` nodes and keyed relations:
  - `(:Frame)-[:HAS_AUDIO_SEGMENT]->(:AudioSegment)`
  - `(:AudioSegment)-[:SPOKEN_BY]->(:Person)`
  - frame-level `SPOKEN_BY` keyed by `segment_key`
  - `PERFORMS` speech relations keyed by `event_key`
- Updated ingestion CLI defaults to use fused root (`data/ava/fused`).

### Output (Latest Successful Verification Snapshot)
- Ingestion command completed: `Ingested 5 graph files from data/ava/fused`.
- Last successful DB validation (when Neo4j was online) showed event-level parity:
  - `AudioSegment`: 2716
  - `HAS_AUDIO_SEGMENT`: 2716
  - segment-level `SPOKEN_BY`: 2716
  - frame-speaker `SPOKEN_BY` with `segment_key`: 2716
  - speech `PERFORMS` with `event_key`: 2716
- Legacy non-keyed speech relationships were cleaned to eliminate old duplicates.

### Relevance
- Neo4j now stores multimodal events with timestamp-level granularity.
- This structure directly supports precise question answering and timestamp retrieval in later phases.

### Inference
- Audio events are preserved at segment granularity (no collapse to one speaker-per-frame), which improves retrieval fidelity for speech-centric queries.
- The MMKG is now suitable for GraphRAG-style retrieval where temporal and semantic precision matters.

---

## 4) Overall Interpretation After Phase 4
- Phase 1 established a reliable modular foundation.
- Phase 2 produced clean multimodal primitives from raw video.
- Phase 3 successfully fused modalities and exported graph-ready/GNN-ready artifacts.
- Phase 4 persisted those multimodal events into Neo4j with event-level correctness.

Net result: We have an end-to-end multimodal knowledge graph pipeline ready for Phase 5 retrieval (NL-to-Cypher + GraphRAG) with strong timestamp grounding.

## 5) Mentor Discussion Points (Suggested)
- Why event-level audio modelling (`segment_key`, `event_key`) matters for avoiding semantic loss.
- Tradeoff between sparse graph construction and relational coverage for scalable reasoning.
- Current dataset coverage and known gaps:
  - train split is active and validated,
  - val fused split not yet populated in current run.
- Next milestone readiness: retrieval evaluation (precision of returned timestamps for natural language queries).

## 6) Note on Current Runtime State
During the latest report generation, Neo4j service was not running in the local session at one check, so DB counts above are explicitly from the latest successful verification snapshot.
