```markdown
# VidQuery: Implementation Plan & Architecture

## Project Overview
VidQuery is a semantic search engine for videos. It utilizes a Two-Layer Architecture: a GNN Reasoning Engine for visual scene graph extraction and a Multimodal Knowledge Graph (MMKG) in Neo4j for storage and retrieval. Natural Language queries are parsed into Cypher queries via Groq Llama 3.1 to retrieve exact video timestamps.

## Tech Stack
* **Core:** Python 3.9+ [cite: 704]
* **Extraction:** OpenCV, YOLOv8, OpenAI Whisper, pyannote.audio 
* **Modelling:** PyTorch, PyTorch Geometric (PyG) 
* **Storage:** Neo4j (Community Edition 5.x), Neo4j Python Driver 
* **Retrieval/LLM:** Groq API (Llama 3.1)
* **Frontend/API:** FastAPI (Backend), React or Streamlit (Frontend)

---

## Directory Structure
```text
VidQuery/
├── api/                    # FastAPI endpoints
├── app/                    # Frontend UI (React/Streamlit)
├── core/                   # Configuration and constants
├── database/               # Neo4j connection and Cypher scripts
├── extraction/             # Audio & Visual processing modules
├── modelling/              # GNN architecture and data alignment
├── retrieval/              # Groq NL2Cypher and GraphRAG logic
├── data/                   # Raw videos, frames, audio buffers (gitignored)
├── requirements.txt
├── .env
└── main.py                 # Application entry point
```

---

## Phase 1: Project Setup & Infrastructure
**Goal:** Initialize the environment, establish configuration, and connect to the database.

1.  **`requirements.txt` & `.env`**
    * Define dependencies: `torch`, `torch-geometric`, `ultralytics`, `openai-whisper`, `pyannote.audio`, `neo4j`, `groq`, `fastapi`, `uvicorn`.
    * Define environment variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GROQ_API_KEY`, `HF_TOKEN` (for Pyannote).
2.  **`core/config.py`**
    * *Function:* Load env vars and set global constants (e.g., `FPS = 1`, `AUDIO_SR = 16000`, `CHUNK_SIZE = 15`).
3.  **`database/neo4j_client.py`**
    * *Class `Neo4jClient`:* * `__init__()`: Establish driver connection.
        * `close()`: Safely close driver.
        * `execute_write()` / `execute_read()`: Wrapper for transactions.

---

## Phase 2: Ingestion & Extraction Layer
**Goal:** Process raw MP4 files into structured text and bounding box data.

1.  **`extraction/preprocessor.py`**
    * *Class `VideoPreprocessor`:*
        * `extract_frames(video_path)`: Uses OpenCV to sample 1 frame per second. Saves to `/data/frames/`.
        * `extract_audio(video_path)`: Uses FFmpeg to extract 16kHz Mono WAV. Saves to `/data/audio/`.
2.  **`extraction/visual.py`**
    * *Class `VisualExtractor`:*
        * `run_yolov8(frame_dir)`: Runs inference[cite: 663].
        * *Output:* Returns a list of dictionaries containing `frame_id`, `timestamp`, `bounding_boxes`, `classes`, `confidence`.
3.  **`extraction/audio.py`**
    * *Class `AudioExtractor`:*
        * `transcribe(audio_path)`: Runs Whisper. Returns text segments with start/end timestamps[cite: 663].
        * `diarize(audio_path)`: Runs pyannote.audio. Returns speaker IDs with start/end timestamps[cite: 663].
        * `merge_audio_data()`: Aligns transcript text with the corresponding speaker ID based on timestamps.

---

## Phase 3: Modelling Layer (GNN & Alignment)
**Goal:** Synthesize extracted data into relationships (Scene Graphs) and align modalities.

1.  **`modelling/gnn_dataset.py`**
    * *Function `build_sparse_graphs(visual_data)`:* Converts YOLO bounding boxes into PyG `Data` objects. Uses spatial proximity to create edge indices (reducing $O(N^2)$ complexity)[cite: 884].
2.  **`modelling/gnn_model.py`**
    * *Class `SceneGraphMPNN(torch.nn.Module)`:* * Implements Message Passing Neural Network[cite: 921].
        * `forward(x, edge_index)`: Predicts edge attributes (Subject-Predicate-Object relationships like "Person -> Writing -> Whiteboard")[cite: 666, 669, 671].
3.  **`modelling/alignment.py`**
    * *Function `fuse_modalities(scene_graphs, audio_data)`:* Maps audio segments (Speaker + Text) to the corresponding visual `scene_id` or `frame_id` based on timestamps. Outputs a unified JSON representation of the video's timeline.

---

## Phase 4: Storage & MMKG Integration
**Goal:** Ingest the fused JSON scene graphs into Neo4j[cite: 644, 648].

1.  **`database/schema.cypher`**
    * Cypher script to create constraints and indexes on `Video`, `Frame`, `Scene`, `Object`, `Person`, `Concept`.
2.  **`database/ingestion.py`**
    * *Class `GraphIngestor`:*
        * `ingest_video_metadata(video_dict)`
        * `ingest_scene_graph(fused_json)`: Parses the JSON and executes Cypher merges to create nodes (`(:Person)`, `(:Object)`) and relationships (`-[:PERFORMS]->`, `-[:HAS_OBJECT]->`, `-[:SPOKEN_BY]->`).

---

## Phase 5: Retrieval Pipeline (GraphRAG)
**Goal:** Translate user natural language queries into Cypher to fetch timestamps[cite: 645, 649].

1.  **`retrieval/prompt_templates.py`**
    * Define the few-shot prompt for Groq Llama 3.1 containing the Neo4j schema and examples of converting NL (e.g., "Find where the professor writes on the board") into Cypher queries.
2.  **`retrieval/query_engine.py`**
    * *Class `SemanticSearchEngine`:*
        * `parse_nl_to_cypher(user_query)`: Calls Groq API (Llama 3.1) with the system prompt to generate Cypher.
        * `execute_search(cypher_query)`: Runs the query via `Neo4jClient`.
        * `format_results(raw_results)`: Extracts specific `video_id` and `timestamp` (e.g., `[00:14:23]`).

---

## Phase 6: API & UI Integration
**Goal:** Expose the pipeline via REST and build the user interface.

1.  **`api/routes.py`** (FastAPI)
    * `POST /upload`: Triggers Phase 2-4 ingestion pipeline asynchronously.
    * `GET /search?q={query}`: Triggers Phase 5 retrieval pipeline. Returns timestamped JSON results.
2.  **`app/ui.py`** (Streamlit/React)
    * *Components:*
        * Search Bar.
        * Results Gallery (list of retrieved segments).
        * Video Player embedded with logic to auto-seek to the queried timestamp URL parameter (e.g., `video.mp4#t=863`).
```