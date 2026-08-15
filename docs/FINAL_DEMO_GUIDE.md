# Final controlled demo guide

## What this demo proves

The demo is a deterministic 25-second architecture meeting with two synthetic Windows voices, an original generated meeting-room image, one laptop, one whiteboard, and five labelled five-second speech windows. It demonstrates ingestion, persistent canonical segments, exact relationship tuples, speaker filtering, transcript retrieval, multimodal ranking, Neo4j mirroring, thumbnails, and range playback.

The checked-in `demo/final_demo_manifest.json` is ground truth. Its entities, relations, transcripts, and speakers are applied as `manual_annotation`, never presented as YOLO, Whisper, diarization, or learned-model output. The generated MP4 is ignored by Git; the original source image and generation prompt are included and contain no third-party footage.

## Prerequisites

- Windows 10/11 with the standard Microsoft David and Zira desktop voices.
- Python 3.11 or 3.12 with `.[dev,database]` installed.
- Docker Desktop for graph/container demonstration.
- `imageio-ffmpeg`, installed by the base package.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,full]"
Copy-Item .env.example .env
```

Set a non-default `NEO4J_PASSWORD` in `.env` before starting Compose.

## 1. Generate the MP4

```powershell
& .\scripts\prepare_final_demo.ps1 -PythonExecutable python
```

Output: `data/demo/final_demo.mp4` (25.000 seconds, 1280×720, 25 fps, H.264 video, mono AAC audio). On the verified Intel laptop, narration plus encoding took about 2.6 seconds. Voice and encoding speed will vary. If SAPI reports access denied in a sandboxed terminal, run the same command from a normal local PowerShell window.

## 2A. Recommended containerized ingestion

```powershell
docker compose up -d --build --wait
$upload = curl.exe -sS -X POST `
  -F "file=@data/demo/final_demo.mp4;type=video/mp4" `
  http://127.0.0.1:8000/api/videos | ConvertFrom-Json
$videoId = $upload.video_id

do {
  Start-Sleep -Milliseconds 500
  $status = Invoke-RestMethod "http://127.0.0.1:8000/api/videos/$videoId/status"
} until ($status.state -in @("READY", "FAILED"))

if ($status.state -ne "READY") { throw $status.failure_message }
docker compose exec -T backend python -m scripts.apply_demo_manifest `
  --video-id $videoId --neo4j
```

On the verified machine, model-disabled canonical processing took under two seconds after upload. The actual first upload returned `UPLOADED`, then polling returned `READY`. Full YOLO/Whisper/Pyannote processing time was not measured for this controlled demo and must not be quoted.

## 2B. Host-only ingestion with compulsory diarization

```powershell
$env:ENABLE_YOLO = "true"
$env:ENABLE_WHISPER = "true"
$env:ENABLE_DIARIZATION = "true"
$env:REQUIRE_DIARIZATION = "true"
# HUGGINGFACE_TOKEN must already be set in .env or this shell.
python -m vidquery.cli process --video data/demo/final_demo.mp4
python -m scripts.apply_demo_manifest --video-id VIDEO_ID
python -m vidquery.cli serve --host 127.0.0.1 --port 8000
```

Add `--neo4j` to the manifest command when the `.env` Neo4j credentials point to a running service.
Audio-bearing demo ingestion intentionally fails if Pyannote cannot load; there
is no silent UNKNOWN-speaker fallback in the required configuration.

## Ground truth

| Time | Speaker | Spoken content |
|---|---|---|
| 00:00–00:05 | SPEAKER_00 | We discuss the software architecture. |
| 00:05–00:10 | SPEAKER_01 | The laptop runs the API with SQLite metadata. |
| 00:10–00:15 | SPEAKER_00 | Deployment uses Docker for the backend and Neo4j graph. |
| 00:15–00:20 | SPEAKER_01 | This whiteboard shows the retrieval architecture. |
| 00:20–00:25 | SPEAKER_00 | The system searches speech, objects, and relationships. |

Every window has manually labelled `person`, `laptop`, and `whiteboard` occurrences. Stored symmetric tuples include `person near laptop` and `person near whiteboard`, with confidence 1.0 and source `manual_annotation`.

## Primary queries and expected first result

| Query | Expected first window | Evidence to point out |
|---|---:|---|
| Find where architecture is discussed. | 00:15–00:20 | transcript; 00:00–00:05 is also relevant |
| Find a person near a laptop. | 00:00–00:05 | exact matched tuple, endpoint IDs/classes, manual source |
| Find where someone discusses deployment while a laptop is visible. | 00:10–00:15 | deployment transcript plus person/laptop entities |
| Find where SPEAKER_01 discusses architecture. | 00:15–00:20 | speaker and transcript evidence |
| Find a person near a whiteboard. | 00:00–00:05 | exact symmetric tuple; all five windows are relevant |

Backup queries: `SQLite metadata`, `Docker Neo4j`, `SPEAKER_00 deployment`, `person near laptop`, and `whiteboard architecture`.

## Presentation flow

1. Open `http://127.0.0.1:8000` and show the green local-index status.
2. Select `final_demo.mp4` in the video filter.
3. Run the architecture query and open the 00:15 result.
4. Run `person near laptop`; point to the tuple chip and match explanation rather than merely the entity chips.
5. Run the deployment multimodal query and seek to 00:10.
6. Run the speaker query.
7. Switch the API request to `retrieval_backend: "neo4j"` in `/docs` to show explicit graph retrieval.
8. Finish with the negative query `zzzxylophone quantum penguin` to show the no-result state.

## Troubleshooting

- `SAPI.SpVoice` error: run PowerShell outside a restricted sandbox and confirm Microsoft David/Zira desktop voices are installed.
- FFmpeg missing: activate the environment that contains `imageio-ffmpeg`, or pass its Python executable with `-PythonExecutable`.
- Duplicate upload: reuse the returned existing video ID; SHA-256 deduplication is expected.
- Relation query has no matched tuple: run `scripts.apply_demo_manifest` after canonical processing, then include `--neo4j` again if graph mode is used.
- Graph request returns 503: confirm `docker compose ps`, the Neo4j password, and `GET /api/health`.
- Thumbnail missing: confirm processing created frames and that the video remains in the persistent volume.
- Learned classifier unavailable in the default container: expected; the checkpoint is ignored and is not baked into the demo image. Train or mount a validated checkpoint explicitly before enabling it.
