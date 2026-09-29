# Demo guide

## Reliable artifact-backed demo

This route avoids model downloads and uses the preserved AVA evidence.

1. Install the base/dev package and copy `.env.example` to `.env`.
2. Run `python -m vidquery import-ava --limit 1`.
3. Run `python -m vidquery serve --host 127.0.0.1 --port 8000`.
4. Open `http://127.0.0.1:8000`.
5. Confirm the header says the local index is available and Library shows `-FaXLcSFjUI` as `READY`.

Suggested narrative:

| Query | Expected leading moment | Evidence to point out |
|---|---:|---|
| `animal across trains` | 15:00–15:05 | transcript and speaker-aligned window |
| `person watching` | 15:00–15:05 | person occurrence plus mapped AVA watch action |
| `person near chair` | 28:20–28:25 | co-occurring entity classes and geometry-labelled `near` |
| `SPEAKER_09 animal` | 15:00–15:05 | speaker constraint plus spoken term |

Click Play to open the HTTP-range-backed video and seek to the returned timestamp. Explain that the evidence chips disclose whether a relationship came from geometry, AVA mapping, or another source.

## Fresh-video demo

Install optional adapters and choose model policy explicitly:

```powershell
python -m pip install -e ".[dev,full]"
$env:ENABLE_YOLO="true"
$env:ENABLE_WHISPER="true"
$env:ALLOW_MODEL_DOWNLOADS="true"  # only for an intentional first-time download
python -m vidquery process --video C:\path\to\demo.mp4 --copy
python -m vidquery serve --host 127.0.0.1 --port 8000
```

In a second terminal, run
`Invoke-RestMethod http://127.0.0.1:8000/api/health`.

Run installation, processing, and serving from the same activated Python
environment. The health response must say `yolo: available_configured` and
`whisper: available_configured`; the word `enabled` alone is not proof that a
dependency or checkpoint is usable. If a video was indexed before the models
were ready, open Library and click **Reprocess** so the old empty evidence is
replaced. Set `ALLOW_MODEL_DOWNLOADS=false` again after caching. Pyannote
diarization is compulsory for audio-bearing uploads: keep `ENABLE_DIARIZATION=true`
and `REQUIRE_DIARIZATION=true`, install the `full` dependencies, and provide
`HUGGINGFACE_TOKEN`. Processing fails explicitly when this requirement is not
usable rather than silently indexing unknown speakers. On CPU, use small videos and the Whisper `tiny` model;
expect processing to take materially longer than video duration on some machines.

Fresh processing currently supplies detector entities, Whisper transcript, optional speaker labels, and geometry relationships. It does not supply trained action recognition or a GNN prediction.

## Neo4j demo

Local search needs no Neo4j. To demonstrate the graph mirror, start Neo4j, set `ENABLE_NEO4J=true` and credentials, then process/reprocess a video. `/api/health` should report `neo4j: available`. Show `Video-[:HAS_SEGMENT]->Segment` and evidence nodes in Neo4j Browser. If the graph is down, processing finishes with a warning and the local index remains searchable.

## Recovery and troubleshooting

- `415` on upload: ensure the extension, MIME/signature, and decoded container are MP4.
- Video becomes `FAILED`: inspect the public stage/message in Library; consult server logs/private SQLite detail locally.
- Model warning: verify the optional package/checkpoint/cache and model-download setting.
- Model says ready but an old upload has no detections/transcript: reprocess that
  video; installing a model does not retroactively change stored segments.
- No thumbnail on imported legacy data: verify preserved frame paths; playback/search still work.
- No result: inspect parsed-query chips and try fewer structured constraints.
- Port 8000 busy: use `--port 8001` and add the origin to `CORS_ORIGINS` if frontend/API origins differ.

## Acceptance checklist

- Upload returns quickly and status transitions are visible.
- A restart preserves Library records and search results.
- Duplicate upload returns the same video ID.
- A search result contains timestamp, score, transcript/evidence, thumbnail URL, and stream URL.
- Playback seeks and a `Range: bytes=0-99` request returns `206`.
- Missing optional models create warnings rather than fabricated evidence.
- GNN health remains `unavailable_untrained`.
