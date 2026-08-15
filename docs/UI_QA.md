# UI and deployment QA

## Outcome

On 12 August 2026, the current host application passed 24/24 reproducible HTTP
and packaged-UI-surface checks after a no-result false-positive fix. The actual
Compose backend ran healthy on host port 8001 alongside the working host server;
it became healthy again after restart. Neo4j remained healthy and both named
volumes remained present.

Manual visual browser QA is **not claimed**. The browser-control runtime exposed no Chrome or in-app browser (`[]` available browsers), so rendered layout, native media playback, and a second-browser pass could not be honestly inspected in this session. No UI screenshots were captured. The exact remaining manual checklist is included below.

Machine-readable evidence:

- `evaluation/results/http-ui-qa.json`
- `evaluation/results/http-ui-qa-after-restart.json`
- `evaluation/results/http-ui-qa-current.json`

Reproduce the automated surface checks with:

```powershell
python -m scripts.run_http_qa `
  --video-id VIDEO_ID `
  --demo-video data/demo/final_demo.mp4
```

## Verified behavior

| Area | Evidence | Result |
|---|---|---|
| Upload | A new 25-second MP4 returned 202/`UPLOADED`; a second independent five-second MP4 also reached `READY` | Pass |
| Validation error | `invalid.txt` multipart upload returned 415 and `Only .mp4 uploads are accepted` | Pass |
| Deduplication | Re-upload returned the same video ID with `duplicate: true` | Pass |
| Processing status | Status polling reached `READY`/`complete` | Pass |
| Library | Four persisted records were listed in the final container volume | Pass |
| Static frontend | `/`, CSS, and JavaScript returned 200; repeat `/` was byte-identical | Pass |
| Search | Transcript, entity, exact relation, speaker, multimodal, SQLite, and Neo4j searches returned expected first windows | Pass |
| No-result state | An impossible three-token query returned an empty result list | Pass |
| Thumbnail | Result thumbnail returned JPEG/200 (165,055 bytes in final run) | Pass |
| Evidence contract | Parsed chips, matched tuple, source method, endpoint IDs/classes, and match explanation were present in API payload; frontend code renders them | Pass at contract/code level |
| Repeated query | Repeated exact-tuple query returned the same ordered segment IDs | Pass |
| Multiple-video filter | API filter returned only the selected ID; missing-ID filter returned no results | Pass |
| Filter control | Missing browser control was identified; an `All videos`/specific-video selector was added and a packaged-frontend regression test asserts its `video_ids` wiring | Fixed and tested |
| Range playback | `Range: bytes=0-127` returned 206, 128 bytes, `Accept-Ranges`, and correct `Content-Range` | Pass at HTTP level |
| API failures | Missing video returned 404; invalid range 416; blank query 422 | Pass |
| Health | SQLite and Neo4j reported `available` | Pass |
| Page refresh | Repeated root fetch returned the same installed page | Pass at HTTP level |

## Actual container verification

Commands executed:

```powershell
docker compose build backend
$env:BACKEND_PORT="8001"
docker compose up -d --force-recreate
docker compose ps
docker compose restart backend
docker compose up -d --wait --wait-timeout 120
docker volume ls --filter name=vidquery
```

Observed after restart:

- `vidquery-backend-1`: healthy, container port 8000 published to host port 8001
  for this non-disruptive verification (`BACKEND_PORT=8001`).
- `vidquery-neo4j-1`: healthy, ports 7474 and 7687 published.
- `vidquery_vidquery_data` and `vidquery_neo4j_data`: present.
- `/api/health`: application `ok`, database `available`, Neo4j `available`.
- All 24 final-image QA checks passed against persisted video `209a32a1-1a34-41e6-addf-0b53495130da`.

The default container correctly reports the action and relationship classifiers
as unavailable because `.pt` checkpoints are deliberately excluded from the
build. It never loads random weights. Validated host checkpoints remain opt-in
and must be explicitly mounted for model-enabled container inference.

The final rebuilt image uses matched CPU packages (`torch==2.5.1+cpu`,
`torchaudio==2.5.1+cpu`, `torchvision==0.20.1+cpu`). Direct imports of Pyannote,
PyG, Ultralytics, Whisper, SentenceTransformers, and EasyOCR passed inside the
running container. The existing `HF_TOKEN` is mapped to the application's
`HUGGINGFACE_TOKEN` setting without exposing its value, and health reports
`diarization: available_configured_required`.

## Issue log

| Issue | Resolution/status |
|---|---|
| Search UI had no multiple-video selector even though the API supported filtering | Added `#video-filter`, populated it from `/api/videos`, preserved selection, and sent `video_ids` with search requests |
| A nonsense semantic query returned low-score results instead of the empty state | Added configurable `SEMANTIC_MIN_SIMILARITY=0.20`, a regression test, and reran 24/24 HTTP QA plus the 62-query evaluation |
| Browser-control runtime had no available browser | External environment limitation; recorded, not worked around or misreported |
| Rendered relationship chips and seeking could not be visually exercised | Still requires manual browser pass |

## Required final manual pass

When Chrome is available, perform and screenshot: upload selection/progress, invalid-file message, READY transition, filtered search, empty result panel, thumbnail rendering, relation chip and match explanation, Play dialog seeking at 00:10, repeated query, refresh, and a simulated API outage. Repeat layout/search/playback in Edge or Firefox. Record browser versions and screenshot paths here; until then these items are unverified visually.
