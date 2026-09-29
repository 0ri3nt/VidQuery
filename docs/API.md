# HTTP API

Base URL for local development: `http://127.0.0.1:8000`. OpenAPI JSON is `/openapi.json`; Swagger UI is `/docs`. Request validation errors use FastAPI's standard `422` envelope, and endpoint errors use `{"detail": "message"}`.

## Upload and lifecycle

### `POST /api/videos`

Multipart field `file`; only validated MP4 is accepted. The body is streamed, hashed, size-limited, sanitized, and probed before registration. Returns `202`:

```json
{"video_id":"uuid","state":"UPLOADED","duplicate":false}
```

An identical content hash returns the existing video with `duplicate: true`. Errors: `413` too large, `415` extension/MIME/signature/probe failure.

### `GET /api/videos`

Returns all persistent `VideoRecord` values, newest first.

### `GET /api/videos/{video_id}` and `/status`

Returns one public `VideoRecord`; `404` when unknown. `failure_detail` is intentionally excluded.

### `POST /api/videos/{video_id}/process`

Schedules reprocessing and returns `202`. Allowed from `UPLOADED`, `READY`, or `FAILED`; returns `409` when already in an intermediate state.
Reprocessing preserves and merges controlled-demo manual overlays and
`legacy_ava_fused_adapter` evidence by timestamp; ordinary stale model output is
replaced. The equivalent all-library CLI is `python -m vidquery refresh-library`.

## Search

### `POST /api/search`

```json
{
  "query": "SPEAKER_09 person watching",
  "video_ids": [],
  "limit": 10,
  "retrieval_backend": "sqlite"
}
```

`query` is 1–500 non-blank characters, at most 100 video IDs may be supplied, and `limit` is 1–50. `retrieval_backend` is `sqlite` by default or may be explicitly set to `neo4j`. The response echoes the backend actually used. Neo4j mode uses a static parameterized graph plan to select segment IDs, then hydrates and ranks their canonical evidence from SQLite; it returns 503 when Neo4j is disabled or unavailable. There is no silent fallback that could mislabel SQLite retrieval as graph retrieval.

The response contains the original query, validated plan, and ranked evidence. A
local parser first handles supported aliases, morphology, plurals, speech cues,
and OCR cues. Only a genuinely ambiguous plan may be sent to the optional Groq
query planner. That planner receives allowlisted vocabularies but no indexed
data, cannot issue database queries, and cannot rank evidence. Its output is
schema-validated and rejects invented labels. `query_planner_status`,
`query_ambiguities`, and `alternative_query_plans` make this routing explicit.
For a bare homonym such as `find drive`, VidQuery safely searches both the
observed AVA action and the spoken concept rather than silently selecting one.

Known entity/action/relationship/speaker/OCR constraints must all be present.
When a subject, predicate, and object can be parsed,
`parsed_query.relationship_tuples` records the normalized tuple and
directionality. A segment then passes the relation constraint only when one
stored edge connects those endpoint classes; symmetric predicates (`near`,
`overlaps_with`) may reverse endpoints, while directional predicates may not.

Each result includes `matched_relationships`. These are the exact stored edges that satisfied parsed tuples, including source/target detection IDs and classes, normalized predicate, directionality, confidence, source method, and timestamp. `match_reason` renders the same tuple. The score uses only those matched edges; unrelated relationships in the segment do not raise the relation component. A result score is a transparent weighted average of active modalities, not a calibrated probability.

Each result also includes `action_evidence` for GNN predictions: action,
confidence, source person ID, optional resolved target ID/predicate, timestamp,
and provenance. `gnn_action_model` means person-action classification only;
`gnn_action_plus_target_resolver` means a separate resolver selected the target.
Action-only search uses stored prediction confidence when ranking matches.

## Playback and evidence images

### `GET /api/videos/{video_id}/stream`

Returns `video/mp4`, `Accept-Ranges: bytes`, and supports single standard ranges including `bytes=0-1023`, `bytes=1024-`, and `bytes=-1024`. Satisfiable ranges return `206` with `Content-Range`; invalid/out-of-file ranges return `416`.

### `GET /api/videos/{video_id}/thumbnail?timestamp=900`

Returns the generated JPEG closest to the requested non-negative timestamp, or `404` if the video/thumbnail is unavailable.

## Health

### `GET /api/health`

```json
{
  "application": "ok",
  "database": "available",
  "neo4j": "disabled",
  "models": {
    "yolo": "available_configured",
    "whisper": "available_configured",
    "diarization": "disabled",
    "gnn_person_action_classifier": "available_validated_enabled",
    "gnn_relationship_prediction": "unavailable_no_relation_labels",
    "person_action_classifier": "available_validated_fallback",
    "query_planner": "available_configured_local_first"
  }
}
```

YOLO and Whisper report `available_configured` only when they are enabled, their
Python dependency is importable, and the configured checkpoint is locally
available (or downloads were explicitly allowed). Missing dependencies and
checkpoints have distinct `unavailable_*` states. The 80-label GNN status is
derived from checkpoint existence, schema, model version, label and feature
contracts, validation metadata, thresholds, and SHA-256 integrity. It ends in
`_inactive` or `_enabled` when valid; invalid/missing states state the reason.
The relationship-model key stays unavailable because the trained GNN predicts
person actions, not explicit relation targets. Adapters still load weights lazily during processing.
`/health` is a hidden compatibility alias.

## Operational caveats

FastAPI `BackgroundTasks` is suitable for a local demo, not durable job execution. Upload progress covers network transfer; processing progress is polled through status. There is no authentication or per-user authorization, so do not expose this service directly to the internet.
