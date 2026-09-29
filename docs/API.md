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
  "retrieval_backend": "sqlite",
  "ranking": "reliability",
  "localize": true,
  "collapse_duplicate_evidence": true,
  "hypothesis_id": null
}
```

The last four fields are the Review-2 controls. Defaults are `ranking: "hybrid"`,
`localize: true`, `collapse_duplicate_evidence: false`, and no `hypothesis_id`,
which reproduces the Review-1 ranking contract. The browser UI sends the values
shown above. An unknown `hypothesis_id` returns `422`. See
[REVIEW2.md](REVIEW2.md) for the algorithms.

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

### Review-2 result and response fields

Each result may also carry:

```json
{
  "localization": {
    "peak_time": 42.1, "start_time": 42.1, "end_time": 42.6,
    "precision": "word", "source": "whisper_word_timing",
    "confidence": 0.95, "evidence": ["word 'deployment' spoken at 42.10s"]
  },
  "evidence": {
    "confidence": 0.71, "verdict": "supported",
    "contributions": [{"modality": "transcript", "raw_score": 0.9,
      "source_method": "whisper_transcript", "reliability_prior": 0.9,
      "query_weight": 1.0, "weighted_score": 0.81}],
    "explanation": "transcript evidence from whisper_transcript ... -> supported."
  },
  "supporting_hypotheses": ["spoken_mention"]
}
```

- `localization.precision` is one of `word`, `frame`, `interval`,
  `utterance_interpolated`, `utterance`, or `segment`. `peak_time` is where the
  player seeks. It always lies inside `[start_time, end_time]`, which lies
  inside the segment.
- `evidence.verdict` is `supported`, `weak`, or `insufficient`. `confidence` is
  a reliability-weighted fusion, not a calibrated probability, and is separate
  from `score`.

The response adds `hypotheses` (each with `hypothesis_id`, `label`, `plan`,
`prior`, `support`, `posterior`, `result_count`), `interpretation` (`single`,
`resolved`, `clarification_suggested`, or `insufficient_evidence`),
`selected_hypothesis_id`, and `clarification_prompt`. `insufficient_evidence`
means the system is abstaining; clients should present it as "not found" rather
than showing the top result as an answer.

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
    "diarization": "available_configured_required",
    "gnn_person_action_classifier": "available_validated_enabled",
    "gnn_relationship_prediction": "available_validated_enabled",
    "person_action_classifier": "unavailable_no_valid_checkpoint",
    "sentence_embedding_retrieval": "available_cached_enabled",
    "ocr": "available_cached",
    "rag_generation": "available_configured",
    "query_planner": "available_configured_local_first",
    "entity_appearance": "available_configured_frozen"
  }
}
```

This is the output of a fully configured local install. The legacy six-label
`person_action_classifier` is unavailable when its optional checkpoint is not
installed; the 80-label GNN replaces it.

YOLO and Whisper report `available_configured` only when they are enabled, their
Python dependency is importable, and the configured checkpoint is locally
available (or downloads were explicitly allowed). Missing dependencies and
checkpoints have distinct `unavailable_*` states. The 80-label GNN status is
derived from checkpoint existence, schema, model version, label and feature
contracts, validation metadata, thresholds, and SHA-256 integrity. It ends in
`_inactive` or `_enabled` when valid; invalid/missing states state the reason.
`gnn_relationship_prediction` reports the VidOR pair-visual relation GNN, which
is validated the same way. Adapters still load weights lazily during processing.
`/health` is a hidden compatibility alias.

## Operational caveats

FastAPI `BackgroundTasks` is suitable for a local demo, not durable job execution. Upload progress covers network transfer; processing progress is polled through status. There is no authentication or per-user authorization, so do not expose this service directly to the internet.
