# Data formats

## Canonical video registry

`VideoRecord` is persisted in SQLite. Public API responses exclude `failure_detail`.

```json
{
  "video_id": "uuid",
  "display_name": "meeting.mp4",
  "original_filename": "meeting.mp4",
  "stored_path": "server-owned absolute path",
  "content_sha256": "64 lowercase hex characters",
  "duration": 120.4,
  "width": 1920,
  "height": 1080,
  "fps": 29.97,
  "has_audio": true,
  "upload_time": "ISO-8601 UTC",
  "updated_time": "ISO-8601 UTC",
  "state": "READY",
  "current_stage": "complete",
  "failure_message": null,
  "warnings": []
}
```

## Canonical segment

The search and graph unit is a half-open, fixed-duration time window. `segment_id` is deterministic for a video and start time. All evidence records retain timestamps even after alignment.

```json
{
  "segment_id": "video-id:00000900.000",
  "video_id": "uuid",
  "start_time": 900.0,
  "end_time": 905.0,
  "frame_ids": ["frame-id"],
  "detections": [
    {
      "detection_id": "detection-id",
      "frame_id": "frame-id",
      "video_id": "uuid",
      "timestamp": 903.0,
      "class_id": 0,
      "class_label": "person",
      "confidence": 0.91,
      "pixel_bbox": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
      "normalized_bbox": {"x1": 0.01, "y1": 0.04, "x2": 0.11, "y2": 0.41},
      "centroid": [60, 120],
      "centroid_normalized": [0.06, 0.22]
    }
  ],
  "entities": ["person"],
  "relationships": [
    {
      "relationship_id": "relationship-id",
      "source_id": "detection-id",
      "target_id": "other-detection-id",
      "predicate": "near",
      "confidence": 0.75,
      "source_method": "bounding_box_geometry",
      "timestamp": 903.0
    }
  ],
  "actions": ["watch"],
  "transcript": "aligned text",
  "transcript_segments": [],
  "speakers": ["SPEAKER_09"],
  "embedding": null,
  "processing_metadata": {
    "frame_sample_rate": 1.0,
    "segment_duration": 5.0,
    "relationship_mode": "bounding_box_geometry"
  }
}
```

Pixel boxes are `[0,width] × [0,height]`; normalized boxes and centroids are clamped to `[0,1]`. Geometry predicates are directional where appropriate (`left_of`, `right_of`, `above`, `below`) and symmetric where defined (`near`, `overlaps_with`, `touches`). Actions imported from AVA identify mapped names and preserve unmapped IDs as `ava_action_<id>`.

## Search plan and response

A request contains query text, optional canonical video IDs, and a bounded limit. The parser emits an inspectable plan:

```json
{
  "intent": "relationship_search",
  "visual_entities": ["person", "chair"],
  "relationships": ["near"],
  "relationship_tuples": [
    {
      "subject": "person",
      "predicate": "near",
      "object": "chair",
      "directionality": "symmetric"
    }
  ],
  "actions": [],
  "spoken_terms": [],
  "ocr_terms": [],
  "speaker": null
}
```

The search request also accepts `retrieval_backend: "sqlite" | "neo4j"`, defaulting to `sqlite`; the response echoes the backend actually used. Structured fields are required constraints. `spoken_terms` search transcript evidence; `ocr_terms` search only OCR evidence. A parsed relationship tuple is matched against a single canonical edge by resolving `source_id` and `target_id` to detection classes. `near` and `overlaps_with` permit reversed endpoint order; `left_of`, `right_of`, `above`, `below`, `inside`, and `contains` preserve order. The response records local/Groq planner status, explicit ambiguities, and any safely merged alternative plans.

Each result contains the canonical interval, score, transcript, evidence arrays, human-readable match reason, thumbnail URL, and stream URL. Its `matched_relationships` array exposes the exact matching edge as `relationship_id`, `source_id`, `source_class`, `predicate`, `target_id`, `target_class`, `directionality`, `confidence`, `source_method`, and `timestamp`. The broader `relationships` array remains available as segment context.

## Existing AVA artifacts

The legacy corpus contains MP4s, extracted JPEGs, per-frame YOLO JSON, transcript and diarization JSON, scene graph JSON, fused JSON, and `.gnn.json` feature exports. Field shapes vary across phases. `vidquery/ava_adapter.py` is the normalization boundary: it resolves box scale, frame dimensions, timestamp keys, action mappings, transcripts, speakers, and explicit provenance before data enters the canonical store.

Newly generated GNN feature records contain `model_status: "untrained_feature_export"`, node features, candidate `edge_index`, and empty `relationship_predictions`. Older cached records may contain random latent `edge_feature_tensor` values from the former exporter; they are not predictions and should be regenerated with `--overwrite` if used for research.

## Evaluation schema

`evaluation/queries.json` stores query ID, category, text, one or more stable source-video/time relevance windows, tolerance seconds, and provenance. `python -m vidquery evaluate` writes aggregate and per-query metrics plus the returned source IDs, times, and scores to JSON. Source filenames—not database UUIDs—make the checked-in labels stable across imports.
