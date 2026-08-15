# Live Neo4j verification

## Current full-pipeline refresh (12 August 2026)

The persisted Neo4j 5.26 Community service was restarted and reported healthy.
The freshly processed 25-second controlled demo is searchable through the
static parameterized Neo4j tuple plan with SQLite result hydration. Current
counts are five segments, 403 relationship-evidence nodes, 70
`relationship_gnn` instances, and 190 temporally smoothed instances. The
separate six-second OCR fixture has four distinct `OCRText` nodes.

`person in front of person` returns five exact learned tuple results through
both SQLite and Neo4j with the same first timestamp. All returned learned edges
have source/target track IDs; stale untracked learned rows are zero. The full
machine-readable proof is
`evaluation/results/full-fresh-video-e2e-proof.json` and can be regenerated with
`python -m scripts.capture_full_pipeline_proof` while the API and Neo4j are up.

## Final live snapshot (10 August 2026)

After the original two-video idempotency exercise, all five evaluation videos and the controlled demo were exercised against the same persistent service. The eight graph IDs in this development snapshot comprise five AVA videos, separate host/container registrations of the controlled demo, and one short derived distractor used for multiple-video QA. They are operational run artifacts, not eight independent evaluation videos. The final evidence snapshot in `evaluation/results/neo4j-live-final.json` reported:

| Node label | Count |
|---|---:|
| Video | 8 |
| Segment | 872 |
| EntityOccurrence | 6,726 |
| RelationshipEvidence | 45,047 |
| TranscriptSegment | 925 |
| Speaker | 83 |
| Action | 57 |

There were 13 uniqueness constraints and 22 indexes. Queries grouping by `video_id` and `segment_id` returned no duplicate keys. The controlled demo returned its stored manual `person near laptop` relationship IDs at 0–5 and 5–10 seconds through the live graph. Both named volumes survived a Neo4j/backend restart, both containers returned healthy, and the post-restart HTTP run still retrieved the same graph tuples.

## Verified environment

This verification was run locally on 2026-08-09, not inferred from Compose validation.

| Component | Verified version/state |
|---|---|
| Docker Engine | 29.2.1 |
| Docker Compose | 5.0.2 |
| Neo4j | `neo4j:5.26-community`, healthy |
| Neo4j Python driver | 6.1.0 |
| VidQuery backend | built from the repository and running |
| API/UI | `http://127.0.0.1:8000`, HTTP 200 |

The tested startup commands were:

```powershell
docker compose up -d neo4j
docker compose up -d --build backend
docker compose ps
```

The final `docker compose ps` showed both services running and Neo4j healthy on ports 7474 and 7687. `GET /api/health` returned SQLite `available`, Neo4j `available`, and the optional ML configuration states. The packaged frontend returned HTTP 200 and contained the expected search page.

## Schema

The application applied `database/schema.cypher` twice through `CanonicalNeo4jIndexer.apply_schema`. Both runs succeeded. Live `SHOW CONSTRAINTS` and `SHOW INDEXES` reported:

- 13 uniqueness constraints;
- 22 indexes in total, including constraint-backed and token lookup indexes;
- canonical constraints `segment_id_unique`, `detection_id_unique`, `transcript_segment_id_unique`, `speaker_key_unique`, `action_name_unique`, and `relationship_evidence_id_unique`;
- canonical range indexes `segment_video_time_idx`, `entity_occurrence_class_idx`, and `relationship_predicate_idx`.

The legacy schema remains present for research compatibility. Canonical application writes use `Video`, `Segment`, `EntityOccurrence`, `TranscriptSegment`, `Speaker`, `Action`, and `RelationshipEvidence`.

## Two-video ingestion and duplicate check

Two independent checked-in AVA examples were adapted into the SQLite system of record and mirrored to Neo4j:

| Source video | Canonical segments |
|---|---:|
| `-FaXLcSFjUI.mp4` | 154 |
| `-IELREHX_js.mp4` | 180 |

The graph was empty before this run. Counts after first ingestion were:

| Node label | Count |
|---|---:|
| Video | 2 |
| Segment | 334 |
| EntityOccurrence | 2,254 |
| RelationshipEvidence | 12,796 |
| TranscriptSegment | 302 |
| Speaker | 26 |
| Action | 40 |

| Edge type | Count |
|---|---:|
| HAS_SEGMENT | 334 |
| HAS_ENTITY | 2,254 |
| HAS_RELATIONSHIP | 12,796 |
| FROM_ENTITY | 12,796 |
| TO_ENTITY | 12,796 |
| HAS_TRANSCRIPT | 455 |
| HAS_SPEAKER | 232 |
| HAS_ACTION | 1,197 |

Both videos were then ingested a second time. Every node and edge count was identical (`idempotent: true`), proving that the uniqueness constraints and `MERGE` identities prevent duplicate canonical records and links for this corpus.

## Graph retrieval

`Neo4jGraphSearchEngine` is an explicit retrieval mode. It compiles a parsed `QueryPlan` through the source-owned `SafeCypherCompiler`, executes only that static parameterized statement, then hydrates and ranks the candidate canonical segments from SQLite. It never executes query text as Cypher.

An end-to-end request was tested through the containerized API:

```json
{
  "query": "person left of chair",
  "retrieval_backend": "neo4j",
  "limit": 3
}
```

The response declared `retrieval_backend: "neo4j"` and returned three graph-filtered results. The first was:

```json
{
  "video_title": "-IELREHX_js.mp4",
  "start_time": 945.0,
  "end_time": 950.0,
  "matched_relationship": {
    "relationship_id": "ava_rel_bf9d1e7983986dc0ab8b",
    "source_id": "ava_det_1ba1d975482f740d8702",
    "source_class": "person",
    "predicate": "left_of",
    "target_id": "ava_det_69b0d8918506b0b1f977",
    "target_class": "chair",
    "directionality": "directed",
    "confidence": 0.65,
    "source_method": "bounding_box_geometry",
    "timestamp": 945.0
  }
}
```

The live integration test also verified symmetric reversal (`chair near person` matched a stored person-to-chair `near` edge), directional rejection (`chair left_of person` did not match a stored person-to-chair `left_of` edge), endpoint IDs/classes, relationship timestamps, and segment start/end timestamps.

Saved parameterized examples are in `database/verification_queries.cypher`. The production graph search statement is owned by `SafeCypherCompiler`; neither file accepts generated Cypher.

## Health, restart, and persistence

Both containers were restarted with:

```powershell
docker compose restart neo4j backend
```

After restart:

- Neo4j returned to `healthy`;
- `/api/health` again reported SQLite and Neo4j `available`;
- the backend volume still contained both video records;
- the Neo4j volume still returned three exact graph results for the sample query;
- the first persisted result still began at 945.0 seconds.

## Storage and retrieval responsibilities

SQLite remains authoritative for upload deduplication, video/job state, canonical segment JSON, processing warnings, media paths, the default local search, result hydration, and playback metadata. A Neo4j outage never deletes or hides the SQLite record, and requests without `retrieval_backend: "neo4j"` continue to use SQLite.

Neo4j is an optional mirror used for explicit allowlisted graph candidate retrieval: entity, action, speaker, transcript-term, video, and exact subject-predicate-object constraints; relationship tuple projection; and segment/timestamp lookup. The graph mode then hydrates candidates from SQLite so public evidence uses the same canonical contract. If Neo4j mode is explicitly requested while disabled or unavailable, the API returns 503 rather than silently claiming graph retrieval.

## Reproducible test command

```powershell
python -m pytest -q tests/test_neo4j_integration.py
```

The test attempts the configured service, passes when Neo4j is available, and skips with an explicit reason when the password, driver, or service is unavailable. Its deterministic two-video records are removed after the test so they do not pollute the AVA verification counts.

## Improved GNN action proof

The validation-selected `ava80-gat-v3-roi` checkpoint was backfilled over the
held-out test video `hbYvDvJrpNk`, then mirrored into the live container twice.
Both ingestions produced identical counts:

| Record | Count |
|---|---:|
| Segment | 179 |
| EntityOccurrence | 3,568 |
| Action vocabulary nodes | 55 |
| RelationshipEvidence | 33,303 |

Parameterized traversal retrieved GNN-only `write` action evidence at
1,265–1,270 and 1,270–1,275 seconds. It also retrieved compatible-entity edges
whose `source_method` is `gnn_action_plus_target_resolver`. This proves that
learned person-action output and separately resolved target provenance survive
canonical serialization and Neo4j ingestion; it does not turn the resolver into
a learned relationship predictor.

Reproduce while Neo4j is healthy:

```powershell
python scripts/verify_ava_improved_neo4j.py
```

The complete counts, sample results, constraint/index totals, and boolean
verification fields are saved in
`evaluation/results/ava80-improved-neo4j-proof.json`.
