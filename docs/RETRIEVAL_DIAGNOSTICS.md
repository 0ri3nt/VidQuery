# Diagnosing action and relationship retrieval

When a query such as "person talking to another person" or "person typing on a
laptop" returns nothing, or the wrong moment, the cause can be anywhere between
the detector and the ranker. `diagnose-query` traces one query through every
stage and classifies the failure so it can be fixed at the right layer rather
than by lowering thresholds.

## Running it

```bash
python -m vidquery diagnose-query "person talking to another person" \
  --video-id VIDEO_UUID
python -m vidquery diagnose-query "find drive" --json --no-planner
```

| Flag | Meaning |
|---|---|
| `--video-id` | restrict to one video; repeat for several; omit for the whole library |
| `--limit` | number of ranked results per ranking mode (default 5) |
| `--json` | machine-readable report |
| `--no-planner` | deterministic parser only; skip the Groq planner |

## What the report contains

- **Deterministic plan**: intent, entities, actions, relationship tuples, and
  spoken or OCR terms the local parser produced.
- **Hypotheses**: any alternative interpretations (see [REVIEW2.md](REVIEW2.md)).
- **Planner status**: whether the Groq planner was used.
- **Index inventory**: every action label stored for the selected videos with
  counts and maximum confidence, and every stored relationship predicate by
  source method.
- **Raw hits**: segments whose stored actions or relationships match the plan,
  before any ranking.
- **Ranking modes**: result counts and top results under both `hybrid` and
  `reliability` ranking.
- **Failure categories** and a one-line summary.

## Failure taxonomy

| Code | Stage | Meaning | Typical fix |
|---|---|---|---|
| `A_DETECTION_FAILURE` | model or indexing | the requested action or relation was never stored | check `/api/health`, reprocess, or accept a model gap |
| `B_STORAGE_OR_D_RETRIEVAL_FAILURE` | storage or filter | the label exists in the inventory but no segment matched | inspect segment payloads and the plan's filters |
| `C_QUERY_UNDERSTANDING_FAILURE` | parser | the query collapsed to transcript terms | add an alias in `ACTION_ALIASES` or `ENTITY_ALIASES` in `vidquery/search.py` |
| `E_RANKING_OR_F_RELIABILITY_FAILURE` | ranking | raw evidence exists but ranked search returns nothing | check score floors and constraint logic |
| `F_RELIABILITY_FAILURE` | reliability | hybrid returns hits but reliability ranking does not | review priors; do not lower them for one video |
| `OK_OR_G_LOCALIZATION_CHECK` | localization | evidence was retrieved | verify peak timestamps against manual ground truth |

## Case study: a custom phone video

A 72-second phone recording returned nothing for action and relationship
queries. The diagnostic showed:

1. **Detection/indexing (A) was dominant.** The first index had no GNN actions
   and no GNN relationships because the AVA and VidOR feature extractors could
   not load their ResNet backbone. The project `TORCH_HOME`
   (`data/app/cache/torch`) was empty, and the download failed on macOS Python
   SSL. Whisper and OpenCLIP had the same download problem.
2. **Query understanding (C) was second.** "typing" and "interacting" were not
   aliased, so they became spoken terms. The parser now maps `typing` to the
   AVA action `work on a computer` and `interacting` to the `talk to` action.
3. **Remaining gaps are model limits.** After caching the weights and
   reprocessing, the index held transcript words, speakers, OCR, appearance
   embeddings, MiniLM vectors, 467 action predictions, and 489 GNN
   relationships. The AVA GNN still never predicted `drink` or
   `work on a computer` on this footage. That is a genuine domain-shift gap, and
   reliability ranking correctly left strong matches in place.

The lesson: check the index inventory before touching ranking. A missing label
in the inventory is never fixed by changing thresholds.

## Related

- [SETUP.md](SETUP.md): model caches and the macOS SSL and Whisper notes that
  caused the indexing failure above.
- [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md): what the action and relation
  models can and cannot detect.
