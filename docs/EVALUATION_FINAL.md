# Final retrieval evaluation

Evidence date: 12 August 2026.

## Independent corpus

`evaluation/queries_independent.json` contains 62 video-scoped queries over five
videos: 52 positives and 10 negatives. Category counts are eight transcript,
eight visual entity, eight spatial relationship, ten AVA action, six speaker,
eight multimodal, four OCR, and ten negative queries.

The corpus contains one strict AVA action-model test video (`hbYvDvJrpNk`) and
four non-training fixtures (`final_demo`, `distractor_demo`, `ocr_demo`, and the
short user MP4 `She How many dates have we been to`). The executable independence
audit reports zero overlap with the AVA model's 24 training videos and zero
overlap with its three validation videos. The only split overlap is the intended
held-out test video.

Query intents and primary intervals were manually selected in
`scripts/prepare_independent_evaluation.py`, then validated against canonical
SQLite evidence. Controlled transcripts/entities/relations use the labelled
fixture manifest; speaker clusters use actual compulsory Pyannote output; OCR
queries use real EasyOCR output; AVA actions and held-out boxes use official
annotations. This remains a capstone evaluation, not an annotator-agreement or
broad-domain study.

## Reproduction

```powershell
python -m scripts.prepare_independent_evaluation
python -m scripts.validate_multivideo_evaluation `
  --manifest evaluation/queries_independent.json
python -m scripts.index_independent_evaluation
python -m vidquery evaluate `
  --dataset evaluation/queries_independent.json `
  --output evaluation/results/independent-final.json `
  --limit 5 --neo4j --gnn-actions
```

The semantic baseline requires a locally cached
`sentence-transformers/all-MiniLM-L6-v2`. Neo4j must be healthy. The learned
method refuses an invalid checkpoint and uses the configured validation-selected
AVA v5 model.

## Methods

| ID | Result key | Meaning |
|---|---|---|
| A | `transcript_lexical` | Transcript token overlap only. |
| B | `transcript_semantic` | Frozen MiniLM transcript embeddings. |
| C | `visual_entity` | Strict requested-entity retrieval. |
| D | `hybrid_without_relationship_tuples` | Multimodal ranker with exact tuple constraints removed. |
| E | `hybrid` | Exact tuple-aware local SQLite hybrid retrieval. |
| F | `exact_relationship_aware_graph` | Static parameterized Neo4j candidate traversal with SQLite hydration. |
| G | `learned_model_enhanced` | Actual v5 GNN person-action predictions on eligible held-out action queries. |

## Overall machine-generated results

Positive metrics use 52 queries except method G, whose declared supported subset
has 14 queries. P@5 always uses a denominator of five.

| Method | P@1 | P@5 | R@5 | MRR | timestamp accuracy | temporal IoU | latency ms | negative rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A transcript lexical | 0.4231 | 0.0885 | 0.3462 | 0.4231 | 0.4231 | 0.4231 | 98.4527 | 0.9000 |
| B transcript semantic | 0.5000 | 0.1885 | 0.4692 | 0.5000 | 0.5000 | 0.5000 | 9.0403 | 0.3000 |
| C visual entity | 0.3462 | 0.3500 | 0.3617 | 0.4109 | 0.5192 | 0.5192 | 93.1440 | 0.9000 |
| D hybrid, tuple ablated | **0.9615** | 0.5692 | 0.7706 | **0.9615** | **0.9615** | 0.9423 | 105.1575 | 0.9000 |
| E exact local hybrid | **0.9615** | **0.5769** | **0.7738** | **0.9615** | **0.9615** | 0.9423 | 106.7754 | 0.9000 |
| F exact Neo4j graph | 0.8654 | 0.5462 | 0.6776 | 0.8654 | 0.8654 | 0.8654 | 128.7383 | **1.0000** |
| G learned v5, supported subset | 0.5000 | 0.3714 | 0.2181 | 0.5000 | 0.5000 | 0.5000 | 523.2959 | n/a |

Method G covers 14/62 queries (22.58%), all on the strict held-out action video:
ten AVA-action and four multimodal queries. Its AVA-action slice has P@1/MRR
0.6000/0.6000; the multimodal slice has 0.2500/0.2500. It is an honest learned
inference result, not a superiority result.

## Exact relationship slice

| Method | spatial P@1 | spatial P@5 | spatial R@5 | spatial MRR | latency ms |
|---|---:|---:|---:|---:|---:|
| Tuple ablation | 1.0000 | 0.9500 | 0.6553 | 1.0000 | 118.8394 |
| Exact local tuple | 1.0000 | 1.0000 | 0.6757 | 1.0000 | 184.0075 |
| Exact Neo4j graph | 1.0000 | 1.0000 | 0.6757 | 1.0000 | 132.7082 |

Exact tuples improve P@5 and R@5 slightly on this slice, but P@1/MRR tie the
co-occurrence ablation. That is the safe conclusion; this corpus does not show a
P@1 superiority for graph tuples. Exact matching is still required for semantic
correctness because it rejects endpoints not connected by the requested edge.

## Interpretation

- The local hybrid has the strongest overall P@1/R@5 in this corpus. The Neo4j
  graph path is slower and has lower overall P@1 because its static plan is a
  strict candidate filter, but it perfectly rejects all ten negatives.
- The standalone MiniLM baseline beats lexical transcript retrieval overall here
  but rejects only 30% of negative queries. The production hybrid API now applies
  a configurable `SEMANTIC_MIN_SIMILARITY=0.20` no-result floor; calibrating the
  standalone baseline itself remains future work.
- Visual-only and transcript-only methods are evaluated on all categories, so
  their low overall numbers are expected and useful as modality ablations.
- OCR is exercised through the hybrid methods; the dedicated transcript-only
  and visual-only baselines correctly score zero for OCR queries.
- Full category and per-video slices, raw rankings, latency values, and every
  query result are stored in `evaluation/results/independent-final.json`.
- The earlier `queries_multivideo.json` / `multivideo-final.json` files are
  preserved as historical source-derived comparisons, not the final independent
  benchmark.
