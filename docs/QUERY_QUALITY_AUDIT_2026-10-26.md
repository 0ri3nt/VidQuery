# Query quality audit — 26–27 October 2026

This audit covers three areas: the YOLO detector, the Groq explanations and query plans, and
retrieval and scoring. Every number comes from a controlled run listed under "Reproduce".

The following were not changed:

- production code and defaults;
- checkpoints, the frozen test set, and existing result files.

All fixes below were measured in-process, by monkeypatching inside the audit scripts. **None is
applied.** Each needs your approval.

## Summary

| # | Finding | Status | Measured effect | Recommendation |
|---|---|---|---|---|
| 1 | YOLOv8s beats the current YOLOv8n in the production default (strict) settings | confirmed on 50 validation videos | P@1 0.111 → 0.239, **+0.128 [0.041, 0.203]**; +1 s indexing per video | confirm on all 150 validation videos, then one test run, then decide |
| 2 | Larger or newer detectors (v8m, YOLO11 s/m, YOLO26 s) bring no measurable gain | measured | all differences not significant; YOLO11/26 index 40–50% slower | don't adopt |
| 3 | Verdicts ignore unmatched parts of the query; matching more of the query can *lower* confidence | **bug, confirmed** | "supported" was right 17–23% of the time; a penalty fix raises in-domain rejection +0.08 on test | fix verdict fusion (see 3.2) after threshold refit |
| 4 | Whisper hallucinates on silent clips; the hallucinations are indexed as speech | **bug, confirmed** | 56% of Charades segments carry hallucinated text; removing it gives P@1 +0.026 [0.000, 0.052] (upper bound) | gate transcripts with speech activity (VAD/diarization) |
| 5 | Function words and unknown nouns become "spoken terms"; 82% of queries are half speech searches | **bug, confirmed** | no measurable retrieval effect (P@1 0.000 [−0.009, 0.008]) | fix for correctness; low priority |
| 6 | The grounded-answer validator checks citation IDs only | **gap, confirmed** | accepts wrong times, unsupported actions, and "supported" answers over insufficient evidence | add claim checks (3.3) |
| 7 | The Groq free tier is exhausted after about 60 planner calls a day; the UI then falls back silently | **confirmed** | 77% of queries route to Groq; 71 of 84 audit searches hit the limit | shrink the planner prompt, cache plans on disk, show fallback status |
| 8 | Multi-reading results ignore reliability ranking | code inconsistency | 0 of 137 affected validation queries changed | low priority |

## 1. YOLO detector

### Method

The audit used the first 50 validation videos of `evaluation/charades_sta_navmr_benchmark.json`:

- 307 queries: 117 positives, 117 in-domain negatives and 73 out-of-domain negatives;
- the test split was never read.

For each detector, every video was fully re-indexed with the same `det-ava8-vidor4` profile and
only `YOLO_MODEL` changed. AVA v8, VidOR v4 and deterministic Whisper stayed the same.

Each index was evaluated twice with frozen queries and thresholds:

- **T0:** the production default (strict, no frame similarity);
- **T1:** the selected/demo settings (soft constraints, frame similarity).

Comparisons are paired video-bootstrap CIs (2,000 resamples, seed 2026) against a YOLOv8n
rebuild of the same videos. Rebuild noise, YOLOv8n rebuild against the original index:

- T0: exactly 0;
- T1: P@1 +0.009 [−0.018, 0.039].

Weights:

- **YOLOv8 n/s/m:** from `data/app/models/yolo/`;
- **YOLO11 n/s/m and YOLO26 s:** downloaded from the Ultralytics release assets to `data/app/benchmark/yolo_audit_2026-10-26/weights/`; not installed anywhere else.

### Results

| Detector | T0 P@1 | Δ vs v8n [95% CI] | T1 P@1 | Δ vs v8n | Index s/video | Detections/segment | Relations/segment |
|---|---:|---|---:|---|---:|---:|---:|
| YOLOv8n (current) | 0.111 | — | 0.308 | — | 9.6 | 13.9 | 58.8 |
| **YOLOv8s** | **0.239** | **+0.128 [0.041, 0.203]** | 0.359 | +0.051 [−0.041, 0.139] | 10.6 | 17.5 | 82.8 |
| YOLOv8m | 0.154 | +0.043 [−0.024, 0.107] | 0.299 | −0.009 [−0.129, 0.093] | 10.3 | 17.4 | 73.2 |
| YOLO11s | 0.162 | +0.051 [−0.010, 0.111] | 0.359 | +0.051 [−0.036, 0.124] | 13.3 | 14.9 | 59.8 |
| YOLO11m | 0.145 | +0.034 [−0.027, 0.092] | 0.316 | +0.009 [−0.090, 0.103] | 14.7 | 16.7 | 72.5 |
| YOLO26s | 0.128 | +0.017 [−0.052, 0.089] | 0.299 | −0.009 [−0.084, 0.070] | 14.6 | 15.5 | 67.1 |

- **No change on other metrics:** R@5@0.5 and in-domain rejection differences are not significant for any detector.
- **The v8s result survives multiple comparisons:** z ≈ 3.1, so it stays significant after a Bonferroni correction for 10 comparisons.
- **Why strict mode benefits:** strict mode drops any segment missing a parsed object, so detecting more cups, chairs and doors keeps more correct segments in play.
- **Detector latency is negligible:** 8–18 ms per frame for every model on the RTX 4060 (`detector_microbenchmark.json`), about 0.3–0.5 s per video. Indexing time is dominated by other stages; YOLO11/26 add 3–5 s per video in the full pipeline.
- **Distribution shift is included:** the AVA action GNN and VidOR relation GNN were trained on v8n detections. Changing the detector shifts their inputs, and these results include that effect.

**Limits:**

- 117 positives give wide intervals.
- The v8s gain in the demo settings (T1) is not significant.
- Retraining the GNNs on v8s boxes was not tried.

## 2. Groq explanations and query plans

### 2.1 Rate limits decide what users get

These numbers are from the account's response headers and a 429 error body:

| Limit | Value |
|---|---|
| Tokens per minute | 8,000 |
| Requests per day | 1,000 |
| **Tokens per day** | **200,000** (197,570 used when measured) |

- **Planner requests are large:** about 3,100 tokens each, because every request carries the full allowed vocabulary. So the planner alone exhausts the daily budget after about 60 calls.
- **Most Charades-style queries go to Groq:** the planner calls Groq for every query it considers ambiguous, which is 705 of 915 validation queries (77%).
- **Silent degradation:** when Groq returns 429, the planner falls back to the local parser (`fallback_provider_unavailable`) and the answer is dropped (`provider_unavailable`).
- **The UI never says so:** `frontend/assets/app.js` reads neither `query_planner_status` nor `rag_status`.
- **Plans don't persist:** they are cached only in process memory, so every server restart spends the budget again.

**Observed consequence:** the same query gives different results depending on quota. On the demo library, "person cooking at the stove." gave the same peak (37.99 s) either way:

- with a Groq plan: **supported, 0.456**;
- with the fallback: **weak, 0.325**.

That run is recorded in `docs/LIBRARIES.md`.

### 2.2 Answer audit

**Query set:**

- `groq_queries_demo.json`: 28 demo-library queries, with intervals annotated before any run;
- `groq_queries_production.json`: 14 `final_demo.mp4` queries with ground truth.

Each query ran twice with Groq (fresh app each time) and once without a key.

- **Coverage:** only **3 of 84** Groq-enabled searches got an answer. 71 hit the daily cap, 10 had no result above the evidence threshold, and every planner call failed. **A hallucination rate could not be measured**, and 3 answers are not a sample.
- **Results unaffected:** ranked results were identical with and without the answer, in all 42 query pairs.

| Query | Answer | Check |
|---|---|---|
| I2DHG "person holding a cup." | "Yes, the person is holding a cup." | correct (cup and carry/hold, inside the true 14–21.5 s interval), but states as certain what the system rated **weak** in all three cited results |
| final_demo "Find where architecture is discussed." | cites 0–5 s and 15–20 s and quotes both utterances | times and quotes exactly match the evidence and ground truth |
| final_demo "Find a person near a laptop." | "Yes, the evidence shows a person near a laptop." | correct |

**What the answer service does not check** (`vidquery/rag.py`, confirmed by the scripted-provider
test in `rag-validator-probe.json`):

| Provider answer | Accepted as supported? |
|---|---|
| valid citation, text says "at 00:42" for a 15–20 s segment | **yes** |
| valid citation, text says "cooking" with no cook action in the evidence | **yes** |
| "supported" while every result is `insufficient` and the interpretation is `insufficient_evidence` | **yes** |
| invented evidence ID | no (`rejected_invalid_evidence_ids`) |
| supported without citations | no |

Two more gaps:

- **Evidence selection:** results go to Groq by retrieval score ≥ 0.20, not by verdict.
- **Interpretation is hidden:** the system's interpretation is never sent to Groq or checked against the answer.

### 2.3 Recommended fixes (not applied)

- **Validate answer text before showing it:**
  - every time mentioned must lie inside a cited segment;
  - object and action words must appear in the cited evidence;
  - downgrade to unsupported when the interpretation is `insufficient_evidence` or every cited verdict is insufficient.

  `groq_audit.py check` implements these checks offline and can be reused.
- **Planner budget:**
  - send a shorter vocabulary, for example only the candidate labels near the query's words;
  - persist plans in a disk cache keyed by the normalized query;
  - skip Groq for queries that the deterministic parser already covers.
- **UI:** show "local planning — Groq unavailable" and the answer status, so users know which path produced the result.
- **Rerun the audit when the quota allows:** `groq_run.sh` is paced for the per-minute limit. The daily cap needs a fresh 24-hour window. About 170 calls are needed.

## 3. Retrieval and scoring

### 3.1 Query routing (`StructuredQueryParser`)

Validation results (`diagnose-validation-T1.json`, 915 queries):

- **Spoken terms everywhere:** 753 queries (82%) get "spoken terms", but only **1** of those terms appears in that video's transcript.
- **What they are:** most common are `its` (79), `their` (52), `into` (44), `takes`, `puts`, `through`, `off`, `floor`, `out`, `sneezing`, `cabinet`, `glass`, `shoes`, `door`, `closet` and `he`.
- **Cause:** the stopword list lacks common function words. Every word that is not a known entity, action or relation becomes a spoken term. Examples:
  - "he walks into the room." → spoken terms `he, into, room`;
  - "person opens the closet door." → `closet`.
- **Effect on intent:** 710 queries become `multimodal_search`. The verdict then expects transcript evidence (weight 1.0) that cannot exist.

Pre-registered fix tested (`EXTRA_STOPWORDS` in `scoring_audit.py`): P@1 0.000 [−0.009, 0.008] and rejection 0.000. It is a correctness fix with no measured retrieval benefit. Visual nouns outside COCO (door, closet, shelf, floor) still need a route: frame similarity, or a "visual concept" field.

### 3.2 Verdicts and confidence (`reliability.assess_evidence`)

**The bug:** confidence is a weighted average over the evidence that is present.

- With soft constraints, an unmatched action, object or relation emits no contribution, so it never lowers confidence.
- A matched action from the weak action model (prior 0.30) *does* enter the average.

**Controlled probes** (`scoring-probes.json`, synthetic segments):

| Soft constraints on | Segment | Query | Verdict |
|---|---|---|---|
| ✓ | person only (no cooking) | "person cooking." | **supported 0.728** |
| ✓ | person + action `cook` | "person cooking." | weak 0.386 |
| ✓ | person only (no laptop) | "person near a laptop." | **supported 0.728** |
| ✓ | person + laptop | "person near a laptop." | supported 0.684 |
| ✗ | person + action `cook` | "person cooking." | weak 0.386 (strict mode has the same inversion once the action matches) |

**On validation, demo settings:**

- "supported" was assigned to 50 positives and 53 in-domain negatives.
- 85 of those 103 had a top result missing every parsed action.
- 338 of 349 positive explanations say entity evidence (essentially "person") dominates.
- In-domain rejection is 0.6%.

**Variants** (validation; T1; paired video bootstrap):

| Variant | P@1 | In-domain rejected | Positives rejected | AUROC pos vs in-domain | "supported" (precision) |
|---|---:|---:|---:|---:|---|
| baseline (reproduces the original run exactly) | 0.350 | 0.6% | 0.6% | 0.594 | 103 (17%) |
| missing-constraint penalty | 0.341 (−0.009 [−0.022, 0.003]) | **8.0% (+0.075 [0.047, 0.105])** | 1.7% | **0.676 (+0.082 [0.043, 0.125])** | 9 |
| penalty on verdict only, ranking unchanged | 0.350 (0.000) | 1.4% (+0.009 [0.000, 0.020]) | 1.2% | **0.664 (+0.070 [0.033, 0.109])** | 9 |
| penalty + stopwords + no transcripts | 0.361 (+0.012 [−0.015, 0.039]) | 7.5% (+0.069) | 0.9% | 0.675 (+0.080) | 9 |

**Single pre-registered test confirmation** (penalty variant, 300 test videos; `test-confirmation-preregistration.txt`):

| Metric | Baseline | Penalty | Difference [95% CI] |
|---|---:|---:|---|
| In-domain rejected | 0.8% | 8.8% | **+0.080 [0.059, 0.103]** |
| "supported" precision | 23% (230) | 56% (27) | |
| P@1 | 0.404 | 0.386 | **−0.019 [−0.031, −0.007]** |
| Positives rejected | 0.9% | 3.2% | +0.023 [0.010, 0.040] |
| AUROC | 0.603 | 0.628 | +0.025 [−0.005, 0.056] |

The rejection gain replicates. On test it costs about 2 points of P@1, which validation did not show.

**Recommendation:**

1. Fix the fusion so unmatched parsed constraints count as zero evidence, applied to the verdict and confidence only. On validation, ranking is then unchanged and the confidence becomes more informative (AUROC +0.070).
2. Refit the weak and supported thresholds on validation for the new confidence. The current thresholds were fitted to the old one, which is why the verdict-only variant barely rejects more.
3. Confirm once on test before changing any default.

### 3.3 Hallucinated transcripts

1,642 of 2,935 Charades segments (56%) have non-empty transcripts. Most of these clips have no speech. The most frequent texts are Whisper silence hallucinations:

- "I'm going to do a little bit of a little bit of…" (102 segments);
- "1.5 cm." and its variants (over 300);
- "I'm going to make a video of the video." (59);
- "1.5mm", ",", "...", "I'm sorry.".

They are indexed as speech with a fixed reliability of 0.90 and feed the sentence-embedding evidence.

- **Whisper's own confidence does not separate them:** hallucinations have a *higher* median, 0.41 vs 0.32.
- **Speaker labels do not separate them either:** every utterance carries one.

Counterfactual without transcript evidence (an upper bound for a perfect speech-activity gate on clips without speech): P@1 +0.026 [0.000, 0.052], with no change in rejection.

**Recommendation:**

- keep only utterances that overlap pyannote speech turns or a VAD;
- drop high-compression repeats;
- re-index, then re-measure.

Real speech videos (lectures, `final_demo.mp4`) must keep their transcripts, so the gate needs its own check there.

### 3.4 Other checks

| Component | Result |
|---|---|
| Candidate fetching | all segments of the requested videos are scored. Neo4j only narrows candidates, and those are re-scored through SQLite. No defect found. |
| Ranking | `reliability` ranking = 0.5 × score + 0.5 × confidence, so the verdict bug also moves ranking. This is why the full penalty costs P@1 and the verdict-only variant does not. |
| Multi-reading queries (`fuse_hypotheses`) | ordered by raw score, not reliability, and not re-deduplicated across readings. On validation 137 queries take this path, with **0** order changes and **0** near-duplicate pairs. Code inconsistency only. |
| Thresholds | the weak threshold of 0.15 is below the confidence of a lone "person" detection (0.728), so in-domain negatives are almost never rejected in the demo settings. |
| Interpretation | `insufficient_evidence` only when every returned result is insufficient. Consistent with the API and the evaluator. |

## Reproduce

```bash
# 1. YOLO (validation subset; isolated indexes under data/app/benchmark/yolo_audit_2026-10-26/)
bash evaluation/results/yolo-audit-2026-10-26/run_scripts/yolo_run.sh
python evaluation/results/yolo-audit-2026-10-26/run_scripts/compare_yolo.py evaluation/results/yolo-audit-2026-10-26/comparison.json

# 2. Groq (needs daily quota) and the offline validator test
bash evaluation/results/query-quality-audit-2026-10-27/run_scripts/groq_run.sh
python evaluation/results/query-quality-audit-2026-10-27/run_scripts/groq_audit.py check --input <records...> --output groq-checks.json
PYTHONPATH=. python evaluation/results/query-quality-audit-2026-10-27/run_scripts/rag_validator_probe.py rag-validator-probe.json

# 3. Scoring: probes, diagnostics, variants (copy of the frozen integrated index)
PYTHONPATH=. python evaluation/results/query-quality-audit-2026-10-27/run_scripts/scoring_probes.py scoring-probes.json
bash evaluation/results/query-quality-audit-2026-10-27/run_scripts/variants_run.sh
python evaluation/results/query-quality-audit-2026-10-27/run_scripts/compare_variants.py \
  evaluation/charades_sta_navmr_benchmark.json validation compare-validation-T1-all.json baseline=... <variant>=...
```

Audit indexes and the database copies live under `data/app/benchmark/yolo_audit_2026-10-26/`
and `data/app/benchmark/query_audit_2026-10-27/`. They are local only; `data/app/` is not in git.
