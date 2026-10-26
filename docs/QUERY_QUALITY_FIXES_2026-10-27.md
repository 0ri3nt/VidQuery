# Query quality fixes — 27 October 2026

This document covers the fixes for the confirmed issues in
[QUERY_QUALITY_AUDIT_2026-10-26.md](QUERY_QUALITY_AUDIT_2026-10-26.md), and the YOLOv8s
validation. Every number comes from a run listed under "Reproduce". The decision rules were
written before measurement, in
[`preregistration.md`](../evaluation/results/query-quality-fixes-2026-10-27/preregistration.md).

**Production behaviour is unchanged by default.** Fixes that alter indexing, verdicts or the
detector are opt-in settings that default to the previous behaviour. Promoting them needs your
approval.

## Summary

| # | Issue | Change | Default | Evidence | Recommendation |
|---|---|---|---|---|---|
| 3 | Function words parsed as speech-search terms | stopword list extended | **on** | queries with a function-word spoken term 395 → 0 of 915; validation P@1 unchanged (0.3496 → 0.3496); test P@1 −0.001 (n.s.); 19/19 demo queries reproduce | **fixed and verified** |
| 4a | Groq answers checked by citation ID only | claim checks plus verdict gate | **on** | scripted bad answers now rejected; real correct answers pass; correct answers over weak evidence are downgraded | **fixed and verified** (offline); live Groq re-audit **unverified**, quota exhausted |
| 4b | Groq fallback silent; fallback cached | reason logged, provider failures not cached, UI notice | **on** | regression tests; real-browser check | **fixed and verified** |
| 4c | Test suite called the real Groq API | tests default to no key | **on** | no provider call in the suite | **fixed and verified** |
| 1 | Verdict ignores unmatched query constraints | `UNMATCHED_CONSTRAINT_EVIDENCE` plus fitted thresholds | off | inversion fixed (tests); ranking identical; test: in-domain false acceptance −2.9 pts [−4.2, −1.7], P@1 −0.002 (n.s.), AUROC +0.035 | **awaiting approval**, demo profile only |
| 2 | Whisper hallucinations indexed as speech | `TRANSCRIPT_SPEECH_FILTER` | off | silence/noise hallucinations 23 → 0 words; real speech kept (WER unchanged); Charades transcript words −95%; retrieval +0.012 P@1 (n.s.); production speech queries unchanged | **awaiting approval** (needs reprocessing) |
| 5 | YOLOv8s vs YOLOv8n | evaluation only; nothing changed | v8n | labelled recall +7 pts; 150-video strict P@1 +0.032 (n.s.) | **requires more evidence** |

## 3. Query parsing

The stopword list in `vidquery/search.py` now covers:

- pronouns: he, she, his, their, its, …
- prepositions and particles: into, out, off, through, …
- auxiliaries and fillers: has, been, start, begin, …
- the possessive remainder "s", as in "person's".

This is the word list fixed before the audit's measurement.

| Validation queries (915) | Before | After |
|---|---:|---:|
| with a function word as a spoken term | 395 | **0** |
| with any spoken term | 753 | 732 |
| T1 P@1 | 0.3496 | 0.3496 |
| Test P@1, T1 / T0 (control vs integrated) | 0.404 / 0.132 | 0.403 / 0.131 (−0.001, n.s.) |

**Regression tests:** `tests/test_query_quality_fixes.py`.

- Function words never become spoken terms, for example:
  - "he walks into the room." → `room` only;
  - "person's phone …" → no "s".
- Speech queries keep their content words:
  - "Find where SPEAKER_01 discusses architecture." → `architecture` plus `SPEAKER_01`;
  - "when does someone say deployment" → `deployment`;
  - "SQLite metadata" → `sqlite metadata`.

**Unresolved:** verbs the action lexicon misses (takes, puts) and nouns outside the detector vocabulary (floor, cabinet, door, closet) are still spoken terms. They could be genuine speech, so routing them needs its own design (for example, a visual-concept field scored by frame similarity).

## 4. Groq answers and fallback

### 4a. Answer verification (`vidquery/rag.py`)

Every answer Groq marks `supported` is checked against the results it cites:

| Check | On failure |
|---|---|
| Timestamps (mm:ss, "N s", "A–B s") inside a cited segment (±1 s) | rejected |
| Objects (project entity vocabulary) in cited entities, relations, appearance, transcript or OCR | rejected |
| Actions (project action lexicon) in cited actions or transcript | rejected |
| Speakers (`SPEAKER_nn`) among cited speakers | rejected |
| Quoted speech/text found verbatim in cited transcripts or OCR | rejected |
| Confidence or score numbers (0.xx or N%) equal to a cited value (±0.011) | rejected |
| Mentions speech or OCR while no cited result has any | rejected |
| System interpretation `insufficient_evidence`, or no cited result has verdict `supported` | **downgraded**: `supported=false`, with a deterministic summary of the real verdicts and closest moments |

- **New statuses:** `rejected_unverified_claims`, `downgraded_weak_evidence`, `downgraded_insufficient_evidence`.
- **Prompt:** the request now carries the system interpretation and tells the model to claim support only when a cited verdict is `supported`.
- **Authority:** retrieval and verdicts never depend on Groq. The answer can only be accepted, rejected or downgraded.

**Measured:**

| Case | Before | After |
|---|---|---|
| Scripted: valid citation, "drinks at 00:42" for a 15–20 s segment | accepted as supported | `rejected_unverified_claims` |
| Scripted: valid citation, "cooking" with no cook action | accepted | rejected |
| Scripted: supported over an `insufficient` result | accepted | rejected (unsupported action) or downgraded (`insufficient_evidence`) |
| Real Groq answer, "architecture … 0–5 s … 15–20 s" with an exact quote | accepted | accepted (`generated_supported`) |
| Real Groq answer, "person holding a cup" over three `weak` results | accepted as supported | `downgraded_weak_evidence`, citing I2DHG 00:15–00:20 |

All are regression tests.

**Unverified:** a live re-run of the Groq audit. The daily cap of 200,000 tokens was spent on 26 October, and the rolling 24-hour window had not reopened when this work ended. The harness, `evaluation/results/query-quality-audit-2026-10-27/run_scripts/groq_run.sh`, is ready to run once it does.

### 4b. Fallback visibility (`vidquery/query_planner.py`, `frontend/`)

- **Logs:** the planner logs the reason, for example `Constrained query planning unavailable (RAGProviderError: Groq request failed: HTTPError (HTTP 429)); using the local parser`.
- **Retry after recovery:** provider failures are no longer cached, so a query is planned by Groq again once the provider recovers. Rejected provider plans stay cached, so a bad answer is not paid for twice.
- **UI:** the search view shows a notice whenever:
  - the local planner was used;
  - no answer was generated;
  - an answer was rejected or downgraded.

  Downgraded answers show their closest moments, labelled as unconfirmed.
- **Browser check:** a real Chrome run against a server with an invalid Groq key showed "Local query planner used: Groq unavailable (rate limit or network)" and "No generated answer: Groq unavailable", and the server logged `HTTP 401`.

### 4c. Tests no longer spend Groq quota

`tests/conftest.py` picked up the developer's real `GROQ_API_KEY` from `.env`, so API tests sent live planner requests. The failing test log showed a real `HTTP 429`. Test settings now default to no key. A run of the RAG, API and annotation tests logs no provider call.

## 1. Verdict fusion (`UNMATCHED_CONSTRAINT_EVIDENCE`, default off)

**What changes when enabled:**

- With soft constraints, each parsed object, action or relation a segment lacks adds a zero-valued contribution (`unmatched_constraint`) to the verdict confidence.
- Ranking keeps the confidence without those contributions (`EvidenceAssessment.ranking_confidence`), so candidate order and scores are unchanged.
- In strict mode the flag has no effect, because a segment missing a constraint is already discarded.

**Controlled probes** (regression tests):

| Segment | Query | Before | After |
|---|---|---|---|
| person, no cooking | "person cooking." | supported 0.728 | lower than the cooking segment, not supported |
| person + cook action | "person cooking." | weak 0.386 | now above the person-only segment |
| ranking, all queries | — | — | identical order and scores |

**Contradictory evidence:** the index records evidence that is present, not explicit absence or contradiction. "Missing" is now counted. Positive contradiction (a confidently detected *different* action) is **not modelled** and remains open.

**Validation** (150 videos, demo settings T1; paired video bootstrap):

| Configuration | P@1 | P@5 (R@5@0.5) | R@1@0.7 | In-domain false acceptance | Positives falsely rejected | AUROC pos vs in-domain | "supported" |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.350 | 0.831 | 0.161 | 99.4% | 0.6% | 0.596 | 109 (18% correct) |
| flag on, default thresholds | 0.347 (−0.003 [−0.009, 0.000]) | 0.831 (0) | 0.161 (0) | 98.9% | 1.2% | **0.667 (+0.071 [0.037, 0.109])** | 9 |
| flag on, fitted thresholds | 0.347 (−0.003 [−0.009, 0.000]) | 0.831 (0) | 0.161 (0) | **97.7% (rejection +0.017 [0.005, 0.032])** | 1.4% (+0.009 [0.000, 0.023]) | 0.667 (+0.071 [0.035, 0.111]) | 0 |

**Thresholds, fitted on validation only** (`evaluation/configs/verdict_thresholds_unmatched_2026-10-27.json`):

- **Weak: 0.155** (current 0.15). This uses the existing P@1-preserving rule. Validation positives and in-domain negatives overlap so much that little extra rejection is possible without losing P@1.
- **Supported: 1.0.** No confidence level reaches 50% top-1 precision with at least 20 answered queries, so on Charades the calibrated system never says "supported".

  This threshold is global. Promoting it would also turn the correct, labelled `final_demo` speech results (confidence 0.65–0.93) from supported to weak. **Do not promote the fitted supported threshold as a global default.**

**Demo effect:** with the flag on and default thresholds, all 19 demo peaks are unchanged. Five verdicts change; three correct answers drop from supported to weak because the action model did not detect the action:

- I2DHG "drinking from a cup";
- I2DHG "sitting on the bed";
- GG4UR "watching television on the sofa".

**Frozen-test confirmation** (one run of exactly the pre-registered configuration, 300 test
videos; against the existing integrated test result):

| Test | P@1 | P@5 | R@1@0.7 | In-domain false acceptance | OOD false acceptance | Positives falsely rejected | AUROC | "supported" |
|---|---|---|---|---|---|---|---|---|
| T1 demo, baseline | 0.404 | 0.866 | 0.181 | 99.2% | 0.7% | 0.9% | 0.603 | 230 (23% correct) |
| T1 demo, fitted | 0.402 (−0.002 [−0.008, 0.003]) | 0.867 (+0.001 [0.000, 0.004]) | 0.182 (+0.001) | **96.3% (rejection +0.029 [0.017, 0.042])** | 0.0% | 1.3% (+0.004 [0.000, 0.008]) | **0.638 (+0.035 [0.009, 0.061])** | 0 |
| T0 strict, baseline | 0.132 | 0.366 | 0.056 | 39.8% | 0.8% | 49.6% | 0.563 | 38 (37% correct) |
| T0 strict, fitted | 0.131 (−0.001 [−0.004, 0.000]) | unchanged | unchanged | unchanged | unchanged | 49.8% | 0.567 | 0 |

**Pre-registered criterion met** for T1: P@1 not significantly lower, in-domain false acceptance
significantly lower. The control in §1-control separates the parsing change from this.

**§1-control.** The code changed after the integrated baseline (the parsing fix), so one test
run of the current default configuration (all flags off, default thresholds) was added. It was
recorded in the pre-registration and selects nothing.

| Test comparison | P@1 | In-domain rejection | Positives falsely rejected | AUROC |
|---|---|---|---|---|
| control vs integrated (parsing fix alone), T1 | −0.001 [−0.006, 0.004] | 0 | 0 | −0.000 |
| control vs integrated (parsing fix alone), T0 | −0.001 [−0.004, 0.000] (1 of 863 queries) | 0 | 0 | +0.004 [0.001, 0.007] |
| fitted vs control (verdict configuration alone), T1 | −0.001 [−0.004, 0.000] | **+0.029 [0.017, 0.042]** | +0.004 [0.000, 0.008] | **+0.035 [0.008, 0.063]** |
| fitted vs control, T0 | 0 | 0 | +0.002 [0.000, 0.007] | 0 |

The parsing fix has no measurable effect on test either. The verdict gains belong to the verdict configuration.

**Recommendation: awaiting your approval, and only for the soft-constraint (demo) profile.**

- **Benefit:**
  - in-domain false acceptance falls 2.9 points;
  - confidence separates true from false answers better (AUROC +0.035);
  - "supported" stops being wrong 77% of the time.
- **Cost:**
  - "supported" disappears on Charades-like content;
  - 3 correct demo answers become weak;
  - the gain is modest: 96% of in-domain negatives are still accepted.
- **Do not set the fitted thresholds as global defaults.** In strict production mode they only remove "supported", including from correct labelled speech results.
- **To roll out on the demo profile only**, add to its `run_with_profile` command:
  - `--set UNMATCHED_CONSTRAINT_EVIDENCE=true`;
  - `--set RELIABILITY_PRIORS_PATH=evaluation/configs/verdict_thresholds_unmatched_2026-10-27.json`.

  Re-record the 19 demo expectations afterwards, since 5 verdicts change.

## 2. Whisper speech filter (`TRANSCRIPT_SPEECH_FILTER`, default off)

An utterance is kept only if all three hold:

1. it has a letter or digit;
2. its zlib compression ratio is ≤ 2.4 (Whisper's own repetition threshold);
3. when diarization completed, it overlaps a diarized speech turn.

Without completed diarization, rule 3 is skipped. The outcome is recorded per segment as `processing_metadata.transcript_speech_filter`.

**Why diarization:** in the Charades index, 98.8% of pattern-matched hallucinations (1,024 of 1,036) overlap no diarized speech, against 63.7% of other utterances.

This corrects an audit statement. Utterances without overlap are labelled `"UNKNOWN"`, and the audit had counted that label as a speaker.

**Representative clips** (pipeline Whisper and pyannote; `speech-filter-clips.json`):

| Clip | Words before → after | WER before → after |
|---|---|---|
| labelled speech (`final_demo`) | 36 → 36 | 0.057 → 0.057 |
| + pink noise, 10 dB SNR | 39 → 39 | 0.229 → 0.229 |
| + pink noise, 0 dB SNR | 30 → 30 | 0.571 → 0.571 |
| movie dialogue, three 60 s AVA excerpts (en, en, it) | 202 → 202, 130 → 130, 146 → 143 | no reference |
| digital silence 30 s | 1 → 0 ("you") | — |
| room tone −50 dBFS 30 s | 22 → 0 ("1.5mm", "3.5mm", …) | — |
| pink noise −25 dBFS 30 s | 0 → 0 | — |

The only real speech dropped was one three-word Italian line ("Non ho detto.").

**Replay on index copies, before any rebuild** (`replay_speech_filter.py`):

| Index | Transcript words | Utterances dropped |
|---|---|---|
| Charades (450 videos) | 67,025 → 3,496 | 2,221 no diarized speech, 340 repetition, 133 no words; 1,304 of 1,642 speech segments emptied |
| Production library (13 videos) | 35,094 → 34,990 | 25; the 5 AVA films have no recorded diarization, so rule 3 is skipped |

**Retrieval** (validation, paired video bootstrap):

| Setting | P@1 | P@5 | R@1@0.7 | In-domain false acceptance |
|---|---|---|---|---|
| T1 demo | +0.012 [−0.009, 0.034] | +0.012 [0.000, 0.025] | −0.006 [−0.020, 0.008] | +0.003 |
| T0 strict | +0.012 [−0.009, 0.033] | +0.003 [0.000, 0.009] | +0.003 [−0.013, 0.020] | 0 |

**Production speech queries:** all 10 `final_demo` speech and visual queries stay correct, with identical top-1 windows.

As pre-registered, no test run, because the validation effect is not significant.

## 5. YOLOv8s vs YOLOv8n

**Setup:**

- **Videos:** all 150 validation videos (915 queries: 349 positives, 349 in-domain negatives, 217 out-of-domain).
- **Indexing:** the same `det-ava8-vidor4` profile and pipeline code, with only `YOLO_MODEL` changed. 50 videos per detector were reused from the audit's identical builds; 100 were added (`index_run.sh`).
- **Evaluation:** the same evaluator and code for both, paired video bootstrap.
- **Rebuild noise** (v8n rebuild vs original index, T0): P@1 +0.006 [−0.008, 0.024].

**Labelled detection quality** (VidOR held-out validation and test, 100 videos, 600 frames; class-aware IoU ≥ 0.5 matching as in the earlier 27.5% figure; `vidor-detection-quality.json`):

| | YOLOv8n | YOLOv8s | Difference [95% CI] |
|---|---:|---:|---|
| Object recall | 52.4% | 59.7% | **+7.3 pts [5.8, 8.7]** |
| Relation recall (both endpoints found) | 32.7% | 39.7% | **+7.0 pts [5.1, 8.8]** |
| Detections matching a labelled object (precision proxy; VidOR labels are incomplete) | 67.6% | 62.7% | −4.9 pts [−7.1, −2.8] |

**Retrieval** (validation, 150 videos):

| Setting | Metric | YOLOv8n | YOLOv8s | Difference [95% CI] |
|---|---|---:|---:|---|
| T0 strict (production default) | P@1 | 0.152 | 0.183 | +0.032 [−0.012, 0.075] |
| | P@5 | 0.433 | 0.456 | +0.023 [−0.009, 0.059] |
| | R@1@0.7 | 0.043 | 0.069 | +0.026 [0.000, 0.052] |
| | positives falsely rejected | 43.8% | 40.7% | **−0.032 [−0.061, −0.003]** |
| | in-domain false acceptance | 41.0% | 41.6% | +0.006 [−0.015, 0.029] |
| T1 demo | P@1 | 0.355 | 0.373 | +0.017 [−0.028, 0.064] |
| | P@5 | 0.840 | 0.860 | +0.020 [−0.005, 0.046] |
| | in-domain false acceptance | 99.4% | 100% | +0.006 |

**Cost:**

| | YOLOv8n | YOLOv8s |
|---|---|---|
| Weights | 6.5 MB | 22.6 MB |
| Detector time per frame (186 frames, RTX 4060) | 18.3 ms | 17.8 ms |
| Detector process peak RAM | 1.21 GB | 1.25 GB |
| Peak GPU memory | 82 MB | 97 MB |
| Full indexing, uncontended (50-video audit) | 9.6 s/video | 10.6 s/video |
| Index for 150 videos | 381 MB | 398 MB (database 69 → 86 MB) |

The 150-video indexing times are not comparable: v8n's new videos ran alongside other jobs.

**Decision, as pre-registered:** the 50-video audit's strict-mode gain (+0.128) does not hold on all 150 videos (+0.032, not significant). So **no frozen-test run and no promotion change** were made.

YOLOv8s detects more of the labelled objects and wrongly rejects fewer positives in strict mode, at a small cost. Retrieval gains are not established.

**Recommendation: requires more evidence.** Either:

- a larger strict-mode sample, for example adding the 300 test videos only after deciding to treat them as a new validation set; or
- retraining the action and relation GNNs on v8s boxes, since their inputs shift with the detector.

**If later approved, rollout is reversible:**

1. Back up the library database.
2. Set `YOLO_MODEL=data/app/models/yolo/yolov8s.pt` in `.env`.
3. Run `python -m vidquery refresh-library`.

To roll back, restore the backup and the previous `.env` line. The weights are already in `data/app/models/yolo/`.

## Unresolved issues

- **Live Groq answer quality is unmeasured.** The daily token cap was exhausted. The planner request alone is about 3,100 tokens, which allows about 60 planner calls a day. Shrinking the prompt and caching plans on disk are not done.
- **The answer checks are conservative.** "Someone says …" about a segment with no detected person is rejected, because "someone" maps to `person`. Negations ("the person is not cooking") are not parsed.
- **Contradictory evidence** (a confidently detected *different* action) does not lower confidence. Only missing evidence does.
- **Speaker attribution is per 5 s segment:** "Find where SPEAKER_01 discusses Docker." returns `supported`, but SPEAKER_00 says "Docker". Both speakers appear in that segment. This exists before and after the fixes.
- **Routing:** non-detector nouns (floor, door, closet) and unparsed verbs (takes, puts) are still spoken terms.
- **Indexer memory grows with every video.** The batch indexer's committed memory reached 10.7 GB after about 100 videos and pushed the pagefile into the disk (free space fell to 6 GB). Index long runs in batches (`index_videos_batch --limit`, fresh process per batch) until this is found.
- **YOLOv8s:** retrieval benefit not established (§5).

## Failed or aborted runs (kept for the record)

- **First Charades replay:** printed nothing because output was filtered, and left a partial copy. Moved to the scratchpad and rerun.
- **First launch of the fitted-threshold evaluations:** a shell edit did not apply and the commands ran without their variables. All four exited at argument parsing (exit 2) **before any evaluation**, so no test result was computed. Logs set aside, wrapper fixed, relaunched.
- **One validation strict-mode run of the fitted configuration (exit 127):** stopped deliberately. Free disk fell to 6 GB as the YOLO indexer's committed memory grew to 10.7 GB and the pagefile expanded. Rerun after indexing.

## Rollback

| Change | Rollback |
|---|---|
| Parsing, answer verification, fallback notices, test isolation | `git revert` of the fix commit; no data change |
| `UNMATCHED_CONSTRAINT_EVIDENCE`, fitted thresholds | unset the flag and `RELIABILITY_PRIORS_PATH`; no data change |
| `TRANSCRIPT_SPEECH_FILTER` | unset the flag. Indexes processed while it was on keep filtered transcripts: restore the database backup taken before reprocessing, or reprocess with the flag off |
| YOLOv8s (if approved) | see §5 |

Existing indexes, checkpoints, weights, benchmark splits and historical results were not modified. Audit copies are under `data/app/benchmark/{query_audit_2026-10-27,speech_filter_2026-10-27,yolo_val150_2026-10-27}/`.

## Reproduce

All scripts are in `evaluation/results/query-quality-fixes-2026-10-27/run_scripts/` and `evaluation/results/yolo-validation150-2026-10-27/run_scripts/`.

| Script | Purpose |
|---|---|
| `eval_run.sh NAME DB GENERATED T0\|T1 SPLIT [KEY=VALUE …]` | one evaluator run (optional `PRIORS=thresholds.json`) |
| `compare_runs.py MANIFEST SPLIT OUT name=result.json …` | paired comparison |
| `fit_verdict_thresholds.py VALIDATION_RESULT OUT` | pre-registered threshold fit |
| `speech_filter_clips.py WORK OUT` | representative clips |
| `replay_speech_filter.py SOURCE TARGET SUMMARY` | filter replay on an index copy |
| `production_queries.py QUERIES OUT name=db …` | production query comparison |
| `index_run.sh`, `vidor_detection_quality.py`, `detector_resources.py` (YOLO folder); `compare_runs.py` on the `validation-*-yolo150-*.json` results | YOLO |
