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
| 4a | Groq answers checked by citation ID only | claim checks plus verdict gate | **on** | scripted bad answers now rejected; real correct answers pass; correct answers over weak evidence are downgraded | **fixed and verified** (offline); live re-audit in round 2 covers 24 of 42 queries (see Round 2) |
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

**Live re-run:** done in round 2 for 24 of the 42 audit queries before the daily cap was reached again. No answer made a claim absent from its cited evidence. See [§Groq re-audit](#groq-re-audit).

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

## Round 2: the unresolved issues

The issues left open by round 1 were worked through on 27 October. Decision rules are in
[`round-2 preregistration`](../evaluation/results/query-quality-fixes2-2026-10-27/preregistration.md),
and results are in `evaluation/results/query-quality-fixes2-2026-10-27/`.

| Issue | What changed | Default | Evidence | Status |
|---|---|---|---|---|
| Speaker attribution | 1. A named speaker scopes the spoken words to that speaker's own utterances (scoring and localization). 2. A refresh no longer mixes labelled utterances with fresh diarization labels: labelled utterances win, overlapping model copies are dropped, and repeated refreshes stop accumulating copies. | **on** | Regression tests. On a production copy with `final_demo` re-merged: "SPEAKER_01 discusses Docker" goes from supported 0.687 (10–15 s) to weak 0.375; all 10 positive production queries unchanged. | **fixed and verified**; the production library still holds the old merged `final_demo` until it is refreshed (**awaiting approval**) |
| Conservative answer checks | Speech in a cited segment implies a person. Negated mentions ("is not cooking", "no laptop", "isn't drinking from a cup") are not treated as claims. | **on** | Regression tests; un-negated false claims are still rejected | **fixed and verified** |
| Groq budget | Planner request carries only aliases present in the query (12,430 → about 5,200 characters, about 3,100 → 1,300 tokens). Provider aliases are mapped to labels locally. Optional on-disk response cache keyed by the full request (`QUERY_PLAN_CACHE_PATH`). | prompt on; cache off | Regression tests (size bound, alias mapping, cache across restarts). Live: planner prompts measured at 1,157–1,174 tokens (median 1,163). Groq's rate limiter put a full-alias request at 3,011–3,060 tokens, including the 700-token completion allowance. Whether the full and reduced prompts give the same plans was not measured, because the cap was reached. | **fixed and verified**; plan equivalence **requires more evidence** |
| Live Groq answer quality | Live re-audit with the fixed code: all 14 production queries and 10 of the 28 demo queries before the daily token cap | — | No answer made a claim, time or object absent from its cited evidence. Every in-domain and out-of-domain negative was declined, except "SPEAKER_01 discusses Docker", which the speaker fix (above) corrects. All 10 production positives were answered; 2 of 7 demo positives were answered, the rest declined over weak evidence. Repeated queries gave identical plans and answers (4 of 4). See §Groq re-audit. | **verified on 24 of 42 queries**; the remaining 18 demo queries **require more evidence** (blocked by the daily cap) |
| Empty Groq refusal shown as an outage (found in the re-audit) | A refusal with an empty answer (`{"answer": "", "supported": false}`) is now a "not supported" answer, not `provider_unavailable` | **on** | Regression test built from the live response; an empty *supported* answer is still rejected | **fixed and verified** (offline); live confirmation blocked by the cap |
| Rejected Groq plan labelled "Groq unavailable" (found in the re-audit) | Not changed | — | 3 of 13 planner responses were rejected by validation (invalid speaker label ×2, unsupported action label ×1). The local parser was used correctly, but the status and UI notice say "Groq unavailable (rate limit or network)". An existing test pins this status. | **awaiting approval**: add a `fallback_invalid_plan` status and notice |
| Routing of visible words | `VISUAL_CONCEPT_ROUTING`: visible nouns and verbs outside the detector and action vocabularies become `visual_concepts` (no transcript requirement). Visible-concept-only queries keep a spoken reading. Groq plans are routed too. | **off** | Validation: P@1 −0.003 (T1) / +0.003 (T0), not significant. In strict mode, "supported" verdicts rise from 9 (44% correct) to 181 (16% correct), because the absent-transcript zero had been damping overconfident verdicts. In the demo profile with the verdict fix: no gain (AUROC −0.020, not significant). | implemented, **not recommended** |
| Contradictory evidence | `CONTRADICTION_EVIDENCE`: a required pose (sit, stand, walk, …) contradicted at every sampled moment by a different pose (confidence ≥ 0.6, at least 2 moments) adds an extra zero to the verdict | **off** | Fires on 42% of in-domain-negative vs 25% of positive segments for pose queries. Only 48 positives and 43 in-domain negatives involve a pose, so validation shows no measurable effect (AUROC +0.003, rejection 0). | implemented, **no measured benefit** |
| Indexer memory | 1. The OpenCLIP text-embedding cache is bounded (LRU, 2,048 texts); search encodes every query there, so it grew for the life of the server. 2. `scripts/index_videos_batch.py --release-gpu-cache` frees cached CUDA blocks after each video. | 1 **on**; 2 **off** | The 10.7 GB was **not reproduced** in a clean single-process run: committed memory plateaus at 7.5 GB once the models load, then grows 3–5 MB per video. `--release-gpu-cache` lowers the plateau to 4.6 GB but makes every video slower (+28% mean) and leaves the slope unchanged. See §Indexer memory. | cache bound **fixed and verified**; GPU-cache release opt-in, **awaiting approval** if wanted; residual 3–5 MB/video **requires more evidence** |
| YOLOv8s | No code change: this is an evidence question, not a defect | v8n | §5 | **decision needed**: retraining the GNNs on v8s boxes is the remaining experiment |

### Indexer memory

Each run indexes the first 20 validation videos into a scratch library in one process (the
production profile, `det-ava8-vidor4`), with nothing else using the GPU. Committed memory is
read after each video.

| Run | Committed after video 3 | After video 20 | Slope (videos 3–20) | Median s/video |
|---|---|---|---|---|
| Default | 7.50 GB | 7.59 GB | 3.1 MB/video | 8.1 |
| `--release-gpu-cache` | 4.64 GB | 4.73 GB | 4.7 MB/video | 9.9 |

- Models load during the first two videos (4.6 → 7.5 GB). After that the growth is a few MB per
  video, about 0.3–0.5 GB per 100 videos.
- Resident memory stays at 2.3 GB, and live Python objects grow by about 30 per video.
- With allocation tracing, the Python heap grew by 0.7 MB over 10 videos. The largest item was the
  appearance text cache (`vidquery/appearance.py`), which is now bounded. The rest of the slope is
  native memory and has not been attributed.
- Releasing the GPU cache made all 20 videos slower, so it is opt-in rather than the default.
- The 10.7 GB observed in round 1 happened while another evaluation shared the GPU. The likely
  cause is CUDA allocations spilling into system memory under contention, but that was not
  measured directly. `--release-gpu-cache` is the mitigation to try if it recurs.

Results: `index-memory-before-fix.json`, `index-memory-after-fix.json` and
`index-memory-traced.json`, produced by `run_scripts/index_memory_profile.py`.

### Groq re-audit

The re-audit used the same queries, libraries and settings as round 1: the production copy with
`.env` defaults, and the demo copy with the demo profile. Requests were paced at 6,000
tokens/minute. Token use was read from each response; no keys or headers were stored.

| Library | Queries answered live | Positives answered | Negatives declined | Claim-check issues |
|---|---|---|---|---|
| Production copy | 14 of 14 (plus 4 repeats) | 10 of 10 | 3 of 4 | 3 flags, all false positives (the speech verb "discusses" checked as a visual action) |
| Demo copy (OM2NF only) | 10 of 28 | 2 of 7 | 3 of 3 | 0 |

- **The only wrong answer:** "Find where SPEAKER_01 discusses Docker" was answered as supported.
  That happens because the production copy still has the triplicated `final_demo` utterances.
  On the copy with `final_demo` re-merged, the evidence is weak (0.375) and Groq declined. That
  refusal arrived as `{"answer": "", "supported": false}`, which the app reported as a provider
  failure; this is now fixed (see the table above).
- **Declined demo positives:** four had only weak top evidence, so declining follows the answer
  policy. One ("cooking at the stove") had a supported verdict, but the evidence passed to Groq
  shows a microwave and no cooking action, so it declined. That loses recall but is grounded.
- **Grounding:** "holding a cup" cited three supported segments, and every time it gave is inside
  them. One of those times (27 s) falls outside the labelled moments: a detector false positive,
  not an invented claim.
- **Determinism:** the 4 repeated production queries gave the same plan, top result and answer
  text.
- **Token use:** about 2.8k tokens per production query and up to 5.3k per demo query (planner
  plus answer). Answer prompts were 983–3,859 tokens.
- **Daily cap:** the demo run hit Groq's tokens-per-day limit after 10 queries ("Limit 200000, Used
  198773"); the round-1 audit from the previous night was still inside the rolling window. The
  remaining 18 demo queries and the full-vs-reduced prompt comparison need about 110k tokens.
  To resume once the window allows:

  ```bash
  bash evaluation/results/query-quality-fixes2-2026-10-27/run_scripts/groq_run.sh demo-rest
  bash evaluation/results/query-quality-fixes2-2026-10-27/run_scripts/groq_run.sh prompt
  ```

Records: `groq-production-records.json`, `groq-production-speakerfix-records.json`,
`groq-demo-records.json` (the queries after the cap are recorded as rate-limited) and
`groq-demo-records-completed.json`. Checks: `groq-production-checks.json`, `groq-demo-checks.json`.
The blocked prompt comparison is kept as `groq-prompt-equivalence-ratelimited.json`; every row in
it is a 429 fallback, not a measured plan difference.

## Failed or aborted runs (kept for the record)

- **First Charades replay:** printed nothing because output was filtered, and left a partial copy. Moved to the scratchpad and rerun.
- **First launch of the fitted-threshold evaluations:** a shell edit did not apply and the commands ran without their variables. All four exited at argument parsing (exit 2) **before any evaluation**, so no test result was computed. Logs set aside, wrapper fixed, relaunched.
- **One validation strict-mode run of the fitted configuration (exit 127):** stopped deliberately. Free disk fell to 6 GB as the YOLO indexer's committed memory grew to 10.7 GB and the pagefile expanded. Rerun after indexing.
- **Round 2, first production Groq launch (exit 127):** printed only an import warning and made no Groq calls; cause not determined. Relaunched unchanged, and it completed.
- **Round 2, one-query probe:** the planner call failed with a transient `URLError`, and the retry succeeded. A separate diagnostic request without the app's User-Agent got Cloudflare error 1010 (403). The app always sends its own User-Agent, so the app was not affected.
- **Round 2, demo Groq run and prompt comparison:** stopped by the tokens-per-day cap (HTTP 429) after 10 demo queries. Rate-limited records are kept in `groq-demo-records.json` and excluded from the analysis.

## Rollback

| Change | Rollback |
|---|---|
| Parsing, answer verification, fallback notices, test isolation | `git revert` of the fix commit; no data change |
| `UNMATCHED_CONSTRAINT_EVIDENCE`, fitted thresholds | unset the flag and `RELIABILITY_PRIORS_PATH`; no data change |
| `TRANSCRIPT_SPEECH_FILTER` | unset the flag. Indexes processed while it was on keep filtered transcripts: restore the database backup taken before reprocessing, or reprocess with the flag off |
| YOLOv8s (if approved) | see §5 |
| Round 2: speaker scoping, refresh merge, answer-check negation and speech-implies-person, reduced planner prompt, bounded text cache | `git revert` of the round-2 commit; no data change. A library refreshed with the new merge keeps its merged utterances: restore the database backup taken before the refresh |
| `QUERY_PLAN_CACHE_PATH`, `CONTRADICTION_EVIDENCE`, `VISUAL_CONCEPT_ROUTING`, `--release-gpu-cache` | unset or omit; no data change (delete the cache file if one was written) |

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

Round 2 scripts are in `evaluation/results/query-quality-fixes2-2026-10-27/run_scripts/`:

| Script | Purpose |
|---|---|
| `eval_run.sh`, `compare_runs.py` | routing and contradiction validation runs and paired comparisons |
| `index_memory_profile.py OUT N [--trace] [--release]` | per-video memory profile on a scratch library |
| `groq_run.sh production\|demo\|demo-rest\|prompt` | live Groq re-audit (`groq_reaudit.py`), token-paced and capped |
| `../../query-quality-audit-2026-10-27/run_scripts/groq_audit.py check` | round-1 answer checks applied to the round-2 records |
