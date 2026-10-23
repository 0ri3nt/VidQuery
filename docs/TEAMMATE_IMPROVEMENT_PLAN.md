# VidQuery improvement plan

This is the work plan for taking VidQuery from a working capstone to a demo
that survives examiners' live queries and a paper that survives review. It is
written to be followed phase by phase, with an AI coding agent doing most of
the editing. Read the whole file once before starting.

Background reading, in this order: [../README.md](../README.md),
[ARCHITECTURE.md](ARCHITECTURE.md), [REVIEW2.md](REVIEW2.md),
[GNN_STATUS.md](GNN_STATUS.md), and [RUN_AND_TEST.md](RUN_AND_TEST.md).

## Why this plan exists

An audit in October 2026 found the following.

1. **The search logic is good; the perception is the weak link.** Speech
   queries land on the exact word, and the system correctly refuses trick
   questions. But Whisper `tiny` mishears technical terms ("binary search tree"
   became "Vandri search tree", "Neo4j" became "4.js"), so those queries cannot
   match.
2. **The headline benchmark is not end-to-end.** The README's P@1 0.96 was
   measured on an index whose evidence mostly came from labels, not models: the
   synthetic `final_demo` video had its transcript, objects, and relations
   inserted from `demo/final_demo_manifest.json`, and the AVA clip
   `hbYvDvJrpNk` was indexed with official AVA human action labels. The only
   fully model-driven result is P@1 0.50 on 14 action queries. This must not be
   presented as end-to-end accuracy.
3. **The Review-2 benchmark (timestamps, ambiguity, abstention) has no data
   yet.** The annotation page that creates it is now on `main`.
4. **The Groq planner was failing on every call** (HTTP 400: the `gpt-oss`
   reasoning model ran out of tokens). This is fixed on `main` by sending
   `reasoning_effort: low`. The free tier is still limited to 8,000 tokens per
   minute, and one planner call uses about 2,900.
5. **Research position.** VidQuery cannot beat state-of-the-art accuracy (AVA
   action detection is about 40–46 mAP; ours is 8.7). Its defensible novelty is
   **reliability-aware abstention for a modular multimodal search system**:
   weighting each perception model's evidence by how trustworthy that model is,
   searching seen, said, and written interpretations of ambiguous queries
   separately, and refusing rather than answering confidently and wrongly. Today
   the trust values are hand-picked constants. Learning them from data is the
   core research contribution of this plan (Phase 3).

## Ground rules: do not break the code

Give these rules to your AI coding agent at the start of every session.

1. **Work on a branch per phase** (`phase-1-demo-hardening`, and so on). Merge
   to `main` only when every check passes. Never force-push `main`.
2. **Record the baseline first.** Before any change, run the checks below and
   save the output. On `main` at the time of writing: `185 passed, 2 skipped`,
   ruff clean, and mypy with 13 pre-existing errors (missing library stubs and
   unused `type: ignore` comments). A change must not reduce passing tests or
   add new mypy errors.

   ```bash
   python -m pytest -q
   python -m ruff check .
   python -m mypy vidquery scripts
   node --check frontend/assets/app.js
   node --check frontend/assets/annotate.js
   ```

3. **New behaviour goes behind a setting whose default keeps today's
   behaviour.** Add the setting to `vidquery/config.py` and `.env.example`.
   `SearchRequest` defaults intentionally reproduce the Review-1 ranking; do not
   change them.
4. **Every change gets a test** in `tests/`. Use the existing fixtures in
   `tests/conftest.py` (`settings_factory` keeps all files in a temporary
   directory). Groq must be mocked in tests; never spend API quota in tests.
5. **Never commit** `.env`, anything under `data/app/` (models, uploads,
   SQLite), raw datasets, or virtual environments. They are gitignored; keep it
   that way.
6. **Never tune on the test split.** Anything learned or chosen (priors,
   thresholds, model sizes) is chosen on `calibration` and reported once on
   `test`.
7. **Do not edit frozen evidence files.** Leave `evaluation/results/*.json`,
   `evaluation/queries_independent.json`, and the trained checkpoints untouched.
   Write new results to new files.
8. **Keep the safety boundaries.** Groq output is never executed, only
   validated against allowlists. Neo4j queries stay static and parameterized.
9. **Small commits with clear messages**, one idea each.

## Phase 0: Set up and reproduce

Goal: the system runs on your laptop exactly as on `main`.

1. Clone and install (see [SETUP.md](SETUP.md); Windows commands are there):

   ```bash
   git clone https://github.com/0ri3nt/VidQuery.git && cd VidQuery
   python3.12 -m venv .venv && source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev,full]"
   cp .env.example .env
   ```

   With an NVIDIA GPU, install the CUDA build of `torch`, `torchvision`, and
   `torchaudio` from the same release before the line above. A mismatched
   `torchaudio` breaks Pyannote.

2. Fill in `.env`: `HUGGINGFACE_TOKEN` (the account must accept the
   `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` terms)
   and `GROQ_API_KEY`. For the first run set `ALLOW_MODEL_DOWNLOADS=true` and
   `HF_HUB_OFFLINE=0`.

3. **Copy the trained GNN checkpoints.** They are not in git and cannot be
   downloaded. Get these from a teammate's `data/app/models/` and place them at
   the same paths:

   - `data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.pt`
     and `.metadata.json`
   - `data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt`
     and `.metadata.json`

   Then cache the ResNet-18 backbone as shown in [SETUP.md](SETUP.md), section 3.

4. Start the server and check health. Every key should match the table in
   [RUN_AND_TEST.md](RUN_AND_TEST.md#2-check-health-before-every-session).

   ```bash
   python -m vidquery serve --host 127.0.0.1 --port 8000
   curl -s http://127.0.0.1:8000/api/health | python -m json.tool
   ```

5. Run the baseline checks from the ground rules and save the output.

**Done when:** health is all green and the checks match the baseline.

## Phase 1: Demo hardening

Goal: examiners' free-form queries work as often as possible. Each step is
independent. Do them in order and commit after each.

### 1.1 Better speech recognition (largest single improvement)

- Whisper size is set by `WHISPER_MODEL` (default `tiny`). Installed options
  include `tiny`, `base`, `small`, `medium`, `large-v3`, and `turbo`.
- Measure before choosing. Record a 2–3 minute clip with technical vocabulary
  (algorithm and database terms), write the true transcript by hand, and compare
  `tiny`, `small`, `medium`, and `turbo` on word error rate and processing time.
  Use a small script under `scripts/` that only reads the stored transcript;
  do not change the pipeline.
- Apply with `WHISPER_MODEL=<choice>` in `.env`, then
  `python -m vidquery refresh-library`. Existing segments are not rewritten
  until you reprocess.
- Whisper is forced to the CPU on Apple Silicon (MPS); that is expected. On
  CUDA it uses the GPU.
- Confirm that word timestamps still appear (`WHISPER_WORD_TIMESTAMPS=true`):
  speech results should show localization precision `word`.
- Do not change the default in `.env.example` without agreeing it with the
  team, because the 8 GB MacBook cannot run `medium` or larger comfortably.

### 1.2 Better object detection

- `YOLO_MODEL` points at `data/app/models/yolo/yolov8n.pt` (nano). Try
  `yolov8s.pt` or `yolov8m.pt` in the same folder. Always use a path inside
  `data/app/models/yolo/`, or Ultralytics downloads into the repository root.
- The class vocabulary stays the same 80 COCO classes. Do **not** switch to a
  model with different classes: the parser's entity aliases
  (`ENTITY_ALIASES` in `vidquery/search.py`) and the VidOR relation GNN's class
  features assume COCO.
- After reprocessing, confirm that `gnn_relationship_prediction` still produces
  relations (`diagnose-query "person near laptop" --video-id ...`).

### 1.3 Survive the Groq rate limit

The free tier allows 8,000 tokens per minute and one planner call costs about
2,900 tokens, mostly because the prompt in `QueryPlanningService` in
`vidquery/query_planner.py` sends every alias dictionary and allowlist. Options,
in order of safety:

1. **Cache plans per normalized query** (a small in-memory LRU cache in
   `QueryPlanningService`). Examiners repeating a query then cost nothing. Add a
   test with a mock transport asserting a second identical call makes no request.
2. **Shrink the prompt.** Send only the allowlist entries relevant to the
   query's tokens (the full lists are still used for validation). Re-run
   `python -m scripts.evaluate_query_planner` before and after; planner accuracy
   must not drop. Check its `--help` first.
3. If the RAG answer and planner compete for quota, consider disabling RAG
   during demos with `ENABLE_RAG_GENERATION=false`.

Never remove the local fallback: when Groq fails, search must still work.

### 1.4 Parser leftover words

`StructuredQueryParser.parse` in `vidquery/search.py` turns every word it does
not recognize into a spoken term. In "when does the student talk about operating
systems", the words `talk` and `about` became spoken terms, so a segment matching
only "about" was returned as a result.

- First add failing tests in `tests/test_search_and_storage.py` (or a new test
  file) for 10 to 15 realistic examiner phrasings, asserting the expected
  `spoken_terms`, `intent`, and entities.
- Then extend `STOPWORDS` or the speech-cue handling (`SPEECH_CUE_PATTERN`)
  minimally, so that question and filler words ("when", "does", "talk about",
  "mention", "say", "show me", "is there") are consumed.
- Be careful: `talk to` is also an action and relation (`talk_to`). Keep
  "person talking to another person" parsing as a relation; there are existing
  tests for it in `tests/test_action_relationship_query_fix.py`.

### 1.5 Prepare the demo videos

- 3–5 real recordings: clear speech with technical terms, slides with readable
  text, visible COCO objects (laptop, bottle, cell phone, cup, chair, book), and
  two people interacting. Rename the files before upload (for example
  `demo_os_lecture.mp4`).
- Optionally one AVA **test-split** clip for action queries. The split IDs are
  in `data/app/models/ava80_scaling/30/split_manifest.json` (referenced by
  `docs/GNN_STATUS.md`). That file is not in git; get it from whoever trained
  the model. `hbYvDvJrpNk` is a known test-split video. Do not use AVA training
  videos in any reported result.
- Run the examiner test list in [RUN_AND_TEST.md](RUN_AND_TEST.md#4-test-it-like-an-examiner)
  and keep a short list of queries that work well.

**Phase 1 done when:** all checks pass, and the examiner test list works on
every demo video with the chosen models.

## Phase 2: An honest end-to-end benchmark

Goal: numbers that come only from the models, on real videos and real queries.

1. **Index with models only.** Use real videos, not the synthetic fixtures, and
   do not apply `scripts/apply_demo_manifest.py` or any AVA ground-truth import
   to these videos.
2. **Assign splits** in the Annotate tab (`http://127.0.0.1:8000/#annotate`). A
   reasonable start is 2 calibration videos and 3 test videos, with different
   content in each split.
3. **Label about 60 queries per split** from the video itself, without running
   them in Search first. Targets per [REVIEW2.md](REVIEW2.md#labelling-ground-truth):
   at least 15 ambiguous, 15 timestamp, 10 action/relation, and 5 negative.
   - Make negatives of two kinds and say which in the notes field:
     `in-domain` (plausible but false, such as "person riding a bicycle" in a
     lecture) and `out-of-domain` (unrelated). This follows Moment of Untruth
     (link below).
   - Phrase queries the way an examiner would, not the way the parser likes.
   - If two people can label, have both label 15 shared queries and report how
     often they agree.
4. **Run the benchmark:**

   ```bash
   python -m vidquery evaluate-challenge --validate-only
   python -m vidquery evaluate-challenge --split calibration \
     --output evaluation/results/challenge-calibration.json
   python -m vidquery evaluate-challenge --split test \
     --output evaluation/results/challenge-test.json
   ```

   It already reports P@1, MRR, timestamp error, temporal IoU, intent accuracy,
   negative rejection, confident-wrong rate, and 95% bootstrap intervals.
5. **Commit** `evaluation/queries_challenge.json` and the new result files.
6. **Correct the README.** In "Measured results", state plainly that the
   Review-1 numbers were measured on labelled evidence, and add the new
   end-to-end numbers next to them. Do not delete the old numbers; label them.

**Done when:** `challenge-test.json` exists from a real manifest and the README
describes both benchmarks honestly.

## Phase 3: Learned reliability (the research contribution)

Goal: replace the hand-picked trust constants with values measured on the
calibration split, and show that this reduces confident-wrong answers.

### Where the code is

- `vidquery/reliability.py`: `SOURCE_RELIABILITY` (trust per source, for
  example `whisper_transcript` 0.90 and `gnn_action_model` 0.30),
  `QUERY_MODALITY_WEIGHTS`, `assess_evidence()`, which produces confidence and
  verdict, and the thresholds `DEFAULT_SUPPORTED_THRESHOLD` 0.45 and
  `DEFAULT_WEAK_THRESHOLD` 0.15.
- `vidquery/search.py`: `LocalHybridSearchEngine.search` calls
  `assess_evidence` (around line 1048) and `reliability_rank_score` (around line
  1100). `_score_detailed` emits one `RawContribution(modality, raw_score,
  source_method)` per piece of matched evidence.
- `vidquery/challenge_evaluation.py`: benchmark runner with splits and
  confident-wrong rate.
- `tests/test_reliability_and_hypotheses.py`: tests that pin the current
  ordering of priors. They must keep passing with the default configuration.

### Steps

1. **Collect calibration data.** Write `scripts/calibrate_reliability.py`. For
   each calibration query, run search with `ranking="reliability"`, and for each
   returned result record every contribution (`source_method`, `modality`,
   `raw_score`) together with whether that result is correct against the
   labelled moments, using the same matching rule `challenge_evaluation.py`
   uses. Reuse its helpers; do not write a second definition of "correct".
2. **Fit per-source reliability.** The simplest defensible estimator: for each
   `source_method`, the fraction of results with that evidence that were
   correct, smoothed toward the current constant when there is little data
   (for example `(hits + k * prior) / (n + k)` with `k = 5`). Keep the hand
   constant for any source with fewer than about 10 observations, and record
   that. Optionally, also fit the two verdict thresholds on calibration to hit a
   target confident-wrong rate.
3. **Write the result to a file**, for example
   `evaluation/results/reliability-priors.json`, containing the values, the
   number of observations per source, the manifest checksum, the split name,
   and the date.
4. **Load it behind a setting.** Add `RELIABILITY_PRIORS_PATH` (default empty)
   to `vidquery/config.py` and `.env.example`. When empty, behaviour is exactly
   as today. When set, `reliability_prior()` uses the file's values. Validate the
   file: known keys only, values in [0, 1], and a clear error otherwise. Thread
   the values through settings rather than mutating the module-level dictionary
   at import time, so tests stay isolated.
5. **Tests:** default behaviour is unchanged (the existing tests pass), the
   loader rejects bad files, and the fitter returns expected values on a small
   synthetic dataset.
6. **Ablations on the test split, run once at the end:**

   | Variant | How |
   |---|---|
   | Review-1 ranking (no reliability) | `ranking="hybrid"` |
   | Uniform trust | every source 0.5 |
   | Hand-picked trust | today's constants |
   | Learned trust | the calibration file |
   | Learned trust without hypotheses | hypotheses off |
   | Learned trust without localization | `localize=false` |

   If `evaluate-challenge` lacks a switch for a variant, add an optional CLI
   flag with a default that keeps today's behaviour.
7. **Report the right metrics.** Confident-wrong rate is the headline. Add to
   `challenge_evaluation.py`, if missing: a risk-coverage curve and its area
   (AURC), rejection F1, and AUROC for answerable versus unanswerable queries
   (as in Soccer-GMR, linked below). Include bootstrap intervals; with about 30
   test queries, small differences are not significant, so say so.

**The claim to aim for:** "At equal coverage, learned reliability reduces
confident-wrong answers compared with uniform and hand-picked trust." Report it
only if the test split supports it, including if it does not.

## Phase 4 (optional): Speech search that survives transcription errors

Goal: "binary search tree" should still find "Vandri search tree".

- Add a soft phonetic match used only when exact matching misses a query term,
  for example Double Metaphone codes or fuzzy matching on sound codes. Add the
  library to `pyproject.toml`.
- Put it behind `ENABLE_PHONETIC_MATCH=false` by default, and emit its own
  `source_method` (for example `phonetic_transcript`) with its own entry in
  `SOURCE_RELIABILITY`, so Phase 3 learns how much to trust it.
- Risk: fuzzy matching makes trick questions match by accident. Measure negative
  rejection and confident-wrong rate with it on and off.
- Related work: [phonetic embedding for ASR robustness (Interspeech 2022)](https://www.isca-archive.org/interspeech_2022/zhou22h_interspeech.html).

## Phase 5: Paper preparation

**Claims that are safe:**

- The architecture: a modular multimodal index (speech, speakers, OCR, objects,
  appearance, relations, actions) with per-source provenance.
- Reliability-weighted evidence fusion, separate retrieval of ambiguous
  interpretations, abstention, and peak-moment localization, all training-free
  over stored evidence.
- End-to-end numbers from Phase 2 and the ablations from Phase 3, with
  intervals.

**Claims that are not safe:**

- "0.96 accuracy" as end-to-end performance.
- Any comparison claiming better accuracy than state-of-the-art moment retrieval
  or action detection.
- Calling the verdict thresholds calibrated probabilities, unless Phase 3 fits
  and validates them.

**Related work to position against** (read and cite the originals):

- Negative queries and abstention: [Moment of Untruth, WACV 2025](https://arxiv.org/html/2502.08544);
  [Soccer-GMR, 2026](https://arxiv.org/html/2605.02623v1).
- Clarification and ambiguity: [ViReD](https://arxiv.org/html/2205.05739v3);
  [UMIVR, ICCV 2025](https://doi.org/10.1109/iccv51701.2025.02054);
  [FuzzySeek, IUI 2026](https://dlnext.acm.org/doi/10.1145/3742413.3789095).
- Modality routing: [Smart Routing, 2025](https://arxiv.org/html/2507.13374);
  [Unified Interactive Moment Retrieval, 2025](https://arxiv.org/html/2512.12935v1).
- Multimodal video RAG: [Video-RAG](https://github.com/Leon1207/Video-RAG-master);
  [AdaVideoRAG](https://arxiv.org/html/2506.13589);
  [SceneRAG](https://arxiv.org/html/2506.07600).
- Neuro-symbolic search with calibrated perception: [NSVS-TL, ECCV 2024](https://arxiv.org/html/2403.11021v3).
  This is the closest in philosophy; explain that VidQuery fuses many
  heterogeneous sources and handles cross-modal ambiguity.
- Reference accuracy: AVA v2.2 action detection is about 40–46 mAP and
  QVHighlights moment retrieval about 78 R1@0.5. These figures come from the
  SOTA2 aggregator; verify them in the original papers before citing.

Suitable venues are applied, demo, or application tracks (for example ACM
Multimedia demo, ICMR, or WACV applications) rather than main computer-vision
tracks that expect leaderboard wins.

## Out of scope without team agreement

- Deleting legacy code (`extraction/`, `modelling/`, `core/`, `retrieval/`,
  `api/`, `app/`, `main.py`, and the root launcher scripts). Parts are still
  imported by `tests/test_geometry_and_legacy.py` and
  `scripts/process_ava_selection.py`.
- Retraining the GNNs, or switching YOLO to a non-COCO vocabulary.
- Changing the `SearchRequest` defaults or the stored segment format.

## Checklist per phase

- [ ] Branch created from up-to-date `main`
- [ ] Baseline checks recorded
- [ ] Change behind a default-off setting, where it changes behaviour
- [ ] Tests added; `pytest`, `ruff`, and `mypy` show no regressions
- [ ] Health check is green and the examiner test list still works
- [ ] Docs updated (`.env.example`, the relevant file in `docs/`)
- [ ] Merged to `main`
