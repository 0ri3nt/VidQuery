# Review-2: localization, reliability, and ambiguity-aware retrieval

Review-1 ranked fixed five-second canonical segments and returned the segment
start as the answer timestamp. It trusted every matched modality equally and
picked one interpretation for an ambiguous query. Review-2 keeps the Review-1
ranking contract reproducible and adds four deterministic layers on top:

| Layer | Module | What it adds |
|---|---|---|
| Coarse-to-fine localization | `vidquery/localization.py` | a sub-segment `peak_time` and a tight interval inside each ranked segment |
| Reliability-aware evidence fusion | `vidquery/reliability.py` | a per-result confidence and a `supported` / `weak` / `insufficient` verdict |
| Ambiguity-aware hypotheses | `vidquery/hypotheses.py` | explicit alternative interpretations, per-interpretation support, and clarification prompts |
| Challenge benchmark | `vidquery/challenge_evaluation.py` | point-in-time ground truth, timestamp error, and abstention metrics |

None of these layers runs a model. They operate on evidence and timestamps the
pipeline already persisted, so they work on existing indexes. The only
indexing-time change is that Whisper now stores word timings
(`WHISPER_WORD_TIMESTAMPS=true`); videos indexed before that fall back to
utterance-level timing, which is labelled as such.

## Request controls

`POST /api/search` accepts four new optional fields. The defaults preserve the
Review-1 benchmark; the browser UI opts in to the full Review-2 behaviour.

| Field | Default | UI sends | Effect |
|---|---|---|---|
| `ranking` | `"hybrid"` | `"reliability"` | `reliability` re-ranks by `0.5 * score + 0.5 * evidence.confidence` |
| `localize` | `true` | `true` | attach `localization` to each result |
| `collapse_duplicate_evidence` | `false` | `true` | drop adjacent-window duplicates whose peaks are within 0.5 s |
| `hypothesis_id` | `null` | set when the user clicks a chip | pin one interpretation instead of fusing |

## Coarse-to-fine localization

For each ranked segment, the localizer collects candidate moments from the
evidence that actually matched the query plan:

- Whisper word timings for spoken terms (or proportional interpolation inside
  the utterance when word timings are absent);
- per-frame detector observations for visual entities and appearance matches;
- OCR observation intervals;
- action instance timestamps;
- matched relationship observations;
- speaker turns.

Each candidate carries a precision level. Lower rank means a tighter expected
timestamp error:

| Precision | Confidence constant | Meaning |
|---|---:|---|
| `word` | 0.95 | Whisper word boundary |
| `frame` | 0.85 | a sampled frame observation |
| `interval` | 0.75 | an observation interval, or the intersection of several modalities |
| `utterance_interpolated` | 0.55 | word position interpolated inside an utterance |
| `utterance` | 0.40 | whole utterance only |
| `segment` | 0.10 | nothing better than the five-second window |

Candidates are combined as follows. If their intervals overlap, the peak is the
tightest candidate inside the intersection (source suffixed `+agreement`,
confidence boosted 10%). If they overlap but no peak lies inside, the midpoint
of the intersection is used (`multimodal_intersection`). If modalities disagree
in time, the tightest candidate wins, confidence is multiplied by 0.7, and the
source is suffixed `+disagreement`. These constants are design values, not
calibrated probabilities.

## Reliability-aware evidence fusion

The rank `score` is unchanged. A separate `evidence` assessment is computed per
result:

```text
confidence = Σ (query_weight × raw_score × reliability_prior) / Σ query_weight
```

- `raw_score` is how well the stored evidence matched the plan for that
  modality.
- `reliability_prior` is a per-source constant derived from the measured
  results in [GNN_STATUS.md](GNN_STATUS.md) and
  [EVALUATION_FINAL.md](EVALUATION_FINAL.md). Examples: Whisper transcript
  0.90, YOLO and EasyOCR 0.80, Pyannote 0.75, geometry 0.65, MiniLM 0.60,
  OpenCLIP 0.55, VidOR relation GNN 0.35, AVA action GNN 0.30, action plus
  target resolver 0.25, manual annotation 1.00.
- `query_weight` is how central that modality is to the parsed intent. For
  example, `action_search` weights actions 1.0 and entities 0.6, while
  `transcript_search` weights actions 0.2.

The verdict is `supported` at `EVIDENCE_SUPPORTED_THRESHOLD` (0.45) or above,
`weak` at `EVIDENCE_WEAK_THRESHOLD` (0.15) or above, and `insufficient` below.
The explanation string names the dominant modality and flags any
low-reliability model output that was down-weighted.

The priors reflect the fact that the learned action and relation GNNs are weak
(held-out macro F1 of about 0.07 and 0.18). This does not hide GNN hits: a
strong GNN match still ranks, it simply cannot produce a `supported` verdict on
its own unless corroborating evidence is present. The thresholds should not be
lowered to make one video look better; see
[RETRIEVAL_DIAGNOSTICS.md](RETRIEVAL_DIAGNOSTICS.md) for how to tell a
reliability problem from an indexing problem.

## Ambiguity-aware hypotheses

`generate_hypotheses` enumerates validated `QueryPlan` interpretations only
when the query is genuinely ambiguous. Explicit cues such as "says", "mentions",
or "on screen" suppress it. Three patterns are recognised:

| Pattern | Example | Hypotheses (prior) |
|---|---|---|
| Bare action homonym | `find drive` | `visible_action` (0.5), `spoken_mention` (0.5) |
| Entity-only query | `find the laptop scene` | `visible_object` (0.6), `spoken_mention` (0.4) |
| Unclassified concept | `deployment` | `spoken_mention` (0.6), `visible_text` via OCR (0.4) |

Each hypothesis is retrieved separately. Support is measured from its results,
and the posterior is `prior × (0.05 + support)` normalised across hypotheses.

- If the best posterior leads the runner-up by at least
  `HYPOTHESIS_RESOLUTION_MARGIN` (0.25), or no runner-up has results, the query
  is `resolved` and results supporting the winner are ranked first.
- Otherwise the response is `clarification_suggested` with a
  `clarification_prompt`. Results are merged by score and each carries
  `supporting_hypotheses`.
- If every returned result is `insufficient`, or there are none, the response is
  `insufficient_evidence`. This is the system abstaining instead of returning a
  confident wrong timestamp.
- Passing `hypothesis_id` pins one interpretation.

The UI renders hypotheses as chips showing hit count and posterior percentage;
clicking a chip re-runs the search with that `hypothesis_id`.

## Response fields

Per result: `localization` (`peak_time`, `start_time`, `end_time`,
`precision`, `source`, `confidence`, `evidence`), `evidence` (`confidence`,
`verdict`, `contributions`, `explanation`), and `supporting_hypotheses`.

Per response: `hypotheses`, `interpretation` (`single`, `resolved`,
`clarification_suggested`, `insufficient_evidence`), `selected_hypothesis_id`,
and `clarification_prompt`.

The grounded Groq answer receives the localized timestamps, so its citations
point at the peak moment rather than the segment start.

## Challenge benchmark

The Review-1 benchmark labels relevance on the five-second grid, so it cannot
measure sub-segment timestamp error and is already near saturation
(P@1 0.96). The challenge benchmark uses a different manifest whose ground
truth is a `peak_time` plus a tight interval, with deliberately ambiguous,
timestamp-specific, and negative queries.

Three systems are compared on identical inputs:

| System | Behaviour |
|---|---|
| `review1_segment_start` | hybrid ranking, segment start as timestamp, alternatives merged by max score |
| `review2_localized` | Review-1 ranking plus localization (isolates the timestamp gain) |
| `review2_full` | hypotheses, reliability ranking, localization, duplicate collapse, abstention |

Metrics: P@1, Hit@5, MRR, top-1 timestamp error in seconds, within-tolerance
rate (default 2 s), temporal IoU, intent accuracy against
`expected_hypothesis_ids`, negative rejection, and confident-wrong rate.

```bash
python -m vidquery evaluate-challenge --validate-only \
  --dataset evaluation/queries_challenge.template.json
python -m vidquery evaluate-challenge \
  --dataset evaluation/queries_challenge.json \
  --output evaluation/results/challenge-latest.json
```

`--use-groq` allows the constrained planner; by default the run is fully
deterministic.

**Status:** the benchmark code and schema are implemented and unit-tested, and
`evaluation/queries_challenge.template.json` shows the format. A real manifest
needs at least 15 ambiguous, 15 timestamp, and 5 negative queries with manually
annotated peak times on indexed videos. That annotated manifest has not been
checked in, so there are **no measured Review-2 benchmark numbers yet**. Do not
report timestamp-error or abstention improvements until
`evaluation/results/challenge-latest.json` exists from a real manifest.

## Configuration

```dotenv
HYPOTHESIS_RESOLUTION_MARGIN=0.25
EVIDENCE_SUPPORTED_THRESHOLD=0.45
EVIDENCE_WEAK_THRESHOLD=0.15
WHISPER_WORD_TIMESTAMPS=true
```

## Tests

- `tests/test_localization.py`: precision ordering, agreement and disagreement,
  fallbacks, duplicate collapse.
- `tests/test_reliability_and_hypotheses.py`: priors, verdict thresholds,
  hypothesis generation, fusion, pinning, abstention.
- `tests/test_challenge_evaluation.py`: manifest validation and metrics.
- `tests/test_action_relationship_query_fix.py`: parser aliases for action and
  relationship queries.

## Limitations

- Localization can only be as precise as the stored evidence. At the default
  1 fps sampling, visual peaks have roughly one-second granularity.
- Reliability priors and precision confidences are documented design
  constants, not calibrated probabilities.
- Hypothesis generation covers three ambiguity patterns. Other ambiguities
  still go through the constrained Groq planner's `alternative_query_plans`.
- No measured benchmark improvement is claimed yet (see status above).
