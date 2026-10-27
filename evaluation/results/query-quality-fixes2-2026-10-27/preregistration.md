# Pre-registration, round 2 (written before any measurement of these changes), 2026-10-27

## Visual-concept routing (parser default)
- Change: words in VISUAL_CONCEPT_WORDS (vidquery/search.py), unless the query is speech- or
  OCR-cued, become `visual_concepts` instead of spoken terms. They keep the sentence-similarity
  component but no transcript requirement.
- Validation (150 videos), T1 and T0, vs the round-1 baselines
  (evaluation/results/query-quality-fixes-2026-10-27/validation-T{1,0}-baseline.json).
- Becomes the default only if, in both T1 and T0: P@1 and P@5 are not significantly lower and
  in-domain false acceptance is not significantly higher. Then exactly one frozen-test run
  (T1, T0) vs the round-1 test controls with the same rule. Otherwise it stays behind a flag.

## Contradiction evidence (CONTRADICTION_EVIDENCE, opt-in, requires UNMATCHED_CONSTRAINT_EVIDENCE)
- Rule: a required AVA pose action (bend/bow, crawl, crouch/kneel, dance, fall down, get up,
  jump/leap, lie/sleep, martial art, run/jog, sit, stand, swim, walk) that the segment lacks is
  contradicted when, at every sampled timestamp with pose predictions (at least 2), the most
  confident pose is a different one with confidence >= 0.6. A contradicted constraint adds one
  extra zero-valued action contribution to the verdict; ranking is unchanged.
- Validation T1 with the round-1 fitted thresholds, vs UNMATCHED_CONSTRAINT_EVIDENCE alone with
  the same thresholds. One frozen-test run only if in-domain false acceptance is significantly
  lower with no significant P@1 loss.

## Amendment (after the routing validation, before the runs below)
- Routing met the registered criteria on validation, but an unregistered metric showed harm in
  strict mode: "supported" verdicts 9 (44% correct) -> 181 (16% correct). It is therefore NOT made
  the default; it becomes the opt-in VISUAL_CONCEPT_ROUTING, and no frozen-test run is made for it.
- Additional validation-only comparison (no test run without approval): the demo-profile
  combination VISUAL_CONCEPT_ROUTING + UNMATCHED_CONSTRAINT_EVIDENCE + round-1 fitted thresholds,
  T1, vs UNMATCHED_CONSTRAINT_EVIDENCE + fitted thresholds.

## Note (after the runs): measurements without a registered decision rule
- The indexer memory profiles and the live Groq re-audit are descriptive measurements; no
  decision rule was registered for them before they ran.
- The per-video GPU-cache release was added to scripts/index_videos_batch.py before its profile
  and made opt-in (--release-gpu-cache) after it, because it slowed every video. This keeps the
  previous default rather than promoting a change.
- The empty-refusal handling in vidquery/rag.py was written after the live audit observed the
  response; it is supported by a regression test built from that response, not by a held-out
  measurement.
