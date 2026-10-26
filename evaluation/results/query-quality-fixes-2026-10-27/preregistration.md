# Pre-registration (written before any measurement of these fixes), 2026-10-27

## Verdict fix (UNMATCHED_CONSTRAINT_EVIDENCE)
- Variant: unmatched parsed entities/actions/relations add zero evidence to the verdict;
  ranking uses the confidence without them (candidate order and scores unchanged).
- Thresholds are fitted on validation only (review2_full|expand:10, selected settings T1):
  - weak threshold: existing `scripts/fit_moment_abstention.py --objective p1_preserving
    --max-p1-loss 0.01` (highest negative rejection with P@1 within 0.01 of no abstention);
  - supported threshold: lowest confidence at which answered validation queries
    (positives counted correct when top-1 IoU >= 0.5; any answered negative wrong)
    reach precision >= 0.5 with at least 20 answered queries; 1.0 if none does.
- One frozen-test confirmation of exactly this configuration, compared with the existing
  integrated test result. Promote only if test P@1 is not significantly lower and
  false acceptance of in-domain negatives is significantly lower.

## Speech filter (TRANSCRIPT_SPEECH_FILTER)
- Rule: keep an utterance iff it has a letter/digit, zlib compression ratio <= 2.4, and,
  when diarization completed, it overlaps diarized speech (speaker != UNKNOWN).
- Representative clips: labelled real speech (final_demo), real movie dialogue (AVA
  excerpts), digital silence, low room tone, pink noise, speech mixed with noise at
  10 dB and 0 dB SNR.
- Downstream: replay the rule on copies of the validation index and the production
  library; evaluate T0 and T1 on validation and the documented production queries.
  No frozen-test run for this fix unless validation shows a significant effect.

## YOLOv8s
- All 150 validation videos, both detectors indexed with the same profile and code.
- Promotion candidate only if validation strict-mode (T0) P@1 gain is significant;
  then exactly one frozen-test confirmation (v8s test index vs existing v8n test index).

## Amendment (2026-10-27, before any verdict-fix result existed)
- Thresholds set through RELIABILITY_PRIORS_PATH apply in every search mode, so the
  validation check and the single frozen-test confirmation report both T1 (selected
  settings, where the unmatched-constraint fix acts) and T0 (strict production default,
  where only the new thresholds act). Both runs use exactly the fitted configuration.
- The supported threshold is fitted on each query's top-1 result confidence (the verdict
  shown on that result); the weak threshold on the maximum confidence of the top 5
  (the system abstains only when every returned result is below weak).
queued 2026-10-26T21:23:38Z: verdict configuration = UNMATCHED_CONSTRAINT_EVIDENCE=true + evaluation/configs/verdict_thresholds_unmatched_2026-10-27.json (weak 0.155, supported 1.0); validation T1/T0 then the single test confirmation T1/T0
control added 2026-10-26T22:04:27Z, after the T1 test confirmation and before reading it in detail: one test run of the current default configuration (flag off, default thresholds, T1 and T0) because the code changed (parsing fix) since the integrated baseline; it selects nothing.
