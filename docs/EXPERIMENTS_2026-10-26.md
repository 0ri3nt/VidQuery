# Experiments — 26 October 2026

Three experiments, each evaluated on the task its model is built for:
deterministic Whisper indexing (index reproducibility), AVA action
classification at about 100 videos, and VidOR relation classification at
2,000 videos. Charades-STA is not used to judge either model (see
[EVALUATION_CHARADES_STA.md](EVALUATION_CHARADES_STA.md); its 41.4% P@1
stays the moment-retrieval baseline). Nothing here changes production
defaults or promotes a model.

## 1. Deterministic Whisper indexing

Whisper now decodes greedily at temperature 0 (`WHISPER_TEMPERATURE_FALLBACK=false`,
the new default); `true` restores the sampling fallback. Segment metadata
records `whisper_decoding`.

| Check (`evaluation/results/whisper-determinism-2026-10-26/`) | Greedy, temperature 0 | Temperature fallback (old default) |
|---|---:|---:|
| 40 Charades clips × 3 repeats, identical transcripts | **40/40** | 17/40 |
| Same 40 clips, second process vs first | **40/40** | — |
| Two full pipeline rebuilds of 20 test videos, segments identical in every field (131 segments) | **131/131** | transcripts differ in 66, actions in 45, relations in 54 |

Existing benchmark indexes were built with the fallback and were not rebuilt.

## 2. AVA action classification: 24 → 94 training videos

### Protocol

- **Data.** 61 new AVA v2.2 videos chosen by the existing class-aware
  selector (registry videos excluded), 44.58 GB downloaded in 32 batches, all
  61 shards validated (`expansion-selection.json`; state in
  `data/app/features/ava80_shards/expansion_state_scale100.json`). One DNS
  outage stopped the run at 97/100; the resumable expansion retried the two
  affected videos with no data loss.
- **Split** (`evaluation/ava80_scale100_split.json`). Validation: the same 3
  calibration videos as the baseline. Test: the baseline's 3 held-out videos
  plus the registry's 3 untouched videos (`2PpxiG0WU18`, `NO2esmws190`,
  `uNT6HrrnqPU`), whose shards were built into a separate evaluation-only
  directory that only the test split can read. Train: the baseline's 24
  videos plus 70 others. No overlap; no registry video in train or
  validation (the builder and trainer refuse both).
- **Models.** One streaming trainer and one set of hyperparameters for every
  new model (AdamW, learning rate 3e-4, batch 32, up to 60 epochs, patience
  10, seed 2026), selected on validation macro F1 then mAP. The existing
  leakage-free baseline (`ava80-gat-v7-noleak-lr0.0003`, legacy trainer) and
  its MLP are included as trained.
- **Test.** Scored once, after `validation-decision.json` was written. Primary
  metric: macro F1 over supported classes, all 10,459 supervised test
  persons (70 classes present). Paired 95% CIs from two bootstraps sharing
  resamples across models: keyframes resampled within each video (captures
  within-video noise), and whole videos (6 videos; conservative).

### Results (`evaluation/results/ava80-scale100-2026-10-26/test-comparison.json`)

| Model | Train videos | Validation macro F1 | Test macro F1 | Test mAP | Test micro F1 |
|---|---:|---:|---:|---:|---:|
| Existing leakage-free baseline (GAT, legacy trainer) | 24 | 0.132 | 0.0731 | 0.0706 | 0.507 |
| Existing MLP baseline (legacy trainer) | 24 | 0.116 | 0.0751 | 0.0730 | 0.475 |
| GAT, streaming trainer | 24 | 0.123 | 0.0719 | 0.0738 | 0.488 |
| MLP, streaming trainer | 24 | 0.154 | 0.0682 | 0.0707 | 0.467 |
| **GAT, streaming trainer (validation choice)** | 94 | **0.164** | **0.0881** | **0.0890** | 0.500 |
| MLP, streaming trainer | 94 | 0.161 | 0.0864 | 0.0880 | 0.485 |
| Active v5 (leaking features; reference only) | 24 | 0.125 | 0.0711 | 0.0768 | 0.453 |

Paired differences, test, all persons. Each cell shows the keyframe CI
first and the video CI second; `*` marks a CI that excludes 0.

| Comparison | Macro F1 | mAP | Micro F1 |
|---|---|---|---|
| GAT 94 − GAT 24 (same trainer) | +0.016 [0.012, 0.021]* · [0.007, 0.027]* | +0.015 [0.012, 0.020]* · [0.008, 0.026]* | +0.012 [0.007, 0.018]* · [−0.008, 0.042] |
| MLP 94 − MLP 24 (same trainer) | +0.018 [0.015, 0.023]* · [0.009, 0.029]* | +0.017 [0.012, 0.028]* · [0.013, 0.028]* | +0.018 [0.013, 0.023]* · [−0.003, 0.045] |
| GAT 94 − existing baseline | +0.015 [0.011, 0.020]* · [0.008, 0.022]* | +0.018 [0.015, 0.023]* · [0.012, 0.028]* | −0.007 [−0.012, −0.002]* · [−0.024, 0.013] |
| MLP 94 − existing baseline | +0.013 [0.010, 0.018]* · [0.008, 0.025]* | +0.017 [0.011, 0.028]* · [0.011, 0.029]* | −0.022 [−0.027, −0.017]* · [−0.038, −0.006]* |
| MLP 94 − existing MLP baseline | +0.011 [0.008, 0.016]* · [0.008, 0.018]* | +0.015 [0.009, 0.026]* · [0.011, 0.027]* | +0.009 [0.004, 0.015]* · [−0.007, 0.027] |
| MLP 94 − GAT 94 | −0.002 [−0.006, 0.003] · [−0.005, 0.007] | −0.001 [−0.007, 0.008] · [−0.006, 0.004] | −0.015 [−0.020, −0.010]* · [−0.036, 0.004] |

Detector-only persons (10,100) give the same conclusions (in the result file).

Per video, test macro F1:

| Video | Test group | Persons | Existing baseline | GAT 24 | GAT 94 | MLP 24 | MLP 94 |
|---|---|---:|---:|---:|---:|---:|---:|
| hbYvDvJrpNk | base | 2,037 | 0.085 | 0.095 | 0.101 | 0.083 | 0.097 |
| t0V4drbYDnc | base | 1,694 | 0.099 | 0.109 | 0.128 | 0.102 | 0.150 |
| zR725veL-DI | base | 1,543 | 0.143 | 0.122 | 0.151 | 0.136 | 0.148 |
| 2PpxiG0WU18 | untouched | 1,079 | 0.162 | 0.138 | 0.187 | 0.120 | 0.197 |
| NO2esmws190 | untouched | 2,084 | 0.139 | 0.160 | 0.152 | 0.151 | 0.150 |
| uNT6HrrnqPU | untouched | 2,022 | 0.116 | 0.115 | 0.118 | 0.107 | 0.120 |

GAT 94 beats GAT 24 on 5 of 6 videos and the existing baseline on 6 of 6.
Mean AP rises on every video for both architectures.

### What this supports

- **More data helps, significantly.** Going from 24 to 94 training videos raises test macro F1 by about
  0.016–0.018 and mAP by about 0.015–0.017 for both architectures. These are
  relative gains of about 22–27%. The CIs exclude 0 under both bootstraps.
- **Against the existing baseline**, the 94-video models are significantly better on macro F1 and mAP.
  Micro F1 is slightly lower: the new models trade some accuracy on frequent
  classes for better rare-class recall.
- **GAT vs MLP at 94 videos**: no significant difference in macro F1 or mAP. The GAT's message
  passing still adds nothing measurable. Validation picked the GAT
  (0.164 vs 0.161).
- **Absolute quality stays low** (test macro F1 0.088). Validation (3 videos) overestimates test by about
  2×. No promotion is proposed until runtime integration and the AVA
  end-to-end benchmark are re-checked.

### Runtime and resources (measured)

| Step | Wall time | Peak Python RSS | Notes |
|---|---:|---:|---|
| Evaluation-only shards (3 videos, 1.31 GB) | 8.5 min | 2.0 GB | separate directory, 17 MB |
| Expansion (61 videos, 44.58 GB) | 209 min + 49 min after the DNS outage | 2.15 GB | 316 MB of new shards; free disk ≥ 20.0 GB (8 GB floor never reached) |
| Train MLP 24 / GAT 94 / MLP 94 (GPU) | 2.4 / 14.8 / 9.9 min | 1.34 / 1.44 / 1.39 GB | 73,863 training graphs at 94 videos |
| Test comparison (7 models, 2 × 2,000 resamples × 2 subsets) | 13.2 min | — | inference 49 s on GPU |

Free system RAM dropped to 0.6–1.3 GB for a few minutes while other
applications were open; no step failed.

## 3. VidOR relation classification: 400 → 1,900 training videos

### Protocol

- **Selection** (`split-manifest-scale2000.json`): the next 1,500 videos under the existing
  rare-predicate ranking (the 500-video selection is its exact prefix), all
  added to train. The 50 validation and 50 test videos are unchanged.
- **Frozen frame choice**: frame-selection counts stay at the 500-video values. All 500 earlier
  videos keep their frames (checked on the selection), the 3,000 copied
  frames match by SHA-256, and all 600 validation and test graphs in the new
  cache are byte-identical to the 500-video cache, with a largest visual
  feature difference of 0.0 (`graph-invariant-validation-test.json`).
- **Models**: MLP and GAT trained from scratch (`v4-scale2000`), and both
  warm-started from the matching `v3-scale500` checkpoint
  (`v4-scale2000-warm`) on the same cached graphs. Test was withheld during
  training.
- **Test**: scored once, after `validation-decision.json`, on the 50-video
  test graphs of the 500-video cache. These are the graphs every earlier
  VidOR comparison used; the v2 and v3 numbers below reproduce
  `vidor-scale500-comparison.json` exactly. Paired video-level bootstrap,
  2,000 resamples.

### Validation (same 50 videos) and test

| Model | Train videos | Validation macro F1 | Validation mAP | Test macro F1 | Test mAP | Test micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| GAT v2 (previous) | 57 | — (own 6-video split) | — | 0.071 | 0.088 | 0.316 |
| MLP v2 | 57 | — | — | 0.072 | 0.081 | 0.289 |
| GAT v3-scale500 (active in local `.env`) | 400 | 0.240 | 0.202 | 0.171 | 0.187 | 0.328 |
| MLP v3-scale500 | 400 | 0.245 | 0.211 | 0.166 | 0.172 | 0.374 |
| **GAT v4-scale2000 (runtime-eligible validation choice)** | 1,900 | 0.288 | 0.251 | **0.218** | **0.242** | 0.355 |
| MLP v4-scale2000 | 1,900 | 0.287 | 0.260 | 0.233 | 0.239 | 0.387 |
| GAT v4-scale2000-warm | 1,900 | 0.268 | 0.243 | 0.214 | 0.237 | 0.349 |
| MLP v4-scale2000-warm (overall validation choice) | 1,900 | 0.292 | 0.262 | 0.231 | 0.255 | 0.386 |

The runtime predictor requires a GNN, so the GAT is the only model that can
be activated.

Paired differences, test (video bootstrap; `*` = CI excludes 0):

| Comparison | Macro F1 | mAP | Micro F1 |
|---|---|---|---|
| GAT v4 − GAT v2 | +0.147 [0.104, 0.185]* | +0.154 [0.114, 0.200]* | +0.039 [0.001, 0.076]* |
| GAT v4-warm − GAT v2 | +0.143 [0.098, 0.187]* | +0.149 [0.105, 0.195]* | +0.034 [−0.001, 0.068] |
| **GAT v4 − GAT v3-scale500** | **+0.048 [0.015, 0.073]*** | **+0.056 [0.021, 0.085]*** | **+0.027 [0.006, 0.047]*** |
| GAT v4-warm − GAT v3-scale500 | +0.044 [0.016, 0.071]* | +0.050 [0.020, 0.074]* | +0.021 [0.006, 0.038]* |
| MLP v4 − MLP v3-scale500 | +0.067 [0.029, 0.089]* | +0.067 [0.030, 0.092]* | +0.013 [−0.007, 0.033] |
| MLP v4-warm − MLP v3-scale500 | +0.065 [0.032, 0.094]* | +0.083 [0.039, 0.104]* | +0.012 [−0.005, 0.031] |
| GAT v4-warm − GAT v4 (warm start) | −0.004 [−0.023, 0.021] | −0.006 [−0.033, 0.019] | −0.005 [−0.025, 0.012] |
| MLP v4-warm − MLP v4 (warm start) | −0.002 [−0.017, 0.022] | +0.016 [−0.010, 0.028] | −0.001 [−0.012, 0.011] |
| MLP v4 − GAT v4 | +0.015 [−0.011, 0.032] | −0.003 [−0.029, 0.018] | +0.032 [0.017, 0.048]* |

Per video (50 test videos, macro F1): GAT v4 beats GAT v3-scale500 on 34
(15 worse) and GAT v2 on 47. Per-video metrics for every model are in
`test-comparison.json`.

### What this supports

- **More data helps, significantly.** Going from 400 to 1,900 training videos raises test macro F1 by 0.048 and
  mAP by 0.056 for the GAT (+28% and +30%), and by about 0.065 for the MLP.
  Every CI excludes 0.
- **Returns are diminishing but still positive.** GAT test macro F1 at 57 → 99 → 400 → 1,900 training videos is
  0.071 → 0.095 → 0.171 → 0.218. The last 4.75× step added 0.047; the step
  before it, 4×, added 0.076.
- **Warm-starting from v3-scale500 gives no measurable benefit.** It also gives no measurable harm; every CI includes 0.
- **MLP vs GAT**: no significant macro F1 or mAP difference. The MLP has a higher micro F1.
  As with AVA, message passing adds nothing measurable.
- Validation overestimates test by 0.05–0.07 macro F1 for every model. So
  validation ranking among close candidates (within about 0.02) is not
  reliable. Validation preferred the warm MLP over the scratch MLP by 0.005;
  on test the order reversed by 0.002.

### Runtime and resources (measured)

| Step | Wall time | Notes |
|---|---:|---|
| Frame staging, 8 archive parts (1,500 new videos; 500 reused from copies) | 111 min | one Hugging Face read timeout in part 5; retried; 12,000 frames, 325 MB |
| Graph build (12,000 graphs, 297,420 candidate edges) + MLP + GAT, scratch | 28 min | MLP 69 s, GAT 105 s on GPU; cache 843 MB |
| Warm-start MLP + GAT (reused cache) | 4 min | GAT 168 s |
| Test comparison (8 models, 2,000 video resamples) | 25.5 min | CPU-bound |

Peak Python RSS 4.17 GB, during the graph build. Free system RAM fell to
0.73 GB, with browsers open, and no step failed. Free disk was at least
22.0 GB.

## Recommended next steps

1. Review before promoting anything; production defaults are unchanged.
   - **VidOR**: `gat-v4-scale2000` is the runtime-eligible candidate. It is significantly better than
     the active v3-scale500 on relation classification. Activating it means
     changing three `.env` lines and keeping a backup. Existing indexes keep
     their old relation evidence until reprocessed.
   - **AVA**: `gat-v8-noleak-scale100` is significantly better on classification, but its
     absolute macro F1 is 0.088. It also needs a runtime check of the
     leakage-free schema in the indexing path before any activation.
2. **Data scaling continues to pay for both models.** For AVA, the next step is the remaining
   eligible AVA videos (about 196, about 100 GB). For VidOR, the next step is
   about 4,000 videos. The graph builder's RAM (4.2 GB at 2,000) would need
   streaming or sharding for the full 7,000.
3. **Larger test sets would tighten the AVA CIs.** AVA has only 6 test videos.
