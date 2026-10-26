# Integrated evaluation — 26 October 2026

The question: what do the new AVA action model (`ava80-gat-v8-noleak-scale100-lr0.0003`),
the new VidOR relation model (`vidor-relation-pair-visual-gat-v4-scale2000`),
and deterministic Whisper change in **end-to-end moment retrieval and
answerability** (Charades-STA + NA-VMR)? Each change is isolated with a
full-rebuild ablation.

Model classification results are in
[EXPERIMENTS_2026-10-26.md](EXPERIMENTS_2026-10-26.md). The benchmark design is in
[EVALUATION_CHARADES_STA.md](EVALUATION_CHARADES_STA.md).

## Configurations

All indexes were rebuilt from the raw MP4s by the full pipeline, into
versioned folders under `data/app/benchmark/charades_integrated_2026-10-26/`.
Profiles are in `evaluation/configs/model_profiles_2026-10-26.json`, applied by
`scripts/run_with_profile.py`, which validates each checkpoint before running.

| Index | Whisper | AVA action GNN | VidOR relation GNN | Videos |
|---|---|---|---|---:|
| `orig` (earlier, the 41.4% run) | temperature fallback | v5 | v2 | 300 test |
| `rerun` (earlier control) | temperature fallback | v5 | v2 | 300 test |
| `rerun_v3` (earlier) | temperature fallback | v5 | v3-scale500 | 300 test |
| `det_ava5_vidor2` | greedy, T = 0 | v5 | v2 | 300 test |
| `det_ava8_vidor2` | greedy, T = 0 | **v8** | v2 | 300 test |
| `det_ava5_vidor4` | greedy, T = 0 | v5 | **v4** | 300 test |
| **`det_ava8_vidor4` (integrated)** | greedy, T = 0 | **v8** | **v4** | 450 (150 validation + 300 test) |

**Provenance check:** every segment of each new index records its Whisper decoding mode, the model versions, and the SHA-256 of each checkpoint. All 2,935 segments of the integrated index show `greedy_temperature_0`, AVA v8 (`ad0063c6…`), and VidOR v4 (`efd44bec…`).

**Search settings**, identical for every run:

| Setting | T0 | T1 | T2 |
|---|---|---|---|
| Soft constraints, frame-level OpenCLIP similarity | off | on | on |
| Thresholds | default | default (0.15 / 0.45) | P@1-preserving weak threshold 0.1595, fitted on the **old** validation index (not refitted) |

All settings use Review-2 ranking, 10 s evaluation spans, and the rule-based parser only.

Nothing was tuned on test. Queries, plans, and thresholds are identical
across runs. This was checked by `scripts/compare_moment_runs.py`, which
asserts the query sequence and the plans.

## Results — test (300 videos, 863 positives, 863 in-domain + 855 out-of-domain negatives)

### Selected settings (T1)

| Index | P@1 [95% CI] | R@1@0.5 | R@1@0.7 | R@5@0.5 | In-domain rej. | Out-of-domain rej. | Rejection F1 | AUROC in / out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orig | 0.414 [0.381, 0.445] | 0.416 | 0.187 | 0.870 | 0.008 | 0.997 | 0.665 | 0.600 / 0.9995 |
| rerun | 0.399 [0.365, 0.430] | 0.402 | 0.185 | 0.870 | 0.008 | 0.997 | 0.665 | 0.602 / 0.9996 |
| rerun_v3 | 0.407 [0.374, 0.439] | 0.409 | 0.175 | 0.871 | 0.008 | 0.994 | 0.664 | 0.600 / 0.9994 |
| det_ava5_vidor2 | 0.403 [0.371, 0.436] | 0.407 | 0.185 | 0.869 | 0.008 | 0.997 | 0.665 | 0.607 / 0.9995 |
| det_ava8_vidor2 | 0.410 [0.375, 0.443] | 0.414 | 0.185 | 0.863 | 0.008 | 0.997 | 0.665 | 0.605 / 0.9994 |
| det_ava5_vidor4 | 0.403 [0.371, 0.436] | 0.407 | 0.182 | 0.868 | 0.008 | 0.997 | 0.665 | 0.604 / 0.9994 |
| **det_ava8_vidor4** | **0.404 [0.371, 0.437]** | 0.408 | 0.181 | 0.866 | 0.008 | 0.993 | 0.663 | 0.603 / 0.9994 |

With the validation-fitted P@1-preserving threshold (T2), positives are
unchanged and out-of-domain rejection is 1.000 for every deterministic index.

### Paired differences (video bootstrap, 2,000 resamples; `*` = CI excludes 0)

| Effect | Comparison | P@1 (T1) | R@1@0.7 (T1) | In-domain rej. (T0) | P@1 (T0) |
|---|---|---|---|---|---|
| Rebuild noise (fallback Whisper) | rerun − orig | −0.015 [−0.038, 0.006] | −0.001 [−0.014, 0.012] | +0.001 [−0.005, 0.008] | −0.005 [−0.021, 0.012] |
| **Deterministic Whisper** | det_ava5_vidor2 − rerun | +0.005 [−0.015, 0.025] | 0.000 [−0.010, 0.010] | 0.000 [−0.009, 0.008] | −0.004 [−0.017, 0.013] |
| **AVA v8** (VidOR v2) | det_ava8_vidor2 − det_ava5_vidor2 | +0.007 [−0.021, 0.033] | 0.000 [−0.019, 0.019] | **+0.045 [0.020, 0.072]*** | −0.009 [−0.034, 0.016] |
| **AVA v8** (VidOR v4) | det_ava8_vidor4 − det_ava5_vidor4 | +0.001 [−0.026, 0.027] | −0.001 [−0.021, 0.018] | **+0.046 [0.022, 0.073]*** | −0.002 [−0.028, 0.023] |
| **VidOR v4** (AVA v5) | det_ava5_vidor4 − det_ava5_vidor2 | 0.000 [−0.012, 0.011] | −0.004 [−0.011, 0.003] | +0.004 [−0.002, 0.010] | −0.007 [−0.020, 0.005] |
| **VidOR v4** (AVA v8) | det_ava8_vidor4 − det_ava8_vidor2 | −0.006 [−0.017, 0.005] | −0.005 [−0.014, 0.004] | +0.005 [−0.001, 0.012] | 0.000 [−0.011, 0.010] |
| All three | det_ava8_vidor4 − det_ava5_vidor2 | +0.001 [−0.026, 0.029] | −0.005 [−0.026, 0.015] | **+0.050 [0.024, 0.077]*** | −0.009 [−0.038, 0.017] |
| vs the 41.4% run | det_ava8_vidor4 − orig | −0.009 [−0.036, 0.017] | −0.006 [−0.026, 0.012] | **+0.051 [0.026, 0.077]*** | −0.017 [−0.048, 0.012] |

**Per-video P@1:** the integrated index is better than the deterministic baseline on 39 videos and worse on 34 (T1). It is better than `orig` on 39 and worse on 42. All per-video values are in `comparison-test-*.json`.

**Default strict settings (T0) P@1:**

| Index | P@1 |
|---|---:|
| orig | 0.150 |
| det_ava5_vidor2 | 0.141 |
| det_ava8_vidor2 | 0.132 |
| det_ava5_vidor4 | 0.134 |
| det_ava8_vidor4 | 0.132 |

No pairwise difference is significant.

### What changed in the index (`index-evidence/`, 1,935 test segments)

| Comparison | Transcripts | Actions | Relations | Detections |
|---|---:|---:|---:|---:|
| fallback rerun vs deterministic (same models) | 1,163 | 828 | 914 | 0 |
| AVA v5 → v8 | 0 | 1,793 | 1,615 | 0 |
| VidOR v2 → v4 | 7 | 6 | 1,592 | 0 |
| v5/v2 → v8/v4 | 0 | 1,793 | 1,673 | 5 |

- **AVA v8:** changes action evidence in 93% of segments. It also changes the action-target relations derived from those actions.
- **VidOR v4:** changes the learned relations.
- **Rebuild noise:** deterministic rebuilds are not perfectly bit-identical at this scale. 7 transcripts and 5 detection sets differ out of 1,935 segments (≤ 0.4%), against 60% of transcripts with the old fallback. The likely cause is GPU floating-point nondeterminism; it was not investigated further.

## Results — validation (150 videos, integrated vs the earlier validation index)

| Index | P@1 | R@1@0.5 | R@5@0.5 |
|---|---:|---:|---:|
| earlier (fallback, AVA v5, VidOR v2) | 0.358 | 0.364 | 0.860 |
| integrated (det, AVA v8, VidOR v4) | 0.350 | 0.355 | 0.834 |
| Paired difference | −0.009 [−0.049, 0.034] | — | −0.026 [−0.050, −0.003]* |

## What this supports

1. **Neither new model improves Charades moment retrieval.** No P@1 or R@1 difference is significant, under either the selected or the default settings. The integrated system (0.404) sits inside the rebuild-noise band of the original 41.4% run. **Do not claim a retrieval improvement from the new models.** Their gains are on their own classification tasks (EXPERIMENTS_2026-10-26.md).
2. **AVA v8 makes strict mode reject more in-domain negatives,** +0.045 to +0.051 (significant), with no significant P@1 cost. The new action labels match fewer unrelated Charades sentences. In the selected (soft-constraint) setting this effect disappears, because missing actions no longer veto a segment.
3. **Deterministic Whisper removes most rebuild noise.** It doesn't change accuracy (+0.005, not significant), but it makes the numbers repeatable.
4. **Relation queries are rare in this benchmark:** 12 of 2,581 queries are parsed as relationship searches. So VidOR v4 cannot move Charades retrieval much. Judge it on VidOR classification instead.
5. **One significant validation decrease:** R@5@0.5 is −0.026. It's small, it doesn't appear in test R@5 (−0.004, not significant), and it's noted rather than explained.

## Runtime and resources (measured)

| Index | Indexing | Mean per video | Evaluation runs |
|---|---:|---:|---:|
| det_ava8_vidor4, 450 videos | 64 min | 8.5 s | 23.5 min (6 runs) |
| det_ava5_vidor2, 300 videos | 68 min | 13.6 s | 12.7 min |
| det_ava8_vidor2, 300 videos | 49 min | 9.7 s | 29.1 min |
| det_ava5_vidor4, 300 videos | 67 min | 13.4 s | 11.0 min |

- **Per-video speed:** the slower runs overlapped with other jobs (library reprocessing, demo server).
- **Memory:** peak Python RSS 3.0 GB.
- **Disk:** 0.85 GB per 300-video index and 1.3 GB for the integrated one.
- **System resources dipped twice:**
  - free RAM fell to 0.3 GB with browsers open;
  - free disk dipped to 4.6 GB between 13:58 and 15:00 UTC, then recovered to 24.7 GB.

  The disk dip is not explained by these runs, which write at most 1.3 GB per index; Windows pagefile growth under low RAM is the likely cause. No step failed.

## Limitations

- **Experimental settings:** the selected settings (soft constraints, frame similarity) and the 10 s evaluation span are not production defaults. The production defaults correspond to T0.
- **Test set:** 300 Charades test videos only.
- **Thresholds:** T2 reuses the threshold fitted on the earlier validation index; it was not refitted.
- **In-domain negatives** remain unsolved (AUROC about 0.60).

## Reproduce

```bash
# rebuild one profile's index (the profile validates its checkpoints first)
python -m scripts.run_with_profile --profile det-ava8-vidor4 \
  --set DATABASE_PATH=<dir>/index.sqlite3 --set UPLOAD_DIR=<dir>/uploads --set GENERATED_DIR=<dir>/generated \
  -- python -m scripts.index_videos_batch --manifest evaluation/charades_sta_navmr_benchmark.json \
     --splits validation test --video-dir data/source/charades/videos --progress <dir>/progress.jsonl
# evaluate (T1 shown), then compare runs
SOFT_CONSTRAINT_MATCHING=true ENABLE_FRAME_SEMANTIC_RETRIEVAL=true python -m scripts.run_with_profile \
  --profile det-ava8-vidor4 --set DATABASE_PATH=<dir>/index-snapshot.sqlite3 --set GENERATED_DIR=<dir>/generated \
  -- python -m scripts.evaluate_moment_benchmark --split test --spans native segment expand:10 --output T1.json
python -m scripts.compare_moment_runs --run a=A.json --run b=B.json --pair a:b --output comparison.json
```

The exact scripts used are in `evaluation/results/charades-integrated-2026-10-26/run_scripts/`.
