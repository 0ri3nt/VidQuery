# Library refresh verification

Verified on 2026-08-10 against the local SQLite library with YOLOv8n, Whisper
tiny, and the validated person-action checkpoint enabled.

## Reproducible command

```powershell
python -m vidquery refresh-library
```

Pass one or more `--video-id UUID` arguments to limit the refresh. The command
re-runs configured model inference. Windows imported through the AVA adapter and
windows carrying the controlled demo's manual overlay are merged by timestamp,
so fresh predictions do not erase trusted annotations, transcripts, speakers,
entities, relationships, or AVA action provenance.

## Audited result

| Video | Windows | YOLO | Whisper | Learned model | Preserved windows | Warnings |
|---|---:|---:|---:|---:|---:|---:|
| `distractor_demo.mp4` | 2 | 2 | 2 | 2 | 0 | 0 |
| `She How many dates have we been to.mp4` | 2 | 2 | 2 | 2 | 0 | 0 |
| `final_demo.mp4` | 5 | 5 | 5 | 5 | 5 | 0 |
| `0f39OWEqJ24.mp4` | 1,159 | 1,159 | 1,159 | 1,159 | 175 | 0 |
| `-XpUuIgyUHE.mp4` | 1,083 | 1,083 | 1,083 | 1,083 | 175 | 0 |
| `-OyDO1g74vc.mp4` | 836 | 836 | 836 | 836 | 176 | 0 |
| `-IELREHX_js.mp4` | 521 | 521 | 521 | 521 | 180 | 0 |
| `-FaXLcSFjUI.mp4` | 602 | 602 | 602 | 602 | 154 | 0 |
| **Total** | **4,210** | **4,210** | **4,210** | **4,210** | **865** | **0** |

The final library contains 48,049 stored detections, 16,773 action labels, and
3,190 windows with searchable transcript text. `preserved_evidence_source`,
`action_sources`, `fresh_whisper_transcript`, and the individual model status
fields retain provenance inside each merged canonical segment.

## Evaluation after refresh

```powershell
python -m vidquery evaluate --dataset evaluation/queries_multivideo.json `
  --output evaluation/results/multivideo-refreshed.json --limit 5 --learned-actions
```

The machine-readable result is
`evaluation/results/multivideo-refreshed.json`. On the 35-query controlled
corpus, hybrid retrieval with exact tuples achieved P@1 `0.8667`, Recall@5
`0.8849`, and MRR `0.8778`; the same hybrid without relationship tuples achieved
P@1 `0.7667`, Recall@5 `0.8516`, and MRR `0.8056`. The learned method achieved
P@1/MRR `1.0` only on its supported five-query subset (`14.29%` query coverage),
not on the full corpus. Neo4j was not supplied to this refresh evaluation.

## Decoder notes

Some AVA H.264 sources emitted recoverable `mmco: unref short failure` messages
while OpenCV sought through damaged reference frames. Every job nevertheless
finished `READY / complete`, produced the expected duration-aligned windows, and
recorded no public processing warning.
