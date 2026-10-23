# Run and test VidQuery

A practical guide for running the system locally, checking that it is healthy,
testing it the way an examiner will, and reporting problems. For a first-time
install and model downloads, follow [SETUP.md](SETUP.md) first.

## 1. Start the system

```bash
cd VidQuery
source .venv/bin/activate
python -m vidquery serve --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The page has four tabs:

| Tab | Use |
|---|---|
| Search | type a query, see ranked moments, click to play from the exact second |
| Library | indexed videos, their state, and the reprocess button |
| Upload | add a new MP4 |
| Annotate | label ground-truth moments for the benchmark (see section 6) |

Leave the terminal open; it shows warnings and errors while you work.

## 2. Check health before every session

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
```

Everything you rely on should be available. If a model shows `unavailable_*`,
the state names the missing piece; see the troubleshooting section of
[SETUP.md](SETUP.md).

| Key | Expected |
|---|---|
| `database` | `available` |
| `yolo`, `whisper` | `available_configured` |
| `diarization` | `available_configured_required` |
| `gnn_person_action_classifier` | `available_validated_enabled` |
| `gnn_relationship_prediction` | `available_validated_enabled` |
| `sentence_embedding_retrieval` | `available_cached_enabled` |
| `ocr`, `entity_appearance` | `available_*` |
| `rag_generation`, `query_planner` | `available_configured*` (needs `GROQ_API_KEY`) |
| `person_action_classifier` | `unavailable_no_valid_checkpoint` is expected; this legacy model is replaced by the GNN |
| `neo4j` | `disabled` is fine; SQLite is the default backend |

## 3. Add videos

Rename the file before uploading. The benchmark keys every video by its
original filename without the extension, so `lecture_os_01.mp4` becomes
`lecture_os_01`.

Upload from the Upload tab, or from the command line:

```bash
python -m vidquery process --video path/to/lecture_os_01.mp4 --copy
```

The Library tab shows the stage (`PREPROCESSING`, `VISUAL_PROCESSING`,
`AUDIO_PROCESSING`, ..., `READY`). Search only works once a video is `READY`.

On a MacBook Air M3 with 8 GB RAM, expect several minutes per minute of video.
Speaker diarization (Pyannote) on CPU is the slowest stage. Close other heavy
apps while processing. A second click on reprocess while a job is running
returns `409 Conflict`; that does not mean the job failed.

## 4. Test it like an examiner

Examiners type their own queries. Test each kind below on every demo video,
and note which ones fail.

| Kind | Example | What a good result looks like |
|---|---|---|
| Spoken words | `where is deployment discussed` | jumps to the word, precision `word`, verdict `supported` |
| Speaker | `SPEAKER_01 talks about docker` | only segments from that speaker |
| Object | `find a laptop` | frames where YOLO saw a laptop |
| Relation | `person holding a bottle` | the moment the relation is visible |
| Action | `person sitting`, `two people talking` | often `weak`; the action model is the weakest component |
| Appearance | `person in a blue shirt` | similarity-based, can be imperfect |
| On-screen text | `slide that says Kubernetes` | requires text on screen in the video |
| Ambiguous | `find drive` | either resolves or asks "Did you mean ..." |
| Trick question | `find a dog` in a lecture | no confident result, verdict `insufficient` |

How to read a result:

- **Verdict.** `supported` means strong evidence, `weak` means some evidence
  from less reliable models, and `insufficient` means the system is admitting it
  does not know. Saying `insufficient` on a trick question is correct
  behaviour, not a bug.
- **Precision.** `word` and `frame` are exact; `utterance` and `segment` mean the
  answer is somewhere in that span.
- **Why.** Each result lists the evidence that matched. If the reason looks wrong,
  that is the most useful thing to report.

### Known limits to keep in mind

- **Objects** are limited to YOLO's 80 COCO classes (person, laptop, chair,
  bottle, cell phone, book, car, dog, ...). There is no `whiteboard`,
  `projector`, or `pen`. Queries for those fall back to spoken words.
- **Speech search** is only as good as the transcript. The default Whisper
  `tiny` model mishears technical terms ("binary search tree" became "Vandri
  search tree"). See section 7.
- **Speakers** are `SPEAKER_00`, `SPEAKER_01`, ..., not names.
- **Groq** helps only with ambiguous queries and the written answer. Search works
  without it.

## 5. Diagnose a failing query

When a query returns the wrong moment or nothing:

```bash
python -m vidquery diagnose-query "person holding a bottle" --video-id VIDEO_UUID
```

The output shows how the query was parsed, what the index contains for that
video, the raw hits, and how both ranking modes scored them. The video UUID is
shown in the Library tab. See [RETRIEVAL_DIAGNOSTICS.md](RETRIEVAL_DIAGNOSTICS.md).

## 6. Label ground truth (Annotate tab)

This builds the honest end-to-end benchmark. Full rules are in
[REVIEW2.md](REVIEW2.md#labelling-ground-truth). Summary:

1. Pick a `READY` video and assign it to `calibration` or `test`. A video
   belongs to one split only.
2. Watch the video and mark the moment with `[` (start), `P` (peak), and `]`
   (end).
3. Enter the query, choose the category, and tick the readings that are
   actually present.
4. Save. Entries go to `evaluation/queries_challenge.json`.

Label from the video, never from the system's output: do not run a query in
Search before labelling it.

Run the benchmark when you have about 60 queries:

```bash
python -m vidquery evaluate-challenge --validate-only
python -m vidquery evaluate-challenge --split calibration \
  --output evaluation/results/challenge-calibration.json
python -m vidquery evaluate-challenge --split test \
  --output evaluation/results/challenge-test.json
```

## 7. Demo-day checklist

**One or two days before:**

- [ ] Choose 3–5 demo videos: your own recordings with clear speech, slides with
      text, and visible everyday objects (laptop, bottle, phone, chair, cup).
      Add one AVA test-split movie clip for action queries if available.
- [ ] Rename the files, upload them, and confirm every one reaches `READY`.
- [ ] Optional, better transcripts: set `WHISPER_MODEL=small` in `.env` (with
      `ALLOW_MODEL_DOWNLOADS=true` once to fetch it), then reprocess with
      `python -m vidquery refresh-library`. Expect slower processing; on 8 GB
      RAM, `small` is the practical limit.
- [ ] Run the examiner test list in section 4 on every video and write down
      which queries work well.
- [ ] Run `python -m pytest -q`; all tests should pass.

**On the day:**

- [ ] Start the server about 10 minutes early and check `/api/health`.
- [ ] Run one warm-up query so the models load before the examiners start.
- [ ] Set `ALLOW_MODEL_DOWNLOADS=false` and `HF_HUB_OFFLINE=1` so nothing tries
      to download in the middle of the demo.
- [ ] Have the Wi-Fi working if you want Groq answers. Without the internet,
      search still works.

**Groq rate limit.** The free Groq tier allows 8,000 tokens per minute, and one
planner call uses about 2,900 tokens. After two or three ambiguous queries in a
minute, Groq returns HTTP 429. Search still works, but the query planner falls
back to the local parser and no written answer appears. If examiners type
quickly, pause briefly between ambiguous queries, or upgrade the Groq plan.

**What to say if asked:**

- The system abstains instead of guessing: `insufficient` is deliberate.
- The action and relation GNNs are weak on unseen video (macro F1 0.07 and
  0.18), so their results are down-weighted and marked `weak`.
- The 0.96 P@1 in the README was measured with labelled evidence, not raw model
  output. The end-to-end action result is P@1 0.50 on 14 queries.

## 8. Reporting an issue

Send these details so the problem can be reproduced:

1. The exact query text.
2. The video (filename or UUID) and the moment you expected (for example "around
   1:42, the speaker says 'deadlock'").
3. What happened: top result times, verdict, and the "why" text, or a
   screenshot.
4. The output of `diagnose-query` for that query, if you can run it.
5. Any error lines from the server terminal.
