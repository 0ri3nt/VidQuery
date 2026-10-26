# Video libraries, the demo library, and removing videos

## What a library is

A library is one SQLite database plus the two directories that hold its files:

| Setting | Holds | Production default |
|---|---|---|
| `DATABASE_PATH` | videos, canonical segments, entity appearances | `data/app/vidquery.sqlite3` |
| `UPLOAD_DIR` | uploaded media, `<video_id>.mp4` | `data/app/uploads` |
| `GENERATED_DIR` | per-video frames, audio, frame embeddings, `<video_id>/` | `data/app/generated` |

A server serves exactly one library. Environment variables override `.env`
(`load_dotenv(override=False)`), so `scripts/run_with_profile.py --set ...`
can point a server at another library without editing `.env`.

A video may also point at media **outside** `UPLOAD_DIR` when it was
registered in place (`python -m vidquery process --video <file>` without
`--copy`). In production, 10 of the 13 videos do this: AVA clips in
`data/ava/videos`, the controlled demo in `data/demo`, and one appearance
fixture. Those files are shared sources. Removal never moves or modifies them.

Give every library its own `UPLOAD_DIR` and `GENERATED_DIR`. Removal moves
`GENERATED_DIR/<video_id>` into the library's trash. A second library that
shared the directory would lose those files until the video was restored.

## Libraries in this workspace

| Library | Database | Videos | Served on |
|---|---|---:|---|
| production | `data/app/vidquery.sqlite3` | 13 | `:8000` (unchanged by this work) |
| demo | `data/app/demo_library/demo.sqlite3` | 4 | `:8001` |
| benchmark snapshots | `data/app/benchmark/charades_integrated_2026-10-26/*/index-snapshot.sqlite3` | 300–450 | not served; frozen evaluation inputs |

### The demo library

`data/app/demo_library/` is self-contained (about 40 MB):

```
demo.sqlite3              4 videos, 35 segments, 71 entity appearances
uploads/<video_id>.mp4    copies; each matches its recorded content_sha256
generated/<video_id>/     frames, audio, frame_clip_embeddings.npz
library_manifest.json     where every copy came from, with row counts
profile.json              the exact model profile and overrides of the :8001 server
```

| Video | Source |
|---|---|
| OM2NF.mp4, I2DHG.mp4, GG4UR.mp4 | Charades validation clips, indexed with AVA v8, VidOR v4 and deterministic Whisper (benchmark index `det-ava8-vidor4`) |
| Untitled design.mp4 | the production library |

It was built by copying, so the sources and the production library are
untouched:

```bash
python -m scripts.build_demo_library --output-dir data/app/demo_library --database-name demo.sqlite3 \
  --video <source db> <source generated dir> <video_id>   # once per video
```

The script refuses an existing library. It verifies each media copy against
`content_sha256` and each generated directory by file count and size. It
rewrites `stored_path` and every segment's `thumbnail_path` to point at the
copies, then runs SQLite `integrity_check` and `foreign_key_check`.

Start the demo server (same experimental profile as before the migration):

```bash
python -m scripts.run_with_profile --profile det-ava8-vidor4 \
  --set DATABASE_PATH=data/app/demo_library/demo.sqlite3 \
  --set UPLOAD_DIR=data/app/demo_library/uploads \
  --set GENERATED_DIR=data/app/demo_library/generated \
  --set SOFT_CONSTRAINT_MATCHING=true --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=true \
  --set FRAME_SEMANTIC_WEIGHT=1.0 --set FRAME_SEMANTIC_FLOOR=0.18 --set FRAME_SEMANTIC_CEILING=0.32 \
  --record data/app/demo_library/profile.json \
  -- python -m vidquery serve --host 127.0.0.1 --port 8001
```

Verify it with the 19 pre-registered demo queries. Each must reproduce its
recorded segment, peak, verdict and confidence:

```bash
python evaluation/results/demo-2026-10-26/run_scripts/verify_demo_library.py http://127.0.0.1:8001 3 out.json
```

On 2026-10-26 the result was 19/19 on all 3 runs
([demo-library-verification.json](../evaluation/results/demo-2026-10-26/demo-library-verification.json)).
All 4 videos streamed bytes matching their `content_sha256`, Range requests
returned `206`, and thumbnails at the start, middle and end were JPEGs.

One query depends on the Groq query planner. **"person cooking at the stove."**
is ambiguous to the local parser, so the server asks Groq for a plan. The
recorded outcome (`weak`, 0.3252) is the local fallback used when Groq does
not answer. In one run Groq did answer, and the same peak (37.988 s) came back
`supported` at 0.4563. With `ENABLE_QUERY_PLANNER=false` all 19 queries
reproduce exactly.

### Migration record and rollback

The `:8001` demo previously served a 450-video copy of the benchmark index.
That copy was moved, not deleted, with its empty WAL and shared-memory files, to
`data/app/quarantine-2026-10-26/demo-450-copy/`. Its `ROLLBACK.md` gives the
three rollback steps.

The library's thumbnail paths were rewritten in place after the first build.
The database before that rewrite is
`data/app/backups/demo_library-before-thumbnail-relocation-2026-10-26.sqlite3`.

## Removing a video from a library

In the UI, open **Library** and choose **Remove** on a video. A dialog names
the video. **Cancel**, or Escape, closes it without sending a request.
**Remove video** sends `DELETE /api/videos/{video_id}` (see [API.md](API.md)).

- On success, the row and the search-filter option disappear at once, results from that video are dropped from the page, and a message confirms the removal.
- On failure, the server's reason is shown and the row stays. Typical reasons: the video is processing, or a file is open in another program.

Remove is disabled while a video is processing.

The server then:

1. exports the video's rows to a new trash entry;
2. moves the media file into the entry, but only if it lies inside `UPLOAD_DIR` and no other video uses it;
3. moves `GENERATED_DIR/<video_id>` into the entry;
4. deletes the rows in one transaction, children before the video, and checks the counts;
5. drops the video's cached frame embeddings and, when Neo4j is enabled, its graph nodes.

If step 2, 3 or 4 fails, every file already moved is moved back and the rows
stay. The entry is then marked `rolled_back`. Moves are same-volume renames
with no copy fallback, so a move happens whole or not at all. On Windows, a
file open in another program makes the rename fail with
`409 … in use`, and nothing changes.

### Trash location and format

```
<directory of DATABASE_PATH>/library_trash/<database file stem>/<UTC timestamp>-<video_id>/
    manifest.json   status, video, database and directories, per-file outcome, restore command
    rows.json       every row of the video, column for column
    media/          the moved media file (if it was inside UPLOAD_DIR)
    generated/<video_id>/
```

| Library | Trash |
|---|---|
| production | `data/app/library_trash/vidquery/` |
| demo | `data/app/demo_library/library_trash/demo/` |

`manifest.json` `status` is one of the following:

| Status | Meaning |
|---|---|
| `removing` | the removal was interrupted, for example the server was killed; restorable |
| `removed` | complete; restorable |
| `rolled_back` | failed and fully undone; nothing to restore |
| `rollback_incomplete` | failed and some files could not be moved back; they are listed under `stranded` |
| `restored` | restored; kept as a record |

Nothing empties the trash. Deleting an entry permanently is a deliberate
manual step.

### Restoring

```bash
python -m vidquery list-removed  --database <db> --upload-dir <uploads> --generated-dir <generated>
python -m vidquery restore-video <entry> --database <db> --upload-dir <uploads> --generated-dir <generated>
```

Every removal response and manifest carries the exact `restore_command` for
its library. Without the flags, the commands use the settings from the
environment and `.env`, which is the production library.

`restore-video` moves the files back and re-inserts the rows in one
transaction. A running server shows the video again without a restart. It
refuses, changing nothing, when:

- the entry belongs to another library's trash or database;
- the entry is not `removed` or `removing`;
- a file already exists at the original path;
- the same content was uploaded again under a new id (remove that copy first);
- a path in the manifest points outside the entry or the library's directories.

Column names in `rows.json` are checked against the live schema before any
insert. With Neo4j enabled, the restored video is mirrored again.

### Limits

- The processing reservation is per server process. A separate CLI `process` run on the same database is excluded only by the state check (`409` while the video is in a processing state).
- The trash must be on the same volume as `UPLOAD_DIR` and `GENERATED_DIR`. Otherwise the move fails and the removal rolls back.
- Ids that do not match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` (no `..`) cannot be removed through the API. All current ids match.

### Tests

- `tests/test_library_removal.py`:
  - rows, files and the trash entry;
  - 404, 409 and 422 handling;
  - repeated and concurrent requests;
  - shared sources and `..` path escapes;
  - missing files;
  - database and file failures with rollback, incomplete rollback and an interrupted removal;
  - exact restore and restore refusals;
  - SQL-injection attempts through `rows.json`;
  - cross-library isolation;
  - the Neo4j mirror and the frame-embedding cache;
  - the frontend assets.
- `tests/e2e/library_removal.e2e.cjs` drives the real UI in Chrome against a disposable library. It checks:
  - cancel and Escape;
  - a successful removal;
  - a 404 for a video removed elsewhere while the dialog was open;
  - a 409 for a file held open by another process.

  It needs `playwright-core` (see the header of the file). Never point it at a library you want to keep.
