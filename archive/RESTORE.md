# Full local archive and restoration

The primary source archive is the `dev_aryaman` branch of
`https://github.com/0ri3nt/VidQuery.git`. The versioned Release tagged
`archive-dev-aryaman-2026-11-26` stores the local-only model weights,
checkpoints, application SQLite state, generated media, feature caches,
and experiment records. Raw training datasets and benchmark videos are
excluded at the owner's request. `RELEASE-2026-11-26.json` lists every Release volume and its
SHA-256 hash. `backup-files.jsonl.gz` lists every archived file and its hash.

The public archive deliberately excludes `.env`, which contains API tokens
and a database password. Restore those values separately from a secure
credential store using `.env.example` as a guide. Local virtual environments,
Git internals, and generated Python/lint caches are also omitted because
they are machine-specific and can be recreated.

## Restore on another machine

1. Install Git, Python 3.11 or 3.12, and the platform-specific runtime tools
   described in the root README. Ensure at least 80 GiB of free space.
2. Clone the archival branch:

   ```sh
   git clone --branch dev_aryaman https://github.com/0ri3nt/VidQuery.git
   cd VidQuery
   ```

3. Check out the commit named in `archive/RELEASE-2026-11-26.json` if you need
   the exact source snapshot. The later manifest commit only adds archive
   metadata, so the branch head also contains the archived source.
4. Run the streaming restore command. It downloads each Release volume,
   extracts it into this clone, reassembles split checkpoints, and checks both
   the complete tar SHA-256 and every extracted part's SHA-256. It places
   `__capstone_extras__` files in
   the clone's parent directory, where they were on the original machine.

   ```sh
   python scripts/archive_release.py restore --destination .
   ```

5. Create a new environment and install `requirements.txt`, then
   `requirements-ml.txt` and any optional groups you use. Recreate `.env`
   securely. Follow the README's startup instructions. The trained model
   checkpoints and necessary derived artifacts are restored by step 4; model
   retraining is not required.

## Verify without writing extracted files

After cloning the branch, run:

```sh
python scripts/archive_release.py verify
```

This reads every remote Release volume, parses every tar member, and checks
the full volume and individual file hashes against the branch manifest.

Do not remove the original local project until this verification has passed
against the published Release and you have securely retained your `.env`
credentials. A Release and branch can still be deleted by repository owners;
for long-term redundancy, keep a second copy in independent storage.
