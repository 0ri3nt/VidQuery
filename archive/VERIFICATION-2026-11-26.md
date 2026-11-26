# Archive verification: 26 November 2026

- Branch: `dev_aryaman` in `0ri3nt/VidQuery`.
- Release: `archive-dev-aryaman-2026-11-26` (Release ID `459285474`).
- Archive inventory: 106,573 distinct local-only files, represented by
  106,652 tar members in 147 Release assets. Raw content: 11.69 GiB.
- The GitHub API reported all 147 assets as uploaded. Each remote asset's
  size and server-computed SHA-256 matched `RELEASE-2026-11-26.json`.
- Public, unauthenticated downloads of assets 001, 030, 059, 089, 118, and
  147 were stream-parsed; all 88 sampled tar members matched the per-file
  hashes in `backup-files.jsonl.gz`.
- Asset 001 was extracted into an isolated local folder and all 50 files
  passed hash verification.
- Assets 002 and 003 were extracted into an isolated local folder. Their
  split SQLite backup was reassembled; the 112,029,696-byte result had the
  same SHA-256 as the original local file.
- The repository test suite passed with 506 tests passing and one Neo4j
  integration test skipped because the service was unavailable. Python
  compilation checks passed.

The complete 11.69 GiB archive was not extracted into a second copy on this
machine. The server-computed SHA-256 comparison verifies every remote asset;
the sampled downloads and extractions verify the restoration path.

The archive deliberately omits the raw training datasets and benchmark media
at the owner's request. `.env` credentials were also omitted for safety.
