# Archive

Files kept for the record that the current VidQuery application, tests, and
experiments do not use. Tracked files were moved here with `git mv` (history is
preserved) during the 26 October 2026 cleanup; the superseded results were untracked
and are now tracked here; nothing in `vidquery/`,
`scripts/`, `tests/`, or the Dockerfile references them.

| Folder | Contents | Replaced by |
|---|---|---|
| `legacy_cli/` | One-line entry scripts for the original AVA research pipeline (`extract_frames.py`, `detect_objects.py`, `build_scene_graphs.py`, `whisper_transcribe.py`, `speaker_diarization.py`, `visualize_graphs.py`, `visualize_videos.py`) | `python -m vidquery ...` commands and `scripts/process_ava_selection.py`. The pipeline modules they call (`extraction/`, `modelling/`) are still in use and were not moved. Run a script from the repository root with `PYTHONPATH=.` if needed. |
| `legacy_retrieval/` | Former `retrieval/` package: compatibility wrappers around `vidquery.search` and a disabled LLM-to-Cypher prompt module kept to document that decision | `vidquery.search` (`StructuredQueryParser`, `SafeCypherCompiler`, `LocalHybridSearchEngine`); not packaged, imported, or used by Docker |
| `superseded_results/v2026-10-25b/` | Three AVA end-to-end test runs (B lexicon, C no-leak GNN, E combined) made before the run-metadata provenance fix; they were re-run about 15 minutes later and the committed `evaluation/results/v2026-10-25b/` files and summary use the re-runs | `evaluation/results/v2026-10-25b/test-v2-{B,C,E}-*.json` |
| `legacy_ava_downloader/` | Go AVA downloader (`download.go`, `go.mod`) and its video lists (`ava_train_v2.2.txt`, `ava_test_v2.2.txt`) | `python -m vidquery expand-ava-dataset` (storage-aware, resumable downloads) and `scripts/download_ava_selection.py` |

To restore a file: `git mv archive/<folder>/<file> .`
