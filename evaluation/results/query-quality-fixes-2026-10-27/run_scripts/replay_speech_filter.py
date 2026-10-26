"""Apply vidquery.speech_filter to a COPY of an index, as re-indexing with the filter would.

Usage: replay_speech_filter.py SOURCE_DB TARGET_DB SUMMARY.json
The source is opened read-only. For each segment the kept utterances replace
transcript_segments, the transcript text and speakers are rebuilt like
vidquery.segments does, and the cached sentence embedding is re-encoded from the
same text composition as vidquery.pipeline. Diarization gating applies only to
segments whose processing_metadata records diarization_status == "completed".
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from vidquery.config import get_settings
from vidquery.domain import CanonicalSegment
from vidquery.search import configured_text_encoder
from vidquery.speech_filter import filter_transcripts


def main(source: Path, target: Path, summary_path: Path) -> None:
    if target.exists():
        raise SystemExit(f"{target} exists")
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    src.backup(dst)
    src.close()
    encoder = configured_text_encoder(get_settings())
    stats: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    words_before = words_after = 0
    rows = dst.execute("select segment_id, payload_json from segments").fetchall()
    with dst:
        for segment_id, payload in rows:
            segment = CanonicalSegment.model_validate_json(payload)
            gate = segment.processing_metadata.get("diarization_status") == "completed"
            kept, summary = filter_transcripts(segment.transcript_segments, diarization_completed=gate)
            for reason, count in summary["dropped"].items():
                dropped[reason] += count
            stats["segments"] += 1
            stats[f"gate_{summary['speech_gate']}"] += 1
            words_before += len(segment.transcript.split())
            if len(kept) == len(segment.transcript_segments):
                words_after += len(segment.transcript.split())
                continue
            transcript = " ".join(item.text.strip() for item in kept if item.text.strip()).strip()
            words_after += len(transcript.split())
            stats["segments_changed"] += 1
            if not transcript and segment.transcript:
                stats["segments_transcript_emptied"] += 1
            metadata = {**segment.processing_metadata, "transcript_speech_filter": summary}
            text = " ".join([transcript, *segment.entities, *segment.actions,
                             *(r.predicate for r in segment.relationships),
                             *(o.text for o in segment.ocr_evidence)]).strip()
            updated = segment.model_copy(update={
                "transcript": transcript,
                "transcript_segments": kept,
                "speakers": sorted({i.speaker for i in kept if i.speaker and i.speaker != "UNKNOWN"}),
                "embedding": encoder.encode(text) if segment.embedding is not None else None,
                "processing_metadata": metadata,
            })
            dst.execute("update segments set transcript = ?, payload_json = ? where segment_id = ?",
                        (transcript, updated.model_dump_json(), segment_id))
    dst.close()
    report = {"source": source.as_posix(), "target": target.as_posix(), **stats,
              "utterance_drops_by_reason": dict(dropped),
              "transcript_words_before": words_before, "transcript_words_after": words_after}
    summary_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
