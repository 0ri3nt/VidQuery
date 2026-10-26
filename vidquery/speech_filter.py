"""Keep only transcript utterances that look like real speech.

Whisper hallucinates fluent text on silent or noise-only audio ("1.5 cm.",
"I'm going to do a little bit of a little bit of ...").  An utterance is kept
only if all of these hold:

* it contains a letter or digit (drops "," and "...");
* its zlib compression ratio is at most ``max_compression_ratio`` (Whisper's
  own repetition threshold, 2.4), which drops looping text;
* when diarization completed, it overlaps a diarized speech turn, i.e. its
  speaker is not ``UNKNOWN``.  Without completed diarization this check is
  skipped, so speech is never dropped merely because diarization was missing.
"""

from __future__ import annotations

import zlib
from collections import Counter

from .domain import TranscriptSegment

DEFAULT_MAX_COMPRESSION_RATIO = 2.4


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def filter_transcripts(
    transcripts: list[TranscriptSegment],
    *,
    diarization_completed: bool,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    """Return the kept utterances and a summary for segment metadata."""

    kept: list[TranscriptSegment] = []
    dropped: Counter[str] = Counter()
    for item in transcripts:
        text = item.text.strip()
        if not any(character.isalnum() for character in text):
            dropped["no_words"] += 1
        elif compression_ratio(text) > max_compression_ratio:
            dropped["repetition"] += 1
        elif diarization_completed and item.speaker == "UNKNOWN":
            dropped["no_diarized_speech"] += 1
        else:
            kept.append(item)
    summary: dict[str, object] = {
        "applied": True,
        "speech_gate": "diarization" if diarization_completed else "skipped_no_diarization",
        "kept": len(kept),
        "dropped": dict(sorted(dropped.items())),
    }
    return kept, summary
