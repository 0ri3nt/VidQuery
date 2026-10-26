from __future__ import annotations

import pytest

from vidquery.domain import TranscriptSegment
from vidquery.media import AudioTrackExtractor
from vidquery.speech_filter import compression_ratio, filter_transcripts


def _utterance(text: str, speaker: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(segment_id=f"t{start}", video_id="v", start_time=start,
                             end_time=start + 2, text=text, speaker=speaker)


def test_hallucinations_on_non_speech_are_dropped():
    # Real texts Whisper produced on silent Charades audio, none overlapping diarized speech.
    silent = [
        _utterance("1.5 cm.", "UNKNOWN", 0),
        _utterance("I'm going to make a video of the video.", "UNKNOWN", 2),
        _utterance("...", "UNKNOWN", 4),
        _utterance(",", "SPEAKER_00", 6),
        _utterance("I'm going to do a little bit of " + "a little bit of " * 40, "SPEAKER_00", 8),
    ]
    kept, summary = filter_transcripts(silent, diarization_completed=True)
    assert kept == []
    assert summary["dropped"] == {"no_diarized_speech": 2, "no_words": 2, "repetition": 1}
    assert summary["speech_gate"] == "diarization"


def test_real_speech_is_kept():
    speech = [
        _utterance("We discuss the software architecture.", "SPEAKER_00", 0),
        _utterance("The laptop runs the API with SQLite metadata.", "SPEAKER_01", 5),
        _utterance("Non ho detto niente a nessuno.", "SPEAKER_02", 10),
    ]
    kept, summary = filter_transcripts(speech, diarization_completed=True)
    assert kept == speech
    assert summary["kept"] == 3 and summary["dropped"] == {}


def test_speech_gate_is_skipped_without_completed_diarization():
    # Diarization disabled or failed: no speaker information, so nothing is dropped for it.
    unknown = [_utterance("Hello there, how are you?", "UNKNOWN", 0),
               _utterance("Fine.", "UNKNOWN", 2)]
    kept, summary = filter_transcripts(unknown, diarization_completed=False)
    assert kept == unknown
    assert summary["speech_gate"] == "skipped_no_diarization"


class _Audio(AudioTrackExtractor):
    def extract(self, video, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake wav")
        return output_path


class _Transcriber:
    def transcribe(self, audio_path, video_id):
        return [
            TranscriptSegment(segment_id="real", video_id=video_id, start_time=0.1,
                              end_time=0.4, text="the project budget"),
            TranscriptSegment(segment_id="hallucinated", video_id=video_id, start_time=1.2,
                              end_time=1.8, text="1.5 cm."),
        ]


class _Diarizer:
    def diarize(self, audio_path, video_id):
        from vidquery.domain import SpeakerSegment

        return [SpeakerSegment(segment_id="sp", video_id=video_id, start_time=0.0,
                               end_time=0.5, speaker="SPEAKER_00")]


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_applies_the_filter_only_when_enabled(settings_factory, sample_video, enabled):
    from vidquery.media import VideoIngestionService
    from vidquery.pipeline import VideoProcessingService
    from vidquery.storage import SQLiteRepository

    settings = settings_factory(enable_whisper=True, enable_diarization=True,
                                transcript_speech_filter=enabled)
    repository = SQLiteRepository(settings.database_path)
    record = VideoIngestionService(settings, repository).register_existing(sample_video)[0]
    processor = VideoProcessingService(settings, repository, audio_extractor=_Audio(),
                                       transcriber=_Transcriber(), diarizer=_Diarizer())
    processor.process(record.video_id)
    segments = repository.list_segments([record.video_id])
    texts = {item.text for segment in segments for item in segment.transcript_segments}
    metadata = segments[0].processing_metadata["transcript_speech_filter"]
    if enabled:
        assert texts == {"the project budget"}
        assert metadata == {"applied": True, "speech_gate": "diarization", "kept": 1,
                            "dropped": {"no_diarized_speech": 1}}
    else:
        assert texts == {"the project budget", "1.5 cm."}
        assert metadata == {"applied": False}


def test_compression_ratio_matches_whisper_threshold_behavior():
    assert compression_ratio("We discuss the software architecture.") < 2.4
    assert compression_ratio("a little bit of " * 30) > 2.4
    assert compression_ratio("") == 0.0
