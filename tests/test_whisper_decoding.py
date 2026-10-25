from __future__ import annotations

from dataclasses import replace

import pytest

from vidquery import models
from vidquery.config import Settings
from vidquery.models import WhisperTranscriber, whisper_decoding_mode


class _RecordingWhisper:
    def __init__(self):
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": " hello", "avg_logprob": -0.1,
                              "no_speech_prob": 0.1}]}


def _transcriber(monkeypatch, **overrides):
    settings = replace(Settings(), enable_whisper=True, **overrides)
    transcriber = WhisperTranscriber(settings)
    fake = _RecordingWhisper()
    transcriber._model = fake
    monkeypatch.setattr(models, "_decode_audio_for_whisper", lambda path: [0.0])
    return transcriber, fake


def test_default_decoding_is_greedy_temperature_zero(monkeypatch, tmp_path):
    transcriber, fake = _transcriber(monkeypatch)
    segments = transcriber.transcribe(tmp_path / "audio.wav", "video-1")
    assert segments[0].text.strip() == "hello"
    assert fake.calls[0]["temperature"] == 0.0
    assert whisper_decoding_mode(transcriber.settings) == "greedy_temperature_0"


def test_fallback_option_restores_library_default(monkeypatch, tmp_path):
    transcriber, fake = _transcriber(monkeypatch, whisper_temperature_fallback=True)
    transcriber.transcribe(tmp_path / "audio.wav", "video-1")
    assert "temperature" not in fake.calls[0]
    assert whisper_decoding_mode(transcriber.settings) == "temperature_fallback_sampling"


def test_environment_variable_controls_fallback(monkeypatch):
    monkeypatch.delenv("WHISPER_TEMPERATURE_FALLBACK", raising=False)
    assert Settings().whisper_temperature_fallback is False
    monkeypatch.setenv("WHISPER_TEMPERATURE_FALLBACK", "true")
    assert Settings().whisper_temperature_fallback is True


@pytest.mark.parametrize("fallback", [False, True])
def test_old_whisper_without_word_timestamps_keeps_decoding_options(
    monkeypatch, tmp_path, fallback
):
    transcriber, _ = _transcriber(monkeypatch, whisper_temperature_fallback=fallback)
    calls: list[dict] = []

    class _OldWhisper:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            if "word_timestamps" in kwargs:
                raise TypeError("unexpected keyword argument 'word_timestamps'")
            return {"segments": []}

    transcriber._model = _OldWhisper()
    transcriber.transcribe(tmp_path / "audio.wav", "video-1")
    assert ("temperature" in calls[-1]) is (not fallback)
