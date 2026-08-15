from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from .config import PROJECT_ROOT, Settings
from .domain import (
    BoundingBox,
    Detection,
    FrameRecord,
    SpeakerSegment,
    TranscriptSegment,
)
from .geometry import centroid, interval_overlap, normalize_bbox
from .media import resolve_ffmpeg_binary

LOGGER = logging.getLogger(__name__)


class OptionalComponentUnavailable(RuntimeError):
    pass


class VisualDetector(Protocol):
    def detect(self, frames: list[FrameRecord]) -> list[Detection]: ...


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, video_id: str) -> list[TranscriptSegment]: ...


class Diarizer(Protocol):
    def diarize(self, audio_path: Path, video_id: str) -> list[SpeakerSegment]: ...


class NoopVisualDetector:
    def detect(self, frames: list[FrameRecord]) -> list[Detection]:
        return []


class NoopTranscriber:
    def transcribe(self, audio_path: Path, video_id: str) -> list[TranscriptSegment]:
        return []


class NoopDiarizer:
    def diarize(self, audio_path: Path, video_id: str) -> list[SpeakerSegment]:
        return []


def _stable_id(prefix: str, *parts: object) -> str:
    value = ":".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:20]}"


def _resolve_device(requested: str) -> str | int:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    LOGGER.warning("YOLO is running on CPU; inference will be slower")
    return "cpu"


def _whisper_device(requested: str) -> str:
    device = _resolve_device(requested)
    if device == 0:
        return "cuda"
    return str(device)


def _decode_audio_for_whisper(audio_path: Path):
    """Decode with the resolved FFmpeg binary instead of assuming `ffmpeg` is on PATH."""
    import numpy as np

    result = subprocess.run(
        [
            resolve_ffmpeg_binary(),
            "-nostdin",
            "-threads",
            "0",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0


class YoloVisualDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None

    def _model_reference(self) -> str:
        configured = Path(self.settings.yolo_model)
        candidates = [configured]
        if not configured.is_absolute():
            candidates.insert(0, PROJECT_ROOT / configured)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        if self.settings.allow_model_downloads:
            return self.settings.yolo_model
        raise OptionalComponentUnavailable(
            f"YOLO checkpoint '{self.settings.yolo_model}' is not available locally"
        )

    def _load(self):
        if not self.settings.enable_yolo:
            raise OptionalComponentUnavailable("YOLO is disabled by configuration")
        if self._model is None:
            cache_dir = self.settings.data_dir / "cache" / "ultralytics"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise OptionalComponentUnavailable("ultralytics is not installed") from exc
            self._model = YOLO(self._model_reference())
        return self._model

    def detect(self, frames: list[FrameRecord]) -> list[Detection]:
        if not frames:
            return []
        model = self._load()
        device = _resolve_device(self.settings.device)
        detections: list[Detection] = []
        for frame in frames:
            results = model.predict(
                source=frame.path,
                conf=self.settings.detection_confidence,
                iou=self.settings.detection_iou,
                device=device,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            names = getattr(result, "names", {})
            for index, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                coordinates = [float(value) for value in box.xyxy[0].tolist()]
                pixel_box = BoundingBox(
                    x1=coordinates[0],
                    y1=coordinates[1],
                    x2=coordinates[2],
                    y2=coordinates[3],
                )
                normalized_box = normalize_bbox(pixel_box, frame.width, frame.height)
                class_label = str(names.get(class_id, class_id))
                detections.append(
                    Detection(
                        detection_id=_stable_id(
                            "det",
                            frame.frame_id,
                            index,
                            class_id,
                            *(round(value, 2) for value in coordinates),
                        ),
                        frame_id=frame.frame_id,
                        video_id=frame.video_id,
                        timestamp=frame.timestamp,
                        class_id=class_id,
                        class_label=class_label,
                        confidence=min(1.0, max(0.0, float(box.conf.item()))),
                        pixel_bbox=pixel_box,
                        normalized_bbox=normalized_box,
                        centroid=centroid(pixel_box),
                        centroid_normalized=centroid(normalized_box),
                    )
                )
        return detections


def yolo_component_status(settings: Settings) -> str:
    """Report whether YOLO can actually be invoked, not only whether its flag is set."""

    if not settings.enable_yolo:
        return "disabled"
    if importlib.util.find_spec("ultralytics") is None:
        return "unavailable_missing_dependency"
    try:
        YoloVisualDetector(settings)._model_reference()
    except OptionalComponentUnavailable:
        return "unavailable_missing_checkpoint"
    return "available_configured"


class WhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._cache_dir = settings.whisper_cache_dir

    def _verify_local_model(self, whisper_module) -> None:
        configured = Path(self.settings.whisper_model)
        if configured.is_file():
            return
        if self.settings.whisper_model not in whisper_module._MODELS:
            raise OptionalComponentUnavailable(
                f"Unknown Whisper model '{self.settings.whisper_model}'"
            )
        filename = Path(urlparse(whisper_module._MODELS[self.settings.whisper_model]).path).name
        candidates = [
            self.settings.whisper_cache_dir / filename,
            Path.home() / ".cache" / "whisper" / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._cache_dir = candidate.parent
                return
        if not self.settings.allow_model_downloads:
            raise OptionalComponentUnavailable(
                f"Whisper model '{self.settings.whisper_model}' is not cached; "
                "set ALLOW_MODEL_DOWNLOADS=true to fetch it explicitly"
            )

    def _load(self):
        if not self.settings.enable_whisper:
            raise OptionalComponentUnavailable("Whisper is disabled by configuration")
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:
                raise OptionalComponentUnavailable("openai-whisper is not installed") from exc
            self._verify_local_model(whisper)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = whisper.load_model(
                self.settings.whisper_model,
                device=_whisper_device(self.settings.device),
                download_root=str(self._cache_dir),
            )
        return self._model

    def transcribe(self, audio_path: Path, video_id: str) -> list[TranscriptSegment]:
        model = self._load()
        audio = _decode_audio_for_whisper(audio_path)
        result = model.transcribe(audio, verbose=False)
        segments: list[TranscriptSegment] = []
        for index, item in enumerate(result.get("segments", [])):
            start = max(0.0, float(item.get("start", 0.0)))
            end = max(start, float(item.get("end", start)))
            log_probability = float(item.get("avg_logprob", 0.0))
            speech_probability = 1.0 - float(item.get("no_speech_prob", 0.0))
            confidence = min(
                1.0,
                max(
                    0.0,
                    math.exp(min(0.0, log_probability)) * speech_probability,
                ),
            )
            segments.append(
                TranscriptSegment(
                    segment_id=_stable_id("tx", video_id, index, f"{start:.3f}", f"{end:.3f}"),
                    video_id=video_id,
                    start_time=start,
                    end_time=end,
                    text=str(item.get("text", "")).strip(),
                    confidence=confidence,
                    speaker="UNKNOWN",
                )
            )
        return segments


def whisper_component_status(settings: Settings) -> str:
    """Report whether configured Whisper inference is locally runnable."""

    if not settings.enable_whisper:
        return "disabled"
    if importlib.util.find_spec("whisper") is None:
        return "unavailable_missing_dependency"
    try:
        whisper_module = importlib.import_module("whisper")
        WhisperTranscriber(settings)._verify_local_model(whisper_module)
    except OptionalComponentUnavailable:
        return "unavailable_missing_checkpoint"
    except Exception:
        return "unavailable_load_error"
    return "available_configured"


def diarization_component_status(settings: Settings) -> str:
    """Report mandatory Pyannote configuration without loading the large model."""

    if not settings.enable_diarization:
        return "required_but_disabled" if settings.require_diarization else "disabled"
    if not settings.huggingface_token:
        return "unavailable_missing_token"
    try:
        if importlib.util.find_spec("pyannote.audio") is None:
            return "unavailable_missing_dependency"
    except (ImportError, ModuleNotFoundError):
        return "unavailable_missing_dependency"
    suffix = "required" if settings.require_diarization else "optional"
    return f"available_configured_{suffix}"


class PyannoteDiarizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline: Any | None = None

    def _load(self):
        if not self.settings.enable_diarization:
            raise OptionalComponentUnavailable("Speaker diarization is disabled")
        if not self.settings.huggingface_token:
            raise OptionalComponentUnavailable("HUGGINGFACE_TOKEN is required for diarization")
        if self._pipeline is None:
            try:
                import torch
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise OptionalComponentUnavailable("pyannote.audio is not installed") from exc
            pipeline = Pipeline.from_pretrained(
                self.settings.pyannote_model,
                use_auth_token=self.settings.huggingface_token,
                cache_dir=self.settings.pyannote_cache_dir,
            )
            if pipeline is None:
                raise OptionalComponentUnavailable("Pyannote pipeline could not be loaded")
            device = self.settings.diarization_device.strip().lower()
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                LOGGER.warning("Pyannote is running on CPU; diarization will be slow")
            pipeline.to(torch.device(device))
            self._pipeline = pipeline
        return self._pipeline

    @staticmethod
    def _annotation(result):
        if hasattr(result, "itertracks"):
            return result
        if hasattr(result, "speaker_diarization"):
            return result.speaker_diarization
        if hasattr(result, "to_annotation"):
            return result.to_annotation()
        raise TypeError(f"Unsupported Pyannote output: {type(result)}")

    def diarize(self, audio_path: Path, video_id: str) -> list[SpeakerSegment]:
        result = self._load()(str(audio_path))
        segments: list[SpeakerSegment] = []
        for index, (turn, _, speaker) in enumerate(
            self._annotation(result).itertracks(yield_label=True)
        ):
            start = max(0.0, float(turn.start))
            end = max(start, float(turn.end))
            segments.append(
                SpeakerSegment(
                    segment_id=_stable_id("spk", video_id, index, f"{start:.3f}", f"{end:.3f}"),
                    video_id=video_id,
                    start_time=start,
                    end_time=end,
                    speaker=str(speaker),
                )
            )
        return segments


def assign_speakers(
    transcripts: list[TranscriptSegment], speakers: list[SpeakerSegment]
) -> list[TranscriptSegment]:
    aligned: list[TranscriptSegment] = []
    for transcript in transcripts:
        best_label = "UNKNOWN"
        best_overlap = 0.0
        for speaker in speakers:
            overlap = interval_overlap(
                transcript.start_time,
                transcript.end_time,
                speaker.start_time,
                speaker.end_time,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = speaker.speaker
        aligned.append(transcript.model_copy(update={"speaker": best_label}))
    return aligned
