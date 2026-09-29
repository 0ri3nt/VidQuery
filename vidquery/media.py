from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2

from .config import Settings
from .domain import FrameRecord, ProcessingState, VideoRecord, utc_now
from .storage import SQLiteRepository


class MediaValidationError(ValueError):
    pass


class UploadTooLargeError(MediaValidationError):
    pass


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def sanitize_filename(filename: str) -> str:
    basename = Path(filename.replace("\\", "/")).name
    normalized = unicodedata.normalize("NFKC", basename)
    stem = Path(normalized).stem
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._")
    if not stem:
        stem = "video"
    return f"{stem[:120]}.mp4"


def resolve_ffmpeg_binary() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on host install
        raise MediaValidationError(
            "FFmpeg is required. Install ffmpeg or the imageio-ffmpeg package."
        ) from exc


def _looks_like_mp4(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(32)
    return len(header) >= 12 and b"ftyp" in header[4:16]


def probe_video(path: Path) -> MediaMetadata:
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise MediaValidationError("Only MP4 files are accepted")
    if not path.is_file() or path.stat().st_size == 0:
        raise MediaValidationError("Uploaded video is empty or missing")
    if not _looks_like_mp4(path):
        raise MediaValidationError("File content is not a recognizable MP4 container")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise MediaValidationError("OpenCV could not open the MP4 video stream")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        ok, _ = capture.read()
        if not ok or width <= 0 or height <= 0 or fps <= 0:
            raise MediaValidationError("MP4 does not contain a readable video stream")
        duration = frame_count / fps if frame_count > 0 else 0.0
    finally:
        capture.release()

    ffmpeg = resolve_ffmpeg_binary()
    inspect_result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
        check=False,
    )
    stream_description = (inspect_result.stderr or "") + (inspect_result.stdout or "")
    has_audio = "Audio:" in stream_description

    validation = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            "0.25",
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
        check=False,
    )
    if validation.returncode != 0:
        detail = (validation.stderr or "media decode failed").strip().splitlines()[-1]
        raise MediaValidationError(f"FFmpeg could not decode the video stream: {detail}")

    return MediaMetadata(
        duration=max(0.0, duration),
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
    )


class VideoIngestionService:
    def __init__(self, settings: Settings, repository: SQLiteRepository):
        self.settings = settings
        self.repository = repository
        self.settings.ensure_directories()

    def ingest_stream(
        self,
        stream: BinaryIO,
        filename: str,
        content_type: str | None,
    ) -> tuple[VideoRecord, bool]:
        if Path(filename.replace("\\", "/")).suffix.lower() != ".mp4":
            raise MediaValidationError("Only .mp4 uploads are accepted")
        if content_type and content_type.lower() not in {
            "video/mp4",
            "application/mp4",
            "application/octet-stream",
        }:
            raise MediaValidationError("Upload content type must be video/mp4")

        video_id = str(uuid.uuid4())
        temporary_path = self.settings.upload_dir / f"{video_id}.upload.mp4"
        final_path = self.settings.upload_dir / f"{video_id}.mp4"
        digest = hashlib.sha256()
        size = 0

        try:
            with temporary_path.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.max_upload_size:
                        raise UploadTooLargeError(
                            f"Upload exceeds {self.settings.max_upload_size} bytes"
                        )
                    digest.update(chunk)
                    output.write(chunk)

            content_sha256 = digest.hexdigest()
            duplicate = self.repository.find_by_hash(content_sha256)
            if duplicate is not None:
                temporary_path.unlink(missing_ok=True)
                return duplicate, True

            metadata = probe_video(temporary_path)
            os.replace(temporary_path, final_path)
            now = utc_now()
            original = Path(filename.replace("\\", "/")).name[:255] or "video.mp4"
            record = VideoRecord(
                video_id=video_id,
                display_name=sanitize_filename(original),
                original_filename=original,
                stored_path=str(final_path),
                content_sha256=content_sha256,
                duration=metadata.duration,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
                has_audio=metadata.has_audio,
                upload_time=now,
                updated_time=now,
                state=ProcessingState.UPLOADED,
                current_stage="stored",
            )
            return self.repository.create_video(record), False
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def register_existing(
        self, path: Path, *, copy_into_uploads: bool = False
    ) -> tuple[VideoRecord, bool]:
        path = Path(path).resolve()
        if not path.is_file():
            raise MediaValidationError(f"Video does not exist: {path}")
        if path.stat().st_size > self.settings.max_upload_size:
            raise UploadTooLargeError(f"Video exceeds {self.settings.max_upload_size} bytes")

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        content_sha256 = digest.hexdigest()
        duplicate = self.repository.find_by_hash(content_sha256)
        if duplicate is not None:
            return duplicate, True

        metadata = probe_video(path)
        video_id = str(uuid.uuid4())
        stored_path = path
        if copy_into_uploads:
            stored_path = self.settings.upload_dir / f"{video_id}.mp4"
            shutil.copyfile(path, stored_path)

        now = utc_now()
        record = VideoRecord(
            video_id=video_id,
            display_name=sanitize_filename(path.name),
            original_filename=path.name,
            stored_path=str(stored_path),
            content_sha256=content_sha256,
            duration=metadata.duration,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            has_audio=metadata.has_audio,
            upload_time=now,
            updated_time=now,
            state=ProcessingState.UPLOADED,
            current_stage="registered",
        )
        return self.repository.create_video(record), False


class FrameExtractor:
    def __init__(self, sample_rate: float = 1.0):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate

    def extract(self, video: VideoRecord, output_dir: Path) -> list[FrameRecord]:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(video.stored_path)
        if not capture.isOpened():
            raise MediaValidationError("Could not open video for frame extraction")

        interval = 1.0 / self.sample_rate
        timestamp = 0.0
        records: list[FrameRecord] = []
        frame_index = 0
        try:
            while timestamp < video.duration + 1e-6:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    break
                position_ms = capture.get(cv2.CAP_PROP_POS_MSEC) or timestamp * 1000.0
                actual_time = float(position_ms) / 1000.0
                frame_id = f"{video.video_id}_{int(round(timestamp * 1000)):012d}"
                frame_path = output_dir / f"{frame_id}.jpg"
                if not frame_path.exists() and not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Could not write extracted frame: {frame_path}")
                height, width = frame.shape[:2]
                records.append(
                    FrameRecord(
                        frame_id=frame_id,
                        video_id=video.video_id,
                        frame_index=frame_index,
                        timestamp=max(0.0, actual_time),
                        path=str(frame_path),
                        width=width,
                        height=height,
                    )
                )
                timestamp += interval
                frame_index += 1
        finally:
            capture.release()
        return records


class AudioTrackExtractor:
    def extract(self, video: VideoRecord, output_path: Path) -> Path | None:
        if not video.has_audio:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = resolve_ffmpeg_binary()
        result = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                video.stored_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            shell=False,
            timeout=max(60, int(video.duration * 2 + 30)),
            check=False,
        )
        if result.returncode != 0 or not output_path.exists():
            output_path.unlink(missing_ok=True)
            return None
        return output_path
