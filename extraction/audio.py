import argparse
import csv
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt


def find_media_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts and p.is_file()])


class AudioExtractor:
    def __init__(
        self,
        whisper_model: str = "small",
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        hf_token: str | None = None,
    ):
        self.whisper_model_name = whisper_model
        self.diarization_model_name = diarization_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self._whisper_model = None
        self._diarization_pipeline = None

    @staticmethod
    def _ensure_ffmpeg_in_path() -> None:
        if shutil.which("ffmpeg"):
            return

        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = Path(ffmpeg_exe).parent
            shim_dir = ffmpeg_dir / "whisper_ffmpeg_shim"
            shim_dir.mkdir(parents=True, exist_ok=True)
            shim_exe = shim_dir / "ffmpeg.exe"

            if not shim_exe.exists():
                shutil.copyfile(ffmpeg_exe, shim_exe)

            os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        except Exception as exc:
            raise RuntimeError(
                "ffmpeg executable not found for Whisper. Install ffmpeg or imageio-ffmpeg."
            ) from exc

    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper

            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def _load_diarization(self):
        if self._diarization_pipeline is None:
            if not self.hf_token:
                raise ValueError("HF_TOKEN is required for pyannote diarization")

            warnings.filterwarnings(
                "ignore",
                message=(
                    r"(?s).*torchcodec is not installed correctly so built-in audio "
                    r"decoding will fail.*"
                ),
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"pyannote\.audio\.core\.io",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"pyannote\.audio\.models\.blocks\.pooling",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"numpy\._core\..*",
            )

            import torch
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(self.diarization_model_name, token=self.hf_token)
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            pipeline.to(device)
            print(f"  [INFO] Diarization device: {device}")
            self._diarization_pipeline = pipeline
        return self._diarization_pipeline

    def transcribe(self, audio_path: Path) -> dict:
        self._ensure_ffmpeg_in_path()
        model = self._load_whisper()
        return model.transcribe(str(audio_path))

    @staticmethod
    def _extract_annotation(diarization_output):
        if hasattr(diarization_output, "itertracks"):
            return diarization_output
        if hasattr(diarization_output, "speaker_diarization"):
            ann = diarization_output.speaker_diarization
            if hasattr(ann, "itertracks"):
                return ann
        if hasattr(diarization_output, "to_annotation"):
            ann = diarization_output.to_annotation()
            if hasattr(ann, "itertracks"):
                return ann
        raise TypeError(f"Unsupported diarization output type: {type(diarization_output)}")

    def diarize(self, audio_path: Path):
        pipeline = self._load_diarization()
        try:
            import soundfile as sf
            import torch
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            # Avoid backend decoder issues by passing in-memory waveform input.
            waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            payload = {
                "waveform": torch.from_numpy(waveform.T),
                "sample_rate": int(sample_rate),
            }
            start = time.perf_counter()
            with ProgressHook() as hook:
                result = pipeline(payload, hook=hook)
            elapsed = time.perf_counter() - start
            print(f"    [OK] Diarization inference finished in {elapsed:.1f}s")
            return result
        except Exception as exc:
            print(f"    [WARN] In-memory diarization path failed ({exc}); retrying file-path mode.")
            return pipeline(str(audio_path))

    @staticmethod
    def diarization_to_segments(diarization_output) -> list[dict]:
        segments = []
        annotation = AudioExtractor._extract_annotation(diarization_output)
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": str(speaker),
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "duration": round(float(segment.end - segment.start), 3),
                }
            )
        return segments

    @staticmethod
    def merge_audio_data(whisper_result: dict, speaker_segments: list[dict]) -> list[dict]:
        merged = []
        for seg in whisper_result.get("segments", []):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()

            best_speaker = "unknown"
            best_overlap = 0.0
            for spk in speaker_segments:
                overlap = max(0.0, min(end, spk["end"]) - max(start, spk["start"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = spk["speaker"]

            merged.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": best_speaker,
                    "text": text,
                }
            )

        return merged


def _seconds_to_srt_time(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_speaker_srt(segments: list[dict], srt_path: Path) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(
                f"{_seconds_to_srt_time(seg['start'])} --> {_seconds_to_srt_time(seg['end'])}\n"
            )
            f.write(f"{seg['speaker']}\n\n")


def write_segments_csv(segments: list[dict], csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "start", "end", "duration"])
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg)


def render_timeline_plot(segments: list[dict], png_path: Path, title: str) -> None:
    speakers = []
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speakers:
            speakers.append(speaker)

    if not speakers:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No speaker segments", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        return

    speaker_to_y = {speaker: idx for idx, speaker in enumerate(speakers)}
    max_end = max(float(seg["end"]) for seg in segments)

    cmap = plt.get_cmap("tab20")
    speaker_colors = {speaker: cmap(i % 20) for i, speaker in enumerate(speakers)}

    fig_h = max(3.0, 1.0 + len(speakers) * 0.7)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    for seg in segments:
        speaker = seg["speaker"]
        y = speaker_to_y[speaker]
        start = float(seg["start"])
        duration = float(seg["duration"])
        ax.broken_barh(
            [(start, duration)],
            (y - 0.35, 0.7),
            facecolors=speaker_colors[speaker],
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set_xlim(0, max_end + 1)
    ax.set_ylim(-1, len(speakers))
    ax.set_yticks(list(speaker_to_y.values()))
    ax.set_yticklabels(list(speaker_to_y.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main_whisper() -> None:
    parser = argparse.ArgumentParser(description="Run Whisper transcription on media files")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/ava/audio_outputs/whisper"))
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--language", type=str, default=None)
    args = parser.parse_args()

    extractor = AudioExtractor(whisper_model=args.model)
    files = find_media_files(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for media_path in files:
        try:
            result = extractor._load_whisper().transcribe(str(media_path), language=args.language)
            stem = media_path.stem
            (args.output_dir / f"{stem}.txt").write_text(
                result.get("text", "").strip() + "\n", encoding="utf-8"
            )
            with open(args.output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[OK] {media_path.name}")
        except Exception as exc:
            print(f"[!] Failed for {media_path.name}: {exc}")


def main_diarization() -> None:
    parser = argparse.ArgumentParser(description="Run speaker diarization on media files")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/ava/audio_outputs/diarization"),
    )
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--model", type=str, default="pyannote/speaker-diarization-3.1")
    args = parser.parse_args()

    extractor = AudioExtractor(diarization_model=args.model, hf_token=args.hf_token)
    files = find_media_files(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for media_path in files:
        try:
            diarization_output = extractor.diarize(media_path)
            annotation = extractor._extract_annotation(diarization_output)
            segments = extractor.diarization_to_segments(diarization_output)

            stem = media_path.stem
            with open(args.output_dir / f"{stem}.diarization.json", "w", encoding="utf-8") as f:
                json.dump({"file": str(media_path), "segments": segments}, f, indent=2)
            with open(args.output_dir / f"{stem}.rttm", "w", encoding="utf-8") as f:
                annotation.write_rttm(f)

            write_segments_csv(segments, args.output_dir / f"{stem}.segments.csv")
            write_speaker_srt(segments, args.output_dir / f"{stem}.speaker_turns.srt")
            render_timeline_plot(
                segments,
                args.output_dir / f"{stem}.timeline.png",
                f"Speaker Timeline: {media_path.name}",
            )
            print(f"[OK] {media_path.name}")
        except Exception as exc:
            print(f"[!] Failed for {media_path.name}: {exc}")


if __name__ == "__main__":
    # Keep default behavior aligned with dedicated wrappers.
    main_whisper()
