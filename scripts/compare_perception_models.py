"""Benchmark Whisper and YOLO variants on fixed local dataset samples."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from vidquery.media import resolve_ffmpeg_binary


def _decode_audio(path: Path, seconds: float) -> np.ndarray:
    import subprocess

    result = subprocess.run(
        [
            resolve_ffmpeg_binary(),
            "-nostdin",
            "-threads",
            "0",
            "-t",
            str(seconds),
            "-i",
            str(path),
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
    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0


def _normalize(text: str) -> list[str]:
    return "".join(char.lower() if char.isalnum() else " " for char in text).split()


def _wer(reference: str, hypothesis: str) -> float | None:
    expected = _normalize(reference)
    observed = _normalize(hypothesis)
    if not expected:
        return None
    previous = list(range(len(observed) + 1))
    for row, token in enumerate(expected, start=1):
        current = [row]
        for column, candidate in enumerate(observed, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(token != candidate),
                )
            )
        previous = current
    return round(previous[-1] / len(expected), 4)


def _reference_from_manifest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(str(item["transcript"]) for item in payload["segments"])


def compare_whisper(
    audio: Path, model_names: list[str], seconds: float, reference: str | None
) -> list[dict]:
    import whisper

    samples = _decode_audio(audio, seconds)
    results: list[dict] = []
    for name in model_names:
        started = time.perf_counter()
        model = whisper.load_model(name, download_root=str(Path("data/app/models/whisper")))
        loaded_at = time.perf_counter()
        output = model.transcribe(samples, verbose=False, word_timestamps=True)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "model": name,
                "audio": str(audio),
                "audio_seconds": round(len(samples) / 16000, 3),
                "runtime_seconds": round(elapsed, 3),
                "model_load_seconds": round(loaded_at - started, 3),
                "word_timestamps": any(item.get("words") for item in output.get("segments", [])),
                "wer": _wer(reference, output.get("text", "")) if reference else None,
                "text": output.get("text", "").strip(),
            }
        )
    return results


def _ensure_yolo_checkpoint(name: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        return target
    source = Path(name)
    if not source.exists():
        from ultralytics import YOLO

        YOLO(name)
    if not source.exists():
        raise FileNotFoundError(f"Ultralytics did not create {name}")
    shutil.copy2(source, target)
    return target


def compare_yolo(frames: list[Path], model_names: list[str], directory: Path) -> list[dict]:
    from ultralytics import YOLO

    results: list[dict] = []
    for name in model_names:
        checkpoint = _ensure_yolo_checkpoint(name, directory)
        started = time.perf_counter()
        model = YOLO(str(checkpoint))
        load_seconds = time.perf_counter() - started
        detection_count = 0
        class_counts: dict[str, int] = {}
        inference_started = time.perf_counter()
        for frame in frames:
            predictions = model(str(frame), verbose=False)
            for prediction in predictions:
                detection_count += len(prediction.boxes)
                for class_id in prediction.boxes.cls.tolist():
                    label = prediction.names[int(class_id)]
                    class_counts[label] = class_counts.get(label, 0) + 1
        results.append(
            {
                "model": name,
                "checkpoint": str(checkpoint),
                "frame_count": len(frames),
                "model_load_seconds": round(load_seconds, 3),
                "inference_seconds": round(time.perf_counter() - inference_started, 3),
                "detections": detection_count,
                "class_counts": dict(sorted(class_counts.items())),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--whisper-models", nargs="+", default=["tiny", "small", "medium", "turbo"])
    parser.add_argument("--whisper-seconds", type=float, default=120.0)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--frame-count", type=int, default=20)
    parser.add_argument(
        "--yolo-models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    )
    parser.add_argument("--yolo-directory", type=Path, default=Path("data/app/models/yolo"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = (
        _reference_from_manifest(args.reference_manifest) if args.reference_manifest else None
    )
    result: dict = {
        "schema_version": "perception-comparison-1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "whisper": compare_whisper(
            args.audio, args.whisper_models, args.whisper_seconds, reference
        ),
    }
    if args.frames_root:
        frames = sorted(args.frames_root.rglob("*.jpg"))[: args.frame_count]
        if not frames:
            raise SystemExit(f"No JPG frames found under {args.frames_root}")
        result["yolo"] = compare_yolo(frames, args.yolo_models, args.yolo_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote comparison results to {args.output}")


if __name__ == "__main__":
    main()
