import argparse
import json
from pathlib import Path

import whisper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI Whisper transcription on audio/video files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file or directory. Supports common audio/video formats.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/ava/audio_outputs/whisper"),
        help="Directory where transcript outputs are saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size: tiny/base/small/medium/large.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code, e.g. en. If omitted, Whisper auto-detects.",
    )
    return parser


def find_media_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts and p.is_file()])


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_file(model, media_path: Path, output_dir: Path, language: str | None) -> None:
    result = model.transcribe(str(media_path), language=language)

    stem = media_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    write_text(txt_path, result.get("text", ""))
    write_json(json_path, result)

    print(f"[OK] {media_path.name} -> {txt_path.name}, {json_path.name}")


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    files = find_media_files(args.input)
    if not files:
        print("No media files found.")
        return

    print(f"Found {len(files)} media file(s).")
    for media_path in files:
        try:
            run_file(model, media_path, args.output_dir, args.language)
        except Exception as exc:
            print(f"[!] Failed for {media_path.name}: {exc}")


if __name__ == "__main__":
    main()
