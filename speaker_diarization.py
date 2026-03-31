import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pyannote.audio import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization on an audio/video file or folder of files."
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
        default=Path("data/ava/audio_outputs/diarization"),
        help="Directory where diarization JSON and RTTM files are saved.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face token. If omitted, reads HF_TOKEN from environment.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Hugging Face model id for diarization pipeline.",
    )
    return parser


def find_media_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".webm"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts and p.is_file()])


def _extract_annotation(diarization_output):
    """Support pyannote outputs across versions."""
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

    raise TypeError(
        f"Unsupported diarization output type: {type(diarization_output)}"
    )


def diarization_to_segments(diarization_output) -> list[dict]:
    segments = []
    annotation = _extract_annotation(diarization_output)
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


def _seconds_to_srt_time(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_segments_csv(segments: list[dict], csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speaker", "start", "end", "duration"]
        )
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg)


def write_speaker_srt(segments: list[dict], srt_path: Path) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(
                f"{_seconds_to_srt_time(seg['start'])} --> "
                f"{_seconds_to_srt_time(seg['end'])}\n"
            )
            f.write(f"{seg['speaker']}\n\n")


def write_summary(segments: list[dict], summary_path: Path, source_file: Path) -> None:
    if not segments:
        summary = (
            f"File: {source_file.name}\n"
            "No speaker segments found.\n"
        )
        summary_path.write_text(summary, encoding="utf-8")
        return

    by_speaker = {}
    total = 0.0
    for seg in segments:
        spk = seg["speaker"]
        dur = float(seg["duration"])
        by_speaker[spk] = by_speaker.get(spk, 0.0) + dur
        total += dur

    lines = [
        f"File: {source_file.name}",
        f"Total speaking time: {total:.2f} seconds",
        "",
        "Per-speaker totals:",
    ]

    for spk, dur in sorted(by_speaker.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * dur / total if total > 0 else 0.0
        lines.append(f"- {spk}: {dur:.2f}s ({pct:.1f}%)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    speaker_colors = {
        speaker: cmap(i % 20) for i, speaker in enumerate(speakers)
    }

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


def run_file(pipeline: Pipeline, media_path: Path, output_dir: Path) -> None:
    diarization_output = pipeline(str(media_path))
    annotation = _extract_annotation(diarization_output)

    stem = media_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.diarization.json"
    rttm_path = output_dir / f"{stem}.rttm"
    csv_path = output_dir / f"{stem}.segments.csv"
    srt_path = output_dir / f"{stem}.speaker_turns.srt"
    summary_path = output_dir / f"{stem}.summary.txt"
    timeline_path = output_dir / f"{stem}.timeline.png"

    segments = diarization_to_segments(diarization_output)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"file": str(media_path), "segments": segments}, f, indent=2)

    with open(rttm_path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)

    write_segments_csv(segments, csv_path)
    write_speaker_srt(segments, srt_path)
    write_summary(segments, summary_path, media_path)
    render_timeline_plot(segments, timeline_path, f"Speaker Timeline: {media_path.name}")

    print(
        f"[OK] {media_path.name} -> "
        f"{json_path.name}, {rttm_path.name}, {csv_path.name}, "
        f"{srt_path.name}, {summary_path.name}, {timeline_path.name}"
    )


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    if not args.hf_token:
        raise ValueError(
            "Hugging Face token is required. Pass --hf_token or set HF_TOKEN in your environment."
        )

    print("Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(args.model, token=args.hf_token)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline.to(device)
    print(f"Using device: {device}")

    files = find_media_files(args.input)
    if not files:
        print("No media files found.")
        return

    print(f"Found {len(files)} media file(s).")
    for media_path in files:
        try:
            run_file(pipeline, media_path, args.output_dir)
        except Exception as exc:
            print(f"[!] Failed for {media_path.name}: {exc}")


if __name__ == "__main__":
    main()
