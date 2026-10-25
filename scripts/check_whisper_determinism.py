"""Measure whether Whisper transcripts repeat exactly on the same audio.

Transcribes each ``audio.wav`` under ``--generated-dir`` ``--repeats`` times in
both decoding modes (greedy temperature 0, and the library's temperature
fallback) and reports the share of files whose transcript text and segment
timings are identical across repeats. ``--hashes-output`` writes the greedy
transcript hashes so a second process can be compared with ``--compare``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from vidquery.config import get_settings
from vidquery.models import WhisperTranscriber


def _fingerprint(segments) -> str:
    payload = [(round(item.start_time, 3), round(item.end_time, 3), item.text) for item in segments]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["greedy", "fallback"],
                        choices=["greedy", "fallback"])
    parser.add_argument("--hashes-output", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None,
                        help="greedy hashes from an earlier process to compare against")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new file name")
    audio = sorted(args.generated_dir.glob("*/audio.wav"))[: args.limit]
    settings = get_settings()
    report: dict = {"files": len(audio), "repeats": args.repeats,
                    "whisper_model": settings.whisper_model,
                    "modes": {}}
    greedy_hashes: dict[str, str] = {}
    for mode in args.modes:
        transcriber = WhisperTranscriber(
            replace(settings, whisper_temperature_fallback=(mode == "fallback"))
        )
        identical = nonempty = 0
        started = time.perf_counter()
        for path in audio:
            prints = [_fingerprint(transcriber.transcribe(path, path.parent.name))
                      for _ in range(args.repeats)]
            identical += len(set(prints)) == 1
            nonempty += prints[0] != _fingerprint([])
            if mode == "greedy":
                greedy_hashes[path.parent.name] = prints[0]
        report["modes"][mode] = {
            "files_identical_across_repeats": identical,
            "share_identical": round(identical / max(1, len(audio)), 4),
            "files_with_speech_text": nonempty,
            "seconds": round(time.perf_counter() - started, 1),
        }
    if args.compare and greedy_hashes:
        earlier = json.loads(args.compare.read_text(encoding="utf-8"))
        shared = sorted(set(earlier) & set(greedy_hashes))
        same = sum(earlier[key] == greedy_hashes[key] for key in shared)
        report["greedy_cross_process"] = {"files": len(shared), "identical": same}
    if args.hashes_output and greedy_hashes:
        args.hashes_output.write_text(json.dumps(greedy_hashes, indent=1) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
