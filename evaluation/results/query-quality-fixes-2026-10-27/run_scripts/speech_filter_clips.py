"""Speech filter on representative clips: real speech, movie dialogue, silence, noise, noisy speech.

Uses the pipeline's own WhisperTranscriber (settings from .env) and PyannoteDiarizer,
then vidquery.speech_filter. Writes every utterance with its keep/drop reason.
Usage: speech_filter_clips.py WORK_DIR OUT.json
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from vidquery.config import get_settings
from vidquery.media import resolve_ffmpeg_binary
from vidquery.models import PyannoteDiarizer, WhisperTranscriber, assign_speakers
from vidquery.speech_filter import compression_ratio, filter_transcripts

RATE = 16000
FINAL_DEMO_TEXT = (
    "We discuss the software architecture. The laptop runs the API with SQLite metadata. "
    "Deployment uses Docker for the backend and Neo4j graph. This whiteboard shows the "
    "retrieval architecture. The system searches speech, objects, and relationships."
)
# 60 s excerpts of real movie dialogue (AVA videos in the library), chosen by the
# densest stored transcript window; no reference transcript exists for them.
AVA_EXCERPTS = [("-IELREHX_js", 990), ("0f39OWEqJ24", 1680), ("-XpUuIgyUHE", 1110)]


def extract(source: Path, target: Path, start: float = 0.0, duration: float | None = None):
    command = [resolve_ffmpeg_binary(), "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source)]
    if duration:
        command += ["-t", str(duration)]
    command += ["-vn", "-ac", "1", "-ar", str(RATE), str(target)]
    subprocess.run(command, check=True)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / max(1, len(ref))


def pink_noise(seconds: float, level_dbfs: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(seconds * RATE))
    spectrum = np.fft.rfft(white) / np.sqrt(np.arange(1, len(white) // 2 + 2))
    pink = np.fft.irfft(spectrum, n=len(white))
    pink /= np.sqrt(np.mean(pink ** 2))
    return pink * 10 ** (level_dbfs / 20)


def main(work: Path, out: Path) -> None:
    settings = get_settings()
    work.mkdir(parents=True, exist_ok=True)
    clips: dict[str, dict] = {}
    demo = work / "final_demo.wav"
    extract(Path("data/demo/final_demo.mp4"), demo)
    speech, _ = sf.read(demo)
    clips["speech_final_demo"] = {"path": demo, "kind": "speech", "reference": FINAL_DEMO_TEXT}
    power = np.sqrt(np.mean(speech ** 2))
    for snr in (10, 0):
        noise = pink_noise(len(speech) / RATE, 0.0, seed=snr)
        noise *= power / 10 ** (snr / 20)
        path = work / f"final_demo_pink_snr{snr}.wav"
        sf.write(path, np.clip(speech + noise, -1, 1), RATE)
        clips[f"speech_plus_pink_noise_snr{snr}"] = {"path": path, "kind": "speech",
                                                     "reference": FINAL_DEMO_TEXT}
    sf.write(work / "silence.wav", np.zeros(30 * RATE), RATE)
    clips["silence_digital"] = {"path": work / "silence.wav", "kind": "non_speech"}
    rng = np.random.default_rng(3)
    sf.write(work / "room_tone.wav", rng.standard_normal(30 * RATE) * 10 ** (-50 / 20), RATE)
    clips["room_tone_-50dBFS"] = {"path": work / "room_tone.wav", "kind": "non_speech"}
    sf.write(work / "pink_noise.wav", pink_noise(30, -25.0), RATE)
    clips["pink_noise_-25dBFS"] = {"path": work / "pink_noise.wav", "kind": "non_speech"}
    for name, start in AVA_EXCERPTS:
        path = work / f"ava_{name}_{start}.wav"
        extract(Path(f"data/ava/videos/{name}.mp4"), path, start, 60)
        clips[f"movie_dialogue_{name}_{start}s"] = {"path": path, "kind": "speech"}

    transcriber, diarizer = WhisperTranscriber(settings), PyannoteDiarizer(settings)
    report = {}
    for name, clip in clips.items():
        transcripts = transcriber.transcribe(clip["path"], name)
        speakers = diarizer.diarize(clip["path"], name)
        aligned = assign_speakers(transcripts, speakers)
        kept, summary = filter_transcripts(aligned, diarization_completed=True)
        kept_ids = {id(item) for item in kept}
        before = " ".join(item.text for item in aligned)
        after = " ".join(item.text for item in kept)
        entry = {
            "kind": clip["kind"], "speaker_turns": len(speakers),
            "speech_seconds_diarized": round(sum(s.end_time - s.start_time for s in speakers), 1),
            "utterances_before": len(aligned), "utterances_after": len(kept),
            "words_before": len(words(before)), "words_after": len(words(after)),
            "filter": summary,
            "utterances": [{"start": round(u.start_time, 2), "end": round(u.end_time, 2),
                            "speaker": u.speaker, "compression": round(compression_ratio(u.text), 2),
                            "kept": id(u) in kept_ids, "text": u.text} for u in aligned],
        }
        if "reference" in clip:
            entry["wer_before"] = round(wer(clip["reference"], before), 3)
            entry["wer_after"] = round(wer(clip["reference"], after), 3)
        report[name] = entry
        print(f"{name:40} turns={entry['speaker_turns']:2} words {entry['words_before']:4} -> "
              f"{entry['words_after']:4}" + (f" WER {entry['wer_before']} -> {entry['wer_after']}"
                                              if "reference" in clip else ""), flush=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
