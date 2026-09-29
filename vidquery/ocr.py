from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio  # type: ignore[import-not-found]

from .config import Settings
from .domain import FrameRecord, OCREvidence, RelationshipSource


def normalize_ocr_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\-./]+", " ", text.strip().lower()).split())


def easyocr_status(settings: Settings) -> str:
    if not settings.enable_ocr:
        return "disabled"
    if importlib.util.find_spec("easyocr") is None:
        return "unavailable_missing_dependency"
    required = (
        settings.ocr_model_dir / "craft_mlt_25k.pth",
        settings.ocr_model_dir / "english_g2.pth",
    )
    if not all(path.is_file() for path in required):
        return "unavailable_missing_checkpoint"
    return "available_cached"


class EasyOCRExtractor:
    def __init__(self, settings: Settings, reader: Any | None = None) -> None:
        self.settings = settings
        self._reader = reader

    def _load(self) -> Any:
        if self._reader is not None:
            return self._reader
        if easyocr_status(self.settings) != "available_cached":
            raise RuntimeError("EasyOCR is enabled but its validated model cache is unavailable")
        import easyocr  # type: ignore[import-not-found]
        import torch

        device = self.settings.ocr_device.strip().lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu: bool | str = device if device != "cpu" else False
        self._reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            model_storage_directory=str(self.settings.ocr_model_dir),
            download_enabled=self.settings.allow_model_downloads,
            verbose=False,
        )
        return self._reader

    def extract(self, frames: list[FrameRecord]) -> list[OCREvidence]:
        reader = self._load()
        active: list[dict[str, Any]] = []
        interval = 1.0 / max(self.settings.frame_sample_rate, 0.001)
        for frame in sorted(frames, key=lambda item: item.timestamp):
            source: Any = frame.path
            if Path(frame.path).is_file():
                import cv2

                grayscale = cv2.imread(frame.path, cv2.IMREAD_GRAYSCALE)
                if grayscale is not None:
                    source = grayscale
            results = reader.readtext(source, detail=1, paragraph=False)
            for _, raw_text, raw_confidence in results:
                text = normalize_ocr_text(str(raw_text))
                confidence = float(raw_confidence)
                if len(text) < 2 or confidence < self.settings.ocr_min_confidence:
                    continue
                match = next(
                    (
                        row
                        for row in reversed(active)
                        if frame.timestamp - float(row["last_seen"]) <= interval * 1.5
                        and ratio(text, str(row["text"])) / 100
                        >= self.settings.ocr_dedup_similarity
                    ),
                    None,
                )
                if match is None:
                    active.append(
                        {
                            "text": text,
                            "start_time": frame.timestamp,
                            "end_time": frame.timestamp + interval,
                            "last_seen": frame.timestamp,
                            "confidences": [confidence],
                            "frame_ids": [frame.frame_id],
                        }
                    )
                else:
                    match["end_time"] = frame.timestamp + interval
                    match["last_seen"] = frame.timestamp
                    match["confidences"].append(confidence)
                    match["frame_ids"].append(frame.frame_id)
        output = []
        for row in active:
            raw_id = f"{row['text']}:{float(row['start_time']):.3f}"
            output.append(
                OCREvidence(
                    ocr_id="ocr_" + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20],
                    text=str(row["text"]),
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    confidence=sum(row["confidences"]) / len(row["confidences"]),
                    frame_ids=list(dict.fromkeys(row["frame_ids"])),
                    source_method=RelationshipSource.EASY_OCR,
                )
            )
        return output
