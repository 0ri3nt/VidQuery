from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from vidquery.domain import FrameRecord, RelationshipSource
from vidquery.ocr import EasyOCRExtractor, normalize_ocr_text
from vidquery.search import HashingTextEncoder, LocalHybridSearchEngine, StructuredQueryParser


class StubReader:
    def readtext(self, path, **_kwargs):
        if path.endswith("000.jpg"):
            return [([], "Kubernetes Deployment", 0.92)]
        if path.endswith("001.jpg"):
            return [([], "Kubernetes  Deployment!", 0.88)]
        return [([], "unreliable", 0.1)]


def _frame(frame_id: str, timestamp: float, path: str) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id="video-1",
        frame_index=int(timestamp),
        timestamp=timestamp,
        path=path,
        width=100,
        height=100,
    )


def test_ocr_normalization_and_temporal_deduplication(settings_factory):
    settings = settings_factory(
        enable_ocr=True,
        frame_sample_rate=1.0,
        ocr_min_confidence=0.25,
        ocr_dedup_similarity=0.9,
    )
    evidence = EasyOCRExtractor(settings, reader=StubReader()).extract(
        [
            _frame("frame-0", 0.0, "frame_000.jpg"),
            _frame("frame-1", 1.0, "frame_001.jpg"),
            _frame("frame-2", 2.0, "frame_002.jpg"),
        ]
    )

    assert normalize_ocr_text("  KUBERNETES!!! Deployment ") == "kubernetes deployment"
    assert len(evidence) == 1
    assert evidence[0].text == "kubernetes deployment"
    assert evidence[0].start_time == 0.0
    assert evidence[0].end_time == 2.0
    assert evidence[0].frame_ids == ["frame-0", "frame-1"]
    assert evidence[0].source_method is RelationshipSource.EASY_OCR

    encoder = HashingTextEncoder()
    engine = LocalHybridSearchEngine(cast(Any, SimpleNamespace()), encoder=encoder)
    query = "Kubernetes"
    score, reasons, _ = engine._score(
        StructuredQueryParser().parse(query),
        encoder.encode(query),
        SimpleNamespace(
            transcript="",
            entities=[],
            actions=[],
            relationships=[],
            speakers=[],
            processing_metadata={},
            embedding=None,
            ocr_evidence=evidence,
        ),
    )
    assert score > 0
    assert "OCR terms kubernetes (easyocr)" in reasons
