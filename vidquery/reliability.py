"""Reliability-aware evidence fusion and abstention.

The Review-1 ranker treated every matched modality as equally trustworthy.
That is not true of this system: transcript token hits are near-certain, the
YOLO detector is strong on COCO classes, and the AVA action GNN / VidOR
relation GNN are weak (held-out macro F1 0.07 / 0.18).  This module keeps the
rank score untouched for reproducibility and computes a *separate* evidence
confidence per result from three factors:

* ``raw_score``          - how well the stored evidence matched the plan;
* ``reliability_prior``  - how much the producing model deserves to be trusted;
* ``query_weight``       - how central the modality is to the query intent.

The confidence feeds a three-way verdict (supported / weak / insufficient) so
the API can abstain instead of returning a confident wrong timestamp.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import (
    EvidenceAssessment,
    EvidenceContribution,
    EvidenceModality,
    EvidenceVerdict,
    QueryPlan,
)

# Priors are documented design constants derived from the measured results in
# docs/GNN_STATUS.md and docs/EVALUATION_FINAL.md. They are *not* calibrated
# probabilities and must be reported as such.
SOURCE_RELIABILITY: dict[str, float] = {
    "whisper_transcript": 0.90,
    "whisper_word_timing": 0.90,
    "easyocr": 0.80,
    "yolo_detection": 0.80,
    "pyannote_diarization": 0.75,
    "bounding_box_geometry": 0.65,
    "sentence_embedding": 0.60,
    "openclip_appearance": 0.55,
    "visual_attribute_model": 0.55,
    "relationship_gnn": 0.35,
    "gnn_action_model": 0.30,
    "gnn_action_plus_target_resolver": 0.25,
    "learned_action_model": 0.30,
    "heuristic_action_mapping": 0.40,
    "ava_ground_truth": 1.00,
    "manual_annotation": 1.00,
    "hashing_embedding": 0.35,
    "unknown": 0.50,
}

# How central each modality is to each retrieval intent.
_ALL_MODALITIES: tuple[EvidenceModality, ...] = (
    "transcript",
    "semantic",
    "ocr",
    "entity",
    "action",
    "relationship",
    "speaker",
    "appearance",
)
QUERY_MODALITY_WEIGHTS: dict[str, dict[EvidenceModality, float]] = {
    "transcript_search": {
        "transcript": 1.0,
        "semantic": 0.5,
        "ocr": 0.3,
        "entity": 0.2,
        "action": 0.2,
        "relationship": 0.2,
        "speaker": 0.6,
        "appearance": 0.2,
    },
    "ocr_search": {
        "transcript": 0.3,
        "semantic": 0.4,
        "ocr": 1.0,
        "entity": 0.3,
        "action": 0.2,
        "relationship": 0.2,
        "speaker": 0.2,
        "appearance": 0.2,
    },
    "visual_search": {
        "transcript": 0.3,
        "semantic": 0.3,
        "ocr": 0.3,
        "entity": 1.0,
        "action": 0.5,
        "relationship": 0.5,
        "speaker": 0.2,
        "appearance": 0.9,
    },
    "action_search": {
        "transcript": 0.3,
        "semantic": 0.3,
        "ocr": 0.2,
        "entity": 0.6,
        "action": 1.0,
        "relationship": 0.5,
        "speaker": 0.2,
        "appearance": 0.4,
    },
    "relationship_search": {
        "transcript": 0.3,
        "semantic": 0.3,
        "ocr": 0.2,
        "entity": 0.6,
        "action": 0.4,
        "relationship": 1.0,
        "speaker": 0.2,
        "appearance": 0.4,
    },
    "speaker_search": {
        "transcript": 0.6,
        "semantic": 0.4,
        "ocr": 0.2,
        "entity": 0.2,
        "action": 0.2,
        "relationship": 0.2,
        "speaker": 1.0,
        "appearance": 0.2,
    },
    "multimodal_search": {
        "transcript": 1.0,
        "semantic": 0.8,
        "ocr": 0.8,
        "entity": 1.0,
        "action": 0.8,
        "relationship": 0.8,
        "speaker": 0.8,
        "appearance": 0.8,
    },
}

DEFAULT_SUPPORTED_THRESHOLD = 0.45
DEFAULT_WEAK_THRESHOLD = 0.15


@dataclass(frozen=True, slots=True)
class RawContribution:
    """Modality evidence emitted by the ranker before reliability weighting."""

    modality: EvidenceModality
    raw_score: float
    source_method: str


def load_reliability_priors(path: Path) -> dict[str, float]:
    """Load validated learned priors without changing the hand-picked defaults."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("priors") if isinstance(payload, dict) and "priors" in payload else payload
    if not isinstance(values, dict) or not values:
        raise ValueError("reliability priors must be a non-empty JSON object")
    unknown = sorted(set(values) - set(SOURCE_RELIABILITY))
    if unknown:
        raise ValueError(f"reliability priors contain unknown sources: {', '.join(unknown)}")
    result: dict[str, float] = {}
    for source, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
            raise ValueError(f"reliability prior for {source!r} must be a number in [0, 1]")
        result[source] = float(value)
    return result


def fit_reliability_priors(
    observations: list[Mapping[str, Any]], *, minimum_observations: int = 10, strength: int = 5
) -> dict[str, float]:
    """Fit source priors from calibration observations with prior smoothing."""

    if minimum_observations < 1 or strength < 0:
        raise ValueError("minimum_observations must be positive and strength non-negative")
    counts: dict[str, tuple[int, int]] = {}
    for observation in observations:
        source = str(observation.get("source_method", "")).strip()
        if source not in SOURCE_RELIABILITY:
            raise ValueError(f"unknown observation source: {source!r}")
        correct = observation.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("reliability observations require a boolean 'correct' field")
        total, hits = counts.get(source, (0, 0))
        counts[source] = (total + 1, hits + int(correct))

    fitted = dict(SOURCE_RELIABILITY)
    for source, (total, hits) in counts.items():
        if total >= minimum_observations:
            prior = SOURCE_RELIABILITY[source]
            fitted[source] = (hits + strength * prior) / (total + strength)
    return {source: round(value, 6) for source, value in fitted.items()}


def reliability_prior(source_method: str, priors: Mapping[str, float] | None = None) -> float:
    key = source_method.strip().lower()
    values = priors or SOURCE_RELIABILITY
    if key in values:
        return values[key]
    if key in SOURCE_RELIABILITY:
        return SOURCE_RELIABILITY[key]
    # Model names such as ``sentence-transformers/all-MiniLM-L6-v2``.
    if "minilm" in key or "sentence" in key or "mpnet" in key:
        return values.get("sentence_embedding", SOURCE_RELIABILITY["sentence_embedding"])
    if "hashing" in key:
        return values.get("hashing_embedding", SOURCE_RELIABILITY["hashing_embedding"])
    if "clip" in key:
        return values.get("openclip_appearance", SOURCE_RELIABILITY["openclip_appearance"])
    return values.get("unknown", SOURCE_RELIABILITY["unknown"])


def modality_weight(intent: str, modality: EvidenceModality) -> float:
    table = QUERY_MODALITY_WEIGHTS.get(intent, QUERY_MODALITY_WEIGHTS["multimodal_search"])
    return float(table.get(modality, 0.5))


def assess_evidence(
    plan: QueryPlan,
    contributions: list[RawContribution],
    *,
    supported_threshold: float = DEFAULT_SUPPORTED_THRESHOLD,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    priors: Mapping[str, float] | None = None,
) -> EvidenceAssessment:
    """Fuse per-modality evidence into one confidence and an abstention verdict."""

    weighted: list[EvidenceContribution] = []
    numerator = 0.0
    denominator = 0.0
    for item in contributions:
        raw = max(0.0, min(1.0, item.raw_score))
        prior = reliability_prior(item.source_method, priors)
        weight = modality_weight(plan.intent, item.modality)
        value = raw * prior
        numerator += weight * value
        denominator += weight
        weighted.append(
            EvidenceContribution(
                modality=item.modality,
                raw_score=round(raw, 4),
                source_method=item.source_method,
                reliability_prior=prior,
                query_weight=weight,
                weighted_score=round(value, 4),
            )
        )
    confidence = numerator / denominator if denominator else 0.0
    confidence = max(0.0, min(1.0, confidence))
    verdict: EvidenceVerdict
    if confidence >= supported_threshold:
        verdict = "supported"
    elif confidence >= weak_threshold:
        verdict = "weak"
    else:
        verdict = "insufficient"

    strongest = sorted(weighted, key=lambda item: -item.weighted_score * item.query_weight)
    if not strongest:
        explanation = "No modality produced evidence for this query."
    else:
        lead = strongest[0]
        explanation = (
            f"{lead.modality} evidence from {lead.source_method} "
            f"(match {lead.raw_score:.2f} x reliability {lead.reliability_prior:.2f}) "
            f"dominates; fused confidence {confidence:.2f} -> {verdict}."
        )
        weak_sources = [
            item
            for item in weighted
            if item.reliability_prior < 0.5 and item.query_weight >= 0.8
        ]
        if weak_sources:
            names = ", ".join(sorted({item.source_method for item in weak_sources}))
            explanation += f" Low-reliability model output ({names}) is down-weighted."
    return EvidenceAssessment(
        confidence=round(confidence, 4),
        verdict=verdict,
        contributions=weighted,
        explanation=explanation,
    )


def reliability_rank_score(score: float, assessment: EvidenceAssessment | None) -> float:
    """Blend the transparent rank score with fused evidence confidence."""

    if assessment is None:
        return score
    return max(0.0, min(1.0, 0.5 * score + 0.5 * assessment.confidence))
