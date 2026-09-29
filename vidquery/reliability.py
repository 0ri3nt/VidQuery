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

from dataclasses import dataclass

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


def reliability_prior(source_method: str) -> float:
    key = source_method.strip().lower()
    if key in SOURCE_RELIABILITY:
        return SOURCE_RELIABILITY[key]
    # Model names such as ``sentence-transformers/all-MiniLM-L6-v2``.
    if "minilm" in key or "sentence" in key or "mpnet" in key:
        return SOURCE_RELIABILITY["sentence_embedding"]
    if "hashing" in key:
        return SOURCE_RELIABILITY["hashing_embedding"]
    if "clip" in key:
        return SOURCE_RELIABILITY["openclip_appearance"]
    return SOURCE_RELIABILITY["unknown"]


def modality_weight(intent: str, modality: EvidenceModality) -> float:
    table = QUERY_MODALITY_WEIGHTS.get(intent, QUERY_MODALITY_WEIGHTS["multimodal_search"])
    return float(table.get(modality, 0.5))


def assess_evidence(
    plan: QueryPlan,
    contributions: list[RawContribution],
    *,
    supported_threshold: float = DEFAULT_SUPPORTED_THRESHOLD,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> EvidenceAssessment:
    """Fuse per-modality evidence into one confidence and an abstention verdict."""

    weighted: list[EvidenceContribution] = []
    numerator = 0.0
    denominator = 0.0
    for item in contributions:
        raw = max(0.0, min(1.0, item.raw_score))
        prior = reliability_prior(item.source_method)
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
