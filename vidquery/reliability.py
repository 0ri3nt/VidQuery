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
    # Sound-alike transcript hits (ENABLE_PHONETIC_MATCH); deliberately below
    # exact transcript evidence until calibration measures it.
    "phonetic_transcript": 0.50,
    "easyocr": 0.80,
    "yolo_detection": 0.80,
    "pyannote_diarization": 0.75,
    "bounding_box_geometry": 0.65,
    "sentence_embedding": 0.60,
    "openclip_appearance": 0.55,
    # Frame-level OpenCLIP text-to-visual similarity (ENABLE_FRAME_SEMANTIC_RETRIEVAL).
    "openclip_frame_similarity": 0.55,
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
    "visual_semantic",
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
        "visual_semantic": 0.1,
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
        "visual_semantic": 0.1,
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
        "visual_semantic": 0.9,
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
        "visual_semantic": 0.9,
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
        "visual_semantic": 0.7,
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
        "visual_semantic": 0.1,
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
        "visual_semantic": 0.9,
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


def load_reliability_thresholds(path: Path) -> tuple[float, float] | None:
    """Calibration-fitted ``(supported, weak)`` verdict thresholds, if the file has them."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds") if isinstance(payload, dict) else None
    if thresholds is None:
        return None
    if not isinstance(thresholds, dict):
        raise ValueError("'thresholds' must be an object")
    supported = thresholds.get("evidence_supported_threshold")
    weak = thresholds.get("evidence_weak_threshold")
    for name, value in (("supported", supported), ("weak", weak)):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} threshold must be a number")
    assert isinstance(supported, int | float) and isinstance(weak, int | float)
    if not 0 <= weak <= supported <= 1:
        raise ValueError("thresholds must satisfy 0 <= weak <= supported <= 1")
    return float(supported), float(weak)


def fit_abstention_thresholds(
    decisions: list[Mapping[str, Any]],
    *,
    supported_precision: float = 0.8,
    wrong_answer_cost: float = 1.0,
) -> dict[str, Any]:
    """Fit verdict thresholds on calibration decision rows.

    Each row needs ``negative``, ``max_confidence`` (``None`` when nothing was
    retrieved) and ``top1_correct``.  The system abstains when every returned
    result is below the weak threshold, i.e. when ``max_confidence < weak``.

    * ``weak`` maximises effective reliability: +1 per correct answer,
      ``-wrong_answer_cost`` per wrong answer (any answer to a negative query is
      wrong), 0 per abstention.  Ties keep the lower threshold (answer more).
    * ``supported`` is the lowest confidence at which answered calibration
      queries reach ``supported_precision``; 1.0 when no level reaches it.
    """

    if not decisions:
        raise ValueError("threshold fitting needs at least one decision row")
    rows = []
    for row in decisions:
        confidence = row.get("max_confidence")
        rows.append(
            (
                float(confidence) if confidence is not None else None,
                bool(row.get("top1_correct")) and not bool(row.get("negative")),
            )
        )
    observed = [value for value, _ in rows if value is not None]
    candidates = sorted({0.0, *observed})
    if observed and max(observed) < 1.0:
        # "Abstain on everything" must be reachable when every answer loses.
        candidates.append(min(1.0, max(observed) + 1e-6))
    best_weak, best_score = 0.0, float("-inf")
    for threshold in candidates:
        score = 0.0
        for confidence, correct in rows:
            if confidence is None or confidence < threshold:
                continue
            score += 1.0 if correct else -wrong_answer_cost
        if score > best_score + 1e-12:
            best_weak, best_score = threshold, score
    supported = 1.0
    for threshold in candidates:
        answered = [
            correct
            for confidence, correct in rows
            if confidence is not None and confidence >= threshold
        ]
        if answered and sum(answered) / len(answered) >= supported_precision:
            supported = threshold
            break
    supported = max(supported, best_weak)
    answered_at_weak = [
        correct
        for confidence, correct in rows
        if confidence is not None and confidence >= best_weak
    ]
    return {
        "evidence_supported_threshold": round(supported, 6),
        "evidence_weak_threshold": round(best_weak, 6),
        "objective": f"effective_reliability(wrong_cost={wrong_answer_cost})",
        "objective_value_per_query": round(best_score / len(rows), 4),
        "supported_precision_target": supported_precision,
        "calibration_coverage": round(len(answered_at_weak) / len(rows), 4),
        "calibration_selective_precision": (
            round(sum(answered_at_weak) / len(answered_at_weak), 4) if answered_at_weak else None
        ),
        "decision_count": len(rows),
    }


def canonical_source(source_method: str) -> str:
    """Map an emitted source name (including model names) onto a prior key.

    The ranker reports semantic and appearance evidence by model name, for
    example ``sentence-transformers/all-MiniLM-L6-v2`` or ``ViT-B-32:openai``.
    Fitting and lookup must agree on one key per source, so both use this.
    """

    key = source_method.strip().lower()
    if key in SOURCE_RELIABILITY:
        return key
    if "minilm" in key or "sentence" in key or "mpnet" in key:
        return "sentence_embedding"
    if "hashing" in key:
        return "hashing_embedding"
    if "clip" in key or key.startswith("vit-"):
        return "openclip_appearance"
    return "unknown"


def fit_reliability_priors(
    observations: list[Mapping[str, Any]], *, minimum_observations: int = 10, strength: int = 5
) -> dict[str, float]:
    """Fit source priors from calibration observations with prior smoothing.

    An observation counts only when its source produced non-zero evidence
    (``raw_score > 0``): a zero contribution means the source did not match,
    so it says nothing about how trustworthy that source's matches are.
    """

    if minimum_observations < 1 or strength < 0:
        raise ValueError("minimum_observations must be positive and strength non-negative")
    counts: dict[str, tuple[int, int]] = {}
    for observation in observations:
        source = canonical_source(str(observation.get("source_method", "")))
        correct = observation.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("reliability observations require a boolean 'correct' field")
        raw_score = observation.get("raw_score", 1.0)
        if isinstance(raw_score, int | float) and float(raw_score) <= 0:
            continue
        total, hits = counts.get(source, (0, 0))
        counts[source] = (total + 1, hits + int(correct))

    fitted = dict(SOURCE_RELIABILITY)
    for source, (total, hits) in counts.items():
        if total >= minimum_observations:
            prior = SOURCE_RELIABILITY[source]
            fitted[source] = (hits + strength * prior) / (total + strength)
    return {source: round(value, 6) for source, value in fitted.items()}


def reliability_prior(source_method: str, priors: Mapping[str, float] | None = None) -> float:
    """Trust for ``source_method``; a partial learned file falls back per source."""

    key = canonical_source(source_method)
    if priors and key in priors:
        return priors[key]
    return SOURCE_RELIABILITY[key]


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
