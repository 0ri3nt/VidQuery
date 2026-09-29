"""Coarse-to-fine timestamp localization.

Retrieval ranks fixed five-second canonical segments because that is cheap
and stable.  This module is the second stage: given a ranked segment and the
query plan that matched it, it inspects the observation-level evidence already
stored inside the segment - Whisper word/utterance timings, per-frame detector
observations, OCR intervals, action instances, relationship observations - and
returns the *peak-evidence* moment instead of the segment boundary.

No model is executed here.  Everything is derived from timestamps the pipeline
already persisted, so localization works retroactively on existing indexes.
Word-level precision only becomes available for videos processed after Whisper
word timestamps were enabled; older utterances fall back to a proportional
word-position interpolation that is labelled as such.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .domain import (
    AppearanceMatchEvidence,
    CanonicalSegment,
    LocalizationPrecision,
    MatchedRelationship,
    QueryPlan,
    SearchResult,
    TemporalLocalization,
    TranscriptSegment,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

# Lower rank means a tighter expected timestamp error.
PRECISION_RANK: dict[LocalizationPrecision, int] = {
    "word": 0,
    "frame": 1,
    "interval": 2,
    "utterance_interpolated": 3,
    "utterance": 4,
    "segment": 5,
}

# Confidence that the returned peak lies within roughly one second of the
# true event, by evidence granularity.  These are documented design constants,
# not calibrated probabilities.
PRECISION_CONFIDENCE: dict[LocalizationPrecision, float] = {
    "word": 0.95,
    "frame": 0.85,
    "interval": 0.75,
    "utterance_interpolated": 0.55,
    "utterance": 0.40,
    "segment": 0.10,
}


@dataclass(frozen=True, slots=True)
class _Candidate:
    peak: float
    start: float
    end: float
    precision: LocalizationPrecision
    confidence: float
    source: str
    evidence: str

    @property
    def rank(self) -> int:
        return PRECISION_RANK[self.precision]


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_word(word: str) -> str:
    return "".join(TOKEN_PATTERN.findall(word.lower()))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _frame_interval(segment: CanonicalSegment) -> float:
    rate = segment.processing_metadata.get("frame_sample_rate")
    try:
        rate_value = float(rate) if rate is not None else 1.0
    except (TypeError, ValueError):
        rate_value = 1.0
    return 1.0 / rate_value if rate_value > 0 else 1.0


def _transcript_candidates(
    terms: list[str], transcripts: list[TranscriptSegment]
) -> list[_Candidate]:
    wanted = [term.lower() for term in terms if term]
    if not wanted:
        return []
    candidates: list[_Candidate] = []
    for utterance in transcripts:
        # 1. Exact word timings (Whisper word_timestamps=True).
        if utterance.words:
            matched = [
                word for word in utterance.words if _normalize_word(word.word) in wanted
            ]
            if matched:
                first = matched[0]
                last = matched[-1]
                probabilities = [
                    word.probability for word in matched if word.probability is not None
                ]
                mean_probability = (
                    sum(probabilities) / len(probabilities) if probabilities else 0.9
                )
                candidates.append(
                    _Candidate(
                        peak=first.start_time,
                        start=first.start_time,
                        end=max(first.start_time, last.end_time),
                        precision="word",
                        confidence=PRECISION_CONFIDENCE["word"] * max(0.5, mean_probability),
                        source="whisper_word_timing",
                        evidence=(
                            f"word '{_normalize_word(first.word)}' spoken at "
                            f"{first.start_time:.2f}s"
                        ),
                    )
                )
                continue
        # 2. Proportional interpolation inside the utterance.
        utterance_tokens = _tokens(utterance.text)
        positions = [index for index, token in enumerate(utterance_tokens) if token in wanted]
        if not positions or not utterance_tokens:
            continue
        span = max(0.0, utterance.end_time - utterance.start_time)
        total = len(utterance_tokens)
        first_position = positions[0]
        last_position = positions[-1]
        estimate = utterance.start_time + span * ((first_position + 0.5) / total)
        estimate_end = utterance.start_time + span * ((last_position + 1.0) / total)
        # Uncertainty grows with utterance length: one token of a 20-word
        # utterance is a much weaker anchor than one token of a 4-word one.
        length_penalty = min(1.0, 4.0 / max(total, 1)) ** 0.25
        candidates.append(
            _Candidate(
                peak=estimate,
                start=estimate,
                end=max(estimate, estimate_end),
                precision="utterance_interpolated",
                confidence=PRECISION_CONFIDENCE["utterance_interpolated"] * length_penalty,
                source="whisper_utterance_interpolation",
                evidence=(
                    f"'{utterance_tokens[first_position]}' is token "
                    f"{first_position + 1}/{total} of the utterance "
                    f"{utterance.start_time:.2f}-{utterance.end_time:.2f}s"
                ),
            )
        )
    return candidates


def _ocr_candidates(terms: list[str], segment: CanonicalSegment) -> list[_Candidate]:
    wanted = {term.lower() for term in terms if term}
    if not wanted:
        return []
    candidates: list[_Candidate] = []
    for evidence in segment.ocr_evidence:
        if not wanted & set(_tokens(evidence.text)):
            continue
        start = max(segment.start_time, evidence.start_time)
        end = min(segment.end_time, evidence.end_time)
        if end < start:
            start, end = evidence.start_time, evidence.end_time
        candidates.append(
            _Candidate(
                peak=start,
                start=start,
                end=max(start, end),
                precision="interval",
                confidence=PRECISION_CONFIDENCE["interval"] * max(0.5, evidence.confidence),
                source="easyocr_interval",
                evidence=f"OCR text '{evidence.text}' visible from {start:.2f}s",
            )
        )
    return candidates


def _entity_candidates(
    required: list[str],
    segment: CanonicalSegment,
    appearance_matches: list[AppearanceMatchEvidence],
) -> list[_Candidate]:
    wanted = [item.lower() for item in required]
    if not wanted:
        return []
    track_filter: dict[str, set[str]] = defaultdict(set)
    for match in appearance_matches:
        track_filter[match.entity_class.lower()].add(match.track_id)

    by_time: dict[float, dict[str, float]] = defaultdict(dict)
    for detection in segment.detections:
        label = detection.class_label.lower()
        if label not in wanted:
            continue
        allowed_tracks = track_filter.get(label)
        if allowed_tracks and detection.track_id not in allowed_tracks:
            continue
        current = by_time[detection.timestamp].get(label, 0.0)
        by_time[detection.timestamp][label] = max(current, detection.confidence)

    complete = {
        timestamp: classes
        for timestamp, classes in by_time.items()
        if all(label in classes for label in wanted)
    }
    if not complete:
        return []
    peak = min(
        complete,
        key=lambda timestamp: (
            -sum(complete[timestamp][label] for label in wanted),
            timestamp,
        ),
    )
    ordered = sorted(complete)
    interval = _frame_interval(segment)
    index = ordered.index(peak)
    run_start = peak
    run_end = peak
    cursor = index
    while cursor > 0 and ordered[cursor] - ordered[cursor - 1] <= interval * 1.5:
        cursor -= 1
        run_start = ordered[cursor]
    cursor = index
    while (
        cursor < len(ordered) - 1 and ordered[cursor + 1] - ordered[cursor] <= interval * 1.5
    ):
        cursor += 1
        run_end = ordered[cursor]
    mean_confidence = sum(complete[peak][label] for label in wanted) / len(wanted)
    return [
        _Candidate(
            peak=peak,
            start=max(0.0, run_start - interval / 2),
            end=run_end + interval / 2,
            precision="frame",
            confidence=PRECISION_CONFIDENCE["frame"] * max(0.5, mean_confidence),
            source="detector_frame_observation",
            evidence=(
                f"{'/'.join(wanted)} co-detected at {peak:.2f}s "
                f"(detector confidence {mean_confidence:.2f})"
            ),
        )
    ]


def _action_candidates(actions: list[str], segment: CanonicalSegment) -> list[_Candidate]:
    wanted = {item.lower() for item in actions}
    if not wanted:
        return []
    instances = list(segment.processing_metadata.get("action_instances", []))
    if not instances:
        instances = list(segment.processing_metadata.get("gnn_action_instances", []))
    best: dict | None = None
    for instance in instances:
        if str(instance.get("action", "")).lower() not in wanted:
            continue
        confidence = instance.get("confidence")
        confidence_value = float(confidence) if confidence is not None else 1.0
        if best is None or confidence_value > float(best.get("_confidence", -1.0)):
            best = {**instance, "_confidence": confidence_value}
    if best is None:
        return []
    timestamp = float(best.get("timestamp", segment.start_time))
    half = _frame_interval(segment) / 2
    provenance = str(best.get("provenance") or best.get("source_method") or "action_model")
    return [
        _Candidate(
            peak=timestamp,
            start=max(0.0, timestamp - half),
            end=timestamp + half,
            precision="frame",
            confidence=PRECISION_CONFIDENCE["frame"] * max(0.3, float(best["_confidence"])),
            source=provenance,
            evidence=(
                f"action '{best.get('action')}' instance at {timestamp:.2f}s "
                f"(confidence {best['_confidence']:.2f})"
            ),
        )
    ]


def _relationship_candidates(
    matched: list[MatchedRelationship], segment: CanonicalSegment
) -> list[_Candidate]:
    if not matched:
        return []
    best = max(matched, key=lambda item: (item.confidence, -item.timestamp))
    half = _frame_interval(segment) / 2
    if best.start_time is not None and best.end_time is not None and best.temporal_smoothed:
        start = max(segment.start_time, best.start_time)
        end = min(segment.end_time, best.end_time)
        if end < start:
            start, end = best.start_time, best.end_time
        peak = _clip(best.timestamp, start, end)
        precision: LocalizationPrecision = "interval"
    else:
        peak = best.timestamp
        start = max(0.0, peak - half)
        end = peak + half
        precision = "frame"
    return [
        _Candidate(
            peak=peak,
            start=start,
            end=max(start, end),
            precision=precision,
            confidence=PRECISION_CONFIDENCE[precision] * max(0.4, best.confidence),
            source=f"relationship_{best.source_method.value}",
            evidence=(
                f"{best.source_class} {best.predicate} {best.target_class} observed at "
                f"{peak:.2f}s"
            ),
        )
    ]


def _speaker_candidates(speaker: str | None, segment: CanonicalSegment) -> list[_Candidate]:
    if not speaker:
        return []
    turns = [item for item in segment.transcript_segments if item.speaker == speaker]
    if not turns:
        return []
    first = min(turns, key=lambda item: item.start_time)
    start = max(segment.start_time, first.start_time)
    end = min(segment.end_time, first.end_time)
    if end < start:
        start, end = first.start_time, first.end_time
    return [
        _Candidate(
            peak=start,
            start=start,
            end=max(start, end),
            precision="utterance",
            confidence=PRECISION_CONFIDENCE["utterance"],
            source="pyannote_speaker_turn",
            evidence=f"{speaker} speaks from {start:.2f}s",
        )
    ]


def _fallback_transcript(segment: CanonicalSegment) -> list[_Candidate]:
    """Semantic-only transcript hits have no lexical anchor; use the utterance."""

    if not segment.transcript_segments:
        return []
    first = min(segment.transcript_segments, key=lambda item: item.start_time)
    start = max(segment.start_time, first.start_time)
    end = min(segment.end_time, first.end_time)
    if end < start:
        start, end = first.start_time, first.end_time
    return [
        _Candidate(
            peak=start,
            start=start,
            end=max(start, end),
            precision="utterance",
            confidence=PRECISION_CONFIDENCE["utterance"] * 0.75,
            source="whisper_utterance",
            evidence=f"semantically matched utterance starts at {start:.2f}s",
        )
    ]


def _segment_fallback(segment: CanonicalSegment) -> TemporalLocalization:
    return TemporalLocalization(
        peak_time=segment.start_time,
        start_time=segment.start_time,
        end_time=segment.end_time,
        precision="segment",
        source="segment_window",
        confidence=PRECISION_CONFIDENCE["segment"],
        evidence=["no observation-level evidence; returning the coarse segment"],
    )


def _combine(candidates: list[_Candidate], segment: CanonicalSegment) -> TemporalLocalization:
    if not candidates:
        return _segment_fallback(segment)
    if len(candidates) == 1:
        only = candidates[0]
        return TemporalLocalization(
            peak_time=only.peak,
            start_time=only.start,
            end_time=only.end,
            precision=only.precision,
            source=only.source,
            confidence=round(_clip(only.confidence, 0.0, 1.0), 4),
            evidence=[only.evidence],
        )

    ordered = sorted(candidates, key=lambda item: (item.rank, -item.confidence, item.peak))
    intersection_start = max(item.start for item in candidates)
    intersection_end = min(item.end for item in candidates)
    mean_confidence = sum(item.confidence for item in candidates) / len(candidates)
    evidence = [item.evidence for item in ordered]

    if intersection_end >= intersection_start:
        inside = [
            item for item in ordered if intersection_start <= item.peak <= intersection_end
        ]
        if inside:
            chosen = inside[0]
            return TemporalLocalization(
                peak_time=chosen.peak,
                start_time=intersection_start,
                end_time=intersection_end,
                precision=chosen.precision,
                source=f"{chosen.source}+agreement",
                confidence=round(_clip(min(1.0, mean_confidence * 1.1), 0.0, 1.0), 4),
                evidence=evidence,
            )
        midpoint = (intersection_start + intersection_end) / 2
        return TemporalLocalization(
            peak_time=midpoint,
            start_time=intersection_start,
            end_time=intersection_end,
            precision="interval",
            source="multimodal_intersection",
            confidence=round(_clip(mean_confidence, 0.0, 1.0), 4),
            evidence=evidence,
        )

    # Modalities disagree in time. Trust the tightest evidence but say so.
    chosen = ordered[0]
    return TemporalLocalization(
        peak_time=chosen.peak,
        start_time=chosen.start,
        end_time=chosen.end,
        precision=chosen.precision,
        source=f"{chosen.source}+disagreement",
        confidence=round(_clip(mean_confidence * 0.7, 0.0, 1.0), 4),
        evidence=evidence + ["modalities do not overlap in time; tightest evidence chosen"],
    )


def localize_segment(
    plan: QueryPlan,
    segment: CanonicalSegment,
    *,
    matched_relationships: list[MatchedRelationship] | None = None,
    appearance_matches: list[AppearanceMatchEvidence] | None = None,
) -> TemporalLocalization:
    """Return the peak-evidence moment for ``segment`` under ``plan``."""

    candidates: list[_Candidate] = []
    transcript_hits = _transcript_candidates(plan.spoken_terms, segment.transcript_segments)
    if transcript_hits:
        candidates.append(max(transcript_hits, key=lambda item: (-item.rank, item.confidence)))
    elif plan.spoken_terms and plan.intent in {"transcript_search", "multimodal_search"}:
        ocr_for_concept = (
            _ocr_candidates(plan.spoken_terms, segment)
            if plan.intent != "transcript_search"
            else []
        )
        if ocr_for_concept:
            candidates.append(max(ocr_for_concept, key=lambda item: item.confidence))
        else:
            candidates.extend(_fallback_transcript(segment))
    ocr_hits = _ocr_candidates(plan.ocr_terms, segment)
    if ocr_hits:
        candidates.append(max(ocr_hits, key=lambda item: item.confidence))
    candidates.extend(
        _entity_candidates(plan.visual_entities, segment, list(appearance_matches or []))
    )
    candidates.extend(_action_candidates(plan.actions, segment))
    candidates.extend(_relationship_candidates(list(matched_relationships or []), segment))
    candidates.extend(_speaker_candidates(plan.speaker, segment))
    return _combine(candidates, segment)


def collapse_duplicate_results(
    results: list[SearchResult], *, min_gap_seconds: float = 0.5
) -> list[SearchResult]:
    """Drop adjacent-window duplicates that localize to the same moment.

    A transcript utterance or a tracked relationship that straddles a segment
    boundary is stored in both windows and would otherwise appear twice.
    ``results`` must already be in rank order; the first occurrence wins.
    """

    kept: list[SearchResult] = []
    for result in results:
        duplicate = False
        for existing in kept:
            if existing.video_id != result.video_id:
                continue
            if existing.localization is None or result.localization is None:
                continue
            if (
                abs(existing.localization.peak_time - result.localization.peak_time)
                <= min_gap_seconds
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(result)
    return kept
