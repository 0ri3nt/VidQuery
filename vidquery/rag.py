from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field

from .config import Settings
from .domain import (
    AnswerCitation,
    GroundedAnswer,
    SearchResponse,
    SearchResult,
    StrictModel,
)

LOGGER = logging.getLogger(__name__)
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DECLINED_ANSWER = "The retrieved evidence does not support an answer to this query."

# Claim checks on generated answers (see verify_answer_claims).
TIME_TOLERANCE_SECONDS = 1.0
NUMBER_TOLERANCE = 0.011
_CLOCK = re.compile(r"(?<![\d.:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d+))?(?![\d:])")
_UNIT = r"\s*(?:s|sec|secs|second|seconds)\b"
_SECONDS_RANGE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)" + _UNIT, re.I
)
_SECONDS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)" + _UNIT, re.I)
_DECIMAL = re.compile(r"(?<![\d.:])(0?\.\d+|1\.0+)(?!\d|\.\d)(?!\s*(?:s|sec|secs|seconds?)\b)")
_PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s?%")
_SPEAKER = re.compile(r"\bspeaker[_ ]?(\d+)\b", re.I)
_QUOTE = re.compile("[\"“”]([^\"“”]{3,})[\"“”]")
_TRANSCRIPT_CUES = re.compile(r"\b(?:transcript|says?|said|saying|spoken|speech|utters?)\b", re.I)
_OCR_CUES = re.compile(r"\b(?:ocr|on-screen text|on screen text|written|caption)\b", re.I)
# A negator and up to four words it governs: "is not cooking", "no laptop",
# "isn't drinking from a cup". Applied to normalized text ("isn't" -> "isn t").
_NEGATED = re.compile(
    r"\b(?:no|not|never|without|none|nobody|neither|nor|cannot|"
    r"(?:isn|aren|doesn|don|didn|wasn|weren|can|won) t)\b(?:\s+\w+){0,4}"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9_ ]+", " ", text.lower()).split())


def _mentioned_times(text: str) -> list[float]:
    times: list[float] = []
    for match in _CLOCK.finditer(text):
        first, second, third, fraction = match.groups()
        if third is None:
            value = int(first) * 60 + int(second)
        else:
            value = int(first) * 3600 + int(second) * 60 + int(third)
        times.append(value + (float(f"0.{fraction}") if fraction else 0.0))
    remainder = _CLOCK.sub(" ", text)
    for match in _SECONDS_RANGE.finditer(remainder):
        times.extend((float(match.group(1)), float(match.group(2))))
    remainder = _SECONDS_RANGE.sub(" ", remainder)
    times.extend(float(match.group(1)) for match in _SECONDS.finditer(remainder))
    return times


REPEATED_EVIDENCE_MIN_WORDS = 4


def _without_repeated_evidence(text: str, evidence: str) -> str:
    """Drop runs of at least four words the answer repeats verbatim from the evidence.

    Numbers inside repeated speech or on-screen text ("processes the video into 5 second
    segments") are content, not timestamp or score claims; quotes are checked separately.
    """

    words = text.split()
    normalized = [_normalize(word) for word in words]
    haystack = f" {_normalize(evidence)} "
    keep = [True] * len(words)
    start = 0
    while start < len(words):
        end, best = start, start
        while end < len(words):
            end += 1
            phrase = " ".join(item for item in normalized[start:end] if item)
            if phrase and f" {phrase} " not in haystack:
                break
            if len(phrase.split()) >= REPEATED_EVIDENCE_MIN_WORDS:
                best = end
        if best > start:
            keep[start:best] = [False] * (best - start)
            start = best
        else:
            start += 1
    return " ".join(word for word, kept in zip(words, keep, strict=True) if kept)


def verify_answer_claims(text: str, cited: list[SearchResult]) -> list[str]:
    """Reasons a generated answer's text is not backed by its cited results.

    Checks timestamps, objects, actions, speakers, quoted speech/text,
    confidence numbers and named evidence sources against the cited results
    only. An empty list means every checked claim is backed.
    """

    from .action_lexicon import match_actions
    from .search import ENTITY_ALIASES, StructuredQueryParser

    reasons: list[str] = []
    spoken = " ".join(
        [item.transcript for item in cited]
        + [ocr.text for item in cited for ocr in item.ocr_evidence]
    )
    claims = _without_repeated_evidence(text, spoken)
    windows = [(item.start_time, item.end_time) for item in cited]
    bad_times = [
        value
        for value in _mentioned_times(claims)
        if not any(
            start - TIME_TOLERANCE_SECONDS <= value <= end + TIME_TOLERANCE_SECONDS
            for start, end in windows
        )
    ]
    if bad_times:
        reasons.append(f"timestamps outside every cited segment: {bad_times}")

    normalized_spoken = _normalize(spoken)
    evidence_entities = {entity.lower() for item in cited for entity in item.entities}
    for item in cited:
        for relation in item.matched_relationships:
            evidence_entities.update({relation.source_class.lower(), relation.target_class.lower()})
        evidence_entities.update(match.entity_class.lower() for match in item.appearance_matches)
    evidence_entities.update(
        StructuredQueryParser._extract_aliases(normalized_spoken, ENTITY_ALIASES)
    )
    if any(item.transcript.strip() or item.speakers for item in cited):
        evidence_entities.add("person")  # someone is speaking, so a person is present
    # Negated mentions ("the person is not cooking", "no laptop") are not claims.
    asserted = _NEGATED.sub(" ", _normalize(text))
    claimed_entities = StructuredQueryParser._extract_aliases(asserted, ENTITY_ALIASES)
    missing_entities = sorted(set(claimed_entities) - evidence_entities)
    if missing_entities:
        reasons.append(f"objects not in the cited evidence: {missing_entities}")

    evidence_actions = {action.lower() for item in cited for action in item.actions}
    evidence_actions.update(match.label for match in match_actions(normalized_spoken))
    claimed_actions = {match.label for match in match_actions(asserted)}
    missing_actions = sorted(claimed_actions - evidence_actions)
    if missing_actions:
        reasons.append(f"actions not in the cited evidence: {missing_actions}")

    evidence_speakers = {speaker.upper() for item in cited for speaker in item.speakers}
    claimed_speakers = {f"SPEAKER_{int(match.group(1)):02d}" for match in _SPEAKER.finditer(text)}
    if claimed_speakers - evidence_speakers:
        reasons.append(
            f"speakers not in the cited evidence: {sorted(claimed_speakers - evidence_speakers)}"
        )

    # A quoted title of a cited video ("In "final_demo.mp4" at 11 s") names it; it is
    # not a quotation of speech or on-screen text.
    titles = {_normalize(item.video_title) for item in cited} | {
        _normalize(item.video_title.rsplit(".", 1)[0]) for item in cited
    }
    misquoted = [
        quote.strip()
        for quote in _QUOTE.findall(text)
        if _normalize(quote)
        and _normalize(quote) not in normalized_spoken
        and _normalize(quote) not in titles
    ]
    if misquoted:
        reasons.append(f"quotes not found in cited transcripts or on-screen text: {misquoted}")

    known_numbers = [
        value
        for item in cited
        for value in (
            item.score,
            item.evidence.confidence if item.evidence is not None else None,
            item.localization.confidence if item.localization is not None else None,
            *(relation.confidence for relation in item.matched_relationships),
        )
        if value is not None
    ]
    claimed_numbers = [float(value) for value in _DECIMAL.findall(claims)] + [
        float(value) / 100 for value in _PERCENT.findall(claims)
    ]
    unbacked = [
        value
        for value in claimed_numbers
        if not any(abs(value - known) <= NUMBER_TOLERANCE for known in known_numbers)
    ]
    if unbacked:
        reasons.append(f"scores or confidences not in the cited evidence: {unbacked}")

    if _TRANSCRIPT_CUES.search(text) and not any(item.transcript.strip() for item in cited):
        reasons.append("cites speech but no cited segment has a transcript")
    if _OCR_CUES.search(text) and not any(item.ocr_evidence for item in cited):
        reasons.append("cites on-screen text but no cited segment has OCR evidence")
    return reasons


def _clock(seconds: float) -> str:
    whole = int(max(0.0, seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def _seconds(value: float) -> str:
    return f"{value:.1f}".removesuffix(".0") + " s"


def located_answer(answer: str, cited: list[SearchResult]) -> str:
    """Say where the cited evidence is when a supported answer does not.

    The added location comes from the cited results themselves (title, peak or
    start time, segment range, speaker), so it cannot add an unsupported claim.
    An answer that only repeats a cited transcript becomes an attributed quote.
    """

    text = answer.strip()
    spoken = " ".join(
        [item.transcript for item in cited]
        + [ocr.text for item in cited for ocr in item.ocr_evidence]
    )
    if not cited or _mentioned_times(_without_repeated_evidence(text, spoken)):
        return text
    for item in cited:
        words = _normalize(text).split()
        leftover = _normalize(_without_repeated_evidence(text, item.transcript)).split()
        if len(words) >= REPEATED_EVIDENCE_MIN_WORDS and len(leftover) <= 2:
            peak = item.localization.peak_time if item.localization else item.start_time
            speaker = item.speakers[0] if len(item.speakers) == 1 else "the speaker"
            quote = text.replace('"', "").rstrip(".")
            return (
                f"In {item.video_title} at {_seconds(peak)} "
                f"({_seconds(item.start_time).removesuffix(' s')}–{_seconds(item.end_time)}), "
                f'{speaker} says: "{quote}."'
            )
    places: dict[str, list[str]] = {}
    for item in cited[:3]:
        peak = item.localization.peak_time if item.localization else item.start_time
        places.setdefault(item.video_title, []).append(_seconds(peak))

    def joined(values: list[str]) -> str:
        return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]

    where = joined([f"{title} at {joined(times)}" for title, times in places.items()])
    first = text.split(" ", 1)[0]
    if first in {"A", "An", "The"} or (first[1:].isalpha() and first[1:].islower()):
        text = text[0].lower() + text[1:]  # "A person ..." but not "SPEAKER_00 ..." or "I"
    return f"In {where}: {text}"


def weak_evidence_answer(response: SearchResponse, cited: list[SearchResult]) -> str:
    """Deterministic wording for an answer the system's verdicts do not support."""

    verdicts = sorted({item.evidence.verdict if item.evidence else "unassessed" for item in cited})
    best = max((item.evidence.confidence for item in cited if item.evidence), default=0.0)
    moments = "; ".join(
        f"{item.video_title} {_clock(item.start_time)}-{_clock(item.end_time)}"
        + (f" (peak {_clock(item.localization.peak_time)})" if item.localization else "")
        for item in cited[:3]
    )
    prefix = (
        "No retrieved moment has sufficient evidence"
        if response.interpretation == "insufficient_evidence"
        else "The retrieved evidence does not confirm this"
    )
    return (
        f"{prefix} (system verdict: {'/'.join(verdicts)}, best confidence {best:.2f}). "
        f"Closest moments: {moments}."
    )


class RAGProviderError(RuntimeError):
    """Raised when the optional answer provider cannot return validated output."""


def reasoning_options(model: str) -> dict[str, str]:
    # gpt-oss reasoning tokens count against max_completion_tokens; at the default
    # effort they exhaust the budget before the JSON document is emitted. Groq
    # rejects the parameter for non-reasoning models.
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


class ProviderTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]: ...


class UrllibProviderTransport:
    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                body = response.read()
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            status = getattr(exc, "code", None)
            suffix = f" (HTTP {status})" if status is not None else ""
            raise RAGProviderError(
                f"Groq request failed: {type(exc).__name__}{suffix}"
            ) from exc
        try:
            value = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise RAGProviderError("Groq returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RAGProviderError("Groq returned an invalid response envelope")
        return value


class ProviderGroundedAnswer(StrictModel):
    answer: str = Field(min_length=1, max_length=4000)
    supported: bool
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)


@dataclass(frozen=True, slots=True)
class RAGOutcome:
    answer: GroundedAnswer | None
    status: str


def rag_component_status(settings: Settings) -> str:
    if not settings.enable_rag_generation:
        return "disabled"
    if settings.rag_provider.strip().lower() != "groq":
        return "unavailable_unsupported_provider"
    if not settings.groq_api_key:
        return "unavailable_missing_api_key"
    if not settings.rag_model.strip():
        return "unavailable_missing_model"
    return "available_configured"


class GroundedAnswerService:
    """Synthesizes only over already-ranked VidQuery evidence.

    This service has no repository or Neo4j dependency. It receives immutable
    search results, sends an allowlisted evidence projection to Groq, validates
    returned evidence IDs, and converts them back to exact media intervals.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: ProviderTransport | None = None,
    ) -> None:
        self.enabled = settings.enable_rag_generation
        self.provider = settings.rag_provider.strip().lower()
        self.model = settings.rag_model.strip()
        self.api_key = settings.groq_api_key
        self.top_k = max(1, min(20, settings.rag_top_k))
        self.min_evidence_score = max(0.0, min(1.0, settings.rag_min_evidence_score))
        self.timeout = max(1.0, min(60.0, settings.rag_timeout_seconds))
        self.transport = transport or UrllibProviderTransport()

    def generate(self, query: str, response: SearchResponse) -> RAGOutcome:
        status = self._configuration_status()
        if status != "available_configured":
            return RAGOutcome(answer=None, status=status)

        eligible = [
            result
            for result in response.results
            if result.score >= self.min_evidence_score
        ][: self.top_k]
        if not eligible:
            return RAGOutcome(
                answer=GroundedAnswer(
                    answer="The retrieved evidence is insufficient to answer this query.",
                    supported=False,
                ),
                status="insufficient_evidence",
            )

        evidence = {
            f"E{index}": self._project_evidence(result)
            for index, result in enumerate(eligible, start=1)
        }
        try:
            envelope = self.transport.post_json(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "VidQuery/0.2 grounded-rag",
                },
                payload=self._request_payload(query, evidence, response.interpretation),
                timeout=self.timeout,
            )
            generated = self._parse_provider_answer(envelope)
        except Exception as exc:
            LOGGER.warning("Grounded answer generation unavailable: %s", exc)
            return RAGOutcome(answer=None, status="provider_unavailable")

        unique_ids = list(dict.fromkeys(generated.evidence_ids))
        invalid_ids = [item for item in unique_ids if item not in evidence]
        result_by_id = {
            f"E{index}": result for index, result in enumerate(eligible, start=1)
        }
        if invalid_ids or (generated.supported and not unique_ids):
            return RAGOutcome(
                answer=GroundedAnswer(
                    answer=(
                        "The generated answer was rejected because its citations did not "
                        "match retrieved evidence."
                    ),
                    supported=False,
                ),
                status="rejected_invalid_evidence_ids",
            )
        if not generated.supported:
            return RAGOutcome(
                answer=GroundedAnswer(
                    answer=generated.answer,
                    supported=False,
                ),
                status="generated_unsupported",
            )

        cited = [result_by_id[evidence_id] for evidence_id in unique_ids]
        citations = [
            self._citation(evidence_id, result_by_id[evidence_id])
            for evidence_id in unique_ids
        ]
        reasons = verify_answer_claims(generated.answer, cited)
        if reasons:
            LOGGER.warning("Generated answer rejected: %s", "; ".join(reasons))
            return RAGOutcome(
                answer=GroundedAnswer(
                    answer=(
                        "The generated answer was rejected because the retrieved evidence "
                        "does not back it: " + "; ".join(reasons) + "."
                    ),
                    supported=False,
                ),
                status="rejected_unverified_claims",
            )
        # The deterministic verdicts stay authoritative: a generated answer may
        # only claim support that at least one cited result's verdict gives.
        # (Results without an assessment carry no verdict to contradict.)
        assessed = [item for item in cited if item.evidence is not None]
        if response.interpretation == "insufficient_evidence" or (
            assessed
            and not any(
                item.evidence is not None and item.evidence.verdict == "supported"
                for item in assessed
            )
        ):
            status = (
                "downgraded_insufficient_evidence"
                if response.interpretation == "insufficient_evidence"
                else "downgraded_weak_evidence"
            )
            LOGGER.info("Generated answer %s", status)
            return RAGOutcome(
                answer=GroundedAnswer(
                    answer=weak_evidence_answer(response, cited),
                    supported=False,
                    evidence_ids=unique_ids,
                    citations=citations,
                ),
                status=status,
            )
        return RAGOutcome(
            answer=GroundedAnswer(
                answer=located_answer(generated.answer, cited),
                supported=True,
                evidence_ids=unique_ids,
                citations=citations,
            ),
            status="generated_supported",
        )

    def _configuration_status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.provider != "groq":
            return "unavailable_unsupported_provider"
        if not self.api_key:
            return "unavailable_missing_api_key"
        if not self.model:
            return "unavailable_missing_model"
        return "available_configured"

    @staticmethod
    def _project_evidence(result: SearchResult) -> dict[str, Any]:
        matched_relationships = [
            {
                "subject": item.source_class,
                "predicate": item.predicate,
                "object": item.target_class,
                "confidence": item.confidence,
                "source_method": item.source_method.value,
                "timestamp": item.timestamp,
            }
            for item in result.matched_relationships
        ]
        relationships = matched_relationships or [
            {
                "source_id": item.source_id,
                "predicate": item.predicate,
                "target_id": item.target_id,
                "confidence": item.confidence,
                "source_method": item.source_method.value,
                "timestamp": item.timestamp,
            }
            # Canonical segments can contain hundreds of frame-level geometry
            # observations.  A small deterministic projection preserves every
            # evidence modality while keeping the grounded-answer request
            # inside provider context/TPM limits. Exact tuple matches above are
            # always preferred and are not truncated by this fallback.
            for item in result.relationships[:8]
        ]
        return {
            "segment_id": result.segment_id,
            "video_id": result.video_id,
            "video_title": result.video_title,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "retrieval_score": result.score,
            "peak_time": (
                result.localization.peak_time if result.localization is not None else None
            ),
            "localization_precision": (
                result.localization.precision if result.localization is not None else None
            ),
            "evidence_confidence": (
                result.evidence.confidence if result.evidence is not None else None
            ),
            "evidence_verdict": (
                result.evidence.verdict if result.evidence is not None else None
            ),
            "transcript": result.transcript,
            "speakers": result.speakers,
            "ocr": [
                {
                    "text": item.text,
                    "confidence": item.confidence,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                }
                for item in result.ocr_evidence
            ],
            "entities": result.entities,
            "actions": result.actions,
            "relationships": relationships,
            "appearance": [
                {
                    "entity_class": item.entity_class,
                    "track_id": item.track_id,
                    "attributes": {
                        key: {
                            "value": evidence.value,
                            "confidence": evidence.confidence,
                            "source_method": evidence.source_method.value,
                        }
                        for key, evidence in item.attributes.items()
                    },
                    "description": item.description,
                    "similarity": item.similarity,
                    "matched_via": item.matched_via,
                    "observation_count": item.observation_count,
                }
                for item in result.appearance_matches
            ],
        }

    def _request_payload(
        self,
        query: str,
        evidence: dict[str, dict[str, Any]],
        interpretation: str = "single",
    ) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "supported": {"type": "boolean"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer", "supported", "evidence_ids"],
            "additionalProperties": False,
        }
        return {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 500,
            **reasoning_options(self.model),
            "tool_choice": "none",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's query directly, in one or two sentences of your own "
                        "words: name the video_title (without quotes), say when it happens "
                        "(the peak_time, or "
                        "start_time to end_time, in seconds such as '11.7 s'), and say what the "
                        "evidence shows (who speaks, what is visible or happening). Do not just "
                        "repeat the transcript; quote at most one short exact phrase from it in "
                        "double quotes. "
                        "Answer only from the supplied VidQuery evidence. Do not infer facts "
                        "outside it. Cite only supplied evidence IDs. If the evidence does not "
                        "directly support an answer, set supported=false and evidence_ids=[]. "
                        "Set supported=true only if a cited item's evidence_verdict is "
                        "'supported'. Mention only times inside the cited segments, quote "
                        "transcripts exactly, and do not state scores you were not given. "
                        "Return only the required JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "system_interpretation": interpretation,
                            "retrieved_evidence": evidence,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vidquery_grounded_answer",
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    @staticmethod
    def _parse_provider_answer(envelope: dict[str, Any]) -> ProviderGroundedAnswer:
        try:
            content = envelope["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if (
                isinstance(parsed, dict)
                and parsed.get("supported") is False
                and not str(parsed.get("answer") or "").strip()
            ):
                # Groq declines with an empty answer for weak evidence (seen live); that is
                # a valid "not supported" answer, not a provider failure.
                parsed = {**parsed, "answer": DECLINED_ANSWER}
            return ProviderGroundedAnswer.model_validate(parsed)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RAGProviderError("Groq returned an invalid structured answer") from exc

    @staticmethod
    def _citation(evidence_id: str, result: SearchResult) -> AnswerCitation:
        return AnswerCitation(
            evidence_id=evidence_id,
            segment_id=result.segment_id,
            video_id=result.video_id,
            video_title=result.video_title,
            start_time=result.start_time,
            end_time=result.end_time,
            stream_url=result.stream_url,
            peak_time=(
                result.localization.peak_time if result.localization is not None else None
            ),
        )
