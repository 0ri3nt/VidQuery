from __future__ import annotations

import json
import logging
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


class RAGProviderError(RuntimeError):
    """Raised when the optional answer provider cannot return validated output."""


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
                payload=self._request_payload(query, evidence),
                timeout=self.timeout,
            )
            generated = self._parse_provider_answer(envelope)
        except Exception as exc:
            LOGGER.warning("Grounded answer generation unavailable: %s", exc)
            return RAGOutcome(answer=None, status="provider_unavailable")

        unique_ids = list(dict.fromkeys(generated.evidence_ids))
        invalid_ids = [item for item in unique_ids if item not in evidence]
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

        result_by_id = {
            f"E{index}": result for index, result in enumerate(eligible, start=1)
        }
        citations = [
            self._citation(evidence_id, result_by_id[evidence_id])
            for evidence_id in unique_ids
        ]
        return RAGOutcome(
            answer=GroundedAnswer(
                answer=generated.answer,
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
        self, query: str, evidence: dict[str, dict[str, Any]]
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
            "tool_choice": "none",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied VidQuery evidence. Do not infer facts "
                        "outside it. Cite only supplied evidence IDs. If the evidence does not "
                        "directly support an answer, set supported=false and evidence_ids=[]. "
                        "Return only the required JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query, "retrieved_evidence": evidence},
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
        )
