"""Ambiguity-aware query hypotheses.

A natural-language query frequently admits more than one *valid* retrieval
interpretation over this evidence store: "find drive" may mean the AVA action
``drive`` or the spoken word "drive"; "find the laptop scene" may mean a
visible laptop or someone talking about a laptop; "deployment" may be spoken
or written on a slide.  Review-1 forced one plan (or merged two by max score
without attribution).  Review-2 enumerates the interpretations explicitly,
retrieves for each, measures how well the evidence supports each one, and
either resolves the ambiguity or asks the user to choose.

Everything here is deterministic and operates only on validated ``QueryPlan``
values, so it composes with the constrained Groq planner without giving the
provider any new authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import InterpretationStatus, QueryHypothesis, QueryPlan, SearchResult
from .search import (
    ACTION_ALIASES,
    ENTITY_ALIASES,
    OCR_CUE_PATTERN,
    SPEECH_CUE_PATTERN,
    STOPWORDS,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
DEFAULT_RESOLUTION_MARGIN = 0.25
_SUPPORT_FLOOR = 0.05


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _is_cued(query: str) -> bool:
    lowered = query.lower()
    return bool(SPEECH_CUE_PATTERN.search(lowered) or OCR_CUE_PATTERN.search(lowered))


def _has_structure(plan: QueryPlan) -> bool:
    return bool(
        plan.visual_entities
        or plan.actions
        or plan.relationships
        or plan.relationship_tuples
        or plan.speaker
        or plan.appearance_constraints
    )


def _entity_surface_tokens(query: str, entities: list[str]) -> list[str]:
    """Return the query tokens that named a visual entity (for spoken search)."""

    tokens: list[str] = []
    for alias, canonical in ENTITY_ALIASES.items():
        if canonical not in entities or canonical == "person":
            continue
        if re.search(rf"\b{re.escape(alias)}\b", query.lower()):
            for token in _tokens(alias):
                if token not in tokens and token not in STOPWORDS:
                    tokens.append(token)
    return tokens


def is_bare_action_homonym(query: str, plan: QueryPlan) -> bool:
    if _is_cued(query):
        return False
    core = [token for token in _tokens(query) if token not in STOPWORDS]
    return len(core) == 1 and core[0] in ACTION_ALIASES and bool(plan.actions)


def generate_hypotheses(query: str, plan: QueryPlan) -> list[QueryHypothesis]:
    """Enumerate distinct, validated interpretations of ``query``.

    Returns an empty list when the deterministic parser is unambiguous.
    The first hypothesis is the default interpretation.
    """

    if _is_cued(query):
        return []

    # 1. Bare action homonym: "find drive", "kissing".
    if is_bare_action_homonym(query, plan):
        action = plan.actions[0]
        action_plan = plan.model_copy(
            update={
                "intent": "action_search",
                "relationships": [],
                "relationship_tuples": [],
                "spoken_terms": [],
                "ocr_terms": [],
            }
        )
        spoken_plan = QueryPlan(intent="transcript_search", spoken_terms=[action])
        return [
            QueryHypothesis(
                hypothesis_id="visible_action",
                label=f"visible action '{action}'",
                description=f"Moments where a person is observed performing '{action}'.",
                plan=action_plan,
                prior=0.5,
            ),
            QueryHypothesis(
                hypothesis_id="spoken_mention",
                label=f"spoken word '{action}'",
                description=f"Moments where '{action}' is said in the audio.",
                plan=spoken_plan,
                prior=0.5,
            ),
        ]

    # 2. Entity-only query: "find the laptop scene", "show the whiteboard".
    if (
        plan.visual_entities
        and not plan.actions
        and not plan.relationships
        and not plan.relationship_tuples
        and not plan.speaker
        and not plan.appearance_constraints
        and not plan.spoken_terms
        and not plan.ocr_terms
    ):
        spoken_tokens = _entity_surface_tokens(query, plan.visual_entities)
        if spoken_tokens:
            entity_text = "/".join(plan.visual_entities)
            return [
                QueryHypothesis(
                    hypothesis_id="visible_object",
                    label=f"visible {entity_text}",
                    description=f"Frames where the detector sees {entity_text}.",
                    plan=plan.model_copy(update={"intent": "visual_search"}),
                    prior=0.6,
                ),
                QueryHypothesis(
                    hypothesis_id="spoken_mention",
                    label=f"spoken '{' '.join(spoken_tokens)}'",
                    description=f"Utterances that mention {' '.join(spoken_tokens)}.",
                    plan=QueryPlan(intent="transcript_search", spoken_terms=spoken_tokens),
                    prior=0.4,
                ),
            ]

    # 3. Unclassified concept: "deployment", "kubernetes", "the meeting".
    if plan.spoken_terms and not _has_structure(plan) and not plan.ocr_terms:
        concept = " ".join(plan.spoken_terms)
        return [
            QueryHypothesis(
                hypothesis_id="spoken_mention",
                label=f"spoken '{concept}'",
                description=f"Utterances that mention {concept}.",
                plan=plan.model_copy(
                    update={"intent": "transcript_search", "ocr_terms": []}
                ),
                prior=0.6,
            ),
            QueryHypothesis(
                hypothesis_id="visible_text",
                label=f"on-screen text '{concept}'",
                description=f"Frames where OCR reads {concept}.",
                plan=QueryPlan(intent="ocr_search", ocr_terms=list(plan.spoken_terms)),
                prior=0.4,
            ),
        ]

    return []


def clarification_prompt(hypotheses: list[QueryHypothesis]) -> str:
    labels = [item.label for item in hypotheses]
    if len(labels) == 2:
        return f"Did you mean the {labels[0]} or the {labels[1]}?"
    return "Did you mean: " + "; ".join(labels) + "?"


@dataclass(frozen=True, slots=True)
class FusionOutcome:
    results: list[SearchResult]
    hypotheses: list[QueryHypothesis]
    interpretation: InterpretationStatus
    selected_hypothesis_id: str | None
    clarification_prompt: str | None


def _support(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    values = []
    for item in results[:3]:
        confidence = item.evidence.confidence if item.evidence is not None else item.score
        values.append(item.score * confidence)
    return max(0.0, min(1.0, max(values)))


def fuse_hypotheses(
    hypotheses: list[QueryHypothesis],
    retrieved: dict[str, list[SearchResult]],
    *,
    limit: int,
    resolution_margin: float = DEFAULT_RESOLUTION_MARGIN,
    pinned_hypothesis_id: str | None = None,
) -> FusionOutcome:
    """Combine per-hypothesis rankings and decide whether the query is resolved."""

    scored: list[QueryHypothesis] = []
    for hypothesis in hypotheses:
        results = retrieved.get(hypothesis.hypothesis_id, [])
        scored.append(
            hypothesis.model_copy(
                update={"support": round(_support(results), 4), "result_count": len(results)}
            )
        )
    weights = {
        item.hypothesis_id: item.prior * (_SUPPORT_FLOOR + item.support) for item in scored
    }
    total = sum(weights.values()) or 1.0
    scored = [
        item.model_copy(update={"posterior": round(weights[item.hypothesis_id] / total, 4)})
        for item in scored
    ]
    by_posterior = sorted(scored, key=lambda item: (-item.posterior, -item.prior))

    interpretation: InterpretationStatus
    if pinned_hypothesis_id is not None:
        selected_id: str | None = pinned_hypothesis_id
        interpretation = "resolved"
        prompt: str | None = None
    else:
        top = by_posterior[0]
        runners = [item for item in by_posterior[1:] if item.result_count > 0]
        if not runners or top.posterior - runners[0].posterior >= resolution_margin:
            interpretation = "resolved"
            prompt = None
        else:
            interpretation = "clarification_suggested"
            prompt = clarification_prompt([top, *runners])
        selected_id = top.hypothesis_id

    merged: dict[str, SearchResult] = {}
    for hypothesis in scored:
        for item in retrieved.get(hypothesis.hypothesis_id, []):
            previous = merged.get(item.segment_id)
            tags = sorted(
                {
                    *(previous.supporting_hypotheses if previous else []),
                    hypothesis.hypothesis_id,
                }
            )
            keep = item if previous is None or item.score > previous.score else previous
            merged[item.segment_id] = keep.model_copy(update={"supporting_hypotheses": tags})

    def order_key(item: SearchResult) -> tuple:
        preferred = 0 if selected_id in item.supporting_hypotheses else 1
        if interpretation == "resolved":
            return (preferred, -item.score, item.start_time, item.video_id)
        return (-item.score, item.start_time, item.video_id)

    results = sorted(merged.values(), key=order_key)[:limit]
    if results and all(
        item.evidence is not None and item.evidence.verdict == "insufficient" for item in results
    ):
        interpretation = "insufficient_evidence"
    elif not results:
        interpretation = "insufficient_evidence"
    return FusionOutcome(
        results=results,
        hypotheses=scored,
        interpretation=interpretation,
        selected_hypothesis_id=selected_id,
        clarification_prompt=prompt,
    )
