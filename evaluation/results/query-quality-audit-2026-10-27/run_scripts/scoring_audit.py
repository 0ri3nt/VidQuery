"""Retrieval/scoring audit: per-query diagnostics and counterfactual scoring variants.

Run under scripts.run_with_profile with the T1 (selected) search settings and an
audit copy of an index. Nothing here changes the package; variants are applied
by monkeypatching inside this process only.

  diagnose  --manifest M --split validation --output OUT.json
  evaluate  --variant {baseline,missing_penalty,hyp_reliability,both} -- <evaluate_moment_benchmark args>
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from vidquery import hypotheses as hypotheses_module
from vidquery import search as search_module
from vidquery.challenge_evaluation import ChallengeMethods, source_video_id
from vidquery.config import get_settings
from vidquery.domain import SearchRequest
from vidquery.localization import collapse_duplicate_results
from vidquery.query_planner import QueryPlanningService
from vidquery.reliability import RawContribution, reliability_rank_score
from vidquery.search import configured_search_engine, matching_relationships
from vidquery.storage import SQLiteRepository


def missing_constraints(plan, segment) -> dict[str, list[str]]:
    entities = {item.lower() for item in segment.entities}
    actions = {item.lower() for item in segment.actions}
    missing = {
        "entities": [item for item in plan.visual_entities if item not in entities],
        "actions": [item for item in plan.actions if item not in actions],
        "relationship_tuples": [
            f"{q.subject} {q.predicate} {q.object}"
            for q in plan.relationship_tuples
            if not matching_relationships(segment, q)
        ],
    }
    if not plan.relationship_tuples and plan.relationships:
        present = {search_module.normalize_predicate(r.predicate) for r in segment.relationships}
        missing["relationships"] = [item for item in plan.relationships if item not in present]
    return missing


def install_missing_penalty() -> None:
    """Unmatched parsed constraints contribute zero evidence instead of nothing."""

    original = search_module.LocalHybridSearchEngine._score_detailed

    def patched(self, plan, query_vector, segment, **kwargs):
        evidence = original(self, plan, query_vector, segment, **kwargs)
        if evidence.score <= 0 or not self.soft_constraints:
            return evidence
        missing = missing_constraints(plan, segment)
        contributions = list(evidence.contributions)
        if plan.visual_entities:
            fraction = 1 - len(missing["entities"]) / len(plan.visual_entities)
            contributions = [
                RawContribution(c.modality, c.raw_score * fraction, c.source_method)
                if c.modality == "entity" else c
                for c in contributions
            ]
        for _ in missing["actions"]:
            contributions.append(RawContribution("action", 0.0, "gnn_action_model"))
        for _ in missing["relationship_tuples"] + missing.get("relationships", []):
            contributions.append(RawContribution("relationship", 0.0, "relationship_gnn"))
        return replace(evidence, contributions=contributions)

    search_module.LocalHybridSearchEngine._score_detailed = patched


# Pre-registered before any variant was evaluated: English function words and
# aspect fillers that the parser currently turns into "spoken terms".
EXTRA_STOPWORDS = {
    "its", "it's", "their", "theirs", "into", "he", "she", "his", "her", "hers", "him", "them",
    "we", "us", "our", "your", "some", "any", "out", "off", "through", "up", "down", "by",
    "over", "under", "onto", "upon", "this", "these", "those", "then", "than", "who", "whom",
    "what", "which", "here", "first", "again", "also", "just", "very", "has", "have", "had",
    "having", "been", "being", "were", "will", "would", "could", "should", "as", "if", "so",
    "or", "but", "not", "after", "before", "around", "across", "toward", "towards", "away",
    "back", "all", "both", "himself", "herself", "themselves", "itself", "start", "starts",
    "started", "starting", "begin", "begins", "began", "beginning", "continues", "continue",
}


def install_stopword_fix() -> None:
    search_module.STOPWORDS.update(EXTRA_STOPWORDS)


def install_missing_verdict_only() -> None:
    """Missing constraints lower the verdict/confidence, but ranking keeps the original confidence."""

    original = search_module.LocalHybridSearchEngine._score_detailed

    def patched(self, plan, query_vector, segment, **kwargs):
        evidence = original(self, plan, query_vector, segment, **kwargs)
        if evidence.score <= 0 or not self.soft_constraints:
            return evidence
        missing = missing_constraints(plan, segment)
        extra = [RawContribution("entity", 0.0, "missing_constraint") for _ in missing["entities"]]
        extra += [RawContribution("action", 0.0, "missing_constraint") for _ in missing["actions"]]
        extra += [RawContribution("relationship", 0.0, "missing_constraint")
                  for _ in missing["relationship_tuples"] + missing.get("relationships", [])]
        return replace(evidence, contributions=[*evidence.contributions, *extra])

    def rank(score, assessment):
        if assessment is None:
            return score
        kept = [c for c in assessment.contributions if c.source_method != "missing_constraint"]
        weight = sum(c.query_weight for c in kept)
        confidence = sum(c.query_weight * c.weighted_score for c in kept) / weight if weight else 0.0
        return max(0.0, min(1.0, 0.5 * score + 0.5 * confidence))

    search_module.LocalHybridSearchEngine._score_detailed = patched
    search_module.reliability_rank_score = rank


def install_no_transcripts() -> None:
    """Counterfactual for a perfect speech-activity gate on speech-free clips: no transcript evidence.

    Segment embeddings are re-encoded from the same text minus the transcript.
    """

    from vidquery.search import configured_text_encoder

    original = SQLiteRepository.list_segments
    cache: dict[str, object] = {}
    encoder_holder: list = []

    def patched(self, video_ids=None):
        if not encoder_holder:
            encoder_holder.append(configured_text_encoder(get_settings()))
        encoder = encoder_holder[0]
        out = []
        for segment in original(self, video_ids):
            if segment.segment_id not in cache:
                text = " ".join([*segment.entities, *segment.actions,
                                 *(r.predicate for r in segment.relationships),
                                 *(e.text for e in segment.ocr_evidence)]).strip()
                cache[segment.segment_id] = segment.model_copy(update={
                    "transcript": "", "transcript_segments": [],
                    "embedding": encoder.encode(text) if segment.embedding is not None else None,
                })
            out.append(cache[segment.segment_id])
        return out

    SQLiteRepository.list_segments = patched


def install_hypothesis_reliability_order() -> None:
    """Order fused multi-reading results like single-reading ones (reliability rank)."""

    original = hypotheses_module.fuse_hypotheses

    def patched(hypotheses, retrieved, *, limit, **kwargs):
        outcome = original(hypotheses, retrieved, limit=10_000, **kwargs)
        selected = outcome.selected_hypothesis_id

        def key(item):
            preferred = 0 if (outcome.interpretation == "resolved"
                              and selected in item.supporting_hypotheses) else 1
            return (preferred, -reliability_rank_score(item.score, item.evidence),
                    -item.score, item.start_time, item.video_id)

        ordered = collapse_duplicate_results(sorted(outcome.results, key=key))[:limit]
        return replace(outcome, results=ordered) if hasattr(outcome, "__dataclass_fields__") \
            else outcome.__class__(**{**outcome.__dict__, "results": ordered})

    hypotheses_module.fuse_hypotheses = patched
    import vidquery.challenge_evaluation as challenge_module
    challenge_module.fuse_hypotheses = patched


def diagnose(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    queries = manifest["splits"][args.split]["queries"]
    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    engine = configured_search_engine(repository, settings)
    planner = QueryPlanningService(replace(settings, enable_query_planner=False))
    by_source = {source_video_id(v): v.video_id for v in repository.list_videos()}
    segments = {s.segment_id: s for s in repository.list_segments(None)}
    transcripts: dict[str, set[str]] = {}
    for segment in segments.values():
        transcripts.setdefault(segment.video_id, set()).update(search_module._tokens(segment.transcript))
    methods = ChallengeMethods(repository, planner, engine=engine,
                               resolution_margin=settings.hypothesis_resolution_margin)
    rows = []
    for item in queries:
        planning = planner.plan(item["query"])
        plan = planning.plan
        video_id = by_source[item["video"]]
        request = SearchRequest(query=item["query"], video_ids=[video_id], limit=5,
                                ranking="reliability", localize=True,
                                collapse_duplicate_evidence=True)
        hypothesis_path = bool(planning.hypotheses)
        order_mismatch = None
        if hypothesis_path:
            retrieved = {h.hypothesis_id: methods._retrieve(h.plan, request) for h in planning.hypotheses}
            fused = hypotheses_module.fuse_hypotheses(
                list(planning.hypotheses), retrieved, limit=5,
                resolution_margin=settings.hypothesis_resolution_margin)
            results = fused.results
            by_rel = sorted(results, key=lambda r: -reliability_rank_score(r.score, r.evidence))
            order_mismatch = bool(results) and by_rel[0].segment_id != results[0].segment_id
            peaks = [r.localization.peak_time for r in results if r.localization]
            near_duplicates = sum(1 for i, a in enumerate(peaks) for b in peaks[i + 1:]
                                  if abs(a - b) <= 0.5)
        else:
            results = engine.search(request, plan_override=plan).results
            near_duplicates = 0
        top = results[0] if results else None
        segment = segments.get(top.segment_id) if top else None
        row = {
            "id": item["id"], "kind": item["kind"], "query": item["query"],
            "intent": plan.intent, "entities": plan.visual_entities, "actions": plan.actions,
            "spoken_terms": plan.spoken_terms, "ocr_terms": plan.ocr_terms,
            "relationship_tuples": len(plan.relationship_tuples),
            "spoken_terms_in_video_transcript": [t for t in plan.spoken_terms
                                                 if t in transcripts.get(video_id, set())],
            "hypothesis_path": hypothesis_path, "hypothesis_order_mismatch": order_mismatch,
            "near_duplicate_pairs_in_top5": near_duplicates,
            "result_count": len(results),
        }
        if top is not None:
            contributions = top.evidence.contributions if top.evidence else []
            row.update({
                "top_verdict": top.evidence.verdict if top.evidence else None,
                "top_confidence": top.evidence.confidence if top.evidence else None,
                "top_score": top.score,
                "top_modalities": sorted({c.modality for c in contributions if c.raw_score > 0}),
                "top_explanation": top.evidence.explanation if top.evidence else None,
                "top_missing": missing_constraints(plan, segment) if segment else None,
            })
        rows.append(row)
    summary = summarize(rows)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def summarize(rows: list[dict]) -> dict:
    out: dict = {"queries": len(rows), "by_kind": dict(Counter(r["kind"] for r in rows)),
                 "intents": dict(Counter(r["intent"] for r in rows))}
    with_spoken = [r for r in rows if r["spoken_terms"]]
    out["queries_with_spoken_terms"] = len(with_spoken)
    out["spoken_terms_found_in_that_videos_transcript"] = sum(
        bool(r["spoken_terms_in_video_transcript"]) for r in with_spoken)
    out["most_common_spoken_terms"] = Counter(t for r in rows for t in r["spoken_terms"]).most_common(25)
    out["hypothesis_path_queries"] = sum(r["hypothesis_path"] for r in rows)
    out["hypothesis_order_mismatch"] = sum(bool(r["hypothesis_order_mismatch"]) for r in rows)
    out["hypothesis_near_duplicate_pairs"] = sum(r["near_duplicate_pairs_in_top5"] for r in rows)
    for kind in ("positive", "in_domain_negative", "out_of_domain_negative"):
        group = [r for r in rows if r["kind"] == kind and r.get("top_verdict")]
        with_actions = [r for r in group if r["actions"]]
        out[kind] = {
            "answered": len(group),
            "verdicts": dict(Counter(r["top_verdict"] for r in group)),
            "action_queries": len(with_actions),
            "action_queries_top1_missing_every_action": sum(
                len(r["top_missing"]["actions"]) == len(r["actions"]) for r in with_actions),
            "supported_top1_missing_every_action": sum(
                r["top_verdict"] == "supported" and len(r["top_missing"]["actions"]) == len(r["actions"])
                for r in with_actions),
            "supported_or_weak_with_only_person_entity_and_no_action": sum(
                r["top_verdict"] in ("supported", "weak") and r["entities"] == ["person"]
                and r["actions"] and len(r["top_missing"]["actions"]) == len(r["actions"])
                for r in with_actions),
            "explanation_dominated_by_entity": sum(
                (r["top_explanation"] or "").startswith("entity evidence") for r in group),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--split", required=True)
    d.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--variant", required=True,
                   help="baseline, or '+'-joined: missing_penalty, stopwords, hyp_reliability")
    e.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "diagnose":
        diagnose(args)
        return
    parts = set(args.variant.split("+")) - {"baseline"}
    unknown = parts - {"missing_penalty", "missing_verdict_only", "stopwords", "hyp_reliability",
                       "no_transcripts"}
    if unknown:
        raise SystemExit(f"unknown variant part(s): {sorted(unknown)}")
    if "missing_penalty" in parts:
        install_missing_penalty()
    if "missing_verdict_only" in parts:
        install_missing_verdict_only()
    if "stopwords" in parts:
        install_stopword_fix()
    if "hyp_reliability" in parts:
        install_hypothesis_reliability_order()
    if "no_transcripts" in parts:
        install_no_transcripts()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    sys.argv = ["evaluate_moment_benchmark", *rest]
    runpy.run_module("scripts.evaluate_moment_benchmark", run_name="__main__")


if __name__ == "__main__":
    main()
