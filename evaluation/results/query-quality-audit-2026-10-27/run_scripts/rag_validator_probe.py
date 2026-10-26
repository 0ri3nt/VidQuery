"""Controlled test of GroundedAnswerService validation with a scripted provider (no network).

Each case returns a fixed provider answer for real-looking search results and
records whether the service accepts it. It tests the validator, not Groq.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

from vidquery.config import get_settings
from vidquery.domain import EvidenceAssessment, SearchResponse, SearchResult, TemporalLocalization
from vidquery.rag import GroundedAnswerService


class Scripted:
    def __init__(self, answer):
        self.answer = answer

    def post_json(self, url, *, headers, payload, timeout):
        return {"choices": [{"message": {"content": json.dumps(self.answer)}}]}


def result(start, peak, verdict, confidence, actions):
    return SearchResult(
        video_id="v1", video_title="OM2NF.mp4", segment_id=f"v1-{start}", start_time=start,
        end_time=start + 5, score=0.5, transcript="", speakers=[], entities=["person", "cup"],
        actions=actions, relationships=[], matched_relationships=[], appearance_matches=[],
        thumbnail_url="/t", stream_url="/s", match_reason="Matched visual entities person.",
        localization=TemporalLocalization(peak_time=peak, start_time=start, end_time=start + 5,
                                          precision="frame", source="detector_frame_observation",
                                          confidence=0.5),
        evidence=EvidenceAssessment(confidence=confidence, verdict=verdict, contributions=[],
                                    explanation="entity evidence dominates"),
    )


CASES = {
    "valid citation, wrong time in text": (
        [result(15.0, 15.99, "supported", 0.5, ["drink"])], "single",
        {"answer": "The person drinks at 00:42.", "supported": True, "evidence_ids": ["E1"]}),
    "valid citation, action absent from evidence": (
        [result(35.0, 37.99, "weak", 0.33, [])], "single",
        {"answer": "The person is cooking at the stove.", "supported": True, "evidence_ids": ["E1"]}),
    "supported answer although every result is 'insufficient'": (
        [result(0.0, 1.0, "insufficient", 0.05, [])], "insufficient_evidence",
        {"answer": "Yes, the person cooks here.", "supported": True, "evidence_ids": ["E1"]}),
    "invented evidence id": (
        [result(15.0, 15.99, "supported", 0.5, ["drink"])], "single",
        {"answer": "The person drinks.", "supported": True, "evidence_ids": ["E7"]}),
    "supported without citations": (
        [result(15.0, 15.99, "supported", 0.5, ["drink"])], "single",
        {"answer": "The person drinks.", "supported": True, "evidence_ids": []}),
}


def main(out: Path) -> None:
    settings = replace(get_settings(), enable_rag_generation=True, rag_provider="groq",
                       groq_api_key="scripted", rag_model="scripted")
    report = {}
    for name, (results, interpretation, answer) in CASES.items():
        service = GroundedAnswerService(settings, transport=Scripted(answer))
        response = SearchResponse(query="q", parsed_query={"intent": "visual_search"}, results=results,
                                  interpretation=interpretation)
        outcome = service.generate("q", response)
        report[name] = {"status": outcome.status,
                        "accepted_as_supported": bool(outcome.answer and outcome.answer.supported),
                        "shown_text": outcome.answer.answer if outcome.answer else None}
        print(f"{name}: {outcome.status} (accepted as supported: {report[name]['accepted_as_supported']})")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
