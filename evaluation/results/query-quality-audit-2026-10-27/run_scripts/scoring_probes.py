"""Controlled probes of query routing and verdict fusion on synthetic segments (no index, no models)."""

import json
from pathlib import Path

from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    SearchRequest,
    VideoRecord,
    ProcessingState,
    utc_now,
)
from vidquery.search import LocalHybridSearchEngine, StructuredQueryParser
from vidquery.storage import SQLiteRepository


def segment(video_id: str, entities: list[tuple[str, float]], actions: list[str]) -> CanonicalSegment:
    detections = [
        Detection(detection_id=f"d{i}", frame_id="f0", video_id=video_id, timestamp=1.0, class_id=i,
                  class_label=label, confidence=conf,
                  pixel_bbox=BoundingBox(x1=64, y1=48, x2=320, y2=432),
                  normalized_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.9),
                  centroid=(192.0, 240.0), centroid_normalized=(0.3, 0.5))
        for i, (label, conf) in enumerate(entities)
    ]
    return CanonicalSegment(
        segment_id=f"{video_id}-s0", video_id=video_id, start_time=0.0, end_time=5.0, transcript="",
        frame_ids=["f0"], detections=detections, entities=[label for label, _ in entities],
        actions=actions,
        processing_metadata={"action_sources": {a: "gnn_action_model" for a in actions},
                             "action_confidences": {a: 0.6 for a in actions}},
    )


def main(out: Path) -> None:
    import tempfile

    parser = StructuredQueryParser()
    routing = {}
    for query in ("person opens the closet door.", "person puts their shoes on.",
                  "person takes a towel out of the closet.", "person starts sneezing.",
                  "he walks into the room."):
        plan = parser.parse(query)
        routing[query] = {"intent": plan.intent, "entities": plan.visual_entities,
                          "actions": plan.actions, "spoken_terms": plan.spoken_terms}

    root = Path(tempfile.mkdtemp())
    repo = SQLiteRepository(root / "probe.sqlite3")
    now = utc_now()
    for vid in ("v1",):
        repo.create_video(VideoRecord(video_id=vid, display_name=f"{vid}.mp4", original_filename=f"{vid}.mp4",
                                      stored_path=str(root / f"{vid}.mp4"), content_sha256=vid * 8,
                                      duration=5.0, width=640, height=480, fps=25.0, has_audio=False,
                                      upload_time=now, updated_time=now, state=ProcessingState.READY))
    verdicts = {}
    cases = {
        "person only, no action": segment("v1", [("person", 0.91)], []),
        "person + matching action 'cook'": segment("v1", [("person", 0.91)], ["cook"]),
        "person + laptop (both present)": segment("v1", [("person", 0.91), ("laptop", 0.8)], []),
    }
    queries = ["person cooking at the stove.", "person cooking.", "person.",
               "person near a laptop.", "person with a laptop and a bottle."]
    for soft in (False, True):
        engine = LocalHybridSearchEngine(repo, soft_constraints=soft)
        for name, seg in cases.items():
            repo.replace_segments("v1", [seg])
            for query in queries:
                response = engine.search(SearchRequest(query=query, video_ids=["v1"], limit=1,
                                                       ranking="reliability", localize=False))
                top = response.results[0] if response.results else None
                verdicts[f"soft={soft} | {name} | {query}"] = (
                    {"verdict": top.evidence.verdict, "confidence": top.evidence.confidence,
                     "score": top.score, "explanation": top.evidence.explanation}
                    if top else "no result (rejected)")
    out.write_text(json.dumps({"routing": routing, "verdicts": verdicts}, indent=1), encoding="utf-8")
    print(json.dumps({"routing": routing}, indent=1))
    for key, value in verdicts.items():
        print(key, "->", value if isinstance(value, str) else
              f"{value['verdict']} {value['confidence']} (score {value['score']})")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
