from __future__ import annotations

from test_search_and_storage import evidence_segment, video_record

from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    MatchedRelationship,
    OCREvidence,
    QueryPlan,
    RelationshipSource,
    SearchRequest,
    TemporalLocalization,
    TranscriptSegment,
    TranscriptWord,
)
from vidquery.localization import collapse_duplicate_results, localize_segment
from vidquery.search import LocalHybridSearchEngine
from vidquery.storage import SQLiteRepository


def _detection(detection_id: str, label: str, timestamp: float, confidence: float = 0.9):
    box = BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.8)
    return Detection(
        detection_id=detection_id,
        frame_id=f"frame-{timestamp}",
        video_id="video-1",
        timestamp=timestamp,
        class_id=0,
        class_label=label,
        confidence=confidence,
        pixel_bbox=BoundingBox(x1=10, y1=10, x2=40, y2=80),
        normalized_bbox=box,
        centroid=(25, 45),
        centroid_normalized=(0.25, 0.45),
    )


def _segment(**overrides) -> CanonicalSegment:
    base = dict(
        segment_id="video-1:40000:45000",
        video_id="video-1",
        start_time=40.0,
        end_time=45.0,
        processing_metadata={"frame_sample_rate": 1.0},
    )
    base.update(overrides)
    return CanonicalSegment(**base)


def test_word_timestamps_localize_speech_to_the_word():
    utterance = TranscriptSegment(
        segment_id="tx",
        video_id="video-1",
        start_time=40.0,
        end_time=45.0,
        text="So the deployment plan is ready",
        words=[
            TranscriptWord(word=" So", start_time=40.0, end_time=40.3, probability=0.9),
            TranscriptWord(word=" the", start_time=40.3, end_time=40.5, probability=0.9),
            TranscriptWord(word=" deployment", start_time=42.3, end_time=42.9, probability=0.95),
            TranscriptWord(word=" plan", start_time=42.9, end_time=43.2, probability=0.9),
        ],
    )
    segment = _segment(transcript=utterance.text, transcript_segments=[utterance])

    result = localize_segment(
        QueryPlan(intent="transcript_search", spoken_terms=["deployment"]), segment
    )

    assert result.precision == "word"
    assert result.peak_time == 42.3
    assert result.source == "whisper_word_timing"
    assert result.confidence > 0.8


def test_utterance_without_words_uses_proportional_interpolation():
    utterance = TranscriptSegment(
        segment_id="tx",
        video_id="video-1",
        start_time=40.0,
        end_time=44.0,
        text="one two three deployment",
    )
    segment = _segment(transcript=utterance.text, transcript_segments=[utterance])

    result = localize_segment(
        QueryPlan(intent="transcript_search", spoken_terms=["deployment"]), segment
    )

    assert result.precision == "utterance_interpolated"
    # token 4 of 4 -> start + 4s * (3.5/4) = 43.5
    assert abs(result.peak_time - 43.5) < 1e-6
    assert result.confidence < 0.6


def test_entity_localizes_to_the_frame_where_all_required_classes_co_occur():
    detections = [
        _detection("p1", "person", 40.0),
        _detection("p2", "person", 41.0),
        _detection("l1", "laptop", 42.0, 0.7),
        _detection("p3", "person", 42.0),
        _detection("l2", "laptop", 43.0, 0.95),
        _detection("p4", "person", 43.0),
    ]
    segment = _segment(detections=detections, entities=["person", "laptop"])

    result = localize_segment(
        QueryPlan(intent="visual_search", visual_entities=["person", "laptop"]), segment
    )

    assert result.precision == "frame"
    assert result.peak_time == 43.0
    assert result.start_time <= 42.0 and result.end_time >= 43.0


def test_action_instance_timestamp_is_used():
    segment = _segment(
        actions=["drive"],
        processing_metadata={
            "frame_sample_rate": 1.0,
            "action_instances": [
                {"action": "drive", "confidence": 0.4, "timestamp": 41.0, "source_method": "x"},
                {"action": "drive", "confidence": 0.8, "timestamp": 44.0, "source_method": "x"},
            ],
        },
    )
    result = localize_segment(QueryPlan(intent="action_search", actions=["drive"]), segment)
    assert result.peak_time == 44.0
    assert result.precision == "frame"


def test_ocr_and_relationship_and_fallback_paths():
    ocr_segment = _segment(
        ocr_evidence=[
            OCREvidence(
                ocr_id="o", text="Kubernetes", start_time=38.0, end_time=47.0, confidence=0.9
            )
        ]
    )
    ocr = localize_segment(QueryPlan(intent="ocr_search", ocr_terms=["kubernetes"]), ocr_segment)
    assert ocr.precision == "interval" and ocr.peak_time == 40.0

    matched = MatchedRelationship(
        relationship_id="r",
        source_id="a",
        source_class="person",
        predicate="near",
        target_id="b",
        target_class="laptop",
        directionality="symmetric",
        confidence=0.8,
        source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
        timestamp=43.0,
    )
    relation = localize_segment(
        QueryPlan(intent="relationship_search"), _segment(), matched_relationships=[matched]
    )
    assert relation.peak_time == 43.0

    empty = localize_segment(QueryPlan(intent="visual_search", visual_entities=["car"]), _segment())
    assert empty.precision == "segment" and empty.peak_time == 40.0


def test_multimodal_agreement_intersects_intervals():
    utterance = TranscriptSegment(
        segment_id="tx",
        video_id="video-1",
        start_time=40.0,
        end_time=45.0,
        text="deployment",
        words=[TranscriptWord(word="deployment", start_time=42.4, end_time=42.9)],
    )
    detections = [_detection(f"l{t}", "laptop", float(t)) for t in range(40, 45)]
    segment = _segment(
        transcript="deployment",
        transcript_segments=[utterance],
        detections=detections,
        entities=["laptop"],
    )
    result = localize_segment(
        QueryPlan(
            intent="multimodal_search", spoken_terms=["deployment"], visual_entities=["laptop"]
        ),
        segment,
    )
    assert result.precision == "word"
    assert result.peak_time == 42.4
    assert result.source.endswith("+agreement")
    assert len(result.evidence) == 2


def test_search_response_carries_localization_and_seeks_thumbnail_to_peak(tmp_path):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    repository.replace_segments("video-1", [evidence_segment()])

    response = LocalHybridSearchEngine(repository).search(
        SearchRequest(query="where is deployment mentioned", video_ids=["video-1"], limit=3)
    )

    top = response.results[0]
    assert isinstance(top.localization, TemporalLocalization)
    assert top.localization.precision == "utterance_interpolated"
    assert 5.0 <= top.localization.peak_time <= 7.0
    assert f"timestamp={top.localization.peak_time}" in top.thumbnail_url

    disabled = LocalHybridSearchEngine(repository).search(
        SearchRequest(query="deployment", video_ids=["video-1"], localize=False)
    )
    assert disabled.results[0].localization is None


def test_duplicate_windows_collapse_to_one_moment(tmp_path):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    first = evidence_segment()
    utterance = first.transcript_segments[0].model_copy(
        update={"start_time": 4.0, "end_time": 6.0}
    )
    early = first.model_copy(
        update={
            "segment_id": "video-1:0:5000",
            "start_time": 0.0,
            "end_time": 5.0,
            "transcript_segments": [utterance],
            "detections": [],
            "entities": [],
            "relationships": [],
        }
    )
    late = first.model_copy(update={"transcript_segments": [utterance]})
    repository.replace_segments("video-1", [early, late])
    engine = LocalHybridSearchEngine(repository)

    plain = engine.search(SearchRequest(query="deployment", video_ids=["video-1"]))
    collapsed = engine.search(
        SearchRequest(query="deployment", video_ids=["video-1"], collapse_duplicate_evidence=True)
    )

    assert len(plain.results) == 2
    assert len(collapsed.results) == 1
    assert collapsed.results == collapse_duplicate_results(plain.results)
