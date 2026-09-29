from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from vidquery.api import create_app
from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    ProcessingState,
    Relationship,
    RelationshipSource,
    SearchRequest,
    TranscriptSegment,
    VideoRecord,
)
from vidquery.evaluation import _learned_action_rank, _source_ids, evaluate
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.search import LocalHybridSearchEngine, SafeCypherCompiler, StructuredQueryParser
from vidquery.storage import SQLiteRepository


def video_record(video_id: str = "video-1") -> VideoRecord:
    now = datetime.now(UTC)
    return VideoRecord(
        video_id=video_id,
        display_name="meeting.mp4",
        original_filename="meeting.mp4",
        stored_path="meeting.mp4",
        content_sha256="a" * 64 if video_id == "video-1" else "b" * 64,
        duration=10,
        width=100,
        height=100,
        fps=10,
        has_audio=True,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
    )


def evidence_segment(video_id: str = "video-1") -> CanonicalSegment:
    box = BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.8)
    person = Detection(
        detection_id="person-1",
        frame_id="frame-1",
        video_id=video_id,
        timestamp=5,
        class_id=0,
        class_label="person",
        confidence=0.9,
        pixel_bbox=BoundingBox(x1=10, y1=10, x2=40, y2=80),
        normalized_bbox=box,
        centroid=(25, 45),
        centroid_normalized=(0.25, 0.45),
    )
    laptop = person.model_copy(
        update={
            "detection_id": "laptop-1",
            "class_id": 63,
            "class_label": "laptop",
        }
    )
    relationship = Relationship(
        relationship_id="rel-1",
        source_id=person.detection_id,
        target_id=laptop.detection_id,
        predicate="near",
        confidence=0.8,
        source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
        timestamp=5,
    )
    transcript = TranscriptSegment(
        segment_id="tx-1",
        video_id=video_id,
        start_time=5,
        end_time=7,
        text="We need to review the project budget and deployment architecture",
        confidence=0.9,
        speaker="SPEAKER_01",
    )
    return CanonicalSegment(
        segment_id=f"{video_id}:5000:10000",
        video_id=video_id,
        start_time=5,
        end_time=10,
        frame_ids=["frame-1"],
        detections=[person, laptop],
        entities=["person", "laptop"],
        relationships=[relationship],
        actions=["stand"],
        transcript=transcript.text,
        transcript_segments=[transcript],
        speakers=["SPEAKER_01"],
    )


def populated_repository(tmp_path) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    repository.replace_segments("video-1", [evidence_segment()])
    return repository


def relation_segment(
    *,
    predicate: str = "near",
    source_label: str = "person",
    target_label: str = "chair",
    confidence: float = 0.8,
) -> CanonicalSegment:
    base = evidence_segment()
    source = base.detections[0].model_copy(
        update={"detection_id": f"{source_label}-1", "class_label": source_label}
    )
    target = base.detections[1].model_copy(
        update={"detection_id": f"{target_label}-1", "class_label": target_label}
    )
    relationship = base.relationships[0].model_copy(
        update={
            "relationship_id": f"{source_label}-{predicate}-{target_label}",
            "source_id": source.detection_id,
            "target_id": target.detection_id,
            "predicate": predicate,
            "confidence": confidence,
        }
    )
    return base.model_copy(
        update={
            "detections": [source, target],
            "entities": sorted({source_label, target_label}),
            "relationships": [relationship],
        }
    )


def repository_with_segment(tmp_path, segment: CanonicalSegment) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "relations.sqlite3")
    repository.create_video(video_record())
    repository.replace_segments("video-1", [segment])
    return repository


def test_repository_persists_video_and_segments(tmp_path):
    repository = populated_repository(tmp_path)
    reopened = SQLiteRepository(repository.path)
    assert reopened.get_video("video-1").display_name == "meeting.mp4"
    assert reopened.list_segments()[0].segment_id == "video-1:5000:10000"


def test_replace_segments_is_idempotent(tmp_path):
    repository = populated_repository(tmp_path)
    repository.replace_segments("video-1", [evidence_segment()])
    repository.replace_segments("video-1", [evidence_segment()])
    assert len(repository.list_segments()) == 1


def test_repository_updates_failure_without_exposing_detail(tmp_path):
    repository = populated_repository(tmp_path)
    result = repository.update_state(
        "video-1",
        ProcessingState.FAILED,
        current_stage="test",
        failure_message="safe message",
        failure_detail="secret stack trace",
    )
    assert result.failure_detail == "secret stack trace"
    assert "failure_detail" not in result.model_dump()


def test_query_parser_extracts_multimodal_plan():
    plan = StructuredQueryParser().parse(
        "Find where SPEAKER_01 discusses architecture while standing beside a laptop"
    )
    assert plan.speaker == "SPEAKER_01"
    assert plan.visual_entities == ["laptop"]
    assert plan.relationships == ["near"]
    assert plan.actions == ["stand"]
    assert "architecture" in plan.spoken_terms


def test_query_parser_maps_people_and_white_board():
    plan = StructuredQueryParser().parse("people near a white board")
    assert plan.visual_entities == ["whiteboard", "person"]
    assert plan.relationships == ["near"]
    assert plan.relationship_tuples[0].model_dump() == {
        "subject": "person",
        "predicate": "near",
        "object": "whiteboard",
        "directionality": "symmetric",
    }


def test_query_parser_normalizes_relationship_aliases():
    parser = StructuredQueryParser()
    assert parser.parse("person beside chair").relationship_tuples[0].predicate == "near"
    assert parser.parse("person next to chair").relationship_tuples[0].predicate == "near"
    assert parser.parse("person talking to chair").relationship_tuples[0].predicate == "talk_to"
    assert parser.parse("person writing on whiteboard").relationship_tuples[0].predicate == (
        "write_on"
    )


def test_query_parser_recognizes_ava_entity_and_action_vocabulary():
    parser = StructuredQueryParser()

    assert parser.parse("a suitcase near a refrigerator").visual_entities == [
        "refrigerator",
        "suitcase",
    ]
    assert parser.parse("person opening").actions == ["open"]
    assert parser.parse("person listening").actions == ["listen to"]
    assert parser.parse("person talking").actions == ["talk to"]
    assert parser.parse("person texting").actions == ["text on phone"]
    assert parser.parse("person carrying").actions == ["carry/hold"]


def test_driving_car_is_an_action_and_entity_query():
    plan = StructuredQueryParser().parse("driving car")

    assert plan.visual_entities == ["car"]
    assert plan.actions == ["drive"]
    assert plan.spoken_terms == []


def test_driving_car_rejects_car_only_distractor(tmp_path):
    repository = SQLiteRepository(tmp_path / "driving.sqlite3")
    repository.create_video(video_record())
    base = evidence_segment()
    car = base.detections[1].model_copy(
        update={"detection_id": "car-1", "class_id": 2, "class_label": "car"}
    )
    car_only = base.model_copy(
        update={
            "segment_id": "video-1:0:5000",
            "start_time": 0,
            "end_time": 5,
            "detections": [base.detections[0], car],
            "entities": ["person", "car"],
            "actions": [],
        }
    )
    driving = car_only.model_copy(
        update={
            "segment_id": "video-1:5000:10000",
            "start_time": 5,
            "end_time": 10,
            "actions": ["drive"],
        }
    )
    repository.replace_segments("video-1", [car_only, driving])

    response = LocalHybridSearchEngine(repository).search(
        SearchRequest(query="driving car")
    )

    assert [result.segment_id for result in response.results] == [driving.segment_id]
    assert "actions drive" in response.results[0].match_reason


def test_relationship_phrase_does_not_also_require_action_label():
    plan = StructuredQueryParser().parse("person talking to chair")

    assert plan.relationship_tuples[0].predicate == "talk_to"
    assert plan.actions == []
    write_plan = StructuredQueryParser().parse("person writing on whiteboard")
    assert write_plan.actions == []


def test_query_parser_keeps_below_direction_and_maps_watching():
    plan = StructuredQueryParser().parse("person watching below a laptop")
    assert plan.relationships == ["below"]
    assert plan.actions == ["watch person"]


def test_safe_cypher_uses_parameters_for_untrusted_values():
    parser = StructuredQueryParser()
    compiled = SafeCypherCompiler().compile(
        parser.parse("budget') MATCH (n) DETACH DELETE n //"), [], 10
    )
    assert "DELETE" not in compiled.cypher
    assert compiled.parameters["limit"] == 10
    assert "budget" in compiled.parameters["spoken_terms"]


def test_safe_cypher_compiles_exact_relationship_tuple_as_parameters():
    compiled = SafeCypherCompiler().compile(
        StructuredQueryParser().parse("chair near person"), [], 5
    )
    assert "FROM_ENTITY" in compiled.cypher
    assert "TO_ENTITY" in compiled.cypher
    assert compiled.parameters["relationship_tuples"] == [
        {"subject": "chair", "predicate": "near", "object": "person", "symmetric": True}
    ]


def test_transcript_search_returns_real_segment_timestamp(tmp_path):
    response = LocalHybridSearchEngine(populated_repository(tmp_path)).search(
        SearchRequest(query="Find where the project budget is discussed")
    )
    assert response.results[0].start_time == 5
    assert "spoken terms" in response.results[0].match_reason


def test_visual_and_relationship_search_rank_matching_segment(tmp_path):
    response = LocalHybridSearchEngine(populated_repository(tmp_path)).search(
        SearchRequest(query="Find a person near a laptop")
    )
    assert response.results[0].score == 0.9
    assert response.results[0].relationships[0].source_method.value == "bounding_box_geometry"
    assert response.results[0].matched_relationships[0].source_id == "person-1"
    assert response.results[0].matched_relationships[0].source_class == "person"
    assert response.results[0].matched_relationships[0].target_id == "laptop-1"
    assert response.results[0].matched_relationships[0].target_class == "laptop"
    assert "person[person-1] near laptop[laptop-1]" in response.results[0].match_reason


def test_person_near_chair_matches_the_exact_edge(tmp_path):
    response = LocalHybridSearchEngine(
        repository_with_segment(tmp_path, relation_segment())
    ).search(SearchRequest(query="person near chair"))
    assert len(response.results) == 1
    match = response.results[0].matched_relationships[0]
    assert (match.source_class, match.predicate, match.target_class) == (
        "person",
        "near",
        "chair",
    )
    assert match.directionality == "symmetric"


def test_search_api_exposes_the_matched_relationship_tuple(tmp_path, settings_factory):
    repository = repository_with_segment(tmp_path, relation_segment())
    client = TestClient(create_app(settings_factory(), repository))
    payload = client.post("/api/search", json={"query": "person near chair"}).json()
    assert payload["results"][0]["matched_relationships"][0] == {
        "relationship_id": "person-near-chair",
        "source_id": "person-1",
        "source_class": "person",
        "predicate": "near",
        "target_id": "chair-1",
        "target_class": "chair",
        "directionality": "symmetric",
        "confidence": 0.8,
            "source_method": "bounding_box_geometry",
            "timestamp": 5.0,
            "model_version": None,
            "start_time": None,
            "end_time": None,
            "source_track_id": None,
            "target_track_id": None,
            "observation_count": 1,
            "temporal_smoothed": False,
        }


def test_person_near_chair_rejects_person_near_table_distractor(tmp_path):
    segment = relation_segment(target_label="dining table")
    chair = segment.detections[1].model_copy(
        update={"detection_id": "chair-1", "class_label": "chair"}
    )
    segment = segment.model_copy(
        update={
            "detections": [*segment.detections, chair],
            "entities": ["chair", "dining table", "person"],
        }
    )
    response = LocalHybridSearchEngine(repository_with_segment(tmp_path, segment)).search(
        SearchRequest(query="person near chair")
    )
    assert response.results == []


def test_chair_near_person_matches_reversed_symmetric_edge(tmp_path):
    response = LocalHybridSearchEngine(
        repository_with_segment(tmp_path, relation_segment())
    ).search(SearchRequest(query="chair near person"))
    assert response.results
    assert response.results[0].matched_relationships[0].directionality == "symmetric"


def test_directional_relationship_does_not_match_in_reverse(tmp_path):
    response = LocalHybridSearchEngine(
        repository_with_segment(tmp_path, relation_segment(predicate="left_of"))
    ).search(SearchRequest(query="chair left of person"))
    assert response.results == []


def test_unrelated_relationship_does_not_increase_exact_relation_score(tmp_path):
    exact = relation_segment(confidence=0.4)
    exact_score = LocalHybridSearchEngine(repository_with_segment(tmp_path, exact)).search(
        SearchRequest(query="person near chair")
    ).results[0].score

    table = exact.detections[1].model_copy(
        update={"detection_id": "table-1", "class_label": "dining table"}
    )
    unrelated = exact.relationships[0].model_copy(
        update={
            "relationship_id": "chair-near-table",
            "source_id": "chair-1",
            "target_id": "table-1",
            "confidence": 1.0,
        }
    )
    with_distractor = exact.model_copy(
        update={
            "detections": [*exact.detections, table],
            "entities": ["chair", "dining table", "person"],
            "relationships": [*exact.relationships, unrelated],
        }
    )
    distractor_repository = SQLiteRepository(tmp_path / "relations-distractor.sqlite3")
    distractor_repository.create_video(video_record())
    distractor_repository.replace_segments("video-1", [with_distractor])
    distractor_score = LocalHybridSearchEngine(distractor_repository).search(
        SearchRequest(query="person near chair")
    ).results[0].score
    assert distractor_score == exact_score


def test_structured_search_rejects_partial_entity_match(tmp_path):
    repository = populated_repository(tmp_path)
    segment = evidence_segment().model_copy(
        update={"entities": ["person"], "detections": evidence_segment().detections[:1]}
    )
    repository.replace_segments("video-1", [segment])
    response = LocalHybridSearchEngine(repository).search(
        SearchRequest(query="person near laptop")
    )
    assert response.results == []


def test_parser_maps_vehicle_queries_to_yolo_coco_entities():
    plan = StructuredQueryParser().parse("Find a police car near a person")

    assert plan.visual_entities == ["car", "person"]
    assert plan.spoken_terms == []
    assert plan.relationship_tuples[0].subject == "car"
    assert plan.relationship_tuples[0].predicate == "near"
    assert plan.relationship_tuples[0].object == "person"


def test_speaker_and_action_search(tmp_path):
    response = LocalHybridSearchEngine(populated_repository(tmp_path)).search(
        SearchRequest(query="Find where SPEAKER_01 is standing")
    )
    assert response.results
    assert response.parsed_query.speaker == "SPEAKER_01"


def test_unrelated_query_returns_no_results(tmp_path):
    response = LocalHybridSearchEngine(populated_repository(tmp_path)).search(
        SearchRequest(query="quantum zebras")
    )
    assert response.results == []


def test_search_respects_video_filter(tmp_path):
    repository = populated_repository(tmp_path)
    response = LocalHybridSearchEngine(repository).search(
        SearchRequest(query="budget", video_ids=["not-this-video"])
    )
    assert response.results == []


def test_canonical_neo4j_ingestion_is_parameterized():
    calls = []
    indexer = CanonicalNeo4jIndexer.__new__(CanonicalNeo4jIndexer)
    indexer.run_query = lambda query, parameters=None: calls.append((query, parameters)) or []
    indexer.ingest(video_record(), [evidence_segment()])
    assert len(calls) == 7
    assert "$segments" in calls[4][0]
    assert calls[4][1]["segments"][0]["segment_id"] == "video-1:5000:10000"
    assert calls[5][1]["appearances"] == []
    assert "meeting.mp4" not in calls[0][0]


def test_evaluation_compares_transcript_and_hybrid_rankers(tmp_path):
    repository = populated_repository(tmp_path)
    dataset = tmp_path / "queries.json"
    dataset.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "visual-person",
                        "category": "visual",
                        "query": "person",
                        "relevant": [
                            {
                                "source_video_id": "meeting",
                                "start_time": 5,
                                "end_time": 10,
                                "tolerance_seconds": 0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = evaluate(repository, dataset)
    assert result["summary"]["transcript_lexical"]["overall"]["precision_at_1"] == 0
    assert result["summary"]["hybrid"]["overall"]["precision_at_1"] == 1
    assert result["summary"]["hybrid"]["by_video"]["meeting"]["precision_at_1"] == 1
    assert result["summary"]["exact_relationship_aware_graph"]["status"] == (
        "not_evaluated"
    )


def test_learned_action_rank_requires_checkpoint_predictions(tmp_path):
    repository = populated_repository(tmp_path)
    video_id = repository.list_videos()[0].video_id
    predictions = {("meeting", 5.0): {"stand": 0.91}}

    ranked = _learned_action_rank(
        repository,
        predictions,
        _source_ids(repository),
        "person standing",
        video_ids=[video_id],
        limit=5,
    )

    assert [(result.start_time, result.end_time) for result in ranked] == [(5.0, 10.0)]
    assert (
        _learned_action_rank(
            repository,
            {},
            _source_ids(repository),
            "person standing",
            video_ids=[video_id],
            limit=5,
        )
        == []
    )
