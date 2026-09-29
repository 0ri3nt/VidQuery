from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from vidquery.api import create_app
from vidquery.appearance import TrackAppearanceExtractor, extract_detection_crop
from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    EntityAppearance,
    FrameRecord,
    ProcessingState,
    Relationship,
    RelationshipSource,
    SearchRequest,
    VideoRecord,
    VisualAttributeEvidence,
)
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.query_planner import QueryPlanningService
from vidquery.search import HashingTextEncoder, LocalHybridSearchEngine, StructuredQueryParser
from vidquery.storage import SQLiteRepository


def _normalized(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


class FakeAppearanceEncoder:
    model_name = "test/frozen-vl"
    model_version = "test-frozen-v1"

    def __init__(self) -> None:
        self.image_calls = 0

    def encode_images(self, images):
        self.image_calls += 1
        output = []
        for image in images:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255.0
            darkness = max(0.0, 1.0 - float(rgb.mean()))
            output.append(_normalized([float(rgb[0]), float(rgb[1]), float(rgb[2]), darkness, 0.0]))
        return output

    def encode_texts(self, texts):
        output = []
        for text in texts:
            value = text.lower()
            if "red" in value:
                vector = [1.0, 0.0, 0.0, 0.0, 0.0]
            elif "blue" in value:
                vector = [0.0, 0.0, 1.0, 0.0, 0.0]
            elif "black" in value or "dark" in value:
                vector = [0.0, 0.0, 0.0, 1.0, 0.0]
            elif "white" in value:
                vector = [0.5, 0.5, 0.5, 0.0, 0.5]
            elif "without" in value:
                vector = [0.0, 0.0, 0.0, 0.0, -1.0]
            else:
                vector = [0.0, 0.0, 0.0, 0.0, 1.0]
            output.append(_normalized(vector))
        return output


class NoopProcessor:
    def process(self, video_id: str) -> None:
        del video_id


def _video() -> VideoRecord:
    now = datetime.now(UTC)
    return VideoRecord(
        video_id="appearance-video",
        display_name="appearance-proof.mp4",
        original_filename="appearance-proof.mp4",
        stored_path="appearance-proof.mp4",
        content_sha256="c" * 64,
        duration=10,
        width=100,
        height=100,
        fps=5,
        has_audio=False,
        upload_time=now,
        updated_time=now,
        state=ProcessingState.READY,
    )


def _detection(
    detection_id: str,
    class_label: str,
    timestamp: float,
    *,
    track_id: str,
    appearance_id: str | None = None,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        frame_id=f"frame-{timestamp}",
        video_id="appearance-video",
        timestamp=timestamp,
        class_id=0,
        class_label=class_label,
        confidence=0.9,
        pixel_bbox=BoundingBox(x1=10, y1=10, x2=80, y2=90),
        normalized_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.8, y2=0.9),
        centroid=(45, 50),
        centroid_normalized=(0.45, 0.5),
        track_id=track_id,
        appearance_id=appearance_id,
    )


def _appearance(
    appearance_id: str,
    track_id: str,
    class_label: str,
    embedding: list[float],
    attributes: dict[str, tuple[str | bool, float]],
) -> EntityAppearance:
    return EntityAppearance(
        appearance_id=appearance_id,
        video_id="appearance-video",
        track_id=track_id,
        class_label=class_label,
        embedding=_normalized(embedding),
        appearance_embedding_model=FakeAppearanceEncoder.model_name,
        appearance_embedding_version=FakeAppearanceEncoder.model_version,
        observation_count=4,
        attributes={
            key: VisualAttributeEvidence(value=value, confidence=confidence)
            for key, (value, confidence) in attributes.items()
        },
    )


def _segment(
    name: str,
    start: float,
    detections: list[Detection],
    *,
    actions: list[str] | None = None,
    relationship: Relationship | None = None,
) -> CanonicalSegment:
    return CanonicalSegment(
        segment_id=f"appearance-video:{name}",
        video_id="appearance-video",
        start_time=start,
        end_time=start + 5,
        frame_ids=[detections[0].frame_id],
        detections=detections,
        entities=sorted({item.class_label for item in detections}),
        actions=actions or [],
        relationships=[relationship] if relationship else [],
    )


def _appearance_repository(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "appearance.sqlite3")
    repository.create_video(_video())
    red_detection = _detection("red-person", "person", 0, track_id="red-track", appearance_id="red")
    black_detection = _detection(
        "black-person", "person", 5, track_id="black-track", appearance_id="black"
    )
    repository.replace_segments(
        "appearance-video",
        [
            _segment("red", 0, [red_detection], actions=["sit"]),
            _segment("black", 5, [black_detection], actions=["sit"]),
        ],
    )
    appearances = [
        _appearance(
            "red", "red-track", "person", [1, 0, 0, 0, 0], {"upper_clothing_color": ("red", 0.91)}
        ),
        _appearance(
            "black",
            "black-track",
            "person",
            [0, 0, 0, 1, 0],
            {"upper_clothing_color": ("black", 0.92)},
        ),
    ]
    repository.replace_entity_appearances("appearance-video", appearances)
    return repository, appearances


def test_crop_extraction_clips_box_and_rejects_small_crop():
    image = Image.new("RGB", (100, 80), "red")
    detection = _detection("crop", "person", 0, track_id="crop").model_copy(
        update={"pixel_bbox": BoundingBox(x1=-5, y1=5, x2=70, y2=75)}
    )
    crop = extract_detection_crop(image, detection)
    assert crop is not None and crop.size == (70, 70)
    assert extract_detection_crop(image, detection, minimum_size=90) is None


def test_track_aggregation_is_cached_and_stable(tmp_path):
    red = tmp_path / "red.jpg"
    orange = tmp_path / "orange.jpg"
    Image.new("RGB", (100, 100), "red").save(red)
    Image.new("RGB", (100, 100), (255, 40, 0)).save(orange)
    frames = [
        FrameRecord(
            frame_id="frame-0",
            video_id="appearance-video",
            frame_index=0,
            timestamp=0,
            path=str(red),
            width=100,
            height=100,
        ),
        FrameRecord(
            frame_id="frame-1",
            video_id="appearance-video",
            frame_index=1,
            timestamp=1,
            path=str(orange),
            width=100,
            height=100,
        ),
    ]
    detections = [
        _detection("p0", "person", 0, track_id="person-track").model_copy(
            update={"frame_id": "frame-0"}
        ),
        _detection("p1", "person", 1, track_id="person-track").model_copy(
            update={"frame_id": "frame-1"}
        ),
    ]
    encoder = FakeAppearanceEncoder()
    extractor = TrackAppearanceExtractor(encoder, attribute_min_confidence=0.2)
    records, attached = extractor.extract(
        video_id="appearance-video", frames=frames, detections=detections
    )
    assert len(records) == 1 and records[0].observation_count == 2
    assert records[0].attributes["upper_clothing_color"].value == "red"
    assert {item.appearance_id for item in attached} == {records[0].appearance_id}
    calls = encoder.image_calls
    cached, _ = extractor.extract(
        video_id="appearance-video", frames=frames, detections=detections, existing=records
    )
    assert cached == records
    assert encoder.image_calls == calls


def test_sqlite_persists_one_appearance_per_track(tmp_path):
    repository, appearances = _appearance_repository(tmp_path)
    loaded = SQLiteRepository(repository.path).list_entity_appearances(["appearance-video"])
    assert loaded == sorted(appearances, key=lambda item: item.track_id)
    assert len(loaded) == 2


def test_planner_extracts_attribute_action_and_relationship_constraints():
    parser = StructuredQueryParser()
    red = parser.parse("red shirt person sitting")
    glasses = parser.parse("person with glasses working on the laptop")
    backpack = parser.parse("black backpack next to the chair")
    assert red.actions == ["sit"]
    assert red.appearance_constraints[0].upper_clothing_color == "red"
    assert red.spoken_terms == []
    assert glasses.actions == ["work on computer"]
    assert glasses.appearance_constraints[0].glasses is True
    assert backpack.relationship_tuples[0].predicate == "near"
    assert backpack.appearance_constraints[0].color == "black"


def test_groq_can_resolve_only_ambiguous_open_appearance(settings_factory):
    class Transport:
        def post_json(self, url, *, headers, payload, timeout):
            del url, headers, timeout
            schema = payload["response_format"]["json_schema"]["schema"]
            assert "appearance" in schema["properties"]
            result = {
                "intent": "visual_search",
                "actions": [],
                "entities": ["person"],
                "relations": [],
                "spoken_concepts": [],
                "ocr_concepts": [],
                "speakers": [],
                "appearance": [
                    {
                        "entity_class": "person",
                        "color": None,
                        "upper_clothing_color": None,
                        "upper_clothing_description": "patterned clothing",
                        "glasses": None,
                        "hat": None,
                        "bag": None,
                        "description": "a person wearing patterned clothing",
                    }
                ],
                "confidence": 0.93,
            }
            return {"choices": [{"message": {"content": json.dumps(result)}}]}

    planner = QueryPlanningService(
        settings_factory(
            enable_query_planner=True, groq_api_key="mock", query_planner_model="mock"
        ),
        transport=Transport(),
    )
    outcome = planner.plan("person in patterned clothing")
    assert outcome.status == "groq_planned_validated"
    assert (
        outcome.plan.appearance_constraints[0].description == "a person wearing patterned clothing"
    )


def test_red_clothing_filters_black_same_action(tmp_path, settings_factory):
    repository, _ = _appearance_repository(tmp_path)
    settings = settings_factory(enable_appearance_features=True, attribute_min_confidence=0.45)
    engine = LocalHybridSearchEngine(
        repository,
        encoder=HashingTextEncoder(),
        appearance_settings=settings,
        appearance_encoder=FakeAppearanceEncoder(),
    )
    response = engine.search(SearchRequest(query="red shirt person sitting", limit=10))
    assert [item.segment_id for item in response.results] == ["appearance-video:red"]
    assert response.results[0].appearance_matches[0].matched_via[0] == "explicit_attribute"
    assert "upper_clothing_color=red" in response.results[0].match_reason


def test_validated_appearance_ranks_above_legacy_missing_cache(tmp_path, settings_factory):
    repository, _ = _appearance_repository(tmp_path)
    legacy = _detection("legacy-person", "person", 10, track_id="legacy-track")
    segments = repository.list_segments(["appearance-video"])
    repository.replace_segments(
        "appearance-video",
        [*segments, _segment("legacy", 10, [legacy], actions=["sit"])],
    )
    settings = settings_factory(enable_appearance_features=True)
    response = LocalHybridSearchEngine(
        repository,
        appearance_settings=settings,
        appearance_encoder=FakeAppearanceEncoder(),
    ).search(SearchRequest(query="red shirt person sitting"))
    assert response.results[0].segment_id == "appearance-video:red"
    assert response.results[-1].segment_id == "appearance-video:legacy"
    assert "fallback ranking" in response.results[-1].match_reason


def test_open_ended_appearance_similarity_selects_dark_jacket(tmp_path, settings_factory):
    repository, _ = _appearance_repository(tmp_path)
    settings = settings_factory(enable_appearance_features=True, appearance_min_similarity=0.2)
    engine = LocalHybridSearchEngine(
        repository,
        appearance_settings=settings,
        appearance_encoder=FakeAppearanceEncoder(),
    )
    response = engine.search(SearchRequest(query="person wearing a dark jacket"))
    assert response.results[0].segment_id == "appearance-video:black"
    assert "appearance_similarity" in response.results[0].appearance_matches[0].matched_via
    assert "appearance match" in response.results[0].match_reason


def test_black_object_color_combines_with_exact_relationship(tmp_path, settings_factory):
    repository = SQLiteRepository(tmp_path / "objects.sqlite3")
    repository.create_video(_video())
    segments = []
    appearances = []
    for index, color in enumerate(("black", "blue")):
        start = float(index * 5)
        backpack = _detection(
            f"{color}-bag",
            "backpack",
            start,
            track_id=f"{color}-bag",
            appearance_id=f"{color}-appearance",
        )
        chair = _detection(f"chair-{index}", "chair", start, track_id=f"chair-{index}")
        relation = Relationship(
            relationship_id=f"rel-{color}",
            source_id=backpack.detection_id,
            target_id=chair.detection_id,
            predicate="near",
            confidence=0.9,
            source_method=RelationshipSource.BOUNDING_BOX_GEOMETRY,
            timestamp=start,
        )
        segments.append(_segment(color, start, [backpack, chair], relationship=relation))
        vector = [0, 0, 0, 1, 0] if color == "black" else [0, 0, 1, 0, 0]
        appearances.append(
            _appearance(
                f"{color}-appearance", f"{color}-bag", "backpack", vector, {"color": (color, 0.9)}
            )
        )
    repository.replace_segments("appearance-video", segments)
    repository.replace_entity_appearances("appearance-video", appearances)
    settings = settings_factory(enable_appearance_features=True)
    response = LocalHybridSearchEngine(
        repository, appearance_settings=settings, appearance_encoder=FakeAppearanceEncoder()
    ).search(SearchRequest(query="black backpack beside chair"))
    assert [item.segment_id for item in response.results] == ["appearance-video:black"]
    assert response.results[0].matched_relationships[0].predicate == "near"


def test_weak_attribute_and_disabled_feature_preserve_existing_search(tmp_path, settings_factory):
    repository, appearances = _appearance_repository(tmp_path)
    weak = appearances[0].model_copy(
        update={
            "attributes": {
                "upper_clothing_color": VisualAttributeEvidence(value="red", confidence=0.1)
            }
        }
    )
    repository.replace_entity_appearances("appearance-video", [weak, appearances[1]])
    enabled = LocalHybridSearchEngine(
        repository,
        appearance_settings=settings_factory(
            enable_appearance_features=True, attribute_min_confidence=0.45
        ),
        appearance_encoder=FakeAppearanceEncoder(),
    ).search(SearchRequest(query="blue shirt person sitting"))
    disabled = LocalHybridSearchEngine(
        repository,
        appearance_settings=settings_factory(enable_appearance_features=False),
    ).search(SearchRequest(query="blue shirt person sitting"))
    assert enabled.results
    assert len(disabled.results) == 2
    assert all(not item.appearance_matches for item in disabled.results)


def test_neo4j_mirrors_searchable_attributes_not_embedding():
    calls = []
    indexer = CanonicalNeo4jIndexer.__new__(CanonicalNeo4jIndexer)
    indexer.run_query = lambda query, parameters=None: calls.append((query, parameters)) or []
    detection = _detection("red-person", "person", 0, track_id="red-track", appearance_id="red")
    appearance = _appearance(
        "red", "red-track", "person", [1, 0, 0, 0, 0], {"upper_clothing_color": ("red", 0.9)}
    )
    indexer.ingest(_video(), [_segment("red", 0, [detection])], [appearance])
    payload = next(
        parameters for _, parameters in calls if parameters and "appearances" in parameters
    )
    assert payload["appearances"][0]["upper_clothing_color"] == "red"
    assert "embedding" not in payload["appearances"][0]


def test_api_returns_appearance_evidence(tmp_path, settings_factory):
    repository, _ = _appearance_repository(tmp_path)
    settings = settings_factory(
        enable_appearance_features=True,
        enable_query_planner=False,
        enable_rag_generation=False,
    )
    app = create_app(settings, repository, NoopProcessor())
    app.state.search = LocalHybridSearchEngine(
        repository,
        appearance_settings=settings,
        appearance_encoder=FakeAppearanceEncoder(),
    )
    response = TestClient(app).post("/api/search", json={"query": "red shirt person sitting"})
    assert response.status_code == 200
    payload = response.json()
    assert (
        payload["results"][0]["appearance_matches"][0]["attributes"]["upper_clothing_color"][
            "value"
        ]
        == "red"
    )
