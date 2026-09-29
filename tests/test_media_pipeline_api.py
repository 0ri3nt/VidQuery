from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vidquery.api import create_app
from vidquery.domain import (
    BoundingBox,
    CanonicalSegment,
    Detection,
    ProcessingState,
    Relationship,
    RelationshipSource,
    SpeakerSegment,
    TranscriptSegment,
)
from vidquery.geometry import centroid, normalize_bbox
from vidquery.media import (
    FrameExtractor,
    MediaValidationError,
    UploadTooLargeError,
    VideoIngestionService,
    probe_video,
    sanitize_filename,
)
from vidquery.models import (
    OptionalComponentUnavailable,
    assign_speakers,
    diarization_component_status,
    whisper_component_status,
    yolo_component_status,
)
from vidquery.pipeline import (
    VideoProcessingService,
    merge_preserved_evidence,
    nearest_cached_frame,
)
from vidquery.search import LocalHybridSearchEngine
from vidquery.storage import SQLiteRepository


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../Lecture One.MP4", "Lecture One.mp4"),
        ("weird<>name.mp4", "weird_name.mp4"),
        ("...mp4", "video.mp4"),
        ("folder\\meeting.mp4", "meeting.mp4"),
    ],
)
def test_filename_sanitization(raw, expected):
    assert sanitize_filename(raw) == expected


def test_probe_video_extracts_metadata(sample_video):
    metadata = probe_video(sample_video)
    assert metadata.width == 160
    assert metadata.height == 120
    assert metadata.fps == pytest.approx(10, rel=0.1)
    assert metadata.duration == pytest.approx(2, rel=0.2)
    assert metadata.has_audio is False


def test_nearest_cached_frame_handles_end_of_video_timestamp_drift(tmp_path):
    frame_5000 = tmp_path / "video_000000005000.jpg"
    frame_5000.write_bytes(b"frame")
    frames = {5000: frame_5000}

    assert nearest_cached_frame(
        frames, 5040, tolerance_millisecond=550
    ) == frame_5000
    assert nearest_cached_frame(frames, 6000, tolerance_millisecond=550) is None


def test_probe_rejects_unreadable_mp4(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(MediaValidationError):
        probe_video(path)


def test_yolo_status_checks_dependency_and_checkpoint(settings_factory, tmp_path, monkeypatch):
    disabled = settings_factory(enable_yolo=False)
    assert yolo_component_status(disabled) == "disabled"

    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"checkpoint")
    enabled = settings_factory(enable_yolo=True, yolo_model=str(checkpoint))
    monkeypatch.setattr("vidquery.models.importlib.util.find_spec", lambda _name: None)
    assert yolo_component_status(enabled) == "unavailable_missing_dependency"

    monkeypatch.setattr("vidquery.models.importlib.util.find_spec", lambda _name: object())
    assert yolo_component_status(enabled) == "available_configured"


def test_whisper_status_checks_dependency_and_checkpoint(
    settings_factory, tmp_path, monkeypatch
):
    disabled = settings_factory(enable_whisper=False)
    assert whisper_component_status(disabled) == "disabled"

    enabled = settings_factory(
        enable_whisper=True,
        whisper_model="test-whisper-status",
        whisper_cache_dir=tmp_path,
        allow_model_downloads=False,
    )
    monkeypatch.setattr("vidquery.models.importlib.util.find_spec", lambda _name: None)
    assert whisper_component_status(enabled) == "unavailable_missing_dependency"

    monkeypatch.setattr("vidquery.models.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr(
        "vidquery.models.importlib.import_module",
        lambda _name: SimpleNamespace(
            _MODELS={"test-whisper-status": "https://models/test-whisper-status.pt"}
        ),
    )
    assert whisper_component_status(enabled) == "unavailable_missing_checkpoint"

    (tmp_path / "test-whisper-status.pt").write_bytes(b"checkpoint")
    assert whisper_component_status(enabled) == "available_configured"


def test_ingestion_stream_deduplicates_content(settings_factory, sample_video):
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    service = VideoIngestionService(settings, repository)
    first, duplicate = service.ingest_stream(
        BytesIO(sample_video.read_bytes()), "meeting.mp4", "video/mp4"
    )
    second, second_duplicate = service.ingest_stream(
        BytesIO(sample_video.read_bytes()), "copy.mp4", "video/mp4"
    )
    assert duplicate is False
    assert second_duplicate is True
    assert first.video_id == second.video_id
    assert len(repository.list_videos()) == 1


def test_ingestion_enforces_upload_limit(settings_factory, sample_video):
    settings = settings_factory(max_upload_size=10)
    service = VideoIngestionService(settings, SQLiteRepository(settings.database_path))
    with pytest.raises(UploadTooLargeError):
        service.ingest_stream(BytesIO(sample_video.read_bytes()), "large.mp4", "video/mp4")


def test_ingestion_rejects_wrong_content_type(settings_factory, sample_video):
    settings = settings_factory()
    service = VideoIngestionService(settings, SQLiteRepository(settings.database_path))
    with pytest.raises(MediaValidationError):
        service.ingest_stream(BytesIO(sample_video.read_bytes()), "clip.mp4", "text/plain")


class FakeDetector:
    def detect(self, frames):
        output = []
        for frame in frames[:1]:
            for index, (label, raw_box) in enumerate(
                [("person", (10, 10, 70, 110)), ("laptop", (65, 40, 130, 100))]
            ):
                box = BoundingBox(x1=raw_box[0], y1=raw_box[1], x2=raw_box[2], y2=raw_box[3])
                normalized = normalize_bbox(box, frame.width, frame.height)
                output.append(
                    Detection(
                        detection_id=f"det-{index}",
                        frame_id=frame.frame_id,
                        video_id=frame.video_id,
                        timestamp=frame.timestamp,
                        class_id=index,
                        class_label=label,
                        confidence=0.9,
                        pixel_bbox=box,
                        normalized_bbox=normalized,
                        centroid=centroid(box),
                        centroid_normalized=centroid(normalized),
                    )
                )
        return output


class FakeAudioExtractor:
    def extract(self, video, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake wav boundary")
        return output_path


class FakeTranscriber:
    def transcribe(self, audio_path, video_id):
        return [
            TranscriptSegment(
                segment_id="tx",
                video_id=video_id,
                start_time=0.1,
                end_time=0.9,
                text="the project budget architecture",
                confidence=0.9,
            )
        ]


class FakeDiarizer:
    def diarize(self, audio_path, video_id):
        return [
            SpeakerSegment(
                segment_id="sp",
                video_id=video_id,
                start_time=0,
                end_time=1,
                speaker="SPEAKER_01",
            )
        ]


class MissingDiarizer:
    def diarize(self, audio_path, video_id):
        raise OptionalComponentUnavailable("pyannote checkpoint unavailable")


class MissingDetector:
    def detect(self, frames):
        raise OptionalComponentUnavailable("detector checkpoint missing")


def register_sample(settings, repository, sample_video):
    return VideoIngestionService(settings, repository).register_existing(sample_video)[0]


def test_transcript_speaker_assignment_uses_greatest_overlap():
    transcript = TranscriptSegment(
        segment_id="tx", video_id="v", start_time=1, end_time=3, text="hello"
    )
    speakers = [
        SpeakerSegment(segment_id="a", video_id="v", start_time=0, end_time=1.5, speaker="A"),
        SpeakerSegment(segment_id="b", video_id="v", start_time=1.5, end_time=3, speaker="B"),
    ]
    assert assign_speakers([transcript], speakers)[0].speaker == "B"


def test_refresh_merges_fresh_models_without_erasing_ava_evidence():
    fresh = CanonicalSegment(
        segment_id="fresh",
        video_id="v",
        start_time=0,
        end_time=5,
        entities=["person"],
        actions=["stand"],
        transcript="fresh whisper text",
        speakers=["UNKNOWN"],
        processing_metadata={
            "yolo_inference_used": True,
            "whisper_inference_used": True,
            "learned_action_predictions_used": True,
            "action_sources": {"stand": "learned_action_model"},
        },
    )
    annotated = CanonicalSegment(
        segment_id="annotated",
        video_id="v",
        start_time=0,
        end_time=5,
        entities=["chair"],
        actions=["sit"],
        transcript="trusted annotation text",
        speakers=["SPEAKER_01"],
        processing_metadata={
            "source": "legacy_ava_fused_adapter",
            "action_sources": {"sit": "ava_ground_truth"},
        },
    )

    result = merge_preserved_evidence([fresh], [annotated])[0]

    assert result.entities == ["chair", "person"]
    assert result.actions == ["sit", "stand"]
    assert result.transcript == "trusted annotation text"
    assert result.speakers == ["SPEAKER_01"]
    assert result.processing_metadata["fresh_whisper_transcript"] == (
        "fresh whisper text"
    )
    assert result.processing_metadata["yolo_inference_used"] is True
    assert result.processing_metadata["action_sources"] == {
        "stand": "learned_action_model",
        "sit": "ava_ground_truth",
    }
    assert result.processing_metadata["preserved_evidence"] is True


def test_refresh_preserves_manual_relation_but_discards_stale_learned_relation():
    def detection(detection_id: str, class_label: str, x1: float, x2: float):
        box = BoundingBox(x1=x1, y1=10, x2=x2, y2=80)
        normalized = normalize_bbox(box, 100, 100)
        return Detection(
            detection_id=detection_id,
            frame_id="frame-0",
            video_id="v",
            timestamp=0,
            class_id=0,
            class_label=class_label,
            confidence=1,
            pixel_bbox=box,
            normalized_bbox=normalized,
            centroid=centroid(box),
            centroid_normalized=centroid(normalized),
        )

    person = detection("manual:person", "person", 10, 30)
    chair = detection("manual:chair", "chair", 35, 55)
    manual = Relationship(
        relationship_id="manual:near",
        source_id=person.detection_id,
        target_id=chair.detection_id,
        predicate="near",
        confidence=1.0,
        source_method=RelationshipSource.MANUAL_ANNOTATION,
        timestamp=0,
    )
    stale = manual.model_copy(
        update={
            "relationship_id": "stale:learned",
            "source_method": RelationshipSource.RELATIONSHIP_GNN,
            "model_version": "old-model",
        }
    )
    annotated = CanonicalSegment(
        segment_id="annotated",
        video_id="v",
        start_time=0,
        end_time=5,
        detections=[person, chair],
        entities=["person", "chair"],
        relationships=[manual, stale],
        processing_metadata={"manual_annotation_overlay": True},
    )
    fresh = CanonicalSegment(
        segment_id="fresh",
        video_id="v",
        start_time=0,
        end_time=5,
        processing_metadata={"relationship_gnn_status": "available_validated"},
    )

    result = merge_preserved_evidence([fresh], [annotated])[0]

    assert [item.relationship_id for item in result.relationships] == ["manual:near"]
    assert {item.detection_id for item in result.detections} == {
        "manual:person",
        "manual:chair",
    }


def test_pipeline_builds_searchable_timestamped_segment(settings_factory, sample_video):
    settings = settings_factory(
        enable_diarization=True,
        require_diarization=True,
        enable_whisper=True,
        enable_yolo=True,
        near_threshold=1.0,
    )
    repository = SQLiteRepository(settings.database_path)
    record = register_sample(settings, repository, sample_video)
    processor = VideoProcessingService(
        settings,
        repository,
        frame_extractor=FrameExtractor(2),
        audio_extractor=FakeAudioExtractor(),
        visual_detector=FakeDetector(),
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
    )
    final = processor.process(record.video_id)
    assert final.state is ProcessingState.READY
    segment = repository.list_segments([record.video_id])[0]
    assert segment.entities == ["laptop", "person"]
    assert segment.speakers == ["SPEAKER_01"]
    assert segment.processing_metadata["visual_detector_status"] == "completed"
    assert segment.processing_metadata["visual_detection_count"] == 2
    assert segment.processing_metadata["yolo_inference_used"] is True
    assert segment.processing_metadata["transcription_status"] == "completed"
    assert segment.processing_metadata["transcript_segment_count"] == 1
    assert segment.processing_metadata["whisper_inference_used"] is True
    assert segment.processing_metadata["diarization_required"] is True
    assert segment.processing_metadata["diarization_status"] == "completed"
    assert segment.processing_metadata["speaker_segment_count"] == 1
    assert any(item.predicate == "near" for item in segment.relationships)
    response = LocalHybridSearchEngine(repository).search(
        __import__("vidquery.domain", fromlist=["SearchRequest"]).SearchRequest(
            query="project budget person near laptop", video_ids=[record.video_id]
        )
    )
    assert response.results[0].start_time == 0


def test_pipeline_uses_pair_visual_relationship_checkpoint_with_real_frame_path(
    settings_factory, sample_video, monkeypatch
):
    settings = settings_factory(
        enable_relation_gnn=True,
        relation_gnn_model_version="vidor-relation-pair-visual-gat-v2",
    )
    repository = SQLiteRepository(settings.database_path)
    video = register_sample(settings, repository, sample_video)
    frames = FrameExtractor(2).extract(
        video, settings.generated_dir / video.video_id / "frames"
    )
    detections = FakeDetector().detect(frames)
    calls = []

    class FakePairVisualPredictor:
        model_version = "vidor-relation-pair-visual-gat-v2"

        def __init__(self, *args, **kwargs):
            pass

        def predict_detections(
            self, frame_detections, *, frame_path, previous_detections=None
        ):
            calls.append((Path(frame_path), previous_detections))
            return [
                {
                    "source_id": frame_detections[0].detection_id,
                    "target_id": frame_detections[1].detection_id,
                    "predicate": "next_to",
                    "confidence": 0.8,
                }
            ]

    monkeypatch.setattr(
        "vidquery.vidor_improved.configured_relation_checkpoint_status",
        lambda *args: "available_validated",
    )
    monkeypatch.setattr(
        "vidquery.vidor_improved.ImprovedVidORRelationPredictor",
        FakePairVisualPredictor,
    )
    processor = VideoProcessingService(settings, repository)

    relationships, status = processor._learned_relationships(
        video, frames, detections, []
    )

    assert status == "available_validated"
    assert len(calls) == 1
    assert calls[0][0].is_file()
    assert [item.predicate for item in relationships] == ["near"]
    assert relationships[0].source_method is RelationshipSource.RELATIONSHIP_GNN
    assert relationships[0].model_version == "vidor-relation-pair-visual-gat-v2"


def test_required_diarization_fails_audio_processing(settings_factory, sample_video):
    settings = settings_factory(
        enable_diarization=True,
        require_diarization=True,
        enable_whisper=True,
    )
    repository = SQLiteRepository(settings.database_path)
    record = register_sample(settings, repository, sample_video)
    processor = VideoProcessingService(
        settings,
        repository,
        frame_extractor=FrameExtractor(2),
        audio_extractor=FakeAudioExtractor(),
        transcriber=FakeTranscriber(),
        diarizer=MissingDiarizer(),
    )

    with pytest.raises(RuntimeError, match="Required Pyannote diarization"):
        processor.process(record.video_id)

    failed = repository.get_video(record.video_id)
    assert failed.state is ProcessingState.FAILED
    assert failed.current_stage == "audio processing"


def test_diarization_health_requires_token_and_dependency(settings_factory, monkeypatch):
    missing_token = settings_factory(
        enable_diarization=True,
        require_diarization=True,
        huggingface_token="",
    )
    assert diarization_component_status(missing_token) == "unavailable_missing_token"

    configured = settings_factory(
        enable_diarization=True,
        require_diarization=True,
        huggingface_token="configured-test-token",
    )
    monkeypatch.setattr("vidquery.models.importlib.util.find_spec", lambda name: object())
    assert diarization_component_status(configured) == "available_configured_required"


def test_pipeline_degrades_when_detector_is_missing(settings_factory, sample_video):
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    record = register_sample(settings, repository, sample_video)
    processor = VideoProcessingService(
        settings,
        repository,
        visual_detector=MissingDetector(),
    )
    final = processor.process(record.video_id)
    assert final.state is ProcessingState.READY
    assert "detector checkpoint missing" in final.warnings


class StubProcessor:
    def __init__(self, repository):
        self.repository = repository

    def process(self, video_id):
        return self.repository.update_state(
            video_id, ProcessingState.READY, current_stage="test complete"
        )


def test_api_health_upload_list_status_and_range(settings_factory, sample_video):
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    app = create_app(settings, repository, StubProcessor(repository))
    client = TestClient(app)
    frontend = client.get("/")
    assert frontend.status_code == 200
    assert "SEARCH THE MOMENT" in frontend.text
    assert 'id="video-filter"' in frontend.text
    javascript = client.get("/assets/app.js")
    assert javascript.status_code == 200
    assert "video_ids: selectedVideo ? [selectedVideo] : []" in javascript.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["models"]["diarization"] == "disabled"
    assert health.json()["models"]["gnn_relationship_prediction"] == "disabled"

    upload = client.post(
        "/api/videos", files={"file": ("sample.mp4", sample_video.read_bytes(), "video/mp4")}
    )
    assert upload.status_code == 202
    video_id = upload.json()["video_id"]
    assert client.get("/api/videos").json()[0]["video_id"] == video_id
    assert client.get(f"/api/videos/{video_id}/status").json()["state"] == "READY"

    ranged = client.get(f"/api/videos/{video_id}/stream", headers={"Range": "bytes=0-15"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 16
    assert ranged.headers["content-range"].startswith("bytes 0-15/")


def test_api_rejects_bad_extension_and_empty_query(settings_factory):
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    client = TestClient(create_app(settings, repository, StubProcessor(repository)))
    upload = client.post(
        "/api/videos", files={"file": ("notes.txt", b"not video", "text/plain")}
    )
    assert upload.status_code == 415
    assert client.post("/api/search", json={"query": "   "}).status_code == 422
    graph_search = client.post(
        "/api/search",
        json={"query": "person near chair", "retrieval_backend": "neo4j"},
    )
    assert graph_search.status_code == 503
    assert graph_search.json()["detail"] == "Neo4j retrieval is disabled"


def test_api_returns_404_for_unknown_video(settings_factory):
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    client = TestClient(create_app(settings, repository, StubProcessor(repository)))
    assert client.get("/api/videos/missing").status_code == 404
    assert client.get("/api/videos/missing/stream").status_code == 404
