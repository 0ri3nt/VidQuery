from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProcessingState(StrEnum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    PREPROCESSING = "PREPROCESSING"
    VISUAL_PROCESSING = "VISUAL_PROCESSING"
    AUDIO_PROCESSING = "AUDIO_PROCESSING"
    ALIGNING = "ALIGNING"
    SCENE_GRAPH_GENERATION = "SCENE_GRAPH_GENERATION"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


class RelationshipSource(StrEnum):
    BOUNDING_BOX_GEOMETRY = "bounding_box_geometry"
    AVA_GROUND_TRUTH = "ava_ground_truth"
    HEURISTIC_ACTION_MAPPING = "heuristic_action_mapping"
    GNN_PREDICTION = "gnn_prediction"
    TRANSCRIPT_EXTRACTION = "transcript_extraction"
    SPEAKER_ALIGNMENT = "speaker_alignment"
    MANUAL_ANNOTATION = "manual_annotation"
    LEARNED_ACTION_MODEL = "learned_action_model"
    GNN_ACTION_MODEL = "gnn_action_model"
    GNN_ACTION_PLUS_TARGET_RESOLVER = "gnn_action_plus_target_resolver"
    RELATIONSHIP_GNN = "relationship_gnn"
    EASY_OCR = "easyocr"
    VISUAL_ATTRIBUTE_MODEL = "visual_attribute_model"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoRecord(StrictModel):
    video_id: str
    display_name: str
    original_filename: str
    stored_path: str
    content_sha256: str
    duration: float = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    fps: float = Field(ge=0)
    has_audio: bool
    upload_time: datetime
    updated_time: datetime
    state: ProcessingState = ProcessingState.UPLOADED
    current_stage: str | None = None
    failure_message: str | None = None
    failure_detail: str | None = Field(default=None, exclude=True)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("video_id", "display_name", "original_filename", "stored_path")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class BoundingBox(StrictModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def ordered(self) -> BoundingBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding-box coordinates must be ordered")
        return self

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class FrameRecord(StrictModel):
    frame_id: str
    video_id: str
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Detection(StrictModel):
    detection_id: str
    frame_id: str
    video_id: str
    timestamp: float = Field(ge=0)
    class_id: int
    class_label: str
    confidence: float = Field(ge=0, le=1)
    pixel_bbox: BoundingBox
    normalized_bbox: BoundingBox
    centroid: tuple[float, float]
    centroid_normalized: tuple[float, float]
    track_id: str | None = None
    appearance_id: str | None = None


class VisualAttributeEvidence(StrictModel):
    value: str | bool
    confidence: float = Field(ge=0, le=1)
    source_method: RelationshipSource = RelationshipSource.VISUAL_ATTRIBUTE_MODEL
    score_interpretation: Literal["relative_prompt_score_not_calibrated_probability"] = (
        "relative_prompt_score_not_calibrated_probability"
    )


class EntityAppearance(StrictModel):
    appearance_id: str
    video_id: str
    track_id: str
    class_label: str
    embedding: list[float] = Field(min_length=1)
    appearance_embedding_model: str
    appearance_embedding_version: str
    observation_count: int = Field(ge=1)
    attributes: dict[str, VisualAttributeEvidence] = Field(default_factory=dict)
    source_method: RelationshipSource = RelationshipSource.VISUAL_ATTRIBUTE_MODEL


class TranscriptSegment(StrictModel):
    segment_id: str
    video_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    speaker: str = "UNKNOWN"

    @model_validator(mode="after")
    def valid_interval(self) -> TranscriptSegment:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class SpeakerSegment(StrictModel):
    segment_id: str
    video_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    speaker: str

    @model_validator(mode="after")
    def valid_interval(self) -> SpeakerSegment:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self


class OCREvidence(StrictModel):
    ocr_id: str
    text: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    frame_ids: list[str] = Field(default_factory=list)
    source_method: RelationshipSource = RelationshipSource.EASY_OCR

    @model_validator(mode="after")
    def valid_interval(self) -> OCREvidence:
        if self.end_time < self.start_time:
            raise ValueError("OCR end_time must not precede start_time")
        return self


class Relationship(StrictModel):
    relationship_id: str
    source_id: str
    target_id: str
    predicate: str
    confidence: float = Field(ge=0, le=1)
    source_method: RelationshipSource
    timestamp: float = Field(ge=0)
    model_version: str | None = None
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    source_track_id: str | None = None
    target_track_id: str | None = None
    observation_count: int = Field(default=1, ge=1)
    temporal_smoothed: bool = False

    @model_validator(mode="after")
    def valid_interval(self) -> Relationship:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("relationship end_time must not precede start_time")
        return self


class RelationshipQuery(StrictModel):
    subject: str
    predicate: str
    object: str
    directionality: Literal["symmetric", "directed"]


class MatchedRelationship(StrictModel):
    relationship_id: str
    source_id: str
    source_class: str
    predicate: str
    target_id: str
    target_class: str
    directionality: Literal["symmetric", "directed"]
    confidence: float = Field(ge=0, le=1)
    source_method: RelationshipSource
    timestamp: float = Field(ge=0)
    model_version: str | None = None
    start_time: float | None = Field(default=None, ge=0)
    end_time: float | None = Field(default=None, ge=0)
    source_track_id: str | None = None
    target_track_id: str | None = None
    observation_count: int = Field(default=1, ge=1)
    temporal_smoothed: bool = False


class ActionPredictionEvidence(StrictModel):
    action: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    resolved_predicate: str | None = None
    provenance: RelationshipSource
    timestamp: float = Field(ge=0)


class AppearanceConstraint(StrictModel):
    entity_class: str
    color: str | None = None
    upper_clothing_color: str | None = None
    upper_clothing_description: str | None = None
    glasses: bool | None = None
    hat: bool | None = None
    bag: bool | None = None
    description: str | None = Field(default=None, max_length=200)


class AppearanceMatchEvidence(StrictModel):
    appearance_id: str
    track_id: str
    entity_class: str
    attributes: dict[str, VisualAttributeEvidence] = Field(default_factory=dict)
    description: str | None = None
    similarity: float | None = Field(default=None, ge=-1, le=1)
    matched_via: list[Literal["explicit_attribute", "appearance_similarity"]]
    appearance_embedding_model: str
    appearance_embedding_version: str
    observation_count: int = Field(ge=1)


class CanonicalSegment(StrictModel):
    segment_id: str
    video_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    frame_ids: list[str] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    transcript: str = ""
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    ocr_evidence: list[OCREvidence] = Field(default_factory=list)
    embedding: list[float] | None = None
    processing_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_interval(self) -> CanonicalSegment:
        if self.end_time <= self.start_time:
            raise ValueError("segment end_time must be greater than start_time")
        return self


class QueryPlan(StrictModel):
    intent: Literal[
        "transcript_search",
        "visual_search",
        "action_search",
        "relationship_search",
        "speaker_search",
        "ocr_search",
        "multimodal_search",
    ] = "multimodal_search"
    visual_entities: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    relationship_tuples: list[RelationshipQuery] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    spoken_terms: list[str] = Field(default_factory=list)
    ocr_terms: list[str] = Field(default_factory=list)
    speaker: str | None = None
    appearance_constraints: list[AppearanceConstraint] = Field(default_factory=list)


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=500)
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=10, ge=1, le=50)
    retrieval_backend: Literal["sqlite", "neo4j"] = "sqlite"

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchResult(StrictModel):
    video_id: str
    video_title: str
    segment_id: str
    start_time: float
    end_time: float
    score: float = Field(ge=0, le=1)
    transcript: str
    speakers: list[str]
    entities: list[str]
    actions: list[str]
    action_evidence: list[ActionPredictionEvidence] = Field(default_factory=list)
    ocr_evidence: list[OCREvidence] = Field(default_factory=list)
    relationships: list[Relationship]
    matched_relationships: list[MatchedRelationship] = Field(default_factory=list)
    appearance_matches: list[AppearanceMatchEvidence] = Field(default_factory=list)
    thumbnail_url: str
    stream_url: str
    match_reason: str


class AnswerCitation(StrictModel):
    evidence_id: str
    segment_id: str
    video_id: str
    video_title: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    stream_url: str

    @model_validator(mode="after")
    def valid_interval(self) -> AnswerCitation:
        if self.end_time < self.start_time:
            raise ValueError("citation end_time must not precede start_time")
        return self


class GroundedAnswer(StrictModel):
    answer: str
    supported: bool
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)


class SearchResponse(StrictModel):
    query: str
    parsed_query: QueryPlan
    results: list[SearchResult]
    retrieval_backend: Literal["sqlite", "neo4j"] = "sqlite"
    generated_answer: GroundedAnswer | None = None
    rag_status: str = "disabled"
    query_planner_status: str = "deterministic"
    query_ambiguities: list[str] = Field(default_factory=list)
    alternative_query_plans: list[QueryPlan] = Field(default_factory=list)


class UploadResponse(StrictModel):
    video_id: str
    state: ProcessingState
    duplicate: bool = False


class HealthResponse(StrictModel):
    application: Literal["ok"] = "ok"
    database: str
    neo4j: str
    models: dict[str, str]


def utc_now() -> datetime:
    return datetime.now(UTC)
