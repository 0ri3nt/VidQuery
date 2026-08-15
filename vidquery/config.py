from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _path_env_alias(primary: str, legacy: str, default: Path) -> Path:
    raw = os.getenv(primary) or os.getenv(legacy)
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _path_env("DATA_DIR", Path("data/app")))
    upload_dir: Path = field(
        default_factory=lambda: _path_env("UPLOAD_DIR", Path("data/app/uploads"))
    )
    generated_dir: Path = field(
        default_factory=lambda: _path_env("GENERATED_DIR", Path("data/app/generated"))
    )
    database_path: Path = field(
        default_factory=lambda: _path_env("DATABASE_PATH", Path("data/app/vidquery.sqlite3"))
    )

    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
    )
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    enable_neo4j: bool = field(default_factory=lambda: _bool_env("ENABLE_NEO4J", False))

    huggingface_token: str = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN", ""))
    )
    yolo_model: str = field(default_factory=lambda: os.getenv("YOLO_MODEL", "yolov8n.pt"))
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "tiny"))
    whisper_cache_dir: Path = field(
        default_factory=lambda: _path_env("WHISPER_CACHE_DIR", Path("data/app/models/whisper"))
    )
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    enable_yolo: bool = field(default_factory=lambda: _bool_env("ENABLE_YOLO", True))
    enable_whisper: bool = field(default_factory=lambda: _bool_env("ENABLE_WHISPER", True))
    enable_diarization: bool = field(
        default_factory=lambda: _bool_env("ENABLE_DIARIZATION", True)
    )
    require_diarization: bool = field(
        default_factory=lambda: _bool_env("REQUIRE_DIARIZATION", True)
    )
    pyannote_model: str = field(
        default_factory=lambda: os.getenv(
            "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"
        )
    )
    diarization_device: str = field(
        default_factory=lambda: os.getenv("DIARIZATION_DEVICE", "auto")
    )
    pyannote_cache_dir: Path = field(
        default_factory=lambda: _path_env(
            "PYANNOTE_CACHE_DIR", Path("data/app/models/pyannote")
        )
    )
    allow_model_downloads: bool = field(
        default_factory=lambda: _bool_env("ALLOW_MODEL_DOWNLOADS", False)
    )
    enable_learned_actions: bool = field(
        default_factory=lambda: _bool_env("ENABLE_LEARNED_ACTIONS", False)
    )
    enable_gnn_actions: bool = field(
        default_factory=lambda: _bool_env("ENABLE_GNN_ACTIONS", False)
    )
    action_model_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "ACTION_MODEL_CHECKPOINT",
            Path("data/app/models/action_classifier/action_model.pt"),
        )
    )
    action_model_metadata: Path = field(
        default_factory=lambda: _path_env(
            "ACTION_MODEL_METADATA",
            Path("data/app/models/action_classifier/action_model.metadata.json"),
        )
    )
    gnn_action_checkpoint: Path = field(
        default_factory=lambda: _path_env_alias(
            "GNN_CHECKPOINT",
            "GNN_ACTION_CHECKPOINT",
            Path("data/app/models/ava80/ava80_gat.pt"),
        )
    )
    gnn_action_metadata: Path = field(
        default_factory=lambda: _path_env_alias(
            "GNN_METADATA",
            "GNN_ACTION_METADATA",
            Path("data/app/models/ava80/ava80_gat.metadata.json"),
        )
    )
    gnn_device: str = field(default_factory=lambda: os.getenv("GNN_DEVICE", "auto"))
    gnn_model_version: str = field(
        default_factory=lambda: os.getenv("GNN_MODEL_VERSION", "ava80-gat-v1")
    )
    gnn_feature_cache_dir: Path = field(
        default_factory=lambda: _path_env(
            "GNN_FEATURE_CACHE_DIR", Path("data/app/models/ava80/feature_cache")
        )
    )
    enable_relation_gnn: bool = field(
        default_factory=lambda: _bool_env("ENABLE_RELATION_GNN", False)
    )
    relation_gnn_checkpoint: Path = field(
        default_factory=lambda: _path_env(
            "RELATION_GNN_CHECKPOINT",
            Path(
                "data/app/models/vidor_relation_pair_visual/"
                "vidor-relation-pair-visual-gat-v2.pt"
            ),
        )
    )
    relation_gnn_metadata: Path = field(
        default_factory=lambda: _path_env(
            "RELATION_GNN_METADATA",
            Path(
                "data/app/models/vidor_relation_pair_visual/"
                "vidor-relation-pair-visual-gat-v2.metadata.json"
            ),
        )
    )
    relation_gnn_device: str = field(
        default_factory=lambda: os.getenv("RELATION_GNN_DEVICE", "auto")
    )
    relation_gnn_model_version: str = field(
        default_factory=lambda: os.getenv(
            "RELATION_GNN_MODEL_VERSION", "vidor-relation-pair-visual-gat-v2"
        )
    )
    semantic_retrieval_mode: str = field(
        default_factory=lambda: os.getenv(
            "SEMANTIC_RETRIEVAL_MODE", "sentence_transformer"
        )
    )
    sentence_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "SENTENCE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    sentence_embedding_device: str = field(
        default_factory=lambda: os.getenv("SENTENCE_EMBEDDING_DEVICE", "auto")
    )
    semantic_min_similarity: float = field(
        default_factory=lambda: float(os.getenv("SEMANTIC_MIN_SIMILARITY", "0.20"))
    )
    enable_rag_generation: bool = field(
        default_factory=lambda: _bool_env("ENABLE_RAG_GENERATION", True)
    )
    rag_provider: str = field(default_factory=lambda: os.getenv("RAG_PROVIDER", "groq"))
    rag_model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        )
    )
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    rag_min_evidence_score: float = field(
        default_factory=lambda: float(os.getenv("RAG_MIN_EVIDENCE_SCORE", "0.20"))
    )
    rag_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("RAG_TIMEOUT_SECONDS", "15"))
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    enable_query_planner: bool = field(
        default_factory=lambda: _bool_env("ENABLE_QUERY_PLANNER", True)
    )
    query_planner_model: str = field(
        default_factory=lambda: os.getenv(
            "QUERY_PLANNER_MODEL",
            os.getenv("RAG_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")),
        )
    )
    query_planner_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("QUERY_PLANNER_MIN_CONFIDENCE", "0.70"))
    )
    query_planner_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("QUERY_PLANNER_TIMEOUT_SECONDS", "10"))
    )
    sentence_embedding_cache_dir: Path = field(
        default_factory=lambda: _path_env(
            "SENTENCE_EMBEDDING_CACHE_DIR", Path("data/app/models/sentence_transformers")
        )
    )
    enable_ocr: bool = field(default_factory=lambda: _bool_env("ENABLE_OCR", True))
    ocr_device: str = field(default_factory=lambda: os.getenv("OCR_DEVICE", "auto"))
    ocr_model_dir: Path = field(
        default_factory=lambda: _path_env("OCR_MODEL_DIR", Path("data/app/models/easyocr"))
    )
    ocr_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("OCR_MIN_CONFIDENCE", "0.25"))
    )
    ocr_dedup_similarity: float = field(
        default_factory=lambda: float(os.getenv("OCR_DEDUP_SIMILARITY", "0.90"))
    )
    enable_appearance_features: bool = field(
        default_factory=lambda: _bool_env("ENABLE_APPEARANCE_FEATURES", False)
    )
    appearance_model: str = field(
        default_factory=lambda: os.getenv("APPEARANCE_MODEL", "ViT-B-32:openai")
    )
    appearance_device: str = field(
        default_factory=lambda: os.getenv("APPEARANCE_DEVICE", "auto")
    )
    appearance_min_similarity: float = field(
        default_factory=lambda: float(os.getenv("APPEARANCE_MIN_SIMILARITY", "0.20"))
    )
    attribute_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("ATTRIBUTE_MIN_CONFIDENCE", "0.45"))
    )
    appearance_cache_dir: Path = field(
        default_factory=lambda: _path_env(
            "APPEARANCE_CACHE_DIR", Path("data/app/models/appearance")
        )
    )
    appearance_max_observations: int = field(
        default_factory=lambda: int(os.getenv("APPEARANCE_MAX_OBSERVATIONS", "6"))
    )

    frame_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("FRAME_SAMPLE_RATE", "1.0"))
    )
    segment_duration: float = field(
        default_factory=lambda: float(os.getenv("SEGMENT_DURATION", "5.0"))
    )
    detection_confidence: float = field(
        default_factory=lambda: float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    )
    detection_iou: float = field(
        default_factory=lambda: float(os.getenv("DETECTION_IOU", "0.45"))
    )
    near_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEAR_THRESHOLD", "0.22"))
    )
    enable_temporal_tracking: bool = field(
        default_factory=lambda: _bool_env("ENABLE_TEMPORAL_TRACKING", True)
    )
    track_iou_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRACK_IOU_THRESHOLD", "0.30"))
    )
    track_centroid_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRACK_CENTROID_THRESHOLD", "0.15"))
    )
    track_max_gap_seconds: float = field(
        default_factory=lambda: float(os.getenv("TRACK_MAX_GAP_SECONDS", "1.50"))
    )
    relationship_smoothing_gap_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("RELATIONSHIP_SMOOTHING_GAP_SECONDS", "1.50")
        )
    )
    max_upload_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE", str(2 * 1024 * 1024 * 1024)))
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _list_env(
            "CORS_ORIGINS", ("http://localhost:8000", "http://127.0.0.1:8000")
        )
    )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.generated_dir,
            self.database_path.parent,
            self.whisper_cache_dir,
            self.pyannote_cache_dir,
            self.action_model_checkpoint.parent,
            self.action_model_metadata.parent,
            self.gnn_action_checkpoint.parent,
            self.gnn_action_metadata.parent,
            self.gnn_feature_cache_dir,
            self.relation_gnn_checkpoint.parent,
            self.relation_gnn_metadata.parent,
            self.sentence_embedding_cache_dir,
            self.ocr_model_dir,
            self.appearance_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    return Settings()
