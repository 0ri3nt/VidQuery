from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import frontend

from .config import Settings, get_settings
from .domain import (
    HealthResponse,
    ProcessingState,
    SearchRequest,
    SearchResponse,
    UploadResponse,
    VideoRecord,
)
from .media import MediaValidationError, UploadTooLargeError, VideoIngestionService
from .pipeline import VideoProcessingService
from .query_planner import QueryPlanningService, query_planner_component_status
from .rag import GroundedAnswerService, rag_component_status
from .search import (
    LocalHybridSearchEngine,
    Neo4jGraphSearchEngine,
    configured_text_encoder,
    sentence_embedding_status,
)
from .storage import SQLiteRepository, VideoNotFoundError

RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")
STATIC_DIR = Path(frontend.__file__).resolve().parent


def create_app(
    settings: Settings | None = None,
    repository: SQLiteRepository | None = None,
    processor: VideoProcessingService | None = None,
) -> FastAPI:
    active_settings = settings or get_settings()
    active_settings.ensure_directories()
    active_repository = repository or SQLiteRepository(active_settings.database_path)
    active_processor = processor or VideoProcessingService(active_settings, active_repository)

    application = FastAPI(
        title="VidQuery API",
        version="0.2.0",
        description="Timestamped multimodal video search with explicit evidence provenance.",
    )
    application.state.settings = active_settings
    application.state.repository = active_repository
    application.state.processor = active_processor
    application.state.ingestion = VideoIngestionService(active_settings, active_repository)
    application.state.search = LocalHybridSearchEngine(
        active_repository,
        encoder=configured_text_encoder(active_settings),
        semantic_min_similarity=active_settings.semantic_min_similarity,
        appearance_settings=active_settings,
    )
    application.state.rag = GroundedAnswerService(active_settings)
    application.state.query_planner = QueryPlanningService(active_settings)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Range"],
    )

    if (STATIC_DIR / "assets").is_dir():
        application.mount(
            "/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets"
        )

    @application.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Frontend is not installed")
        return FileResponse(index)

    @application.get("/api/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        app_settings: Settings = request.app.state.settings
        repo: SQLiteRepository = request.app.state.repository
        neo4j_status = "disabled"
        if app_settings.enable_neo4j:
            try:
                from .neo4j import CanonicalNeo4jIndexer

                indexer = CanonicalNeo4jIndexer.from_settings(app_settings)
                try:
                    neo4j_status = "available" if indexer.health_check() else "unavailable"
                finally:
                    indexer.close()
            except Exception:
                neo4j_status = "unavailable"
        from .action_model import action_model_status
        from .appearance import appearance_component_status
        from .ava_gnn80 import ava80_gnn_status
        from .models import (
            diarization_component_status,
            whisper_component_status,
            yolo_component_status,
        )
        from .ocr import easyocr_status
        from .vidor_improved import configured_relation_checkpoint_status

        learned_action_status = action_model_status(
            app_settings.action_model_checkpoint,
            app_settings.action_model_metadata,
        )
        if learned_action_status == "available_validated":
            if app_settings.enable_gnn_actions:
                learned_action_status += "_fallback"
            else:
                learned_action_status += (
                    "_enabled" if app_settings.enable_learned_actions else "_inactive"
                )
        gnn_action_status = ava80_gnn_status(
            app_settings.gnn_action_checkpoint,
            app_settings.gnn_action_metadata,
            app_settings.gnn_model_version,
        )
        if gnn_action_status == "available_validated":
            gnn_action_status += (
                "_enabled" if app_settings.enable_gnn_actions else "_inactive"
            )
        if app_settings.enable_relation_gnn:
            relationship_status = configured_relation_checkpoint_status(
                app_settings.relation_gnn_checkpoint,
                app_settings.relation_gnn_metadata,
                app_settings.relation_gnn_model_version,
            )
            if relationship_status == "available_validated":
                relationship_status += "_enabled"
        else:
            relationship_status = "disabled"
        return HealthResponse(
            database="available" if repo.health_check() else "unavailable",
            neo4j=neo4j_status,
            models={
                "yolo": yolo_component_status(app_settings),
                "whisper": whisper_component_status(app_settings),
                "diarization": diarization_component_status(app_settings),
                "gnn_person_action_classifier": gnn_action_status,
                "gnn_relationship_prediction": relationship_status,
                "person_action_classifier": learned_action_status,
                "sentence_embedding_retrieval": (
                    sentence_embedding_status(app_settings.sentence_embedding_model)
                    + "_enabled"
                    if app_settings.semantic_retrieval_mode == "sentence_transformer"
                    else "hashing_baseline_enabled"
                ),
                "ocr": easyocr_status(app_settings),
                "rag_generation": rag_component_status(app_settings),
                "query_planner": query_planner_component_status(app_settings),
                "entity_appearance": appearance_component_status(app_settings),
            },
        )

    @application.get("/health", response_model=HealthResponse, include_in_schema=False)
    def legacy_health(request: Request) -> HealthResponse:
        return health(request)

    @application.post(
        "/api/videos",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_video(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency declaration
    ) -> UploadResponse:
        ingestion: VideoIngestionService = request.app.state.ingestion
        try:
            record, duplicate = ingestion.ingest_stream(
                file.file,
                file.filename or "video.mp4",
                file.content_type,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except MediaValidationError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        finally:
            await file.close()

        if not duplicate:
            background_tasks.add_task(request.app.state.processor.process, record.video_id)
        return UploadResponse(video_id=record.video_id, state=record.state, duplicate=duplicate)

    @application.get("/api/videos", response_model=list[VideoRecord])
    def list_videos(request: Request) -> list[VideoRecord]:
        return request.app.state.repository.list_videos()

    @application.get("/api/videos/{video_id}", response_model=VideoRecord)
    def get_video(video_id: str, request: Request) -> VideoRecord:
        return _get_video_or_404(request.app.state.repository, video_id)

    @application.get("/api/videos/{video_id}/status", response_model=VideoRecord)
    def get_video_status(video_id: str, request: Request) -> VideoRecord:
        return _get_video_or_404(request.app.state.repository, video_id)

    @application.post(
        "/api/videos/{video_id}/process",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def reprocess_video(
        video_id: str, request: Request, background_tasks: BackgroundTasks
    ) -> UploadResponse:
        record = _get_video_or_404(request.app.state.repository, video_id)
        if record.state not in {
            ProcessingState.UPLOADED,
            ProcessingState.READY,
            ProcessingState.FAILED,
        }:
            raise HTTPException(status_code=409, detail="Video is already processing")
        background_tasks.add_task(request.app.state.processor.process, video_id)
        return UploadResponse(video_id=video_id, state=record.state)

    @application.post("/api/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        planning = request.app.state.query_planner.plan(payload.query)

        def retrieve(plan) -> SearchResponse:
            response: SearchResponse
            if payload.retrieval_backend == "neo4j":
                app_settings: Settings = request.app.state.settings
                if not app_settings.enable_neo4j:
                    raise HTTPException(status_code=503, detail="Neo4j retrieval is disabled")
                try:
                    from .neo4j import CanonicalNeo4jIndexer

                    indexer = CanonicalNeo4jIndexer.from_settings(app_settings)
                    try:
                        response = Neo4jGraphSearchEngine(
                            request.app.state.repository,
                            indexer,
                            appearance_settings=app_settings,
                        ).search(payload, plan_override=plan)
                    finally:
                        indexer.close()
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(
                        status_code=503, detail="Neo4j retrieval is unavailable"
                    ) from exc
            else:
                try:
                    response = request.app.state.search.search(
                        payload, plan_override=plan
                    )
                except TypeError as exc:
                    # Preserve compatibility with injected test/search adapters.
                    if "plan_override" not in str(exc):
                        raise
                    response = request.app.state.search.search(payload)
            return response

        response = retrieve(planning.plan)
        if planning.alternatives:
            combined = {item.segment_id: item for item in response.results}
            for alternative in planning.alternatives:
                for item in retrieve(alternative).results:
                    previous = combined.get(item.segment_id)
                    if previous is None or item.score > previous.score:
                        combined[item.segment_id] = item
            response = response.model_copy(
                update={
                    "results": sorted(
                        combined.values(),
                        key=lambda item: (-item.score, item.start_time, item.video_id),
                    )[: payload.limit]
                }
            )
        response = response.model_copy(
            update={
                "parsed_query": planning.plan,
                "query_planner_status": planning.status,
                "query_ambiguities": list(planning.ambiguities),
                "alternative_query_plans": list(planning.alternatives),
            }
        )

        try:
            outcome = request.app.state.rag.generate(payload.query, response)
        except Exception:
            # Answer synthesis is optional and must never take ranked retrieval down.
            return response.model_copy(update={"rag_status": "provider_unavailable"})
        return response.model_copy(
            update={
                "generated_answer": outcome.answer,
                "rag_status": outcome.status,
            }
        )

    @application.get("/api/videos/{video_id}/stream")
    def stream_video(
        video_id: str,
        request: Request,
        range_header: str | None = Header(default=None, alias="Range"),
    ):
        record = _get_video_or_404(request.app.state.repository, video_id)
        path = Path(record.stored_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Stored video file is missing")
        file_size = path.stat().st_size
        headers = {"Accept-Ranges": "bytes"}
        if range_header is None:
            headers["Content-Length"] = str(file_size)
            return StreamingResponse(
                _file_chunks(path, 0, file_size - 1),
                media_type="video/mp4",
                headers=headers,
            )

        match = RANGE_PATTERN.fullmatch(range_header.strip())
        if match is None:
            raise HTTPException(status_code=416, detail="Invalid byte range")
        start_text, end_text = match.groups()
        if not start_text and not end_text:
            raise HTTPException(status_code=416, detail="Invalid byte range")
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise HTTPException(status_code=416, detail="Invalid suffix range")
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        if start >= file_size or end < start:
            raise HTTPException(
                status_code=416,
                detail="Requested range is outside the file",
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        end = min(end, file_size - 1)
        length = end - start + 1
        headers.update(
            {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
            }
        )
        return StreamingResponse(
            _file_chunks(path, start, end),
            status_code=206,
            media_type="video/mp4",
            headers=headers,
        )

    @application.get("/api/videos/{video_id}/thumbnail")
    def thumbnail(
        video_id: str,
        request: Request,
        timestamp_seconds: float = Query(default=0.0, ge=0, alias="timestamp"),
    ) -> FileResponse:
        _get_video_or_404(request.app.state.repository, video_id)
        segments = request.app.state.repository.list_segments([video_id])
        candidates: list[tuple[float, Path]] = []
        for segment in segments:
            raw_path = segment.processing_metadata.get("thumbnail_path")
            if raw_path:
                path = Path(str(raw_path))
                if path.is_file():
                    candidates.append((abs(segment.start_time - timestamp_seconds), path))
        if not candidates:
            raise HTTPException(status_code=404, detail="No generated thumbnail is available")
        path = min(candidates, key=lambda item: item[0])[1]
        return FileResponse(path, media_type="image/jpeg")

    return application


def _get_video_or_404(repository: SQLiteRepository, video_id: str) -> VideoRecord:
    try:
        return repository.get_video(video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Video was not found") from exc


def _file_chunks(
    path: Path, start: int, end: int, chunk_size: int = 1024 * 1024
) -> Iterator[bytes]:
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = stream.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


app = create_app()
