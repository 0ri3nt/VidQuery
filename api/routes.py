from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from core.config import get_settings
from database.ingestion import GraphIngestor
from database.neo4j_client import Neo4jClient
from extraction.preprocessor import VideoPreprocessor
from extraction.visual import VisualExtractor
from modelling.scene_graph_builder import SceneGraphBuilder
from retrieval.query_engine import SemanticSearchEngine


settings = get_settings()
app = FastAPI(title="VidQuery API", version="0.1.0")


class UploadRequest(BaseModel):
    video_dir: str = "data/ava/videos"
    train_csv: str = "data/ava/annotations/ava_train_v2.2.csv"
    val_csv: str = "data/ava/annotations/ava_val_v2.2.csv"
    max_videos: int | None = 20


class SearchResponse(BaseModel):
    query: str
    cypher: str
    results: list[dict]


def _build_neo4j_client() -> Neo4jClient:
    if not settings.neo4j_password:
        raise HTTPException(status_code=500, detail="NEO4J_PASSWORD is not set")
    return Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)


def run_ingestion_pipeline(req: UploadRequest) -> None:
    preprocessor = VideoPreprocessor(
        video_dir=Path(req.video_dir),
        train_csv=Path(req.train_csv),
        val_csv=Path(req.val_csv),
        max_videos=req.max_videos,
    )
    preprocessor.run()

    visual = VisualExtractor()
    visual.run_yolov8(
        frames_dir=Path("data/ava/extracted_frames"),
        detections_dir=Path("data/ava/detections"),
        train_csv=Path(req.train_csv),
        val_csv=Path(req.val_csv),
    )

    scene_builder = SceneGraphBuilder()
    scene_builder.run()

    if settings.neo4j_password:
        client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        ingestor = GraphIngestor(client)
        schema_path = Path("database/schema.cypher")
        if schema_path.exists():
            ingestor.apply_schema(schema_path)

        for split in ["train", "val"]:
            split_dir = Path("data/ava/scene_graphs") / split
            if not split_dir.exists():
                continue
            for video_dir in split_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                for graph_file in video_dir.glob("*.json"):
                    import json

                    graph_data = json.loads(graph_file.read_text(encoding="utf-8"))
                    ingestor.ingest_scene_graph(graph_data)
        client.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
def upload(req: UploadRequest, background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(run_ingestion_pipeline, req)
    return {"status": "accepted", "message": "Ingestion pipeline started in background."}


@app.get("/search", response_model=SearchResponse)
def search(q: str):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter q is required")

    client = _build_neo4j_client()
    try:
        engine = SemanticSearchEngine(client)
        return engine.search(q)
    finally:
        client.close()
