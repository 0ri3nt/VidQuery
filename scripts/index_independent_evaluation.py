from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidquery.config import get_settings
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror the independent evaluation videos into Neo4j"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/queries_independent.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = set(manifest["videos"])
    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    videos = [
        video
        for video in repository.list_videos()
        if Path(video.original_filename).stem in sources
    ]
    found = {Path(video.original_filename).stem for video in videos}
    if missing := sorted(sources - found):
        raise ValueError(f"evaluation videos are absent from SQLite: {missing}")

    indexer = CanonicalNeo4jIndexer.from_settings(settings)
    rows = []
    try:
        indexer.apply_schema(Path("database/schema.cypher"))
        for video in sorted(videos, key=lambda item: item.original_filename):
            segments = repository.list_segments([video.video_id])
            indexer.ingest(video, segments)
            counts = indexer.run_query(
                """
                MATCH (:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
                OPTIONAL MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
                OPTIONAL MATCH (s)-[:HAS_OCR]->(ocr:OCRText)
                RETURN count(DISTINCT s) AS segments,
                       count(DISTINCT e) AS relationships,
                       count(DISTINCT ocr) AS ocr_nodes
                """,
                {"video_id": video.video_id},
            )[0]
            rows.append(
                {
                    "source_video_id": Path(video.original_filename).stem,
                    "canonical_video_id": video.video_id,
                    **counts,
                }
            )
    finally:
        indexer.close()
    print(json.dumps({"indexed": rows}, indent=2))


if __name__ == "__main__":
    main()
