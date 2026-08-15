from __future__ import annotations

import argparse
from pathlib import Path

from vidquery.config import get_settings
from vidquery.demo_manifest import apply_demo_manifest
from vidquery.neo4j import CanonicalNeo4jIndexer
from vidquery.storage import SQLiteRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the checked-in manual ground truth to a processed demo video"
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("demo/final_demo_manifest.json"))
    parser.add_argument(
        "--neo4j", action="store_true", help="Mirror the annotated canonical segments to Neo4j"
    )
    args = parser.parse_args()

    settings = get_settings()
    repository = SQLiteRepository(settings.database_path)
    segments = apply_demo_manifest(
        repository,
        video_id=args.video_id,
        manifest_path=args.manifest,
    )
    if args.neo4j:
        indexer = CanonicalNeo4jIndexer.from_settings(settings)
        try:
            indexer.apply_schema(Path("database/schema.cypher"))
            indexer.ingest(repository.get_video(args.video_id), segments)
        finally:
            indexer.close()
    print(
        f"Applied controlled-demo manifest to {args.video_id}: "
        f"{len(segments)} canonical segments" + ("; mirrored to Neo4j" if args.neo4j else "")
    )


if __name__ == "__main__":
    main()
