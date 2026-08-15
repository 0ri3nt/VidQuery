from __future__ import annotations

import argparse
import json
from pathlib import Path

from .action_model import (
    PersonActionPredictor,
    action_model_status,
    build_learned_action_index,
    train_person_action_model,
)
from .ava_adapter import import_fused_ava
from .ava_expansion import expand_ava_dataset
from .ava_gnn80 import (
    AVA80GNNPredictor,
    ava80_gnn_status,
    build_ava80_action_index,
    train_ava80_suite,
)
from .config import get_settings
from .domain import SearchRequest
from .evaluation import evaluate, write_evaluation
from .media import VideoIngestionService
from .neo4j import CanonicalNeo4jIndexer
from .pipeline import VideoProcessingService
from .search import LocalHybridSearchEngine
from .storage import SQLiteRepository


def _services():
    settings = get_settings()
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    return settings, repository


def process_video(path: Path, *, copy: bool) -> str:
    settings, repository = _services()
    ingestion = VideoIngestionService(settings, repository)
    record, duplicate = ingestion.register_existing(path, copy_into_uploads=copy)
    if duplicate and record.state.value == "READY":
        print(f"[READY] Existing video {record.video_id}: {record.display_name}")
        return record.video_id
    final_record = VideoProcessingService(settings, repository).process(record.video_id)
    print(
        f"[{final_record.state.value}] {final_record.video_id}: "
        f"{len(repository.list_segments([final_record.video_id]))} canonical segments"
    )
    for warning in final_record.warnings:
        print(f"[WARN] {warning}")
    return final_record.video_id


def import_ava(limit: int | None, requested_video_id: str | None) -> None:
    settings, repository = _services()
    ingestion = VideoIngestionService(settings, repository)
    fused_root = Path("data/ava/fused")
    video_root = Path("data/ava/videos")
    fused_files = sorted(fused_root.glob("*/*.fused.json"))
    if requested_video_id:
        fused_files = [
            path
            for path in fused_files
            if path.name == f"{requested_video_id}.fused.json"
        ]
    if limit is not None:
        fused_files = fused_files[:limit]
    if not fused_files:
        raise SystemExit("No matching fused AVA files were found")

    for fused_path in fused_files:
        source_id = fused_path.name.removesuffix(".fused.json")
        video_path = video_root / f"{source_id}.mp4"
        record, duplicate = ingestion.register_existing(video_path, copy_into_uploads=False)
        count = import_fused_ava(
            video=record,
            fused_path=fused_path,
            settings=settings,
            repository=repository,
        )
        print(
            f"[IMPORTED] {source_id} -> {record.video_id}: {count} segments"
            + (" (existing registration)" if duplicate else "")
        )


def refresh_library(video_ids: list[str]) -> None:
    """Re-run configured models for registered videos while preserving trusted overlays."""

    settings, repository = _services()
    selected = set(video_ids)
    videos = [
        video
        for video in repository.list_videos()
        if not selected or video.video_id in selected
    ]
    if selected - {video.video_id for video in videos}:
        missing = ", ".join(sorted(selected - {video.video_id for video in videos}))
        raise SystemExit(f"Unknown video IDs: {missing}")
    processor = VideoProcessingService(settings, repository)
    for index, video in enumerate(videos, start=1):
        print(f"[REFRESH {index}/{len(videos)}] {video.display_name} ({video.video_id})")
        final = processor.process(video.video_id)
        count = len(repository.list_segments([video.video_id]))
        print(f"[{final.state.value}] {video.display_name}: {count} canonical segments")
        for warning in final.warnings:
            print(f"[WARN] {warning}")


def backfill_gnn_actions(video_ids: list[str]) -> None:
    settings, repository = _services()
    selected = set(video_ids)
    videos = [
        video
        for video in repository.list_videos()
        if not selected or video.video_id in selected
    ]
    if selected - {video.video_id for video in videos}:
        missing = ", ".join(sorted(selected - {video.video_id for video in videos}))
        raise SystemExit(f"Unknown video IDs: {missing}")
    processor = VideoProcessingService(settings, repository)
    for index, video in enumerate(videos, start=1):
        count = processor.backfill_gnn_actions(video.video_id)
        print(
            f"[GNN {index}/{len(videos)}] {video.display_name}: "
            f"{count} timestamped predictions"
        )


def run_search(query: str, video_ids: list[str], limit: int) -> None:
    _, repository = _services()
    response = LocalHybridSearchEngine(repository).search(
        SearchRequest(query=query, video_ids=video_ids, limit=limit)
    )
    print(response.model_dump_json(indent=2))


def demo(path: Path) -> None:
    video_id = process_video(path, copy=True)
    settings, repository = _services()
    engine = LocalHybridSearchEngine(repository)
    examples = [
        "Find segments containing a person and a laptop",
        "Find where the project budget is discussed",
    ]
    print(f"\nVidQuery demo video: {video_id}")
    for query in examples:
        response = engine.search(SearchRequest(query=query, video_ids=[video_id], limit=3))
        print(json.dumps(response.model_dump(mode="json"), indent=2))
    print("\nStart the UI with: python -m vidquery.cli serve")


def main() -> None:
    parser = argparse.ArgumentParser(description="VidQuery canonical application CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="Run the API and browser frontend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    process = subcommands.add_parser("process", help="Register and process an MP4")
    process.add_argument("--video", type=Path, required=True)
    process.add_argument("--copy", action="store_true", help="Copy the source into UPLOAD_DIR")

    demo_command = subcommands.add_parser("demo", help="Process one MP4 and run sample searches")
    demo_command.add_argument("--video", type=Path, required=True)

    ava = subcommands.add_parser("import-ava", help="Adapt existing fused AVA artifacts")
    ava.add_argument("--limit", type=int, default=None)
    ava.add_argument("--video-id", default=None)

    expansion = subcommands.add_parser(
        "expand-ava-dataset",
        help="Incrementally add class-useful AVA videos through validated feature caches",
    )
    expansion.add_argument("--target-videos", type=int, required=True)
    expansion.add_argument("--batch-size", type=int, default=4)
    expansion.add_argument("--workers", type=int, default=3)
    expansion.add_argument("--target-support", type=int, default=25)
    expansion.add_argument("--device", default="auto")
    expansion.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="Minimum free disk space retained while processing temporary batches",
    )
    expansion.add_argument("--keep-raw-video", action="store_true")
    expansion.add_argument("--keep-frames", action="store_true")
    expansion.add_argument(
        "--state",
        type=Path,
        default=Path("data/app/features/ava80/expansion_state.json"),
    )
    expansion.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("evaluation/results/ava-expansion-incremental-selection.json"),
    )

    refresh = subcommands.add_parser(
        "refresh-library",
        help="Re-run active models for registered videos while preserving annotations",
    )
    refresh.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Refresh only this registered video ID; repeat to select multiple",
    )
    gnn_backfill = subcommands.add_parser(
        "backfill-gnn-actions",
        help="Apply the validated GNN to stored segment evidence without media reprocessing",
    )
    gnn_backfill.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Backfill only this registered video ID; repeat to select multiple",
    )

    search = subcommands.add_parser("search", help="Search locally indexed canonical segments")
    search.add_argument("query")
    search.add_argument("--video-id", action="append", default=[])
    search.add_argument("--limit", type=int, default=10)

    evaluation = subcommands.add_parser("evaluate", help="Run the checked-in retrieval evaluation")
    evaluation.add_argument("--dataset", type=Path, default=Path("evaluation/queries.json"))
    evaluation.add_argument(
        "--output", type=Path, default=Path("evaluation/results/latest.json")
    )

    action_training = subcommands.add_parser(
        "train-action-model", help="Train and validate the reduced AVA person-action model"
    )
    action_training.add_argument(
        "--fused-root", type=Path, default=Path("data/ava/fused/train")
    )
    action_training.add_argument(
        "--video-root", type=Path, default=Path("data/ava/videos")
    )
    action_training.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/app/models/action_classifier/action_model.pt"),
    )
    action_training.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/app/models/action_classifier/action_model.metadata.json"),
    )
    action_training.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/action-model-test.json"),
    )
    action_training.add_argument("--seed", type=int, default=2026)
    action_training.add_argument("--epochs", type=int, default=100)
    action_training.add_argument("--batch-size", type=int, default=128)
    action_training.add_argument("--patience", type=int, default=15)
    action_training.add_argument("--device", default="auto")
    gnn_training = subcommands.add_parser(
        "train-gnn-action-model",
        help="Train the full 80-label AVA MLP baseline, then the PyG GNN",
    )
    gnn_training.add_argument(
        "--fused-root", type=Path, default=Path("data/ava/fused")
    )
    gnn_training.add_argument(
        "--frame-root", type=Path, default=Path("data/ava/extracted_frames")
    )
    gnn_training.add_argument(
        "--annotation",
        type=Path,
        action="append",
        default=[],
        help="AVA v2.2 annotation CSV; repeat for train and validation files",
    )
    gnn_training.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("data/app/models/ava80/feature_cache"),
    )
    gnn_training.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/app/models/ava80"),
    )
    gnn_training.add_argument(
        "--existing-six-label-report",
        type=Path,
        default=Path("evaluation/results/action-model-test.json"),
    )
    gnn_training.add_argument("--seed", type=int, default=2026)
    gnn_training.add_argument("--epochs", type=int, default=100)
    gnn_training.add_argument("--patience", type=int, default=15)
    gnn_training.add_argument("--device", default="auto")
    gnn_training.add_argument("--model-version", default="ava80-gat-v1")
    gnn_training.add_argument("--force-rebuild-cache", action="store_true")
    evaluation.add_argument("--limit", type=int, default=5)
    evaluation.add_argument(
        "--neo4j",
        action="store_true",
        help="Evaluate the exact graph method against the configured live Neo4j service",
    )
    evaluation.add_argument(
        "--learned-actions",
        action="store_true",
        help="Evaluate the validated person-action checkpoint on its supported subset",
    )
    evaluation.add_argument(
        "--gnn-actions",
        action="store_true",
        help="Evaluate the validated GraphSAGE action checkpoint on its supported subset",
    )

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("vidquery.api:app", host=args.host, port=args.port, reload=False)
    elif args.command == "process":
        process_video(args.video, copy=args.copy)
    elif args.command == "demo":
        demo(args.video)
    elif args.command == "import-ava":
        import_ava(args.limit, args.video_id)
    elif args.command == "expand-ava-dataset":
        result = expand_ava_dataset(
            target_videos=args.target_videos,
            batch_size=args.batch_size,
            workers=args.workers,
            target_support=args.target_support,
            state_path=args.state,
            selection_manifest=args.selection_manifest,
            yolo_model=get_settings().yolo_model,
            device=args.device,
            delete_raw=not args.keep_raw_video,
            delete_frames=not args.keep_frames,
            min_free_bytes=int(args.min_free_gb * 1024**3),
        )
        print(
            json.dumps(
                {
                    "target_videos": result["target_videos"],
                    "completed_video_count": result["completed_video_count"],
                    "target_reached": result["target_reached"],
                    "failed_video_ids": result["failed_video_ids"],
                    "state": str(args.state),
                },
                indent=2,
            )
        )
    elif args.command == "refresh-library":
        refresh_library(args.video_id)
    elif args.command == "backfill-gnn-actions":
        backfill_gnn_actions(args.video_id)
    elif args.command == "search":
        run_search(args.query, args.video_id, args.limit)
    elif args.command == "evaluate":
        settings, repository = _services()
        graph_runner = None
        learned_action_index = None
        if args.neo4j:
            graph_runner = CanonicalNeo4jIndexer.from_settings(settings)
        if args.learned_actions:
            status = action_model_status(
                settings.action_model_checkpoint, settings.action_model_metadata
            )
            if status != "available_validated":
                raise SystemExit(f"Learned action model is not evaluable: {status}")
            predictor = PersonActionPredictor(
                settings.action_model_checkpoint, settings.action_model_metadata
            )
            learned_action_index = build_learned_action_index(
                predictor,
                fused_root=Path("data/ava/fused/train"),
                video_root=Path("data/ava/videos"),
                segment_duration=settings.segment_duration,
            )
        if args.gnn_actions:
            status = ava80_gnn_status(
                settings.gnn_action_checkpoint,
                settings.gnn_action_metadata,
                settings.gnn_model_version,
            )
            if status != "available_validated":
                raise SystemExit(f"AVA80 GNN action model is not evaluable: {status}")
            gnn_predictor = AVA80GNNPredictor(
                settings.gnn_action_checkpoint,
                settings.gnn_action_metadata,
                device="cpu",
                expected_model_version=settings.gnn_model_version,
            )
            cache_candidates = [
                settings.gnn_feature_cache_dir / "ava80_improved_graphs.pt",
                settings.gnn_feature_cache_dir / "ava80_graph_dataset.pt",
            ]
            graph_cache_path = next(
                (path for path in cache_candidates if path.is_file()), cache_candidates[-1]
            )
            learned_action_index = build_ava80_action_index(
                gnn_predictor,
                graph_cache_path=graph_cache_path,
                segment_duration=settings.segment_duration,
            )
        try:
            result = evaluate(
                repository,
                args.dataset,
                limit=args.limit,
                graph_runner=graph_runner,
                learned_action_index=learned_action_index,
            )
        finally:
            if graph_runner is not None:
                graph_runner.close()
        write_evaluation(result, args.output)
        print(json.dumps(result["summary"], indent=2))
        print(f"Wrote machine-readable results to {args.output}")
    elif args.command == "train-action-model":
        metadata = train_person_action_model(
            fused_root=args.fused_root,
            video_root=args.video_root,
            checkpoint_path=args.checkpoint,
            metadata_path=args.metadata,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device_name=args.device,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata["test_metrics"], indent=2))
        print(f"Wrote checkpoint to {args.checkpoint}")
        print(f"Wrote machine-readable report to {args.report}")
    elif args.command == "train-gnn-action-model":
        annotation_paths = args.annotation or [
            Path("data/ava/annotations/ava_train_v2.2.csv"),
            Path("data/ava/annotations/ava_val_v2.2.csv"),
        ]
        result = train_ava80_suite(
            fused_root=args.fused_root,
            frame_root=args.frame_root,
            annotation_paths=annotation_paths,
            feature_cache_dir=args.feature_cache_dir,
            output_dir=args.output_dir,
            existing_six_label_report=args.existing_six_label_report,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            device_name=args.device,
            model_version=args.model_version,
            force_rebuild_cache=args.force_rebuild_cache,
        )
        print(json.dumps(result["comparison"], indent=2))
        print(f"Wrote AVA80 training artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
