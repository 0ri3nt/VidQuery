import argparse
import json
from pathlib import Path

from extraction.preprocessor import VideoPreprocessor
from extraction.visual import VisualExtractor
from extraction.audio import (
    AudioExtractor,
    render_timeline_plot,
    write_segments_csv,
    write_speaker_srt,
)
from modelling.alignment import fuse_modalities
from modelling.gnn_exporter import export_gnn_outputs
from modelling.scene_graph_builder import SceneGraphBuilder


def _write_whisper_outputs(output_dir: Path, stem: str, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.txt").write_text(result.get("text", "").strip() + "\n", encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _run_audio_and_alignment(
    max_videos: int | None = 20,
    whisper_model: str = "tiny",
    run_diarization: bool = False,
) -> None:
    from core.config import get_settings

    settings = get_settings()
    preprocessor = VideoPreprocessor(max_videos=max_videos)

    video_files = sorted(preprocessor.video_dir.glob("*.mp4"))
    if max_videos is not None:
        video_files = video_files[:max_videos]

    if not video_files:
        print("[WARN] No video files found for audio pipeline.")
        return

    audio_extractor = AudioExtractor(whisper_model=whisper_model, hf_token=settings.hf_token)

    audio_dir = Path("data/ava/audio")
    whisper_dir = Path("data/ava/audio_outputs/whisper")
    diarization_dir = Path("data/ava/audio_outputs/diarization")
    merged_dir = Path("data/ava/audio_outputs/merged")
    fused_root = Path("data/ava/fused")

    audio_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    fused_root.mkdir(parents=True, exist_ok=True)

    for idx, video_path in enumerate(video_files, start=1):
        stem = video_path.stem
        print(f"[Audio {idx}/{len(video_files)}] {stem}")

        wav_path = preprocessor.extract_audio(video_path)

        whisper_json_path = whisper_dir / f"{stem}.json"
        if whisper_json_path.exists():
            whisper_result = json.loads(whisper_json_path.read_text(encoding="utf-8"))
        else:
            whisper_result = audio_extractor.transcribe(wav_path)
            _write_whisper_outputs(whisper_dir, stem, whisper_result)

        speaker_segments = []
        if run_diarization or bool(settings.hf_token):
            try:
                print(f"  [RUN] Diarization starting for {stem} ...")
                diarization_output = audio_extractor.diarize(wav_path)
                annotation = audio_extractor._extract_annotation(diarization_output)
                speaker_segments = audio_extractor.diarization_to_segments(diarization_output)

                diarization_dir.mkdir(parents=True, exist_ok=True)
                (diarization_dir / f"{stem}.diarization.json").write_text(
                    json.dumps({"file": str(wav_path), "segments": speaker_segments}, indent=2),
                    encoding="utf-8",
                )
                with open(diarization_dir / f"{stem}.rttm", "w", encoding="utf-8") as f:
                    annotation.write_rttm(f)
                write_segments_csv(speaker_segments, diarization_dir / f"{stem}.segments.csv")
                write_speaker_srt(speaker_segments, diarization_dir / f"{stem}.speaker_turns.srt")
                render_timeline_plot(
                    speaker_segments,
                    diarization_dir / f"{stem}.timeline.png",
                    f"Speaker Timeline: {stem}",
                )
                print(
                    f"  [OK] Diarization saved for {stem} "
                    f"({len(speaker_segments)} segments) -> {diarization_dir}"
                )
            except Exception as exc:
                print(f"  [WARN] Diarization skipped for {stem}: {exc}")

        merged_segments = AudioExtractor.merge_audio_data(whisper_result, speaker_segments)
        (merged_dir / f"{stem}.merged.json").write_text(
            json.dumps({"video_id": stem, "segments": merged_segments}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  [OK] Merged audio saved for {stem} -> {merged_dir / f'{stem}.merged.json'}")

        for split in ["train", "val"]:
            graph_dir = Path("data/ava/scene_graphs") / split / stem
            if not graph_dir.exists():
                continue

            graph_files = sorted(graph_dir.glob("*.json"))
            scene_graphs = [json.loads(p.read_text(encoding="utf-8")) for p in graph_files]
            fused = fuse_modalities(scene_graphs, merged_segments)

            out_dir = fused_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{stem}.fused.json").write_text(
                json.dumps(
                    {
                        "video_id": stem,
                        "split": split,
                        "num_frames": len(scene_graphs),
                        "num_audio_segments": len(merged_segments),
                        "frames": fused,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"  [OK] Fused output saved for {stem} ({split}) -> {out_dir / f'{stem}.fused.json'}")


def run_pipeline(
    max_videos: int | None = 20,
    skip_audio: bool = False,
    skip_gnn_export: bool = False,
    whisper_model: str = "tiny",
    run_diarization: bool = False,
) -> None:
    preprocessor = VideoPreprocessor(max_videos=max_videos)
    preprocessor.run()

    visual = VisualExtractor()
    visual.run_yolov8()

    scene_builder = SceneGraphBuilder()
    scene_builder.run()

    gnn_source_root = Path("data/ava/scene_graphs")

    if not skip_audio:
        _run_audio_and_alignment(
            max_videos=max_videos,
            whisper_model=whisper_model,
            run_diarization=run_diarization,
        )
        gnn_source_root = Path("data/ava/fused")

    if not skip_gnn_export:
        export_gnn_outputs(scene_graph_root=gnn_source_root, max_videos=max_videos)


def ingest_graphs_to_neo4j(scene_graph_root: Path) -> None:
    from core.config import get_settings
    from database.ingestion import GraphIngestor
    from database.neo4j_client import Neo4jClient

    settings = get_settings()
    settings.validate_neo4j_config()

    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    ingestor = GraphIngestor(client)
    schema_path = Path("database/schema.cypher")
    if schema_path.exists():
        ingestor.apply_schema(schema_path)

    total_files = 0
    for split in ["train", "val"]:
        split_dir = scene_graph_root / split
        if not split_dir.exists():
            continue

        fused_files = sorted(split_dir.glob("*.fused.json"))
        if fused_files:
            for fused_file in fused_files:
                fused_data = json.loads(fused_file.read_text(encoding="utf-8"))
                fused_data.setdefault("split", split)
                ingestor.ingest_scene_graph(fused_data)
                total_files += 1
            continue

        for video_dir in split_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for graph_file in video_dir.glob("*.json"):
                graph_data = json.loads(graph_file.read_text(encoding="utf-8"))
                graph_data.setdefault("split", split)
                ingestor.ingest_scene_graph(graph_data)
                total_files += 1
    print(f"[OK] Ingested {total_files} graph files from {scene_graph_root}")
    client.close()


def check_neo4j_connection() -> None:
    from core.config import get_settings
    from database.neo4j_client import Neo4jClient

    settings = get_settings()
    settings.validate_neo4j_config()

    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        if not client.health_check():
            raise RuntimeError("Neo4j health check query failed")
        print(f"[OK] Neo4j is reachable at {settings.neo4j_uri}")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VidQuery main entrypoint")
    sub = parser.add_subparsers(dest="command")

    pipe = sub.add_parser("pipeline", help="Run extraction + detection + graph building")
    pipe.add_argument("--max_videos", type=int, default=20)
    pipe.add_argument("--skip_audio", action="store_true", help="Skip audio transcription/fusion stage")
    pipe.add_argument(
        "--skip_gnn_export",
        action="store_true",
        help="Skip exporting persistent GNN tensors",
    )
    pipe.add_argument("--whisper_model", type=str, default="tiny", help="Whisper model for audio stage")
    pipe.add_argument(
        "--run_diarization",
        action="store_true",
        help="Run pyannote diarization if HF token is configured",
    )

    gnn = sub.add_parser("gnn", help="Export persistent GNN tensor outputs from scene graphs")
    gnn.add_argument("--max_videos", type=int, default=20)
    gnn.add_argument(
        "--scene_graph_root",
        type=Path,
        default=Path("data/ava/fused"),
        help="Root folder containing fused multimodal JSON (preferred) or scene-graph frames",
    )
    gnn.add_argument("--output_root", type=Path, default=Path("data/ava/gnn_outputs"))
    gnn.add_argument("--overwrite", action="store_true")

    audio = sub.add_parser("audio", help="Run audio extraction/transcription/fusion only")
    audio.add_argument("--max_videos", type=int, default=20)
    audio.add_argument("--whisper_model", type=str, default="tiny")
    audio.add_argument("--run_diarization", action="store_true")

    ing = sub.add_parser("ingest", help="Ingest scene graphs into Neo4j")
    ing.add_argument(
        "--scene_graph_root",
        type=Path,
        default=Path("data/ava/fused"),
        help="Root folder containing train/val fused JSON files or scene-graph folders",
    )

    sub.add_parser("db-check", help="Check Neo4j connectivity and health")
    sub.add_parser("api", help="Run FastAPI server")

    args = parser.parse_args()
    if args.command == "pipeline":
        run_pipeline(
            max_videos=args.max_videos,
            skip_audio=args.skip_audio,
            skip_gnn_export=args.skip_gnn_export,
            whisper_model=args.whisper_model,
            run_diarization=args.run_diarization,
        )
    elif args.command == "gnn":
        export_gnn_outputs(
            scene_graph_root=args.scene_graph_root,
            output_root=args.output_root,
            max_videos=args.max_videos,
            overwrite=args.overwrite,
        )
    elif args.command == "audio":
        _run_audio_and_alignment(
            max_videos=args.max_videos,
            whisper_model=args.whisper_model,
            run_diarization=args.run_diarization,
        )
    elif args.command == "ingest":
        ingest_graphs_to_neo4j(args.scene_graph_root)
    elif args.command == "db-check":
        try:
            check_neo4j_connection()
        except Exception as exc:
            print(f"[ERR] Neo4j check failed: {exc}")
            raise SystemExit(1) from exc
    elif args.command == "api":
        import uvicorn

        uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, reload=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
