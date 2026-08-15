from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("data/app/cache/ultralytics").resolve()))

from extraction.preprocessor import VideoPreprocessor  # noqa: E402
from extraction.visual import VisualExtractor, load_ava_annotations  # noqa: E402
from modelling.scene_graph_builder import SceneGraphBuilder  # noqa: E402
from vidquery.ava_coverage import load_video_coverage  # noqa: E402

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm"}


def _media_by_id(video_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in video_dir.iterdir()
        if path.suffix.lower() in MEDIA_EXTENSIONS and path.stat().st_size > 1024
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract annotated AVA timestamps, detect objects, and cache fused graphs"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("evaluation/results/ava-expansion-selection-target5.json"),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/ava/videos"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/ava-expansion-processing.json"),
    )
    args = parser.parse_args()
    annotation_paths = [
        Path("data/ava/annotations/ava_train_v2.2.csv"),
        Path("data/ava/annotations/ava_val_v2.2.csv"),
    ]
    coverage = load_video_coverage(annotation_paths)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    requested_ids = set(selection["selected_video_ids"])
    media = _media_by_id(args.video_dir)
    process_ids = sorted(requested_ids.intersection(media))
    unavailable = sorted(requested_ids.difference(media))
    preprocessor = VideoPreprocessor(video_dir=args.video_dir, max_videos=None)
    annotations_by_split = {
        "train": load_ava_annotations(annotation_paths[0]),
        "val": load_ava_annotations(annotation_paths[1]),
    }
    extracted: dict[str, Any] = {}
    for index, video_id in enumerate(process_ids, start=1):
        annotation_split = coverage[video_id].annotation_split
        split = "val" if annotation_split == "validation" else annotation_split
        timestamps = set(annotations_by_split[split][video_id])
        output_dir = Path("data/ava/extracted_frames") / split
        print(
            f"[extract {index}/{len(process_ids)}] {video_id}: "
            f"{len(timestamps)} annotated timestamps",
            flush=True,
        )
        frames = preprocessor.extract_frames(media[video_id], timestamps, output_dir)
        extracted[video_id] = {
            "split": split,
            "requested_timestamps": len(timestamps),
            "extracted_frames": len(frames),
        }

    visual = VisualExtractor(model_size=args.model)
    scene_builder = SceneGraphBuilder()
    for split in ("train", "val"):
        def normalized_split(video_id: str) -> str:
            annotation_split = coverage[video_id].annotation_split
            return "val" if annotation_split == "validation" else annotation_split

        selected_in_split = [
            video_id
            for video_id in process_ids
            if normalized_split(video_id) == split
        ]
        if not selected_in_split:
            continue
        visual.process_split(
            split,
            Path("data/ava/extracted_frames"),
            Path("data/ava/detections"),
            annotations_by_split[split],
        )
        scene_builder.process_split(split)
        for video_id in selected_in_split:
            graph_dir = Path("data/ava/scene_graphs") / split / video_id
            graph_paths = sorted(graph_dir.glob("*.json"))
            frames = [json.loads(path.read_text(encoding="utf-8")) for path in graph_paths]
            for frame in frames:
                frame.setdefault("audio_segments", [])
            output_dir = Path("data/ava/fused") / split
            output_dir.mkdir(parents=True, exist_ok=True)
            fused_path = output_dir / f"{video_id}.fused.json"
            fused_path.write_text(
                json.dumps(
                    {
                        "video_id": video_id,
                        "split": split,
                        "num_frames": len(frames),
                        "num_audio_segments": 0,
                        "frames": frames,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            extracted[video_id]["scene_graphs"] = len(frames)
            extracted[video_id]["fused_path"] = str(fused_path.as_posix())

    report = {
        "schema_version": "ava-expansion-processing-v1",
        "selection": str(args.selection.as_posix()),
        "requested_video_count": len(requested_ids),
        "processed_video_count": len(process_ids),
        "unavailable_video_ids": unavailable,
        "videos": extracted,
        "total_annotated_frames": sum(
            int(item.get("extracted_frames", 0)) for item in extracted.values()
        ),
        "yolo_model": args.model,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "processed_video_count": report["processed_video_count"],
        "unavailable_video_ids": unavailable,
        "total_annotated_frames": report["total_annotated_frames"],
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
