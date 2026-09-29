from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import get_hf_file_metadata, hf_hub_url  # type: ignore[import-not-found]
from remotezip import RemoteZip  # type: ignore[import-not-found]

from vidquery.vidor_relation import _selected_frames


def _required_frames(
    annotation_zip: Path,
    manifest: dict[str, Any],
    maximum: int,
    max_frames_per_video: int,
) -> dict[str, list[int]]:
    selected = manifest["selected_videos"][:maximum]
    support: Counter[str] = Counter()
    for item in selected:
        support.update(item["support"])
    output: dict[str, list[int]] = {}
    with zipfile.ZipFile(annotation_zip) as archive:
        for item in selected:
            payload = json.loads(archive.read(item["member"]))
            output[str(payload["video_path"])] = _selected_frames(
                payload["relation_instances"], support, max_frames_per_video
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Range-extract selected VidOR videos, keep only required JPEG frames"
    )
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--maximum-selected-videos", type=int, default=500)
    parser.add_argument("--max-frames-per-video", type=int, default=6)
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/source/vidor/training-annotation.zip")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/app/models/vidor_relation/split_manifest.json")
    )
    parser.add_argument(
        "--frame-root", type=Path, default=Path("data/source/vidor/selected-frames")
    )
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required = _required_frames(
        args.annotations,
        manifest,
        args.maximum_selected_videos,
        args.max_frames_per_video,
    )
    filename = f"training-video-part{args.part}.zip"
    location = get_hf_file_metadata(
        hf_hub_url("shangxd/vidor", filename, repo_type="dataset")
    ).location
    if not location:
        raise RuntimeError(f"no download location for {filename}")
    staged = 0
    staged_video_ids: list[str] = []
    missing = []
    with RemoteZip(location) as remote:
        names = set(remote.namelist())
        for relative_video, frame_ids in sorted(required.items()):
            member = f"video/{relative_video}"
            if member not in names:
                continue
            group = Path(relative_video).parent.name
            video_id = Path(relative_video).stem
            output_dir = args.frame_root / group / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            if all((output_dir / f"frame_{frame + 1:06d}.jpg").is_file() for frame in frame_ids):
                staged += 1
                staged_video_ids.append(video_id)
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                handle.write(remote.read(member))
                video_path = Path(handle.name)
            try:
                for frame_id in frame_ids:
                    output = output_dir / f"frame_{frame_id + 1:06d}.jpg"
                    if output.is_file():
                        continue
                    command = [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(video_path),
                        "-vf",
                        f"select=eq(n\\,{frame_id})",
                        "-frames:v",
                        "1",
                        str(output),
                    ]
                    subprocess.run(command, check=True)
                staged += 1
                staged_video_ids.append(video_id)
            finally:
                video_path.unlink(missing_ok=True)
    for relative_video in required:
        output_dir = args.frame_root / Path(relative_video).parent.name / Path(relative_video).stem
        if not output_dir.is_dir():
            missing.append(relative_video)
    proof = {
        "schema_version": "vidor-pair-visual-frame-staging-v1",
        "source_repository": "shangxd/vidor",
        "source_archive": filename,
        "requested_selected_videos": min(args.maximum_selected_videos, len(required)),
        "staged_videos": staged,
        "not_in_this_archive": missing,
        "frame_root": args.frame_root.as_posix(),
        "raw_video_retention": "none; temporary MP4 deleted immediately after frame extraction",
    }
    proof_path = args.frame_root / "staging_report.json"
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    staged_set = set(staged_video_ids)
    staged_manifest = {
        "schema_version": "vidor-relation-pair-visual-split-v2",
        "seed": int(manifest["seed"]),
        "source_split_manifest": args.manifest.as_posix(),
        "source_archive": filename,
        "selected_videos": [
            row for row in manifest["selected_videos"] if str(row["video_id"]) in staged_set
        ],
        **{
            name: [video_id for video_id in manifest[name] if video_id in staged_set]
            for name in ("train", "validation", "test")
        },
    }
    (args.frame_root / "split_manifest.json").write_text(
        json.dumps(staged_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
