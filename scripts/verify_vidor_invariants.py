"""Check that growing the VidOR selection left earlier videos' data unchanged.

``frames``: every frame of the base selection exists in the new frame root
with the same SHA-256.

``graphs``: for the given split names (default validation and test), each
graph in the old cache has a counterpart in the new cache with identical
structure, labels, classes, boxes, and motion, and visual features within
``--atol`` (frozen ResNet features may differ in the last float bits when
computed in different GPU batches). The maximum feature difference is
reported either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

EXACT = ("x", "edge_index", "edge_attr", "class_ids", "node_motion", "edge_y", "edge_sample_type")
CLOSE = ("node_visual", "union_visual")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frames(manifest: dict[str, Any], annotations: Path, old_root: Path,
                 new_root: Path, max_frames: int) -> dict[str, Any]:
    from vidquery.vidor_relation import _selected_frames, frame_selection_support

    support = frame_selection_support(manifest)
    checked = 0
    mismatched: list[str] = []
    missing: list[str] = []
    with zipfile.ZipFile(annotations) as archive:
        for row in manifest["selected_videos"]:
            payload = json.loads(archive.read(row["member"]))
            relative = Path(payload["video_path"])
            for frame in _selected_frames(payload["relation_instances"], support, max_frames):
                name = Path(relative.parent.name) / relative.stem / f"frame_{frame + 1:06d}.jpg"
                old, new = old_root / name, new_root / name
                if not old.is_file() or not new.is_file():
                    missing.append(name.as_posix())
                    continue
                checked += 1
                if _sha256(old) != _sha256(new):
                    mismatched.append(name.as_posix())
    return {"frames_checked": checked, "missing": missing, "sha256_mismatched": mismatched,
            "identical": not missing and not mismatched}


def check_graphs(old_cache: Path, new_cache: Path, video_ids: set[str],
                 atol: float) -> dict[str, Any]:
    import torch

    def index(path: Path) -> dict[tuple[str, int], Any]:
        graphs = torch.load(path, map_location="cpu", weights_only=False)["graphs"]
        return {(str(graph.video_id), int(graph.frame_id)): graph
                for graph in graphs if str(graph.video_id) in video_ids}

    old, new = index(old_cache), index(new_cache)
    exact_mismatch: list[str] = []
    max_difference = 0.0
    for key, before in old.items():
        after = new.get(key)
        if after is None:
            exact_mismatch.append(f"{key}: missing")
            continue
        for name in EXACT:
            if not torch.equal(getattr(before, name), getattr(after, name)):
                exact_mismatch.append(f"{key}: {name}")
        if list(before.track_ids) != list(after.track_ids) or list(before.class_names) != list(
            after.class_names
        ):
            exact_mismatch.append(f"{key}: tracks/classes")
        for name in CLOSE:
            difference = float((getattr(before, name) - getattr(after, name)).abs().max())
            max_difference = max(max_difference, difference)
    extra = sorted(set(new) - set(old))
    return {
        "graphs_old": len(old),
        "graphs_new": len(new),
        "extra_graphs_in_new": len(extra),
        "exact_field_mismatches": exact_mismatch[:50],
        "exact_field_mismatch_count": len(exact_mismatch),
        "max_visual_feature_abs_difference": max_difference,
        "visual_atol": atol,
        "identical_within_tolerance": not exact_mismatch and not extra and max_difference <= atol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    frames = sub.add_parser("frames")
    frames.add_argument("--manifest", type=Path,
                        default=Path("data/app/models/vidor_relation/split_manifest.json"))
    frames.add_argument("--annotations", type=Path,
                        default=Path("data/source/vidor/training-annotation.zip"))
    frames.add_argument("--old-root", type=Path, default=Path("data/source/vidor/selected-frames"))
    frames.add_argument("--new-root", type=Path, required=True)
    frames.add_argument("--max-frames-per-video", type=int, default=6)
    graphs = sub.add_parser("graphs")
    graphs.add_argument("--old-cache", type=Path, required=True)
    graphs.add_argument("--new-cache", type=Path, required=True)
    graphs.add_argument("--manifest", type=Path,
                        default=Path("data/app/models/vidor_relation/split_manifest.json"))
    graphs.add_argument("--splits", nargs="+", default=["validation", "test"])
    graphs.add_argument("--atol", type=float, default=1e-4)
    for item in (frames, graphs):
        item.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.mode == "frames":
        report = check_frames(manifest, args.annotations, args.old_root, args.new_root,
                              args.max_frames_per_video)
    else:
        videos = {str(video) for name in args.splits for video in manifest[name]}
        report = check_graphs(args.old_cache, args.new_cache, videos, args.atol)
        report["splits"] = args.splits
    if args.output:
        if args.output.exists():
            raise SystemExit(f"{args.output} exists; choose a new file name")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    ok = report.get("identical", report.get("identical_within_tolerance"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
