import json
from pathlib import Path

import torch

from modelling.gnn_dataset import build_sparse_graphs
from modelling.gnn_model import SceneGraphMPNN


def _to_tensor_pair(graph_obj):
    x = getattr(graph_obj, "x", graph_obj["x"])
    edge_index = getattr(graph_obj, "edge_index", graph_obj["edge_index"])
    return x, edge_index


def _load_split_frames(split_dir: Path) -> list[tuple[str, str, dict]]:
    """
    Return list of tuples: (video_id, frame_stem, frame_dict)

    Supports two layouts:
    1) fused: <split>/*.fused.json where each file contains frames[]
    2) per-frame: <split>/<video_id>/*.json
    """
    fused_files = sorted(split_dir.glob("*.fused.json"))
    if fused_files:
        items: list[tuple[str, str, dict]] = []
        for fused_file in fused_files:
            payload = json.loads(fused_file.read_text(encoding="utf-8"))
            default_video_id = str(payload.get("video_id", fused_file.stem.replace(".fused", "")))
            for frame in payload.get("frames", []):
                video_id = str(frame.get("video_id", default_video_id))
                timestamp = int(frame.get("timestamp", 0))
                frame_stem = f"{video_id}_{timestamp:04d}"
                items.append((video_id, frame_stem, frame))
        return items

    items = []
    for video_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        for frame_path in sorted(video_dir.glob("*.json")):
            frame = json.loads(frame_path.read_text(encoding="utf-8"))
            items.append((video_dir.name, frame_path.stem, frame))
    return items


def export_gnn_outputs(
    scene_graph_root: Path = Path("data/ava/fused"),
    output_root: Path = Path("data/ava/gnn_outputs"),
    splits: tuple[str, ...] = ("train", "val"),
    max_videos: int | None = None,
    overwrite: bool = False,
) -> None:
    """
    Persist runtime GNN tensors for each scene-graph frame.

    Output per frame is written to:
    data/ava/gnn_outputs/<split>/<video_id>/<frame>.gnn.json

    A per-video summary is also written to:
    data/ava/gnn_outputs/<split>/<video_id>/summary.json
    """
    scene_graph_root = Path(scene_graph_root)
    output_root = Path(output_root)

    for split in splits:
        split_scene_dir = scene_graph_root / split
        if not split_scene_dir.exists():
            print(f"[SKIP] Scene-graph split not found: {split_scene_dir}")
            continue

        split_out_dir = output_root / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        all_items = _load_split_frames(split_scene_dir)
        grouped: dict[str, list[tuple[str, dict]]] = {}
        for video_id, frame_stem, frame in all_items:
            grouped.setdefault(video_id, []).append((frame_stem, frame))

        video_ids = sorted(grouped.keys())
        if max_videos is not None:
            video_ids = video_ids[:max_videos]

        print(f"[GNN] Exporting split={split} videos={len(video_ids)} -> {split_out_dir}")

        for idx, video_id in enumerate(video_ids, start=1):
            frame_items = sorted(grouped[video_id], key=lambda item: item[0])
            video_out_dir = split_out_dir / video_id
            video_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [Video {idx}/{len(video_ids)}] {video_id} frames={len(frame_items)}")

            model = None
            in_channels = None
            out_dim = 32

            stats = {
                "total_frames": len(frame_items),
                "frames_with_nodes": 0,
                "frames_with_sparse_edges": 0,
                "frames_with_audio": 0,
                "total_nodes": 0,
                "total_sparse_edges": 0,
            }

            for frame_stem, scene_graph in frame_items:
                out_path = video_out_dir / f"{frame_stem}.gnn.json"

                if out_path.exists() and not overwrite:
                    cached = json.loads(out_path.read_text(encoding="utf-8"))
                    node_count = int(cached.get("node_count", 0))
                    sparse_edge_count = int(cached.get("sparse_edge_count", 0))
                    audio_segment_count = int(cached.get("audio_segment_count", 0))

                    if node_count > 0:
                        stats["frames_with_nodes"] += 1
                    if sparse_edge_count > 0:
                        stats["frames_with_sparse_edges"] += 1
                    if audio_segment_count > 0:
                        stats["frames_with_audio"] += 1

                    stats["total_nodes"] += node_count
                    stats["total_sparse_edges"] += sparse_edge_count
                    continue

                graphs = build_sparse_graphs([scene_graph])

                if not graphs:
                    x = torch.empty((0, 10), dtype=torch.float)
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_features = torch.empty((0, out_dim), dtype=torch.float)
                else:
                    x, edge_index = _to_tensor_pair(graphs[0])
                    if x.ndim != 2:
                        x = x.reshape(x.shape[0], -1)

                    current_in_channels = int(x.shape[1]) if x.ndim == 2 else 0
                    if current_in_channels > 0 and (model is None or in_channels != current_in_channels):
                        model = SceneGraphMPNN(in_channels=current_in_channels)
                        model.eval()
                        in_channels = current_in_channels
                        out_dim = int(model.edge_mlp[-1].out_features)

                    if model is None:
                        edge_features = torch.empty((0, out_dim), dtype=torch.float)
                    else:
                        with torch.no_grad():
                            edge_features = model(x, edge_index)

                node_count = int(x.shape[0])
                node_feature_dim = int(x.shape[1]) if x.ndim == 2 else 0
                sparse_edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
                edge_feature_dim = int(edge_features.shape[1]) if edge_features.ndim == 2 else out_dim
                audio_segments = scene_graph.get("audio_segments", [])
                audio_segment_count = len(audio_segments) if isinstance(audio_segments, list) else 0
                unique_speakers = (
                    len(
                        {
                            str(seg.get("speaker", "unknown")).strip().lower()
                            for seg in audio_segments
                            if isinstance(seg, dict)
                        }
                    )
                    if audio_segment_count > 0
                    else 0
                )

                stats["total_nodes"] += node_count
                stats["total_sparse_edges"] += sparse_edge_count
                if node_count > 0:
                    stats["frames_with_nodes"] += 1
                if sparse_edge_count > 0:
                    stats["frames_with_sparse_edges"] += 1
                if audio_segment_count > 0:
                    stats["frames_with_audio"] += 1

                record = {
                    "video_id": scene_graph.get("video_id", video_id),
                    "timestamp": scene_graph.get("timestamp"),
                    "frame_file": scene_graph.get("frame_file"),
                    "source_root": str(scene_graph_root),
                    "node_count": node_count,
                    "node_feature_dim": node_feature_dim,
                    "sparse_edge_count": sparse_edge_count,
                    "edge_feature_dim": edge_feature_dim,
                    "audio_segment_count": audio_segment_count,
                    "audio_unique_speakers": unique_speakers,
                    "node_feature_tensor": x.detach().cpu().tolist(),
                    "edge_index_tensor": edge_index.detach().cpu().tolist(),
                    "edge_pairs": edge_index.t().detach().cpu().tolist() if sparse_edge_count > 0 else [],
                    "edge_feature_tensor": edge_features.detach().cpu().tolist(),
                }

                out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

            summary = {
                "video_id": video_id,
                "split": split,
                "total_frames": stats["total_frames"],
                "frames_with_nodes": stats["frames_with_nodes"],
                "frames_with_sparse_edges": stats["frames_with_sparse_edges"],
                "frames_with_audio": stats["frames_with_audio"],
                "total_nodes": stats["total_nodes"],
                "total_sparse_edges": stats["total_sparse_edges"],
                "output_dir": str(video_out_dir),
            }

            (video_out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                "    "
                f"saved={len(frame_items)} "
                f"with_nodes={stats['frames_with_nodes']} "
                f"with_sparse_edges={stats['frames_with_sparse_edges']} "
                f"with_audio={stats['frames_with_audio']}"
            )