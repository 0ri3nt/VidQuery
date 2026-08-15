import json
from pathlib import Path

import cv2

from .graph_visualizer import RELATION_COLORS


def draw_graph_on_image(img, graph_data):
    color_node = (0, 255, 0)
    color_text = (255, 255, 255)
    color_edge = (0, 165, 255)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    node_centers = {node["node_id"]: tuple(map(int, node["center"])) for node in nodes}
    node_bboxes = {node["node_id"]: tuple(map(int, node["bbox"])) for node in nodes}
    edge_offsets = {}
    self_edge_offsets = {}

    for edge in edges:
        src_id = edge.get("source")
        tgt_id = edge.get("target")
        if src_id not in node_centers or tgt_id not in node_centers:
            continue

        pt1 = node_centers[src_id]
        pt2 = node_centers[tgt_id]
        rel = edge.get("type", "")
        relation_color = RELATION_COLORS.get(rel, color_edge)

        if src_id == tgt_id:
            x1, y1, _, _ = node_bboxes[src_id]
            idx = self_edge_offsets.get(src_id, 0)
            self_edge_offsets[src_id] = idx + 1
            tx = x1
            ty = max(14, y1 - 22 - (idx * 16))
            label = rel
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + 3), relation_color, -1)
            cv2.putText(
                img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_text,
                1,
                cv2.LINE_AA,
            )
            continue

        cv2.line(img, pt1, pt2, relation_color, 2)
        key = (src_id, tgt_id)
        idx = edge_offsets.get(key, 0)
        edge_offsets[key] = idx + 1
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2 - 5 - (idx * 16)
        label = rel

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (mid_x, mid_y - th - 2), (mid_x + tw, mid_y + 3), relation_color, -1)
        cv2.putText(
            img,
            label,
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_text,
            1,
            cv2.LINE_AA,
        )

    for node in nodes:
        x1, y1, x2, y2 = map(int, node["bbox"])
        label = f"[{node['node_id']}] {node['class_name']} ({node['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), color_node, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color_node, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_text,
            1,
            cv2.LINE_AA,
        )

    return img


def process_video_split(split_name: str):
    frames_dir = Path("data/ava/extracted_frames")
    scene_graphs_dir = Path("data/ava/scene_graphs")
    output_dir = Path("data/ava/visualizations_video")
    video_fps = 12

    split_graph_dir = scene_graphs_dir / split_name
    if not split_graph_dir.exists():
        print(f"[!] Graph split not found: {split_graph_dir}")
        return

    split_frame_dir = frames_dir / split_name
    if not split_frame_dir.exists():
        print(f"[!] Frame split not found: {split_frame_dir}")
        return

    for video_dir in sorted(split_graph_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        json_files = sorted(video_dir.glob("*.json"))
        if not json_files:
            continue

        first_frame_name = json_files[0].stem + ".jpg"
        first_frame_path = split_frame_dir / video_dir.name / first_frame_name
        if not first_frame_path.exists():
            continue

        first_img = cv2.imread(str(first_frame_path))
        if first_img is None:
            continue

        h, w = first_img.shape[:2]
        out_dir = output_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"{video_dir.name}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, video_fps, (w, h))

        print(f"  Rendering video: {out_video} ({len(json_files)} frames)")
        for json_path in json_files:
            frame_name = json_path.stem + ".jpg"
            frame_path = split_frame_dir / video_dir.name / frame_name
            if not frame_path.exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            with open(json_path, encoding="utf-8") as f:
                graph_data = json.load(f)

            img_out = draw_graph_on_image(img, graph_data)
            writer.write(img_out)

        writer.release()
        print(f"    Done: {out_video}")


def main() -> None:
    print("==================================================")
    print("  VidQuery - Video Graph Visualizer")
    print("==================================================")
    for split in ["train", "val"]:
        process_video_split(split)


if __name__ == "__main__":
    main()
