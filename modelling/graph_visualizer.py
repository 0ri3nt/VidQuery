import json
from pathlib import Path

import cv2

RELATION_COLORS = {
    "overlaps_with": (0, 165, 255),
    "touches": (255, 0, 255),
    "near": (255, 0, 0),
    "left_of": (255, 255, 0),
    "right_of": (255, 255, 0),
    "above": (200, 200, 0),
    "below": (200, 200, 0),
    "contains": (0, 150, 150),
    "inside": (0, 150, 150),
    "talks_to": (0, 0, 255),
    "interacts_with": (0, 125, 255),
    "sitting": (40, 220, 40),
    "standing": (40, 255, 120),
    "lying": (120, 200, 120),
    "kissing": (200, 40, 255),
    "hugging": (255, 100, 180),
    "talking_to": (0, 0, 220),
    "looking_left_at": (255, 180, 0),
    "looking_right_at": (255, 180, 0),
}


def draw_graph_on_frame(frame_path: Path, json_path: Path, out_path: Path) -> None:
    color_node = (0, 255, 0)
    color_text = (255, 255, 255)
    color_edge = (0, 165, 255)

    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"  [!] Could not load image: {frame_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    node_centers = {node["node_id"]: tuple(map(int, node["center"])) for node in nodes}
    node_bboxes = {node["node_id"]: tuple(map(int, node["bbox"])) for node in nodes}
    edge_offsets = {}
    self_edge_offsets = {}

    for edge in edges:
        src_id = edge["source"]
        tgt_id = edge["target"]
        if src_id not in node_centers or tgt_id not in node_centers:
            continue

        pt1 = node_centers[src_id]
        pt2 = node_centers[tgt_id]
        relation_color = RELATION_COLORS.get(edge.get("type"), color_edge)

        if src_id == tgt_id:
            x1, y1, _, _ = node_bboxes[src_id]
            idx = self_edge_offsets.get(src_id, 0)
            self_edge_offsets[src_id] = idx + 1
            label = edge.get("type", "?")

            tx = x1
            ty = max(14, y1 - 22 - (idx * 16))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + 3), relation_color, -1)
            cv2.putText(
                img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
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
        mid_y = (pt1[1] + pt2[1]) // 2 - 5 - (idx * 15)
        label = edge.get("type", "?")

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (mid_x, mid_y - th - 2), (mid_x + tw, mid_y + 3), relation_color, -1)
        cv2.putText(
            img,
            label,
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_text,
            1,
            cv2.LINE_AA,
        )

    for node in nodes:
        x1, y1, x2, y2 = map(int, node["bbox"])
        label = f"[{node['node_id']}] {node['class_name']} ({node['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), color_node, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color_node, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_text,
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    frames_dir = Path("data/ava/extracted_frames")
    scene_graphs_dir = Path("data/ava/scene_graphs")
    output_dir = Path("data/ava/visualizations")
    max_frames_per_video = 5

    train_graphs_dir = scene_graphs_dir / "train"
    if not train_graphs_dir.exists():
        print(f"[!] No graphs found in {train_graphs_dir}")
        return

    for video_dir in train_graphs_dir.iterdir():
        if not video_dir.is_dir():
            continue
        print(f"Visualizing {video_dir.name}...")
        json_files = list(video_dir.glob("*.json"))[:max_frames_per_video]
        for json_path in json_files:
            frame_filename = json_path.stem + ".jpg"
            frame_path = frames_dir / "train" / video_dir.name / frame_filename
            out_path = output_dir / video_dir.name / frame_filename
            if frame_path.exists():
                draw_graph_on_frame(frame_path, json_path, out_path)

    print(f"\n[OK] Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
