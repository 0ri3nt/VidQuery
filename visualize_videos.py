import json
import cv2
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FRAMES_DIR = Path("data/ava/extracted_frames")
SCENE_GRAPHS_DIR = Path("data/ava/scene_graphs")
OUTPUT_DIR = Path("data/ava/visualizations_video")
VIDEO_FPS = 12  # adjust as needed

COLOR_NODE = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_EDGE = (0, 165, 255)
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


def draw_graph_on_image(img, graph_data):
    """Draw boxes/edges and return augmented image."""
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
        relation_color = RELATION_COLORS.get(rel, COLOR_EDGE)

        if src_id == tgt_id:
            x1, y1, _, _ = node_bboxes[src_id]
            idx = self_edge_offsets.get(src_id, 0)
            self_edge_offsets[src_id] = idx + 1

            tx = x1
            ty = max(14, y1 - 22 - (idx * 16))
            label = rel
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + 3), relation_color, -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
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
        cv2.putText(img, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    active_relations = sorted({edge.get("type") for edge in edges if edge.get("type")})
    legend_x, legend_y = 10, 20
    for rel_type in active_relations:
        color = RELATION_COLORS.get(rel_type, COLOR_EDGE)
        cv2.putText(img, rel_type, (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        legend_y += 18

    for node in nodes:
        x1, y1, x2, y2 = map(int, node["bbox"])
        label = f"[{node['node_id']}] {node['class_name']} ({node['confidence']:.2f})"

        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_NODE, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), COLOR_NODE, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    return img


def process_video_split(split_name: str):
    split_graph_dir = SCENE_GRAPHS_DIR / split_name
    if not split_graph_dir.exists():
        print(f"[!] Graph split not found: {split_graph_dir}")
        return

    split_frame_dir = FRAMES_DIR / split_name
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
            print(f"  [!] Frame not found for video {video_dir.name}: {first_frame_name}")
            continue

        first_img = cv2.imread(str(first_frame_path))
        if first_img is None:
            print(f"  [!] Could not load first frame for {video_dir.name}")
            continue

        h, w = first_img.shape[:2]
        out_dir = OUTPUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"{video_dir.name}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, VIDEO_FPS, (w, h))

        print(f"  Rendering video: {out_video} ({len(json_files)} frames)")
        for json_path in json_files:
            frame_name = json_path.stem + ".jpg"
            frame_path = split_frame_dir / video_dir.name / frame_name
            if not frame_path.exists():
                print(f"    [!] Missing frame: {frame_name}")
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            img_out = draw_graph_on_image(img, graph_data)
            writer.write(img_out)

        writer.release()
        print(f"    Done: {out_video}")


def main():
    print("==================================================")
    print("  VidQuery — Video Graph Visualizer")
    print("==================================================")

    for split in ["train", "val"]:
        process_video_split(split)

    print(f"[OK] Video visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()