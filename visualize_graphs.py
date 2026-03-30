import json
import cv2
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FRAMES_DIR = Path("data/ava/extracted_frames")
SCENE_GRAPHS_DIR = Path("data/ava/scene_graphs")
OUTPUT_DIR = Path("data/ava/visualizations")

# How many frames to visualize per video (keep it low for quick checks)
MAX_FRAMES_PER_VIDEO = 5 

# Colors (BGR format for OpenCV)
COLOR_NODE = (0, 255, 0)      # Green for bounding boxes
COLOR_TEXT = (255, 255, 255)  # White text
COLOR_EDGE = (0, 165, 255)    # Orange for relationship lines

def draw_graph_on_frame(frame_path: Path, json_path: Path, out_path: Path):
    """Draws nodes (bounding boxes) and edges (lines) on a frame."""
    # 1. Load image and JSON
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"  [!] Could not load image: {frame_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # 2. Draw Edges (Lines between centers)
    # We draw edges first so they sit behind the bounding boxes and text
    node_centers = {node["node_id"]: tuple(map(int, node["center"])) for node in nodes}
    
    for edge in edges:
        src_id = edge["source"]
        tgt_id = edge["target"]
        
        if src_id in node_centers and tgt_id in node_centers:
            pt1 = node_centers[src_id]
            pt2 = node_centers[tgt_id]
            
            # Draw the line
            cv2.line(img, pt1, pt2, COLOR_EDGE, 2)
            
            # Add edge type text at the midpoint
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(img, edge["type"], (mid_x, mid_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EDGE, 1, cv2.LINE_AA)

    # 3. Draw Nodes (Bounding Boxes & Labels)
    for node in nodes:
        x1, y1, x2, y2 = map(int, node["bbox"])
        label = f"[{node['node_id']}] {node['class_name']} ({node['confidence']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_NODE, 2)
        
        # Draw label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), COLOR_NODE, -1)
        
        # Draw text
        cv2.putText(img, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    # 4. Save the annotated image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

def main():
    print("==================================================")
    print("  VidQuery — Graph Visualizer")
    print("==================================================")
    
    # Let's just visualize the training split for a quick check
    train_graphs_dir = SCENE_GRAPHS_DIR / "train"
    if not train_graphs_dir.exists():
        print(f"[!] No graphs found in {train_graphs_dir}")
        return

    for video_dir in train_graphs_dir.iterdir():
        if not video_dir.is_dir():
            continue
            
        print(f"Visualizing {video_dir.name}...")
        json_files = list(video_dir.glob("*.json"))[:MAX_FRAMES_PER_VIDEO]
        
        for json_path in json_files:
            # Construct the path to the original frame
            frame_filename = json_path.stem + ".jpg" 
            frame_path = FRAMES_DIR / "train" / video_dir.name / frame_filename
            
            # Construct output path
            out_path = OUTPUT_DIR / video_dir.name / frame_filename
            
            if frame_path.exists():
                draw_graph_on_frame(frame_path, json_path, out_path)
            else:
                print(f"  [!] Missing frame for {json_path.name}")
                
    print(f"\n[OK] Visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()