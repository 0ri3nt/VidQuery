import json
import math
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────

DETECTIONS_DIR   = Path("data/ava/detections")
SCENE_GRAPHS_DIR = Path("data/ava/scene_graphs")

# Thresholds for heuristic edge generation
IOU_THRESHOLD    = 0.05  # Minimum overlap to be considered "interacting" or "touching"
NEAR_THRESHOLD   = 150.0 # Maximum pixel distance between centers to be considered "near"


# ─────────────────────────────────────────────
# SPATIAL MATH HELPERS
# ─────────────────────────────────────────────

def calculate_iou(boxA: list[float], boxB: list[float]) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_spatial_relationships(node_a: dict, node_b: dict) -> list[dict]:
    """
    Compare two nodes (detections) and return a list of edge dictionaries
    representing their spatial relationship.
    """
    edges = []
    
    boxA, centerA = node_a["bbox"], node_a["center"]
    boxB, centerB = node_b["bbox"], node_b["center"]

    # 1. Overlap (Intersection over Union)
    iou = calculate_iou(boxA, boxB)
    if iou > IOU_THRESHOLD:
        edges.append({
            "source": node_a["node_id"],
            "target": node_b["node_id"],
            "type": "overlaps_with",
            "weight": round(iou, 3)
        })

    # 2. Distance (Euclidean between centers)
    dx = centerA[0] - centerB[0]
    dy = centerA[1] - centerB[1]
    distance = math.sqrt(dx**2 + dy**2)
    
    if distance < NEAR_THRESHOLD:
        edges.append({
            "source": node_a["node_id"],
            "target": node_b["node_id"],
            "type": "near",
            "distance": round(distance, 1)
        })

    # 3. Directionality (Relative Position)
    # Only assign left/right/above/below if they are "near" to avoid 
    # connecting everything to everything else across the whole frame.
    if distance < NEAR_THRESHOLD * 2: 
        if centerA[0] < centerB[0] - 20: # 20px buffer to prevent noise
            edges.append({"source": node_a["node_id"], "target": node_b["node_id"], "type": "left_of"})
        elif centerA[0] > centerB[0] + 20:
            edges.append({"source": node_a["node_id"], "target": node_b["node_id"], "type": "right_of"})
            
    return edges


# ─────────────────────────────────────────────
# GRAPH GENERATION
# ─────────────────────────────────────────────

def process_frame(json_path: Path, output_path: Path):
    """Read YOLO detections, generate graph edges, and save."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    
    # 1. Create Nodes
    nodes = []
    for idx, det in enumerate(detections):
        node = {
            "node_id": idx,
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "center": det["center"],
            # Preserve AVA matching if it exists
            "matched_annotation": det.get("matched_annotation") 
        }
        nodes.append(node)

    # 2. Create Edges (Compare every node to every other node)
    edges = []
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue # Skip self-comparison
            
            spatial_edges = get_spatial_relationships(nodes[i], nodes[j])
            edges.extend(spatial_edges)

    # 3. Compile Graph Structure
    graph_data = {
        "video_id": data["video_id"],
        "timestamp": data["timestamp"],
        "frame_file": data.get("frame_file"),
        "nodes": nodes,
        "edges": edges
    }

    # 4. Save to Disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)


def process_split(split_name: str):
    """Iterate through all videos in a split (train/val)."""
    split_dir = DETECTIONS_DIR / split_name
    if not split_dir.exists():
        print(f"  [!] Directory not found: {split_dir}")
        return

    video_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    print(f"  Found {len(video_dirs)} videos in {split_name.upper()} split.")

    total_graphs = 0
    for video_dir in video_dirs:
        json_files = list(video_dir.glob("*.json"))
        
        for json_path in json_files:
            # Mirror the directory structure in the output folder
            out_path = SCENE_GRAPHS_DIR / split_name / video_dir.name / json_path.name
            
            # Skip if already generated (resume support)
            if out_path.exists():
                total_graphs += 1
                continue
                
            process_frame(json_path, out_path)
            total_graphs += 1
            
        print(f"    Processed {len(json_files)} frames for {video_dir.name}")
        
    print(f"  Done. Total graphs for {split_name}: {total_graphs}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("==================================================")
    print("  VidQuery — Spatial Scene Graph Builder")
    print("==================================================")
    print(f"  Input  : {DETECTIONS_DIR}")
    print(f"  Output : {SCENE_GRAPHS_DIR}")
    print("\nProcessing TRAIN split...")
    process_split("train")
    
    print("\nProcessing VAL split...")
    process_split("val")
    print("\n[OK] Graph generation complete.")

if __name__ == "__main__":
    main()