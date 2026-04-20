import argparse
import json
import math
import re
from pathlib import Path


class SceneGraphBuilder:
    def __init__(
        self,
        detections_dir: Path = Path("data/ava/detections"),
        scene_graphs_dir: Path = Path("data/ava/scene_graphs"),
        iou_threshold: float = 0.05,
        near_threshold: float = 150.0,
    ):
        self.detections_dir = detections_dir
        self.scene_graphs_dir = scene_graphs_dir
        self.iou_threshold = iou_threshold
        self.near_threshold = near_threshold

    @staticmethod
    def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
        if inter_area == 0:
            return 0.0

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter_area / float(box_a_area + box_b_area - inter_area)

    def get_spatial_relationships(self, node_a: dict, node_b: dict) -> list[dict]:
        edges = []
        box_a, center_a = node_a["bbox"], node_a["center"]
        box_b, center_b = node_b["bbox"], node_b["center"]

        iou = self.calculate_iou(box_a, box_b)
        if iou > 0:
            if iou > self.iou_threshold:
                edges.append(
                    {
                        "source": node_a["node_id"],
                        "target": node_b["node_id"],
                        "type": "overlaps_with",
                        "weight": round(iou, 3),
                    }
                )
            else:
                edges.append(
                    {
                        "source": node_a["node_id"],
                        "target": node_b["node_id"],
                        "type": "touches",
                        "weight": round(iou, 3),
                    }
                )

        if (
            box_a[0] <= box_b[0]
            and box_a[1] <= box_b[1]
            and box_a[2] >= box_b[2]
            and box_a[3] >= box_b[3]
        ):
            edges.append(
                {"source": node_a["node_id"], "target": node_b["node_id"], "type": "contains"}
            )
        elif (
            box_b[0] <= box_a[0]
            and box_b[1] <= box_a[1]
            and box_b[2] >= box_a[2]
            and box_b[3] >= box_a[3]
        ):
            edges.append(
                {"source": node_a["node_id"], "target": node_b["node_id"], "type": "inside"}
            )

        dx = center_a[0] - center_b[0]
        dy = center_a[1] - center_b[1]
        distance = math.sqrt(dx**2 + dy**2)
        if distance < self.near_threshold:
            edges.append(
                {
                    "source": node_a["node_id"],
                    "target": node_b["node_id"],
                    "type": "near",
                    "distance": round(distance, 1),
                }
            )

        if distance < self.near_threshold * 2:
            if center_a[0] < center_b[0] - 20:
                edges.append(
                    {"source": node_a["node_id"], "target": node_b["node_id"], "type": "left_of"}
                )
            elif center_a[0] > center_b[0] + 20:
                edges.append(
                    {"source": node_a["node_id"], "target": node_b["node_id"], "type": "right_of"}
                )
            if center_a[1] < center_b[1] - 20:
                edges.append(
                    {"source": node_a["node_id"], "target": node_b["node_id"], "type": "above"}
                )
            elif center_a[1] > center_b[1] + 20:
                edges.append(
                    {"source": node_a["node_id"], "target": node_b["node_id"], "type": "below"}
                )

        return edges

    @staticmethod
    def get_semantic_relationships(node_a: dict, node_b: dict) -> list[dict]:
        edges = []
        ann_a = node_a.get("matched_annotation")
        ann_b = node_b.get("matched_annotation")

        action_a = set(ann_a.get("action_ids", [])) if isinstance(ann_a, dict) else set()
        action_b = set(ann_b.get("action_ids", [])) if isinstance(ann_b, dict) else set()

        if node_a.get("class_name") == "person" and node_b.get("class_name") == "person":
            if 36 in action_a or 36 in action_b:
                edges.append(
                    {"source": node_a["node_id"], "target": node_b["node_id"], "type": "talks_to"}
                )

        if node_a.get("class_name") == "person" and node_b.get("class_name") in {
            "laptop",
            "keyboard",
            "cell phone",
            "book",
            "cup",
        }:
            if 17 in action_a or 44 in action_a or 15 in action_a:
                edges.append(
                    {
                        "source": node_a["node_id"],
                        "target": node_b["node_id"],
                        "type": "interacts_with",
                    }
                )

        if node_b.get("class_name") == "person" and node_a.get("class_name") in {
            "laptop",
            "keyboard",
            "cell phone",
            "book",
            "cup",
        }:
            if 17 in action_b or 44 in action_b or 15 in action_b:
                edges.append(
                    {
                        "source": node_a["node_id"],
                        "target": node_b["node_id"],
                        "type": "interacts_with",
                    }
                )

        return edges

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    @staticmethod
    def _person_posture(node: dict) -> str:
        x1, y1, x2, y2 = node["bbox"]
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        ratio = height / width
        if ratio >= 1.7:
            return "standing"
        if ratio >= 1.1:
            return "sitting"
        return "lying"

    @staticmethod
    def _nearest_other_node(source: dict, nodes: list[dict]) -> dict | None:
        src_center = source["center"]
        best = None
        best_dist = float("inf")

        for node in nodes:
            if node["node_id"] == source["node_id"]:
                continue
            dx = src_center[0] - node["center"][0]
            dy = src_center[1] - node["center"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best = node
        return best

    def get_action_relationships(self, nodes: list[dict]) -> list[dict]:
        edges = []
        for node in nodes:
            if node.get("class_name") != "person":
                continue

            ann = node.get("matched_annotation")
            labels = []
            if isinstance(ann, dict):
                labels = [str(lbl) for lbl in ann.get("action_labels", [])]

            if not labels:
                edges.append({"source": node["node_id"], "target": node["node_id"], "type": self._person_posture(node)})
                continue

            nearest = self._nearest_other_node(node, nodes)
            for label in labels:
                action = self._slug(label)
                if "sit" in action:
                    edges.append({"source": node["node_id"], "target": node["node_id"], "type": "sitting"})
                    continue
                if "stand" in action:
                    edges.append({"source": node["node_id"], "target": node["node_id"], "type": "standing"})
                    continue
                if "kiss" in action and nearest and nearest.get("class_name") == "person":
                    edges.append({"source": node["node_id"], "target": nearest["node_id"], "type": "kissing"})
                    continue
                if "hug" in action and nearest and nearest.get("class_name") == "person":
                    edges.append({"source": node["node_id"], "target": nearest["node_id"], "type": "hugging"})
                    continue
                if "talk" in action and nearest:
                    edges.append({"source": node["node_id"], "target": nearest["node_id"], "type": "talking_to"})
                    continue
                if ("watch" in action or "look" in action) and nearest:
                    src_x = node["center"][0]
                    tgt_x = nearest["center"][0]
                    look_rel = "looking_right_at" if tgt_x > src_x else "looking_left_at"
                    edges.append({"source": node["node_id"], "target": nearest["node_id"], "type": look_rel})
                    continue
                edges.append(
                    {
                        "source": node["node_id"],
                        "target": node["node_id"],
                        "type": f"action_{action}",
                    }
                )

        return edges

    def process_frame(self, json_path: Path, output_path: Path) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detections = data.get("detections", [])
        nodes = []
        for idx, det in enumerate(detections):
            nodes.append(
                {
                    "node_id": idx,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "center": det["center"],
                    "matched_annotation": det.get("matched_annotation"),
                }
            )

        edges = []
        num_nodes = len(nodes)
        action_edges = self.get_action_relationships(nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                edges.extend(self.get_spatial_relationships(nodes[i], nodes[j]))
                edges.extend(self.get_semantic_relationships(nodes[i], nodes[j]))
        edges.extend(action_edges)

        unique = set()
        deduped_edges = []
        for edge in edges:
            key = (edge.get("source"), edge.get("target"), edge.get("type"))
            if key not in unique:
                unique.add(key)
                deduped_edges.append(edge)

        graph_data = {
            "video_id": data["video_id"],
            "timestamp": data["timestamp"],
            "frame_file": data.get("frame_file"),
            "nodes": nodes,
            "edges": deduped_edges,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        return graph_data

    def process_split(self, split_name: str) -> int:
        split_dir = self.detections_dir / split_name
        if not split_dir.exists():
            print(f"  [!] Directory not found: {split_dir}")
            return 0

        video_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(video_dirs)} videos in {split_name.upper()} split.")

        total_graphs = 0
        for video_dir in video_dirs:
            json_files = list(video_dir.glob("*.json"))
            for json_path in json_files:
                out_path = self.scene_graphs_dir / split_name / video_dir.name / json_path.name
                if out_path.exists():
                    total_graphs += 1
                    continue
                self.process_frame(json_path, out_path)
                total_graphs += 1
            print(f"    Processed {len(json_files)} frames for {video_dir.name}")

        print(f"  Done. Total graphs for {split_name}: {total_graphs}")
        return total_graphs

    def run(self) -> None:
        print("==================================================")
        print("  VidQuery - Spatial Scene Graph Builder")
        print("==================================================")
        print(f"  Input  : {self.detections_dir}")
        print(f"  Output : {self.scene_graphs_dir}")

        print("\nProcessing TRAIN split...")
        self.process_split("train")
        print("\nProcessing VAL split...")
        self.process_split("val")
        print("\n[OK] Graph generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene graphs from YOLO detections")
    parser.add_argument("--detections_dir", type=Path, default=Path("data/ava/detections"))
    parser.add_argument("--scene_graphs_dir", type=Path, default=Path("data/ava/scene_graphs"))
    args = parser.parse_args()

    SceneGraphBuilder(args.detections_dir, args.scene_graphs_dir).run()


if __name__ == "__main__":
    main()
