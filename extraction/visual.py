import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


RELEVANT_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    45: "bowl",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    79: "toothbrush",
}

AVA_ACTIONS = {
    12: "sit",
    13: "stand",
    15: "walk",
    17: "write",
    26: "listen_to",
    36: "talk_to",
    44: "use_laptop",
    64: "kiss",
}


def action_label_from_id(action_id: int) -> str:
    return AVA_ACTIONS.get(action_id, f"action_{action_id}")


def load_ava_annotations(csv_path: Path) -> dict:
    annotations = defaultdict(lambda: defaultdict(list))
    if not csv_path.exists():
        return annotations

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue

            video_id = row[0].strip()
            try:
                timestamp = int(float(str(row[1]).strip()))
                x1, y1, x2, y2 = map(float, row[2:6])
                action_id = int(row[6])
                entity_id = int(row[7])
            except ValueError:
                continue

            annotations[video_id][timestamp].append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "action_id": action_id,
                    "action_label": action_label_from_id(action_id),
                    "entity_id": entity_id,
                }
            )

    return annotations


def compute_iou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections_to_annotations(
    detections: list,
    annotations: list,
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.3,
) -> list:
    if not annotations:
        return detections

    entity_annotations = defaultdict(list)
    for ann in annotations:
        entity_annotations[ann["entity_id"]].append(ann)

    for detection in detections:
        best_iou = 0.0
        best_entity = None
        det_box = detection["bbox"]

        for entity_id, anns in entity_annotations.items():
            x1_n, y1_n, x2_n, y2_n = anns[0]["bbox"]
            ava_box = [x1_n * img_w, y1_n * img_h, x2_n * img_w, y2_n * img_h]

            iou = compute_iou(det_box, ava_box)
            if iou > best_iou:
                best_iou = iou
                best_entity = (entity_id, anns, ava_box)

        if best_entity and best_iou >= iou_threshold:
            entity_id, anns, ava_box = best_entity
            detection["matched_annotation"] = {
                "entity_id": entity_id,
                "iou": round(best_iou, 4),
                "ava_bbox": ava_box,
                "action_ids": [a["action_id"] for a in anns],
                "action_labels": [a["action_label"] for a in anns],
            }
        else:
            detection["matched_annotation"] = None

    return detections


class VisualExtractor:
    def __init__(
        self,
        model_size: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ):
        self.model_size = model_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model: YOLO | None = None

    def load_model(self) -> YOLO:
        if self.model is None:
            print(f"Loading YOLO model: {self.model_size}")
            self.model = YOLO(self.model_size)
        return self.model

    def detect_frame(self, frame_path: Path) -> list:
        model = self.load_model()
        results = model(
            str(frame_path),
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )

        detections = []
        result = results[0]
        img_h, img_w = result.orig_shape

        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in RELEVANT_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = round(float(box.conf.item()), 4)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": RELEVANT_CLASSES[class_id],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "bbox_norm": [
                        round(x1 / img_w, 4),
                        round(y1 / img_h, 4),
                        round(x2 / img_w, 4),
                        round(y2 / img_h, 4),
                    ],
                    "confidence": conf,
                    "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                }
            )

        return detections

    @staticmethod
    def _build_frame_record(video_id: str, ts: int, frame_file: str, detections: list) -> dict:
        classes = [str(det.get("class_name", det.get("class_id", "unknown"))) for det in detections]
        bboxes = [det.get("bbox", [0, 0, 0, 0]) for det in detections]
        confidences = [float(det.get("confidence", 0.0)) for det in detections]

        return {
            "video_id": video_id,
            "frame_id": f"{video_id}_{int(ts):04d}",
            "timestamp": int(ts),
            "frame_file": frame_file,
            "bounding_boxes": bboxes,
            "classes": classes,
            "confidence": confidences,
            "detections": detections,
        }

    @staticmethod
    def _has_plan_fields(frame_record: dict) -> bool:
        required = {"frame_id", "timestamp", "bounding_boxes", "classes", "confidence", "detections"}
        return required.issubset(frame_record.keys())

    def process_split(
        self,
        split_name: str,
        frames_dir: Path,
        detections_dir: Path,
        annotations: dict | None = None,
    ) -> dict:
        split_frames_dir = frames_dir / split_name
        split_detections_dir = detections_dir / split_name

        if not split_frames_dir.exists():
            print(f"  [!] Frames dir not found: {split_frames_dir}")
            return {}

        video_dirs = sorted(split_frames_dir.iterdir())
        if not video_dirs:
            print(f"  [!] No video folders found in {split_frames_dir}")
            return {}

        print(f"\n{'=' * 52}")
        print(f"  Detecting objects - split: {split_name.upper()}")
        print(f"  Videos to process: {len(video_dirs)}")
        print(f"{'=' * 52}")

        summary = {}
        t_start = time.time()

        for v_idx, video_dir in enumerate(video_dirs):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name
            frame_files = sorted(video_dir.glob("*.jpg"))
            if not frame_files:
                continue

            out_video_dir = split_detections_dir / video_id
            out_video_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{v_idx + 1}/{len(video_dirs)}] {video_id} - {len(frame_files)} frames")

            video_summary = {
                "total_frames": len(frame_files),
                "total_detections": 0,
                "frames": {},
                "records": [],
            }

            for frame_path in frame_files:
                json_path = out_video_dir / frame_path.with_suffix(".json").name

                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)

                    cached_detections = cached.get("detections", [])
                    try:
                        ts_cached = int(cached.get("timestamp", frame_path.stem.split("_")[-1]))
                    except ValueError:
                        ts_cached = 0

                    if not self._has_plan_fields(cached):
                        cached = self._build_frame_record(
                            video_id,
                            ts_cached,
                            cached.get("frame_file", frame_path.name),
                            cached_detections,
                        )
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(cached, f, indent=2)

                    has_matched_annotations = any(
                        isinstance(det.get("matched_annotation"), dict)
                        for det in cached_detections
                    )
                    if not (annotations and not has_matched_annotations):
                        video_summary["frames"][ts_cached] = cached_detections
                        video_summary["records"].append(
                            {
                                "frame_id": cached["frame_id"],
                                "timestamp": cached["timestamp"],
                                "bounding_boxes": cached["bounding_boxes"],
                                "classes": cached["classes"],
                                "confidence": cached["confidence"],
                            }
                        )
                        video_summary["total_detections"] += len(cached_detections)
                        continue

                try:
                    ts = int(frame_path.stem.split("_")[-1])
                except ValueError:
                    continue

                detections = self.detect_frame(frame_path)

                if annotations and video_id in annotations:
                    ann_for_ts = annotations[video_id].get(ts, [])
                    if ann_for_ts:
                        try:
                            from PIL import Image

                            img = Image.open(frame_path)
                            img_w, img_h = img.size
                        except Exception:
                            img_w, img_h = 1280, 720

                        detections = match_detections_to_annotations(
                            detections, ann_for_ts, img_w, img_h
                        )

                frame_record = self._build_frame_record(video_id, ts, frame_path.name, detections)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(frame_record, f, indent=2)

                video_summary["frames"][ts] = detections
                video_summary["records"].append(
                    {
                        "frame_id": frame_record["frame_id"],
                        "timestamp": frame_record["timestamp"],
                        "bounding_boxes": frame_record["bounding_boxes"],
                        "classes": frame_record["classes"],
                        "confidence": frame_record["confidence"],
                    }
                )
                video_summary["total_detections"] += len(detections)

            summary[video_id] = video_summary
            avg_det = video_summary["total_detections"] / max(video_summary["total_frames"], 1)
            print(
                f"      Detections : {video_summary['total_detections']} total  ({avg_det:.1f} avg/frame)"
            )

        elapsed = time.time() - t_start
        print(f"\n  Done in {elapsed:.1f}s")
        return summary

    def run_yolov8(
        self,
        frames_dir: Path = Path("data/ava/extracted_frames"),
        detections_dir: Path = Path("data/ava/detections"),
        train_csv: Path = Path("data/ava/annotations/ava_train_v2.2.csv"),
        val_csv: Path = Path("data/ava/annotations/ava_val_v2.2.csv"),
    ) -> dict:
        self.load_model()
        detections_dir.mkdir(parents=True, exist_ok=True)

        split_annotations = {
            "train": load_ava_annotations(train_csv),
            "val": load_ava_annotations(val_csv),
        }

        all_results = {}
        for split in ["train", "val"]:
            split_dir = frames_dir / split
            if not split_dir.exists():
                print(f"[SKIP] {split} split not found at {split_dir}")
                continue
            all_results[split] = self.process_split(
                split,
                frames_dir,
                detections_dir,
                split_annotations.get(split),
            )

        return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on extracted AVA frames")
    parser.add_argument("--frames_dir", type=Path, default=Path("data/ava/extracted_frames"))
    parser.add_argument("--detections_dir", type=Path, default=Path("data/ava/detections"))
    parser.add_argument("--train_csv", type=Path, default=Path("data/ava/annotations/ava_train_v2.2.csv"))
    parser.add_argument("--val_csv", type=Path, default=Path("data/ava/annotations/ava_val_v2.2.csv"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    extractor = VisualExtractor(model_size=args.model, conf_thresh=args.conf, iou_thresh=args.iou)
    extractor.run_yolov8(args.frames_dir, args.detections_dir, args.train_csv, args.val_csv)


if __name__ == "__main__":
    main()
