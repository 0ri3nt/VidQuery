import json
import time
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG — should match your extract_frames.py
# ─────────────────────────────────────────────

FRAMES_DIR       = Path("data/ava/extracted_frames")  # root frames dir
DETECTIONS_DIR   = Path("data/ava/detections")        # where JSON goes
TRAIN_CSV        = Path("data/ava/annotations/ava_train_v2.2.csv")
VAL_CSV          = Path("data/ava/annotations/ava_val_v2.2.csv")
MODEL_SIZE       = "yolov8n.pt"                       # nano for testing
CONFIDENCE_THRESH = 0.25                              # min confidence
IOU_THRESH        = 0.45                              # NMS threshold

# YOLO COCO class IDs we actually care about for meeting/lecture videos
# Full list: https://docs.ultralytics.com/datasets/detect/coco/
RELEVANT_CLASSES = {
    0:  "person",
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


# AVA subset + fallback pattern for unknown IDs.
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
    """
    Parse AVA CSV to a nested mapping:
      annotations[video_id][timestamp] -> list of annotations
    """
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

            annotations[video_id][timestamp].append({
                "bbox": [x1, y1, x2, y2],
                "action_id": action_id,
                "action_label": action_label_from_id(action_id),
                "entity_id": entity_id,
            })

    return annotations


# ─────────────────────────────────────────────
# STEP 1: LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_size: str) -> YOLO:
    """
    Load YOLOv8 model. Downloads weights automatically
    on first run and caches them locally.
    """
    print(f"Loading YOLO model: {model_size}")
    model = YOLO(model_size)
    print(f"  [OK] Model loaded — "
          f"{sum(p.numel() for p in model.model.parameters()):,} params")
    return model


# ─────────────────────────────────────────────
# STEP 2: RUN DETECTION ON A SINGLE FRAME
# ─────────────────────────────────────────────

def detect_frame(model: YOLO,
                 frame_path: Path,
                 conf_thresh: float = CONFIDENCE_THRESH,
                 iou_thresh: float  = IOU_THRESH) -> list:
    """
    Run YOLOv8 on a single frame image.

    Returns a list of detection dicts, one per detected object:
        {
            "class_id":    int,
            "class_name":  str,
            "bbox":        [x1, y1, x2, y2],  # absolute pixel coords
            "bbox_norm":   [x1, y1, x2, y2],  # normalized 0-1 (matches AVA)
            "confidence":  float,
            "center":      [cx, cy]            # absolute pixel center
        }

    Only returns detections for classes in RELEVANT_CLASSES.
    """
    results = model(
        str(frame_path),
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False       # suppress per-frame console spam
    )

    detections = []
    result     = results[0]             # single image → single result
    img_h, img_w = result.orig_shape    # original image dimensions

    for box in result.boxes:
        class_id = int(box.cls.item())

        # Skip classes we don't care about
        if class_id not in RELEVANT_CLASSES:
            continue

        # Absolute pixel coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf            = round(float(box.conf.item()), 4)

        detections.append({
            "class_id":   class_id,
            "class_name": RELEVANT_CLASSES[class_id],
            "bbox":       [round(x1, 2), round(y1, 2),
                           round(x2, 2), round(y2, 2)],
            "bbox_norm":  [round(x1 / img_w, 4), round(y1 / img_h, 4),
                           round(x2 / img_w, 4), round(y2 / img_h, 4)],
            "confidence": conf,
            "center":     [round((x1 + x2) / 2, 2),
                           round((y1 + y2) / 2, 2)]
        })

    return detections


# ─────────────────────────────────────────────
# STEP 3: MATCH YOLO DETECTIONS TO AVA ANNOTATIONS
# ─────────────────────────────────────────────

def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    Both boxes in [x1, y1, x2, y2] format.
    Used to match YOLO detections to AVA ground truth boxes.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a       = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b       = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union        = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections_to_annotations(detections: list,
                                     annotations: list,
                                     img_w: int,
                                     img_h: int,
                                     iou_threshold: float = 0.3) -> list:
    """
    Match each YOLO detection to an AVA ground truth annotation
    using IoU overlap. AVA boxes are normalized (0-1), so we
    convert them to absolute pixel coords for comparison.

    Adds 'matched_annotation' key to each detection dict if a
    match is found, containing the AVA action labels for that entity.

    Returns the updated detections list.
    """
    if not annotations:
        return detections

    # Group AVA annotations by entity_id so one entity can have
    # multiple action labels at the same timestamp
    entity_annotations = defaultdict(list)
    for ann in annotations:
        entity_annotations[ann['entity_id']].append(ann)

    for detection in detections:
        best_iou      = 0.0
        best_entity   = None
        det_box       = detection['bbox']   # already absolute

        for entity_id, anns in entity_annotations.items():
            # AVA bbox is normalized — convert to absolute
            x1_n, y1_n, x2_n, y2_n = anns[0]['bbox']
            ava_box = [
                x1_n * img_w, y1_n * img_h,
                x2_n * img_w, y2_n * img_h
            ]

            iou = compute_iou(det_box, ava_box)
            if iou > best_iou:
                best_iou    = iou
                best_entity = (entity_id, anns, ava_box)

        if best_entity and best_iou >= iou_threshold:
            entity_id, anns, ava_box = best_entity
            detection['matched_annotation'] = {
                'entity_id':    entity_id,
                'iou':          round(best_iou, 4),
                'ava_bbox':     ava_box,
                'action_ids':   [a['action_id']   for a in anns],
                'action_labels':[a['action_label'] for a in anns]
            }
        else:
            # Detection has no matching AVA annotation —
            # still useful as a graph node, just unlabeled
            detection['matched_annotation'] = None

    return detections


# ─────────────────────────────────────────────
# STEP 4: PROCESS ALL FRAMES IN A SPLIT
# ─────────────────────────────────────────────

def process_split(model: YOLO,
                  split_name: str,
                  frames_dir: Path,
                  detections_dir: Path,
                  annotations: dict = None) -> dict:
    """
    Run YOLO on every extracted frame in a split directory.

    Directory structure expected (from extract_frames.py):
        frames_dir/
            <split_name>/
                <video_id>/
                    <video_id>_<timestamp>.jpg

    JSON output structure:
        detections_dir/
            <split_name>/
                <video_id>/
                    <video_id>_<timestamp>.json

    Args:
        annotations: optional dict from extract_frames.py results,
                     keyed as results[video_id]['annotations'].
                     If provided, detections are matched to AVA labels.

    Returns:
        summary dict: video_id -> {
            'total_frames':     int,
            'total_detections': int,
            'frames': {
                timestamp -> detection list
            }
        }
    """
    split_frames_dir     = frames_dir / split_name
    split_detections_dir = detections_dir / split_name

    if not split_frames_dir.exists():
        print(f"  [!] Frames dir not found: {split_frames_dir}")
        return {}

    video_dirs = sorted(split_frames_dir.iterdir())
    if not video_dirs:
        print(f"  [!] No video folders found in {split_frames_dir}")
        return {}

    print(f"\n{'='*52}")
    print(f"  Detecting objects — split: {split_name.upper()}")
    print(f"  Videos to process: {len(video_dirs)}")
    print(f"{'='*52}")

    summary = {}
    t_start = time.time()

    for v_idx, video_dir in enumerate(video_dirs):
        if not video_dir.is_dir():
            continue

        video_id   = video_dir.name
        frame_files = sorted(video_dir.glob("*.jpg"))

        if not frame_files:
            print(f"  [{v_idx+1}] {video_id} — no frames found, skipping")
            continue

        # Output dir for this video's JSONs
        out_video_dir = split_detections_dir / video_id
        out_video_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{v_idx+1}/{len(video_dirs)}] {video_id} "
              f"— {len(frame_files)} frames")

        video_summary = {
            'total_frames':     len(frame_files),
            'total_detections': 0,
            'frames':           {}
        }

        for frame_path in frame_files:
            json_path = out_video_dir / frame_path.with_suffix('.json').name

            # Resume support — skip already detected frames
            if json_path.exists():
                with open(json_path) as f:
                    cached = json.load(f)

                # If AVA annotations are available but missing from cached detections,
                # regenerate this frame so action relations can be built downstream.
                has_matched_annotations = any(
                    isinstance(det.get('matched_annotation'), dict)
                    for det in cached.get('detections', [])
                )
                if not (annotations and not has_matched_annotations):
                    ts = cached.get('timestamp')
                    video_summary['frames'][ts] = cached['detections']
                    video_summary['total_detections'] += len(cached['detections'])
                    continue

            # Parse timestamp from filename e.g. "2DUITARAsWQ_0902.jpg"
            try:
                ts = int(frame_path.stem.split('_')[-1])
            except ValueError:
                print(f"      [!] Could not parse timestamp "
                      f"from {frame_path.name}, skipping")
                continue

            # Run YOLO
            detections = detect_frame(model, frame_path)

            # Match to AVA annotations if provided
            if annotations and video_id in annotations:
                ann_for_ts = annotations[video_id].get(ts, [])
                if ann_for_ts:
                    # Get image dimensions from first detection result
                    # (re-probe if needed)
                    try:
                        from PIL import Image
                        img        = Image.open(frame_path)
                        img_w, img_h = img.size
                    except Exception:
                        img_w, img_h = 1280, 720  # safe fallback

                    detections = match_detections_to_annotations(
                        detections, ann_for_ts, img_w, img_h
                    )

            # Build JSON record for this frame
            frame_record = {
                'video_id':   video_id,
                'timestamp':  ts,
                'frame_file': frame_path.name,
                'detections': detections
            }

            with open(json_path, 'w') as f:
                json.dump(frame_record, f, indent=2)

            video_summary['frames'][ts]      = detections
            video_summary['total_detections'] += len(detections)

        summary[video_id] = video_summary

        avg_det = (video_summary['total_detections'] /
                   max(video_summary['total_frames'], 1))
        print(f"      Detections : {video_summary['total_detections']} "
              f"total  ({avg_det:.1f} avg/frame)")
        print(f"      JSONs saved: {out_video_dir}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  VidQuery — YOLO Object Detection")
    print("=" * 52)
    print(f"  Frames dir     : {FRAMES_DIR}")
    print(f"  Detections dir : {DETECTIONS_DIR}")
    print(f"  Model          : {MODEL_SIZE}")
    print(f"  Conf threshold : {CONFIDENCE_THRESH}")
    print(f"  IOU threshold  : {IOU_THRESH}")
    print()

    # Validate frames dir exists
    if not FRAMES_DIR.exists():
        print(f"[ERROR] Frames directory not found: {FRAMES_DIR}")
        print("Run extract_frames.py first.")
        return

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once — reused across both splits
    model = load_model(MODEL_SIZE)

    split_annotations = {
        "train": load_ava_annotations(TRAIN_CSV),
        "val": load_ava_annotations(VAL_CSV),
    }

    # Process both splits
    for split in ["train", "val"]:
        split_dir = FRAMES_DIR / split
        if not split_dir.exists():
            print(f"\n[SKIP] {split} split not found at {split_dir}")
            continue

        results = process_split(
            model          = model,
            split_name     = split,
            frames_dir     = FRAMES_DIR,
            detections_dir = DETECTIONS_DIR,
            annotations    = split_annotations.get(split)
        )

        # Print split summary
        total_frames = sum(r['total_frames']     for r in results.values())
        total_dets   = sum(r['total_detections'] for r in results.values())

        print(f"\n  {split.upper()} SUMMARY")
        print(f"  Videos    : {len(results)}")
        print(f"  Frames    : {total_frames}")
        print(f"  Detections: {total_dets}")
        print(f"  Avg/frame : {total_dets / max(total_frames, 1):.1f}")

    print(f"\n{'='*52}")
    print(f"  DETECTION COMPLETE")
    print(f"  Results saved to: {DETECTIONS_DIR}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()