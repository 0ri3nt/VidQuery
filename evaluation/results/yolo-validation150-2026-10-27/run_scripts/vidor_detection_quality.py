"""Labelled detection quality of YOLOv8n vs YOLOv8s on held-out VidOR frames.

Same frames (vidquery.vidor_relation._selected_frames, 6 per video), thresholds
(DETECTION_CONFIDENCE/IOU from settings) and class-aware greedy IoU >= 0.5 matching
(vidquery.vidor_improved.match_detections_to_annotations) as the earlier
27.5% relation-recall measurement. Held-out = the manifest's validation + test videos.

Usage: vidor_detection_quality.py OUT.json
"""

import json
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

from vidquery.config import get_settings
from vidquery.vidor_improved import _frame_path, match_detections_to_annotations
from vidquery.vidor_relation import _selected_frames, mapped_vidor_class

FRAMES = Path("data/source/vidor/selected-frames")
ANNOTATIONS = Path("data/source/vidor/training-annotation.zip")
DETECTORS = {"yolov8n": "data/app/models/yolo/yolov8n.pt", "yolov8s": "data/app/models/yolo/yolov8s.pt"}


def detect(model, settings, path):
    result = model.predict(source=str(path), conf=settings.detection_confidence,
                           iou=settings.detection_iou, verbose=False)[0]
    height, width = result.orig_shape
    boxes = result.boxes if result.boxes is not None else []
    return [([float(v) for v in box.xyxy[0].tolist()], result.names[int(box.cls.item())],
             float(box.conf.item())) for box in boxes], width, height


def main(out: Path) -> None:
    settings = get_settings()
    manifest = json.loads((FRAMES / "split_manifest.json").read_text(encoding="utf-8"))
    source = json.loads(Path(manifest["source_split_manifest"]).read_text(encoding="utf-8"))
    support: Counter[str] = Counter()
    for row in source["selected_videos"]:
        support.update(row["support"])
    held_out = set(map(str, manifest["validation"])) | set(map(str, manifest["test"]))
    members = sorted(r["member"] for r in manifest["selected_videos"] if str(r["video_id"]) in held_out)
    models = {name: YOLO(path) for name, path in DETECTORS.items()}
    per_video: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    seconds: Counter[str] = Counter()
    frames_used = 0
    with zipfile.ZipFile(ANNOTATIONS) as archive:
        for member in members:
            payload = json.loads(archive.read(member))
            video = str(payload["video_id"])
            categories = {int(i["tid"]): str(i["category"]) for i in payload["subject/objects"]}
            for frame_id in _selected_frames(payload["relation_instances"], support, 6):
                if frame_id >= len(payload["trajectories"]):
                    continue
                observations = sorted((r for r in payload["trajectories"][frame_id]
                                       if int(r["tid"]) in categories), key=lambda r: int(r["tid"]))
                if not observations:
                    continue
                try:
                    path = _frame_path(FRAMES, payload, frame_id)
                except FileNotFoundError:
                    continue
                if not Path(path).is_file():
                    continue
                frames_used += 1
                tids = {int(r["tid"]) for r in observations}
                relations = [r for r in payload["relation_instances"]
                             if int(r["begin_fid"]) <= frame_id < int(r["end_fid"])
                             and {int(r["subject_tid"]), int(r["object_tid"])} <= tids]
                for name, model in models.items():
                    started = time.perf_counter()
                    pixel, width, height = detect(model, settings, path)
                    seconds[name] += time.perf_counter() - started
                    normalized = [([b[0] / width, b[1] / height, b[2] / width, b[3] / height], label,
                                   conf) for b, label, conf in pixel]
                    matched = match_detections_to_annotations(
                        observations, categories, normalized,
                        width=float(payload["width"]), height=float(payload["height"]))
                    counts = per_video[video][name]
                    counts["gt_objects"] += len(observations)
                    counts["matched_objects"] += len(matched)
                    counts["gt_relations"] += len(relations)
                    counts["relations_both_matched"] += sum(
                        int(r["subject_tid"]) in matched and int(r["object_tid"]) in matched
                        for r in relations)
                    counts["mappable_detections"] += sum(
                        mapped_vidor_class(label) is not None for _, label, _ in normalized)
                    counts["detections"] += len(normalized)

    def rates(videos, name):
        total = Counter()
        for v in videos:
            total.update(per_video[v][name])
        return {"object_recall": total["matched_objects"] / max(1, total["gt_objects"]),
                "relation_recall": total["relations_both_matched"] / max(1, total["gt_relations"]),
                "matched_share_of_mappable_detections":
                    total["matched_objects"] / max(1, total["mappable_detections"]),
                "detections_per_frame": total["detections"] / max(1, frames_used)}

    videos = sorted(per_video)
    report = {"held_out_videos": len(videos), "frames": frames_used,
              "confidence": settings.detection_confidence, "iou": settings.detection_iou,
              "metrics": {name: rates(videos, name) for name in DETECTORS},
              "ms_per_frame": {name: round(1000 * seconds[name] / max(1, frames_used), 1)
                               for name in DETECTORS},
              "counts": {name: dict(sum((per_video[v][name] for v in videos), Counter()))
                         for name in DETECTORS}}
    rng = random.Random(2026)
    diffs = defaultdict(list)
    for _ in range(2000):
        sample = [rng.choice(videos) for _ in videos]
        a, b = rates(sample, "yolov8n"), rates(sample, "yolov8s")
        for key in ("object_recall", "relation_recall", "matched_share_of_mappable_detections"):
            diffs[key].append(b[key] - a[key])
    report["yolov8s_minus_yolov8n"] = {
        key: {"difference": round(report["metrics"]["yolov8s"][key] - report["metrics"]["yolov8n"][key], 4),
              "ci95": [round(sorted(v)[50], 4), round(sorted(v)[1949], 4)]}
        for key, v in diffs.items()}
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("held_out_videos", "frames", "metrics", "ms_per_frame",
                                               "yolov8s_minus_yolov8n")}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
