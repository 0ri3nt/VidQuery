import ffmpeg
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# CONFIG — edit these paths before running
# ─────────────────────────────────────────────

VIDEO_DIR        = Path("data/ava/videos")
TRAIN_CSV        = Path("data/ava/annotations/ava_train_v2.2.csv")
VAL_CSV          = Path("data/ava/annotations/ava_val_v2.2.csv")
OUTPUT_FRAME_DIR = Path("data/ava/extracted_frames")
MAX_VIDEOS       = 20   # set to None to process all


# ─────────────────────────────────────────────
# AVA ACTION LABELS (subset of most relevant
# ones for meeting/lecture search)
# ─────────────────────────────────────────────

AVA_ACTIONS = {
    1:  "bend/bow",
    3:  "crouch/kneel",
    5:  "fall down",
    6:  "get up",
    8:  "jump/leap",
    9:  "lie/sleep",
    12: "sit",
    13: "stand",
    15: "walk",
    17: "write",
    18: "answer phone",
    22: "eat",
    26: "listen to",
    27: "open",
    28: "play musical instrument",
    29: "point to",
    30: "press",
    31: "pull",
    32: "push",
    36: "talk to",
    37: "text on phone",
    44: "use laptop",
    79: "carry/hold",
    80: "watch",
}


# ─────────────────────────────────────────────
# STEP 1: PARSE ANNOTATIONS
# ─────────────────────────────────────────────

def load_ava_annotations(csv_path: Path):
    """
    Parse AVA CSV into:
      - timestamp_map:  video_id -> set of integer timestamps
      - annotation_map: (video_id, timestamp) -> list of annotation dicts

    CSV format per row:
      video_id, timestamp, x1, y1, x2, y2, action_id, entity_id
    """
    df = pd.read_csv(csv_path, header=None, names=[
        'video_id',
        'timestamp',    # zero-padded string e.g. "0902"
        'x1', 'y1', 'x2', 'y2',
        'action_id',
        'entity_id'     # track ID for same person across frames
    ])

    # Convert timestamp to int cleanly (handles zero-padding)
    df['timestamp'] = df['timestamp'].astype(str).str.strip()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].astype(int)

    timestamp_map  = defaultdict(set)
    annotation_map = defaultdict(list)

    for _, row in df.iterrows():
        vid = row['video_id'].strip()
        ts  = row['timestamp']

        timestamp_map[vid].add(ts)

        annotation_map[(vid, ts)].append({
            'bbox':      (row['x1'], row['y1'],
                          row['x2'], row['y2']),
            'action_id': int(row['action_id']),
            'action_label': AVA_ACTIONS.get(
                                int(row['action_id']), 'other'),
            'entity_id': int(row['entity_id'])
        })

    print(f"[Annotations] Loaded {len(timestamp_map)} videos "
          f"from {csv_path.name}")
    return timestamp_map, annotation_map


# ─────────────────────────────────────────────
# STEP 2: EXTRACT ANNOTATED FRAMES
# ─────────────────────────────────────────────
import subprocess
import json

def probe_video(video_path: str) -> tuple[int, int]:
    """
    Get video dimensions using ffprobe directly via subprocess.
    Returns (width, height).
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        str(video_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        probe_data = json.loads(result.stdout)
        video_stream = next(
            s for s in probe_data['streams']
            if s['codec_type'] == 'video'
        )
        return int(video_stream['width']), int(video_stream['height'])

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {e.stderr}"
        )
    except StopIteration:
        raise RuntimeError(
            f"No video stream found in {video_path}"
        )

def extract_annotated_frames(video_path: Path,
                              timestamps: set,
                              output_dir: Path) -> dict:
    """
    For a single video, extract only the frames at annotated
    timestamps using FFmpeg and save them to output_dir.

    Output directory structure:
        output_dir/
            <video_id>/
                <video_id>_<timestamp>.jpg

    Returns:
        dict: timestamp (int) -> saved frame path (str)
    """
    video_id   = video_path.stem
    video_out  = output_dir / video_id
    video_out.mkdir(parents=True, exist_ok=True)

    # Probe for dimensions — used to validate frame output
    try:
        probe      = ffmpeg.probe(str(video_path))
        video_info = next(
            s for s in probe['streams']
            if s['codec_type'] == 'video'
        )
        width  = int(video_info['width'])
        height = int(video_info['height'])
    except Exception as e:
        print(f"  [!] Could not probe {video_id}: {e}")
        return {}

    saved_frames = {}

    for ts in sorted(timestamps):
        out_path = video_out / f"{video_id}_{ts:04d}.jpg"

        # Skip if already extracted (allows resuming interrupted runs)
        if out_path.exists():
            saved_frames[ts] = str(out_path)
            continue

        try:
            (
                ffmpeg
                .input(str(video_path), ss=ts)  # fast seek
                .output(str(out_path), vframes=1, q=2)
                .overwrite_output()
                .run(quiet=True)
            )
            saved_frames[ts] = str(out_path)

        except ffmpeg.Error as e:
            print(f"  [!] FFmpeg error at {video_id} "
                  f"t={ts}s: {e.stderr.decode()}")

    return saved_frames


# ─────────────────────────────────────────────
# STEP 3: PROCESS A SPLIT (TRAIN OR VAL)
# ─────────────────────────────────────────────

def process_split(split_name: str,
                  csv_path: Path,
                  video_dir: Path,
                  output_dir: Path,
                  max_videos: int = None) -> dict:
    """
    Process one dataset split (train or val).
    Only processes videos that are physically present in video_dir,
    regardless of how many are listed in the CSV.
    """
    print(f"\n{'='*50}")
    print(f"  Processing split: {split_name.upper()}")
    print(f"{'='*50}")

    timestamp_map, annotation_map = load_ava_annotations(csv_path)

    split_out = output_dir / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    # ── KEY CHANGE: drive the loop from disk, not from CSV ──
    available_videos = list(video_dir.glob("*.mp4"))

    if not available_videos:
        print(f"  [!] No .mp4 files found in {video_dir}")
        return {}

    if max_videos:
        available_videos = available_videos[:max_videos]

    # Warn if a video on disk has no annotations in the CSV
    # (wrong split, or file renamed, etc.)
    no_annotation = [
        p.stem for p in available_videos
        if p.stem not in timestamp_map
    ]
    if no_annotation:
        print(f"  [!] {len(no_annotation)} videos on disk have no "
              f"annotations in {csv_path.name}: {no_annotation}")

    # Only keep videos that have annotations
    matched_videos = [
        p for p in available_videos
        if p.stem in timestamp_map
    ]

    print(f"  Videos found on disk       : {len(available_videos)}")
    print(f"  Videos with CSV annotations: {len(matched_videos)}")

    results = {}

    for i, video_path in enumerate(matched_videos):
        video_id   = video_path.stem
        timestamps = timestamp_map[video_id]

        print(f"  [{i+1}/{len(matched_videos)}] {video_id} "
              f"— {len(timestamps)} annotated timestamps")

        saved_frames = extract_annotated_frames(
            video_path = video_path,
            timestamps = timestamps,
            output_dir = split_out
        )

        annotations = {
            ts: annotation_map[(video_id, ts)]
            for ts in timestamps
            if ts in saved_frames
        }

        results[video_id] = {
            'frames':      saved_frames,
            'annotations': annotations
        }

        print(f"      Saved {len(saved_frames)} frames "
              f"to {split_out / video_id}")

    print(f"\n  Done. Processed {len(results)} videos "
          f"for {split_name} split.")
    return results


def main():
    print("VidQuery — AVA Frame Extraction Pipeline")
    print(f"  Video dir  : {VIDEO_DIR}")
    print(f"  Train CSV  : {TRAIN_CSV}")
    print(f"  Val CSV    : {VAL_CSV}")
    print(f"  Output dir : {OUTPUT_FRAME_DIR}")
    print(f"  Max videos : {MAX_VIDEOS}")

    # Validate paths
    missing = [p for p in [VIDEO_DIR, TRAIN_CSV, VAL_CSV]
               if not p.exists()]
    if missing:
        print(f"\n[ERROR] Missing paths: {missing}")
        print("Please update the CONFIG section at the top.")
        return

    # Quick sanity check — tell the user what's on disk upfront
    all_videos = list(VIDEO_DIR.glob("*.mp4"))
    if not all_videos:
        print(f"\n[ERROR] No .mp4 files found in {VIDEO_DIR}")
        return

    print(f"\n  Found {len(all_videos)} video(s) on disk:")
    for v in all_videos:
        print(f"    - {v.name}")

    OUTPUT_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    train_results = process_split(
        split_name = "train",
        csv_path   = TRAIN_CSV,
        video_dir  = VIDEO_DIR,
        output_dir = OUTPUT_FRAME_DIR,
        max_videos = MAX_VIDEOS
    )

    val_results = process_split(
        split_name = "val",
        csv_path   = VAL_CSV,
        video_dir  = VIDEO_DIR,
        output_dir = OUTPUT_FRAME_DIR,
        max_videos = MAX_VIDEOS
    )

    # Summary
    total_train_frames = sum(
        len(v['frames']) for v in train_results.values()
    )
    total_val_frames = sum(
        len(v['frames']) for v in val_results.values()
    )

    print(f"\n{'='*50}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Train: {len(train_results)} videos, "
          f"{total_train_frames} frames")
    print(f"  Val  : {len(val_results)} videos, "
          f"{total_val_frames} frames")
    print(f"  Frames saved to: {OUTPUT_FRAME_DIR}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()