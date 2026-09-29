import argparse
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

AVA_ACTIONS = {
    1: "bend/bow",
    3: "crouch/kneel",
    5: "fall down",
    6: "get up",
    8: "jump/leap",
    9: "lie/sleep",
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


class VideoPreprocessor:
    def __init__(
        self,
        video_dir: Path = Path("data/ava/videos"),
        train_csv: Path = Path("data/ava/annotations/ava_train_v2.2.csv"),
        val_csv: Path = Path("data/ava/annotations/ava_val_v2.2.csv"),
        output_frame_dir: Path = Path("data/ava/extracted_frames"),
        output_audio_dir: Path = Path("data/ava/audio"),
        max_videos: int | None = 20,
    ):
        self.video_dir = video_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.output_frame_dir = output_frame_dir
        self.output_audio_dir = output_audio_dir
        self.max_videos = max_videos

    @staticmethod
    def _resolve_ffmpeg_binary() -> str:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            return ffmpeg_bin

        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError(
                "ffmpeg executable not found. Install ffmpeg or imageio-ffmpeg."
            ) from exc

    @staticmethod
    def load_ava_annotations(csv_path: Path):
        df = pd.read_csv(
            csv_path,
            header=None,
            names=[
                "video_id",
                "timestamp",
                "x1",
                "y1",
                "x2",
                "y2",
                "action_id",
                "entity_id",
            ],
        )

        df["timestamp"] = df["timestamp"].astype(str).str.strip()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype(int)

        timestamp_map = defaultdict(set)
        annotation_map = defaultdict(list)

        for _, row in df.iterrows():
            vid = row["video_id"].strip()
            ts = row["timestamp"]

            timestamp_map[vid].add(ts)
            annotation_map[(vid, ts)].append(
                {
                    "bbox": (row["x1"], row["y1"], row["x2"], row["y2"]),
                    "action_id": int(row["action_id"]),
                    "action_label": AVA_ACTIONS.get(int(row["action_id"]), "other"),
                    "entity_id": int(row["entity_id"]),
                }
            )

        return timestamp_map, annotation_map

    def extract_frames(
        self,
        video_path: Path,
        timestamps: set[int] | None = None,
        output_dir: Path | None = None,
    ) -> dict[int, str]:
        video_id = video_path.stem
        frame_root = output_dir or self.output_frame_dir
        video_out = frame_root / video_id
        video_out.mkdir(parents=True, exist_ok=True)

        if timestamps is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 30.0

            saved_frames: dict[int, str] = {}
            last_saved_second = -1
            frame_idx = 0

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    second = int(frame_idx / fps)
                    if second > last_saved_second:
                        out_path = video_out / f"{video_id}_{second:04d}.jpg"
                        if not out_path.exists():
                            cv2.imwrite(str(out_path), frame)
                        saved_frames[second] = str(out_path)
                        last_saved_second = second

                    frame_idx += 1
            finally:
                cap.release()

            return saved_frames

        ffmpeg_bin = self._resolve_ffmpeg_binary()

        saved_frames: dict[int, str] = {}
        for ts in sorted(timestamps):
            out_path = video_out / f"{video_id}_{ts:04d}.jpg"
            if out_path.exists():
                saved_frames[ts] = str(out_path)
                continue

            result = subprocess.run(
                [
                    ffmpeg_bin,
                    "-loglevel",
                    "error",
                    "-ss",
                    str(ts),
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-y",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and out_path.is_file():
                saved_frames[ts] = str(out_path)
            else:
                print(f"  [!] FFmpeg error at {video_id} t={ts}s: {result.stderr[-500:]}")

        return saved_frames

    def extract_audio(self, video_path: Path) -> Path:
        self.output_audio_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_audio_dir / f"{video_path.stem}.wav"
        if out_path.exists():
            return out_path

        ffmpeg_bin = self._resolve_ffmpeg_binary()

        result = subprocess.run(
            [
                ffmpeg_bin,
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-y",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr[-500:]}")
        return out_path

    def process_split(self, split_name: str, csv_path: Path) -> dict:
        print(f"\n{'=' * 50}")
        print(f"  Processing split: {split_name.upper()}")
        print(f"{'=' * 50}")

        timestamp_map, annotation_map = self.load_ava_annotations(csv_path)

        split_out = self.output_frame_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)

        available_videos = sorted(
            path
            for path in self.video_dir.iterdir()
            if path.suffix.lower() in {".mp4", ".mkv", ".webm"}
        )
        if not available_videos:
            print(f"  [!] No .mp4 files found in {self.video_dir}")
            return {}

        if self.max_videos:
            available_videos = available_videos[: self.max_videos]

        matched_videos = [p for p in available_videos if p.stem in timestamp_map]

        results = {}
        for i, video_path in enumerate(matched_videos):
            video_id = video_path.stem
            timestamps = timestamp_map[video_id]

            print(
                f"  [{i + 1}/{len(matched_videos)}] {video_id} "
                f"- {len(timestamps)} annotated timestamps"
            )

            saved_frames = self.extract_frames(video_path, timestamps, split_out)
            annotations = {
                ts: annotation_map[(video_id, ts)]
                for ts in timestamps
                if ts in saved_frames
            }

            results[video_id] = {
                "frames": saved_frames,
                "annotations": annotations,
            }

            print(f"      Saved {len(saved_frames)} frames to {split_out / video_id}")

        print(f"\n  Done. Processed {len(results)} videos for {split_name} split.")
        return results

    def run(self) -> tuple[dict, dict]:
        print("VidQuery - AVA Frame Extraction Pipeline")
        print(f"  Video dir  : {self.video_dir}")
        print(f"  Train CSV  : {self.train_csv}")
        print(f"  Val CSV    : {self.val_csv}")
        print(f"  Output dir : {self.output_frame_dir}")
        print(f"  Max videos : {self.max_videos}")

        missing = [
            p for p in [self.video_dir, self.train_csv, self.val_csv] if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing required paths: {missing}")

        self.output_frame_dir.mkdir(parents=True, exist_ok=True)
        train_results = self.process_split("train", self.train_csv)
        val_results = self.process_split("val", self.val_csv)

        return train_results, val_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AVA frames and audio")
    parser.add_argument("--video_dir", type=Path, default=Path("data/ava/videos"))
    parser.add_argument(
        "--train_csv", type=Path, default=Path("data/ava/annotations/ava_train_v2.2.csv")
    )
    parser.add_argument(
        "--val_csv", type=Path, default=Path("data/ava/annotations/ava_val_v2.2.csv")
    )
    parser.add_argument(
        "--output_frame_dir", type=Path, default=Path("data/ava/extracted_frames")
    )
    parser.add_argument("--max_videos", type=int, default=20)
    args = parser.parse_args()

    processor = VideoPreprocessor(
        video_dir=args.video_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_frame_dir=args.output_frame_dir,
        max_videos=args.max_videos,
    )
    processor.run()


if __name__ == "__main__":
    main()
