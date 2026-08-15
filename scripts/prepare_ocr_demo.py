from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    output = Path("data/demo/ocr_demo.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        10.0,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the controlled OCR MP4")
    slides = (
        ("KUBERNETES", "DEPLOYMENT", (34, 76, 52)),
        ("NEO4J", "KNOWLEDGE GRAPH", (45, 48, 70)),
    )
    for title, subtitle, color in slides:
        for _ in range(30):
            frame = np.full((720, 1280, 3), color, dtype=np.uint8)
            cv2.putText(
                frame,
                title,
                (115, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.2,
                (255, 255, 255),
                9,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                subtitle,
                (115, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (190, 245, 255),
                6,
                cv2.LINE_AA,
            )
            writer.write(frame)
    writer.release()
    if output.stat().st_size <= 0:
        raise RuntimeError("controlled OCR MP4 is empty")
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
