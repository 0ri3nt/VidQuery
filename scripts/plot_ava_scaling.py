from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SERIES = (
    ("Validation supported macro F1", "validation_metrics", "macro_f1_supported_classes"),
    ("Test supported macro F1", "test_metrics", "macro_f1_supported_classes"),
    ("Test micro F1", "test_metrics", "micro_f1"),
    ("Test mAP", "test_metrics", "mean_average_precision"),
)


def _metric(run: dict[str, Any], group: str, name: str) -> float:
    return float(run[group][name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render measured AVA scaling CSV and SVG")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/app/models/ava80_scaling/scaling_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    sizes = sorted(int(value) for value in report["runs"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "ava-data-scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "corpus_videos",
                "train_videos",
                "train_graphs",
                "train_supervised_persons",
                "train_action_rows",
                *(name for name, _, _ in SERIES),
            ]
        )
        for size in sizes:
            run = report["runs"][str(size)]
            train = run["corpus_statistics"]["train"]
            writer.writerow(
                [
                    size,
                    train["videos"],
                    train["timestamp_graphs"],
                    train["supervised_persons"],
                    train["action_rows"],
                    *(_metric(run, group, name) for _, group, name in SERIES),
                ]
            )

    width, height = 960, 520
    panel_width, panel_height = 430, 190
    positions = ((55, 55), (505, 55), (55, 295), (505, 295))
    colours = ("#2563eb", "#7c3aed", "#059669", "#d97706")
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<title>AVA action GNN data scaling from 17 to 30 videos</title>",
        (
            "<desc>Four measured validation and held-out test metrics using the same "
            "fixed validation and test videos.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, sans-serif" fill="#111827">',
    ]
    for panel_index, ((label, group, metric_name), (left, top), colour) in enumerate(
        zip(SERIES, positions, colours, strict=True)
    ):
        del panel_index
        values = [_metric(report["runs"][str(size)], group, metric_name) for size in sizes]
        ceiling = max(values) * 1.2 if max(values) > 0 else 1.0
        x1, x2 = left + 75, left + panel_width - 35
        baseline = top + panel_height - 35
        chart_height = panel_height - 75
        points = []
        for index, value in enumerate(values):
            x = x1 if len(values) == 1 else x1 + (x2 - x1) * index / (len(values) - 1)
            y = baseline - chart_height * value / ceiling
            points.append((x, y, value, sizes[index]))
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
        svg.extend(
            [
                f'<text x="{left}" y="{top + 16}" font-size="16" font-weight="600">{label}</text>',
                f'<line x1="{x1}" y1="{baseline}" x2="{x2}" y2="{baseline}" stroke="#d1d5db"/>',
                (
                    f'<polyline points="{point_text}" fill="none" stroke="{colour}" '
                    'stroke-width="3"/>'
                ),
            ]
        )
        for x, y, value, size in points:
            svg.extend(
                [
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>',
                    (
                        f'<text x="{x:.1f}" y="{y - 11:.1f}" font-size="13" '
                        f'text-anchor="middle">{value:.6f}</text>'
                    ),
                    (
                        f'<text x="{x:.1f}" y="{baseline + 22}" font-size="12" '
                        f'text-anchor="middle">{size} videos</text>'
                    ),
                ]
            )
    svg.extend(
        [
            (
                '<text x="480" y="510" font-size="12" text-anchor="middle" '
                'fill="#4b5563">Fixed historical validation/test videos; checkpoint '
                "selection uses validation only.</text>"
            ),
            "</g>",
            "</svg>",
        ]
    )
    svg_path = args.output_dir / "ava-data-scaling.svg"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(json.dumps({"csv": csv_path.as_posix(), "svg": svg_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
