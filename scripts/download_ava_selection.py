from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from vidquery.ava_coverage import load_video_coverage, select_coverage_videos

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm"}


def _valid_media(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as stream:
        header = stream.read(64)
    return b"ftyp" in header or header.startswith(b"\x1aE\xdf\xa3")


def _download(url: str, destination: Path, retries: int = 3) -> dict[str, Any]:
    if _valid_media(destination):
        return {"status": "cached", "bytes": destination.stat().st_size}
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = "download did not start"
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "VidQuery-AVA-Coverage/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = existing > 0 and response.status == 206
                mode = "ab" if append else "wb"
                if not append:
                    existing = 0
                with partial.open(mode) as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if not _valid_media(partial):
                raise RuntimeError("downloaded file is not a valid supported media container")
            os.replace(partial, destination)
            return {
                "status": "downloaded",
                "bytes": destination.stat().st_size,
                "attempts": attempt,
            }
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2**attempt)
    return {
        "status": "failed",
        "error": last_error,
        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
    }


def _download_batch(
    video_ids: list[str], inventory: dict[str, dict[str, Any]], output_dir: Path, workers: int
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download,
                str(inventory[video_id]["url"]),
                output_dir / str(inventory[video_id]["filename"]),
            ): video_id
            for video_id in video_ids
        }
        for future in as_completed(futures):
            video_id = futures[future]
            result = future.result()
            results[video_id] = result
            detail = (
                f" ({result.get('bytes', 0) / 1024**2:.1f} MiB)"
                if result["status"] != "failed"
                else f" - {result['error']}"
            )
            print(
                f"[{len(results)}/{len(video_ids)}] {video_id}: {result['status']}{detail}",
                flush=True,
            )
    return results


def _fetch_inventory() -> dict[str, dict[str, Any]]:
    url = "https://s3.amazonaws.com/ava-dataset?list-type=2&prefix=trainval/"
    request = urllib.request.Request(url, headers={"User-Agent": "VidQuery-AVA-Coverage/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ElementTree.fromstring(response.read())
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    inventory: dict[str, dict[str, Any]] = {}
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", default="", namespaces=namespace)
        suffix = Path(key).suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            continue
        filename = Path(key).name
        video_id = Path(filename).stem
        inventory[video_id] = {
            "filename": filename,
            "bytes": int(item.findtext("s3:Size", default="0", namespaces=namespace)),
            "url": f"https://s3.amazonaws.com/ava-dataset/{key}",
        }
    return inventory


def _available_local_ids(directory: Path) -> set[str]:
    return {
        path.stem
        for path in directory.iterdir()
        if path.suffix.lower() in MEDIA_EXTENSIONS and _valid_media(path)
    } if directory.is_dir() else set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the selected AVA coverage videos")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("evaluation/results/ava-expansion-selection.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ava/videos"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--replacement-rounds", type=int, default=3)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/ava-expansion-download.json"),
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    inventory = _fetch_inventory()
    if args.probe_only:
        video_ids = list(selection["selected_video_ids"])
        report = {
            "videos": [
                {
                    "video_id": video_id,
                    **inventory.get(video_id, {"bytes": None, "error": "not in S3 inventory"}),
                }
                for video_id in video_ids
            ],
            "total_bytes": sum(
                int(inventory.get(video_id, {}).get("bytes", 0)) for video_id in video_ids
            ),
        }
        print(json.dumps(report, indent=2))
        return
    annotations = [
        Path("data/ava/annotations/ava_train_v2.2.csv"),
        Path("data/ava/annotations/ava_val_v2.2.csv"),
    ]
    coverage = load_video_coverage(annotations)
    requested = list(selection["selected_video_ids"])
    unavailable: set[str] = set()
    all_results: dict[str, dict[str, Any]] = {}
    rounds: list[dict[str, Any]] = []

    for round_number in range(args.replacement_rounds + 1):
        pending = [
            video_id
            for video_id in requested
            if video_id not in all_results or all_results[video_id]["status"] == "failed"
        ]
        if not pending:
            break
        missing_from_inventory = [video_id for video_id in pending if video_id not in inventory]
        downloadable = [video_id for video_id in pending if video_id in inventory]
        results = _download_batch(
            downloadable, inventory, args.output_dir, max(1, args.workers)
        )
        results.update({
            video_id: {"status": "failed", "error": "not in official S3 inventory"}
            for video_id in missing_from_inventory
        })
        all_results.update(results)
        failures = {
            video_id
            for video_id, result in results.items()
            if result["status"] == "failed"
        }
        unavailable.update(failures)
        rounds.append({"round": round_number + 1, "requested": pending, "results": results})
        if not failures:
            break
        available = _available_local_ids(args.output_dir)
        replacement = select_coverage_videos(
            coverage,
            available,
            excluded_video_ids=unavailable,
            rare_max_support=int(selection["rare_max_support"]),
            target_support=int(selection["target_support"]),
            max_additional_videos=len(failures) + 10,
        )
        requested = list(replacement["selected_video_ids"])
        if not requested:
            break

    available = _available_local_ids(args.output_dir)
    final_coverage = select_coverage_videos(
        coverage,
        available,
        excluded_video_ids=unavailable,
        rare_max_support=int(selection["rare_max_support"]),
        target_support=int(selection["target_support"]),
        max_additional_videos=0,
    )
    report = {
        "schema_version": "ava-v2.2-expansion-download-v1",
        "selection_manifest": str(args.selection.as_posix()),
        "rounds": rounds,
        "download_results": all_results,
        "unavailable_video_ids": sorted(unavailable),
        "available_video_ids": sorted(available.intersection(coverage)),
        "available_video_count": len(available.intersection(coverage)),
        "actions_present": final_coverage["current_actions_present"],
        "missing_action_ids": final_coverage["missing_action_ids"],
        "extremely_rare_action_ids": final_coverage["extremely_rare_action_ids"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "available_video_count", "actions_present", "missing_action_ids",
        "extremely_rare_action_ids", "unavailable_video_ids"
    )}, indent=2))


if __name__ == "__main__":
    main()
