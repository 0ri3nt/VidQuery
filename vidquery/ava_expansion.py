"""Incremental, storage-safe AVA feature expansion.

Raw AVA media is temporary. A source file is eligible for deletion only after
its improved per-video graph cache has been atomically written, reloaded, and
validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .ava_coverage import VideoCoverage, load_video_coverage
from .ava_labels import AVA_V22_ACTIONS, load_ava_v22_label_map

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm"}
PER_VIDEO_CACHE_SCHEMA = "ava80-incremental-video-cache-v1"
EXPANSION_SCHEMA = "ava80-storage-aware-expansion-v1"
DEFAULT_MIN_FREE_BYTES = 2 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _combined_support(
    coverage: dict[str, VideoCoverage], video_ids: Iterable[str]
) -> Counter[int]:
    support: Counter[int] = Counter()
    for video_id in video_ids:
        if video_id in coverage:
            support.update(coverage[video_id].action_counts)
    return support


def _action_signature(item: VideoCoverage) -> frozenset[int]:
    return frozenset(action_id for action_id, count in item.action_counts.items() if count)


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select_incremental_videos(
    coverage: dict[str, VideoCoverage],
    current_video_ids: Iterable[str],
    *,
    target_videos: int,
    excluded_video_ids: Iterable[str] = (),
    target_support: int = 25,
) -> dict[str, Any]:
    """Select the most useful remaining videos using only annotation metadata."""

    current = sorted(set(current_video_ids).intersection(coverage))
    if target_videos <= len(current):
        needed = 0
    else:
        needed = target_videos - len(current)
    excluded = set(excluded_video_ids)
    remaining = set(coverage).difference(current).difference(excluded)
    support = _combined_support(coverage, current)
    labels = {item.label_id: item for item in load_ava_v22_label_map()}
    existing_signatures = [_action_signature(coverage[video_id]) for video_id in current]
    split_counts = Counter(coverage[video_id].annotation_split for video_id in current)
    selected: list[str] = []
    selection_reasons: dict[str, dict[str, Any]] = {}

    while len(selected) < needed and remaining:
        best_video = ""
        best_score: tuple[float, float, float, float, str] | None = None
        best_reason: dict[str, Any] = {}
        desired_split = min(("train", "validation"), key=lambda item: split_counts[item])
        for video_id in remaining:
            item = coverage[video_id]
            signature = _action_signature(item)
            rare_gain = 0.0
            expected_added: dict[int, int] = {}
            for action_id, count in item.action_counts.items():
                deficit = max(0, target_support - support[action_id])
                if not deficit:
                    continue
                contribution = min(deficit, count)
                rare_gain += contribution / math.sqrt(support[action_id] + 1.0)
                expected_added[action_id] = contribution
            inverse_frequency_breadth = sum(
                1.0 / math.sqrt(support[action_id] + 1.0) for action_id in signature
            )
            max_similarity = max(
                (_jaccard(signature, known) for known in existing_signatures), default=0.0
            )
            diversity = 1.0 - max_similarity
            split_bonus = 1.0 if item.annotation_split == desired_split else 0.0
            score = (
                rare_gain,
                inverse_frequency_breadth,
                diversity,
                split_bonus,
                video_id,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_video = video_id
                best_reason = {
                    "rare_support_gain": round(rare_gain, 6),
                    "inverse_frequency_breadth": round(
                        inverse_frequency_breadth, 6
                    ),
                    "action_signature_diversity": round(diversity, 6),
                    "annotation_split": item.annotation_split,
                    "expected_added_support": {
                        str(action_id): count
                        for action_id, count in sorted(expected_added.items())
                    },
                }
        if not best_video:
            break
        selected.append(best_video)
        selection_reasons[best_video] = best_reason
        selected_item = coverage[best_video]
        support.update(selected_item.action_counts)
        split_counts[selected_item.annotation_split] += 1
        existing_signatures.append(_action_signature(selected_item))
        remaining.remove(best_video)

    action_rows = []
    initial_support = _combined_support(coverage, current)
    selected_support = _combined_support(coverage, selected)
    for action_id in range(1, 81):
        candidates = sorted(
            (
                item
                for item in coverage.values()
                if item.video_id not in current
                and item.video_id not in excluded
                and item.action_counts[action_id]
            ),
            key=lambda item: (-item.action_counts[action_id], item.video_id),
        )
        action_rows.append(
            {
                "action_id": action_id,
                "official_name": labels[action_id].official_name,
                "canonical_name": AVA_V22_ACTIONS[action_id],
                "label_group": labels[action_id].label_type,
                "current_support": initial_support[action_id],
                "expected_added_support": selected_support[action_id],
                "projected_support": initial_support[action_id]
                + selected_support[action_id],
                "selected_videos": [
                    video_id
                    for video_id in selected
                    if coverage[video_id].action_counts[action_id]
                ],
                "candidate_videos": [
                    {
                        "video_id": item.video_id,
                        "support": item.action_counts[action_id],
                        "annotation_split": item.annotation_split,
                    }
                    for item in candidates[:20]
                ],
            }
        )

    cooccurrence: Counter[tuple[int, int]] = Counter()
    for video_id in [*current, *selected]:
        action_ids = sorted(_action_signature(coverage[video_id]))
        for left_index, left in enumerate(action_ids):
            for right in action_ids[left_index + 1 :]:
                cooccurrence[(left, right)] += 1

    return {
        "schema_version": "ava80-class-aware-selection-v2",
        "selection_method": (
            "greedy_rare_support_inverse_frequency_action_signature_diversity"
        ),
        "scene_diversity_proxy": (
            "action-signature Jaccard novelty; pixels are unavailable before download"
        ),
        "target_videos": target_videos,
        "target_support": target_support,
        "current_video_ids": current,
        "current_video_count": len(current),
        "selected_video_ids": selected,
        "selected_video_count": len(selected),
        "projected_video_count": len(current) + len(selected),
        "selection_reasons": selection_reasons,
        "unavailable_video_ids": sorted(excluded),
        "replacement_video_ids": [],
        "label_group_support": {
            group: sum(
                support[action_id]
                for action_id, label in labels.items()
                if label.label_type == group
            )
            for group in sorted({label.label_type for label in labels.values()})
        },
        "top_action_cooccurrence": [
            {"left_action_id": pair[0], "right_action_id": pair[1], "videos": count}
            for pair, count in cooccurrence.most_common(100)
        ],
        "actions": action_rows,
    }


def fetch_ava_inventory() -> dict[str, dict[str, Any]]:
    url = "https://s3.amazonaws.com/ava-dataset?list-type=2&prefix=trainval/"
    request = urllib.request.Request(
        url, headers={"User-Agent": "VidQuery-AVA-Incremental/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ElementTree.fromstring(response.read())
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    inventory: dict[str, dict[str, Any]] = {}
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", default="", namespaces=namespace)
        if Path(key).suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        filename = Path(key).name
        inventory[Path(filename).stem] = {
            "filename": filename,
            "bytes": int(item.findtext("s3:Size", default="0", namespaces=namespace)),
            "url": f"https://s3.amazonaws.com/ava-dataset/{key}",
        }
    return inventory


def valid_media(path: Path, expected_bytes: int | None = None) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    if expected_bytes and path.stat().st_size != expected_bytes:
        return False
    with path.open("rb") as stream:
        header = stream.read(64)
    return b"ftyp" in header or header.startswith(b"\x1aE\xdf\xa3")


def download_ava_media(
    item: dict[str, Any], destination_dir: Path, retries: int = 3
) -> dict[str, Any]:
    destination = destination_dir / str(item["filename"])
    inventory_bytes = int(item["bytes"])
    if valid_media(destination):
        actual_bytes = destination.stat().st_size
        return {
            "status": "cached",
            "path": destination,
            "bytes": actual_bytes,
            "inventory_bytes": inventory_bytes,
            "inventory_size_matches": actual_bytes == inventory_bytes,
        }
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = "not started"
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "VidQuery-AVA-Incremental/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(str(item["url"]), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                append = existing > 0 and response.status == 206
                response_total: int | None = None
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    total_text = content_range.rsplit("/", 1)[-1]
                    if total_text.isdigit():
                        response_total = int(total_text)
                if response_total is None:
                    content_length = response.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        response_total = int(content_length) + (existing if append else 0)
                with partial.open("ab" if append else "wb") as stream:
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
            if not valid_media(partial, response_total):
                raise RuntimeError("downloaded media failed size/container validation")
            os.replace(partial, destination)
            actual_bytes = destination.stat().st_size
            return {
                "status": "downloaded",
                "path": destination,
                "bytes": actual_bytes,
                "inventory_bytes": inventory_bytes,
                "inventory_size_matches": actual_bytes == inventory_bytes,
                "sha256": _sha256(destination),
                "attempts": attempt,
            }
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                _safe_unlink(partial, destination_dir)
            if attempt < retries:
                time.sleep(2**attempt)
    return {
        "status": "failed",
        "path": destination,
        "error": last_error,
        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
    }


def download_ava_media_with_curl(
    item: dict[str, Any], destination_dir: Path, retries: int = 5
) -> dict[str, Any]:
    """Resume one media file with curl without sparse-file preallocation."""

    curl_binary = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_binary:
        return download_ava_media(item, destination_dir, retries=min(retries, 3))
    destination = destination_dir / str(item["filename"])
    partial = destination.with_suffix(destination.suffix + ".part")
    inventory_bytes = int(item["bytes"])
    if valid_media(destination):
        actual_bytes = destination.stat().st_size
        return {
            "status": "cached",
            "path": destination,
            "bytes": actual_bytes,
            "inventory_bytes": inventory_bytes,
            "inventory_size_matches": actual_bytes == inventory_bytes,
            "downloader": "resumable_curl",
        }
    destination_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            curl_binary,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--retry-max-time",
            "1800",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            str(item["url"]),
        ],
        check=False,
        timeout=1900,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and valid_media(partial):
        os.replace(partial, destination)
        actual_bytes = destination.stat().st_size
        return {
            "status": "downloaded",
            "path": destination,
            "bytes": actual_bytes,
            "inventory_bytes": inventory_bytes,
            "inventory_size_matches": actual_bytes == inventory_bytes,
            "sha256": _sha256(destination),
            "downloader": "resumable_curl",
        }
    curl_error = completed.stderr.strip() or f"curl exited {completed.returncode}"
    fallback = download_ava_media(item, destination_dir, retries=3)
    fallback["curl_error"] = curl_error
    fallback["downloader"] = "python_stream_fallback_after_curl"
    return fallback


def _download_batch(
    video_ids: Sequence[str],
    inventory: dict[str, dict[str, Any]],
    destination_dir: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    if shutil.which("curl.exe") or shutil.which("curl"):
        curl_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    download_ava_media_with_curl,
                    inventory[video_id],
                    destination_dir,
                ): video_id
                for video_id in video_ids
                if video_id in inventory
            }
            for video_id in set(video_ids).difference(inventory):
                curl_results[video_id] = {
                    "status": "failed",
                    "error": "not in S3 inventory",
                }
            for future in as_completed(futures):
                video_id = futures[future]
                curl_results[video_id] = future.result()
                print(
                    f"[AVA download {len(curl_results)}/{len(video_ids)}] {video_id}: "
                    f"{curl_results[video_id]['status']}",
                    flush=True,
                )
        return curl_results

    go_binary = shutil.which("go")
    go_source = Path(__file__).resolve().parents[1] / "download.go"
    if go_binary and go_source.is_file():
        destination_dir.mkdir(parents=True, exist_ok=True)
        downloadable = [video_id for video_id in video_ids if video_id in inventory]
        list_path = destination_dir.parent / "download_list.txt"
        list_path.write_text(
            "".join(f"{inventory[video_id]['filename']}\n" for video_id in downloadable),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        go_cache = Path(__file__).resolve().parents[1] / ".go-cache"
        go_cache.mkdir(parents=True, exist_ok=True)
        environment["GOCACHE"] = str(go_cache)
        completed = subprocess.run(
            [
                go_binary,
                "run",
                str(go_source),
                "-list",
                str(list_path.resolve()),
                "-output-dir",
                str(destination_dir.resolve()),
            ],
            cwd=go_source.parent,
            env=environment,
            check=False,
        )
        go_results: dict[str, dict[str, Any]] = {}
        for video_id in video_ids:
            if video_id not in inventory:
                go_results[video_id] = {
                    "status": "failed",
                    "error": "not in S3 inventory",
                }
                continue
            item = inventory[video_id]
            path = destination_dir / str(item["filename"])
            if valid_media(path):
                expected_bytes = int(item["bytes"])
                go_results[video_id] = {
                    "status": "downloaded",
                    "path": path,
                    "bytes": path.stat().st_size,
                    "inventory_bytes": expected_bytes,
                    "inventory_size_matches": path.stat().st_size == expected_bytes,
                    "sha256": _sha256(path),
                    "downloader": "bounded_go_range_downloader",
                }
            else:
                partial = path.with_suffix(path.suffix + ".part")
                _safe_unlink(partial, destination_dir)
                fallback = download_ava_media(item, destination_dir, retries=3)
                fallback["go_error"] = f"Go downloader exited {completed.returncode}"
                fallback["downloader"] = "python_stream_fallback_after_go"
                go_results[video_id] = fallback
        return go_results

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(download_ava_media, inventory[video_id], destination_dir): video_id
            for video_id in video_ids
            if video_id in inventory
        }
        for video_id in set(video_ids).difference(inventory):
            results[video_id] = {"status": "failed", "error": "not in S3 inventory"}
        for future in as_completed(futures):
            video_id = futures[future]
            results[video_id] = future.result()
            print(
                f"[AVA download {len(results)}/{len(video_ids)}] {video_id}: "
                f"{results[video_id]['status']}",
                flush=True,
            )
    return results


def _storage_safe_batch(
    candidates: Sequence[str],
    inventory: dict[str, dict[str, Any]],
    *,
    destination_dir: Path,
    master_cache: Path,
    batch_size: int,
    min_free_bytes: int,
) -> tuple[list[str], dict[str, Any]]:
    """Choose the highest-ranked batch that leaves room for cache rebuilding."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination_dir).free
    master_rebuild_bytes = master_cache.stat().st_size if master_cache.is_file() else 0
    usable_bytes = max(0, free_bytes - min_free_bytes - master_rebuild_bytes)
    selected: list[str] = []
    estimated_bytes = 0
    skipped: list[dict[str, Any]] = []
    ranked_candidates = sorted(
        enumerate(candidates),
        key=lambda item: (
            item[1] not in inventory,
            int(inventory.get(item[1], {}).get("bytes", 0))
            if item[1] in inventory
            else 0,
            item[0],
        ),
    )
    for _, video_id in ranked_candidates:
        if len(selected) >= batch_size:
            break
        item = inventory.get(video_id)
        if item is None:
            selected.append(video_id)
            continue
        source_bytes = int(item.get("bytes", 0))
        filename = str(item.get("filename", f"{video_id}.mp4"))
        destination = destination_dir / filename
        partial = destination.with_suffix(destination.suffix + ".part")
        already_allocated = max(
            destination.stat().st_size if destination.is_file() else 0,
            partial.stat().st_size if partial.is_file() else 0,
        )
        remaining_download = max(0, source_bytes - already_allocated)
        # Feature creation briefly coexists with raw media. Reserve 15% of the
        # source size plus 256 MiB for frames, tensors, and serialization.
        feature_workspace = int(source_bytes * 0.15) + 256 * 1024**2
        required = remaining_download + feature_workspace
        if estimated_bytes + required <= usable_bytes:
            selected.append(video_id)
            estimated_bytes += required
        else:
            skipped.append(
                {
                    "video_id": video_id,
                    "source_bytes": source_bytes,
                    "estimated_required_bytes": required,
                    "reason": "insufficient_free_space_for_media_features_and_master_rebuild",
                }
            )
    return selected, {
        "free_bytes_before_batch": free_bytes,
        "minimum_free_bytes": min_free_bytes,
        "master_rebuild_reserve_bytes": master_rebuild_bytes,
        "usable_batch_bytes": usable_bytes,
        "estimated_batch_bytes": estimated_bytes,
        "within_class_aware_pool_order": "smallest_source_media_first",
        "skipped_candidates": skipped,
    }


def _permanent_download_failure(report: dict[str, Any]) -> bool:
    error = str(report.get("error", "")).lower()
    return "not in s3 inventory" in error or "404" in error or "not found" in error


def _classify_previous_download_failures(
    batches: Sequence[dict[str, Any]], completed: set[str]
) -> tuple[set[str], set[str]]:
    permanent: set[str] = set()
    transient: set[str] = set()
    for batch in batches:
        for video_id, raw_report in batch.get("downloads", {}).items():
            video_id = str(video_id)
            if video_id in completed or not isinstance(raw_report, dict):
                continue
            if raw_report.get("status") != "failed":
                continue
            if _permanent_download_failure(raw_report):
                permanent.add(video_id)
            else:
                transient.add(video_id)
    return permanent, transient.difference(permanent)


def validate_video_feature_cache(
    cache_path: Path, expected_video_id: str | None = None
) -> dict[str, Any]:
    import torch

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != PER_VIDEO_CACHE_SCHEMA:
        raise ValueError(f"invalid per-video cache schema: {cache_path}")
    video_id = str(payload.get("video_id", ""))
    if expected_video_id and video_id != expected_video_id:
        raise ValueError(f"cache video mismatch: expected {expected_video_id}, got {video_id}")
    graphs = list(payload.get("graphs", []))
    if not graphs:
        raise ValueError(f"feature cache contains no graphs: {cache_path}")
    supervised = 0
    action_rows = 0
    for graph in graphs:
        if str(graph.video_id) != video_id:
            raise ValueError("mixed video IDs in per-video feature cache")
        for attribute in (
            "handcrafted_x",
            "roi_x",
            "visual_x",
            "visual_motion_x",
            "edge_attr",
            "y",
            "loss_mask",
        ):
            if not hasattr(graph, attribute):
                raise ValueError(f"graph is missing {attribute}")
            value = getattr(graph, attribute)
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"graph contains non-finite {attribute}")
        expected_dimensions = {
            "handcrafted_x": 138,
            "roi_x": 650,
            "visual_x": 1162,
            "visual_motion_x": 1172,
            "edge_attr": 8,
        }
        if hasattr(graph, "visual_temporal_x"):
            expected_dimensions["visual_temporal_x"] = 1674
            if not bool(torch.isfinite(graph.visual_temporal_x).all()):
                raise ValueError("graph contains non-finite visual_temporal_x")
        for attribute, dimension in expected_dimensions.items():
            value = getattr(graph, attribute)
            if int(value.shape[1]) != dimension:
                raise ValueError(
                    f"graph {attribute} dimension is {value.shape[1]}, expected {dimension}"
                )
        if int(graph.y.shape[1]) != 80:
            raise ValueError("graph target dimension is not 80")
        supervised += int(graph.loss_mask.sum())
        action_rows += int(graph.y[graph.loss_mask].sum())
    if supervised == 0 or action_rows == 0:
        raise ValueError("feature cache contains no supervised AVA actions")
    return {
        "valid": True,
        "video_id": video_id,
        "graphs": len(graphs),
        "nodes": sum(int(graph.num_nodes) for graph in graphs),
        "supervised_persons": supervised,
        "action_rows": action_rows,
        "sha256": _sha256(cache_path),
        "bytes": cache_path.stat().st_size,
    }


def _save_video_feature_cache(
    *,
    video_id: str,
    graphs: Sequence[Any],
    cache_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_id}.pt"
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": PER_VIDEO_CACHE_SCHEMA,
            "video_id": video_id,
            "graphs": list(graphs),
            "provenance": provenance,
        },
        temporary,
    )
    os.replace(temporary, cache_path)
    validation = validate_video_feature_cache(cache_path, video_id)
    metadata = {
        "schema_version": PER_VIDEO_CACHE_SCHEMA,
        "video_id": video_id,
        "cache": cache_path.as_posix(),
        "validation": validation,
        "provenance": provenance,
    }
    _write_json_atomic(cache_path.with_suffix(".metadata.json"), metadata)
    return metadata


def _safe_remove_tree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"refusing to remove path outside temporary root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _safe_unlink(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if root not in resolved.parents:
        raise ValueError(f"refusing to remove path outside temporary root: {resolved}")
    if resolved.is_file():
        resolved.unlink()


def _process_downloaded_batch(
    *,
    media: dict[str, Path],
    coverage: dict[str, VideoCoverage],
    annotation_paths: Sequence[Path],
    batch_root: Path,
    cache_dir: Path,
    embedding_cache_dir: Path,
    yolo_model: str,
    device: str,
) -> dict[str, dict[str, Any]]:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(Path("data/app/cache/ultralytics").resolve())
    )
    from extraction.preprocessor import VideoPreprocessor
    from extraction.visual import VisualExtractor, load_ava_annotations
    from modelling.scene_graph_builder import SceneGraphBuilder

    from .ava_gnn80 import build_ava80_graph_dataset
    from .ava_improved import (
        RESNET18_BACKBONE,
        TEMPORAL_CLIP_ENCODER,
        attach_frozen_temporal_clip_features,
        attach_motion_features,
        attach_pretrained_visual_features,
        materialize_feature_variants,
    )

    frames_root = batch_root / "frames"
    detections_root = batch_root / "detections"
    scene_root = batch_root / "scene_graphs"
    fused_root = batch_root / "fused"
    base_cache = batch_root / "base_cache"
    annotations_by_split = {
        "train": load_ava_annotations(annotation_paths[0]),
        "val": load_ava_annotations(annotation_paths[1]),
    }
    preprocessor = VideoPreprocessor(video_dir=batch_root / "videos", max_videos=None)
    extraction: dict[str, dict[str, Any]] = {}
    for video_id, media_path in media.items():
        annotation_split = coverage[video_id].annotation_split
        split = "val" if annotation_split == "validation" else annotation_split
        timestamps = set(annotations_by_split[split].get(video_id, {}))
        frames = preprocessor.extract_frames(media_path, timestamps, frames_root / split)
        extraction[video_id] = {
            "split": split,
            "requested_timestamps": len(timestamps),
            "extracted_frames": len(frames),
        }

    detector = VisualExtractor(model_size=yolo_model)
    builder = SceneGraphBuilder(
        detections_dir=detections_root, scene_graphs_dir=scene_root
    )
    for split in ("train", "val"):
        if not (frames_root / split).is_dir():
            continue
        detector.process_split(
            split, frames_root, detections_root, annotations_by_split[split]
        )
        builder.process_split(split)
        for video_id, item in extraction.items():
            if item["split"] != split:
                continue
            paths = sorted((scene_root / split / video_id).glob("*.json"))
            scene_frames = [
                json.loads(path.read_text(encoding="utf-8")) for path in paths
            ]
            for scene_frame in scene_frames:
                scene_frame.setdefault("audio_segments", [])
            output = fused_root / split / f"{video_id}.fused.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                output,
                {
                    "video_id": video_id,
                    "split": split,
                    "num_frames": len(scene_frames),
                    "num_audio_segments": 0,
                    "frames": scene_frames,
                },
            )
            item["scene_graphs"] = len(scene_frames)

    base_graphs = build_ava80_graph_dataset(
        fused_root=fused_root,
        frame_root=frames_root,
        annotation_paths=annotation_paths,
        feature_cache_dir=base_cache,
        force_rebuild=True,
    )
    tracked = attach_motion_features(base_graphs)
    visual = attach_pretrained_visual_features(
        tracked,
        frame_root=frames_root,
        cache_dir=embedding_cache_dir,
        device_name=device,
    )
    temporal = attach_frozen_temporal_clip_features(
        visual,
        cache_dir=embedding_cache_dir,
    )
    improved = materialize_feature_variants(temporal)
    by_video: dict[str, list[Any]] = {}
    for graph in improved:
        by_video.setdefault(str(graph.video_id), []).append(graph)

    reports: dict[str, dict[str, Any]] = {}
    for video_id, media_path in media.items():
        graphs = by_video.get(video_id, [])
        if not graphs:
            reports[video_id] = {"status": "failed", "error": "no graphs built"}
            continue
        reports[video_id] = {
            "status": "validated",
            **_save_video_feature_cache(
                video_id=video_id,
                graphs=graphs,
                cache_dir=cache_dir,
                provenance={
                    "source_media_filename": media_path.name,
                    "source_media_bytes": media_path.stat().st_size,
                    "source_media_sha256": _sha256(media_path),
                    "annotation_split": coverage[video_id].annotation_split,
                    "annotation_paths": [path.as_posix() for path in annotation_paths],
                    "timestamps": extraction[video_id],
                    "yolo_model": yolo_model,
                    "visual_backbone": RESNET18_BACKBONE,
                    "temporal_clip_encoder": TEMPORAL_CLIP_ENCODER,
                    "temporal_context_seconds": [-1.0, 0.0, 1.0],
                },
            ),
        }
    return reports


def _master_video_ids(master_cache: Path) -> set[str]:
    if not master_cache.is_file():
        return set()
    import torch

    payload = torch.load(master_cache, map_location="cpu", weights_only=False)
    return {str(graph.video_id) for graph in payload.get("graphs", [])}


def rebuild_incremental_master_cache(
    master_cache: Path, per_video_cache_dir: Path
) -> dict[str, Any]:
    import torch

    from .ava_improved import IMPROVED_DATASET_SCHEMA, save_improved_graph_cache

    by_video: dict[str, list[Any]] = {}
    if master_cache.is_file():
        payload = torch.load(master_cache, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != IMPROVED_DATASET_SCHEMA:
            raise ValueError("existing improved master cache schema mismatch")
        for graph in payload.get("graphs", []):
            by_video.setdefault(str(graph.video_id), []).append(graph)
    for path in sorted(per_video_cache_dir.glob("*.pt")):
        validation = validate_video_feature_cache(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        by_video[validation["video_id"]] = list(payload["graphs"])
    graphs = [graph for video_id in sorted(by_video) for graph in by_video[video_id]]
    graphs.sort(key=lambda graph: (str(graph.video_id), float(graph.timestamp)))
    temporary = master_cache.with_suffix(".rebuild.pt")
    metadata = save_improved_graph_cache(graphs, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if len(reloaded.get("graphs", [])) != len(graphs):
        raise RuntimeError("rebuilt master cache failed graph-count validation")
    os.replace(temporary, master_cache)
    temporary_metadata = temporary.with_suffix(".metadata.json")
    os.replace(temporary_metadata, master_cache.with_suffix(".metadata.json"))
    metadata["cache_path"] = master_cache.as_posix()
    metadata["sha256"] = _sha256(master_cache)
    _write_json_atomic(master_cache.with_suffix(".metadata.json"), metadata)
    return metadata


def expand_ava_dataset(
    *,
    target_videos: int,
    batch_size: int = 4,
    workers: int = 3,
    target_support: int = 25,
    annotation_paths: Sequence[Path] = (
        Path("data/ava/annotations/ava_train_v2.2.csv"),
        Path("data/ava/annotations/ava_val_v2.2.csv"),
    ),
    temporary_root: Path = Path("data/temp/ava_expansion"),
    feature_root: Path = Path("data/app/features/ava80"),
    master_cache: Path = Path(
        "data/app/models/ava80_improved/ava80_improved_graphs.pt"
    ),
    selection_manifest: Path = Path(
        "evaluation/results/ava-expansion-incremental-selection.json"
    ),
    state_path: Path = Path("data/app/features/ava80/expansion_state.json"),
    yolo_model: str = "yolov8n.pt",
    device: str = "auto",
    delete_raw: bool = True,
    delete_frames: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, Any]:
    if target_videos < 1:
        raise ValueError("target_videos must be positive")
    if batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be between 1 and 20")
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must not be negative")
    coverage = load_video_coverage(annotation_paths)
    per_video_cache_dir = feature_root / "videos"
    embedding_cache_dir = feature_root / "embedding_cache"
    per_video_cache_dir.mkdir(parents=True, exist_ok=True)
    completed = _master_video_ids(master_cache)
    incremental_cache_video_ids: set[str] = set()
    for cache_path in per_video_cache_dir.glob("*.pt"):
        try:
            cached_video_id = validate_video_feature_cache(cache_path)["video_id"]
            completed.add(cached_video_id)
            incremental_cache_video_ids.add(cached_video_id)
        except (OSError, ValueError, RuntimeError):
            continue
    state: dict[str, Any] = {
        "schema_version": EXPANSION_SCHEMA,
        "target_videos": target_videos,
        "batch_size": batch_size,
        "initial_validated_video_count": len(completed),
        "completed_video_ids": sorted(completed),
        "failed_video_ids": [],
        "batches": [],
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["failed_video_ids"] = list(previous.get("failed_video_ids", []))
        state["batches"] = list(previous.get("batches", []))
        state["baseline_video_ids"] = list(previous.get("baseline_video_ids", []))
    if not state.get("baseline_video_ids"):
        state["baseline_video_ids"] = sorted(completed - incremental_cache_video_ids)
    baseline_video_ids = set(state["baseline_video_ids"])
    failed, transient_failures = _classify_previous_download_failures(
        state["batches"], completed
    )
    state["failed_video_ids"] = sorted(failed)
    state["transient_failure_video_ids"] = sorted(transient_failures)
    inventory = fetch_ava_inventory() if len(completed) < target_videos else {}
    selected_history = [
        str(video_id)
        for batch_report in state["batches"]
        for video_id in batch_report.get("requested_video_ids", [])
    ]
    attempted: set[str] = set()

    consecutive_failed_batches = 0
    while len(completed) < target_videos:
        selection = select_incremental_videos(
            coverage,
            completed,
            target_videos=target_videos,
            excluded_video_ids=failed | attempted,
            target_support=target_support,
        )
        candidates = list(selection["selected_video_ids"])
        if not candidates:
            break
        if not selected_history:
            _write_json_atomic(selection_manifest, selection)
        batch_index = len(state["batches"]) + 1
        batch_root = temporary_root / f"batch_{batch_index:04d}"
        video_root = batch_root / "videos"
        batch, storage_report = _storage_safe_batch(
            candidates,
            inventory,
            destination_dir=video_root,
            master_cache=master_cache,
            batch_size=batch_size,
            min_free_bytes=min_free_bytes,
        )
        if not batch:
            state["storage_blocked"] = storage_report
            _write_json_atomic(state_path, state)
            break
        attempted.update(batch)
        selected_history.extend(batch)
        downloads = _download_batch(batch, inventory, video_root, workers)
        available_media = {
            video_id: Path(result["path"])
            for video_id, result in downloads.items()
            if result.get("status") in {"cached", "downloaded"}
        }
        batch_report: dict[str, Any] = {
            "batch": batch_index,
            "requested_video_ids": batch,
            "storage": storage_report,
            "downloads": {
                video_id: {
                    key: value.as_posix() if isinstance(value, Path) else value
                    for key, value in result.items()
                }
                for video_id, result in downloads.items()
            },
            "features": {},
            "cleanup": {},
        }
        for video_id, result in downloads.items():
            if result.get("status") != "failed":
                continue
            if _permanent_download_failure(result):
                failed.add(video_id)
            else:
                transient_failures.add(video_id)
        if available_media:
            try:
                feature_reports = _process_downloaded_batch(
                    media=available_media,
                    coverage=coverage,
                    annotation_paths=annotation_paths,
                    batch_root=batch_root,
                    cache_dir=per_video_cache_dir,
                    embedding_cache_dir=embedding_cache_dir,
                    yolo_model=yolo_model,
                    device=device,
                )
            except Exception as exc:
                feature_reports = {
                    video_id: {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for video_id in available_media
                }
            batch_report["features"] = feature_reports
            for video_id, report in feature_reports.items():
                if report.get("status") != "validated":
                    transient_failures.add(video_id)
                    continue
                completed.add(video_id)
                transient_failures.discard(video_id)
                media_path = available_media[video_id]
                if delete_raw:
                    _safe_unlink(media_path, temporary_root)
                    batch_report["cleanup"][video_id] = {
                        "raw_media_deleted_after_validation": True
                    }
                if delete_frames:
                    split = (
                        "val"
                        if coverage[video_id].annotation_split == "validation"
                        else coverage[video_id].annotation_split
                    )
                    _safe_remove_tree(
                        batch_root / "frames" / split / video_id, temporary_root
                    )
                    batch_report["cleanup"].setdefault(video_id, {})[
                        "frames_deleted_after_validation"
                    ] = True
        state["batches"].append(batch_report)
        state["completed_video_ids"] = sorted(completed)
        state["failed_video_ids"] = sorted(failed)
        state["transient_failure_video_ids"] = sorted(transient_failures)
        _write_json_atomic(state_path, state)
        if available_media and any(
            item.get("status") == "validated"
            for item in batch_report["features"].values()
        ):
            state["master_cache"] = rebuild_incremental_master_cache(
                master_cache, per_video_cache_dir
            )
            if (
                delete_raw
                and delete_frames
                and len(batch_report["features"]) == len(batch)
                and all(
                    item.get("status") == "validated"
                    for item in batch_report["features"].values()
                )
            ):
                _safe_remove_tree(batch_root, temporary_root)
                batch_report["temporary_batch_deleted_after_master_validation"] = True
            _write_json_atomic(state_path, state)
        batch_succeeded = any(
            item.get("status") == "validated"
            for item in batch_report["features"].values()
        )
        if batch_succeeded:
            consecutive_failed_batches = 0
            state.pop("download_circuit_breaker", None)
        else:
            consecutive_failed_batches += 1
        if consecutive_failed_batches >= 3:
            state["download_circuit_breaker"] = {
                "open": True,
                "consecutive_failed_batches": consecutive_failed_batches,
                "reason": (
                    "stopped after repeated transfer/feature failures; transient videos "
                    "remain retryable on the next invocation"
                ),
            }
            _write_json_atomic(state_path, state)
            break

    final_selection = select_incremental_videos(
        coverage,
        baseline_video_ids,
        target_videos=target_videos,
        excluded_video_ids=failed,
        target_support=target_support,
    )
    planned_ids = set(final_selection["selected_video_ids"])
    validated_additions = completed.difference(baseline_video_ids)
    validated_history = [
        str(video_id)
        for batch_report in state["batches"]
        for video_id, report in batch_report.get("features", {}).items()
        if isinstance(report, dict)
        and report.get("status") == "validated"
        and str(video_id) in validated_additions
    ]
    validated_history = list(dict.fromkeys(validated_history))
    for video_id in [*final_selection["selected_video_ids"], *sorted(validated_additions)]:
        if video_id in validated_additions and video_id not in validated_history:
            validated_history.append(video_id)
    state.update(
        {
            "completed_video_ids": sorted(completed),
            "completed_video_count": len(completed),
            "failed_video_ids": sorted(failed),
            "transient_failure_video_ids": sorted(transient_failures),
            "selected_video_history": selected_history,
            "target_reached": len(completed) >= target_videos,
            "final_action_coverage": {
                "actions_present": sum(
                    row["current_support"] > 0 for row in final_selection["actions"]
                ),
                "actions_at_or_above_target_support": sum(
                    row["current_support"] >= target_support
                    for row in final_selection["actions"]
                ),
            },
        }
    )
    final_selection["selected_video_history"] = validated_history
    final_selection["attempted_video_count"] = len(set(selected_history))
    final_selection["transient_failure_video_count"] = len(transient_failures)
    final_selection["unavailable_video_ids"] = sorted(failed)
    final_selection["replacement_video_ids"] = [
        video_id
        for video_id in validated_history
        if video_id not in planned_ids
    ]
    final_selection["completed_video_ids"] = sorted(completed)
    _write_json_atomic(selection_manifest, final_selection)
    _write_json_atomic(state_path, state)
    return state
