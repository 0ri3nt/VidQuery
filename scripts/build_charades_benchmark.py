"""Build fixed Charades-STA + NA-VMR evaluation subsets (no retrieval is run here).

Sources (downloaded unchanged into ``data/source/charades/annotations``):

* Charades-STA (Gao et al., ICCV 2017): ``Charades_sta_{train,test}.txt``,
  lines ``VIDEO START END##sentence`` (mirrored in Alvin-Zeng/DRN);
* NA-VMR (Flanagan et al., "Moment of Untruth", WACV 2025,
  keflanagan/MomentofUntruth): in-domain negatives (real Charades-STA
  sentences re-assigned to a video where they do not occur) and out-of-domain
  negatives (unrelated sentences), aligned with the STA train/test videos.

Fixed selection rule, decided before any evaluation:

* ``test``: Charades-STA test videos ordered by SHA-256 of the video id, first
  ``--test-videos``; all their STA positives and NA-VMR test negatives.
* ``validation``: Charades-STA *train* videos in the same order, first
  ``--validation-videos``; their STA positives and NA-VMR train negatives.
  VidQuery trains nothing on Charades, so this split is unseen data used only
  for tuning and diagnosis.

Licences: Charades videos are for non-commercial academic research (AI2
licence, no redistribution). Charades-STA and the NA-VMR repository state no
licence; use them for research evaluation only and cite the papers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "moment-benchmark-1.0"


def _order(video_id: str) -> str:
    return hashlib.sha256(video_id.encode()).hexdigest()


def _sta(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        head, sentence = line.split("##", 1)
        video, start, end = head.split()
        rows.append(
            {"line": number, "video": video, "start": float(start), "end": float(end),
             "query": sentence.strip()}
        )
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_split(
    name: str,
    sta_rows: list[dict[str, Any]],
    negatives: dict[str, list[dict[str, Any]]],
    durations: dict[str, float],
    video_count: int,
) -> dict[str, Any]:
    videos = sorted({row["video"] for row in sta_rows}, key=_order)[:video_count]
    chosen = set(videos)
    queries: list[dict[str, Any]] = []
    skipped_invalid = 0
    for row in sta_rows:
        if row["video"] not in chosen:
            continue
        if row["end"] <= row["start"]:
            skipped_invalid += 1
            continue
        queries.append(
            {
                "id": f"{name}-sta-{row['line']}",
                "video": row["video"],
                "query": row["query"],
                "answerable": True,
                "kind": "positive",
                "start": row["start"],
                "end": min(row["end"], durations.get(row["video"], row["end"])),
            }
        )
    for kind, rows in negatives.items():
        for row in rows:
            if row["vid"] not in chosen:
                continue
            queries.append(
                {
                    "id": f"{name}-{kind}-{row['qid']}",
                    "video": row["vid"],
                    "query": str(row["query"]).strip(),
                    "answerable": False,
                    "kind": kind,
                }
            )
    return {
        "videos": videos,
        "durations": {video: durations.get(video) for video in videos},
        "query_counts": dict(Counter(item["kind"] for item in queries)),
        "skipped_invalid_intervals": skipped_invalid,
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/source/charades/annotations")
    )
    parser.add_argument("--test-videos", type=int, default=300)
    parser.add_argument("--validation-videos", type=int, default=150)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/charades_sta_navmr_benchmark.json")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; refusing to overwrite without --force")
    root = args.annotations
    files = {
        "sta_train": root / "Charades_sta_train.txt",
        "sta_test": root / "Charades_sta_test.txt",
        "durations": root / "Charades_duration.json",
        "navmr_id_neg_test": root / "navmr_charades_id_neg_test.jsonl",
        "navmr_ood_neg_test": root / "navmr_charades_ood_neg_test.jsonl",
        "navmr_id_neg_train": root / "navmr_charades_id_neg_train.jsonl",
        "navmr_ood_neg_train": root / "navmr_charades_ood_neg_train.jsonl",
    }
    durations = {key: float(value) for key, value in json.loads(
        files["durations"].read_text(encoding="utf-8")).items()}
    test = build_split(
        "test",
        _sta(files["sta_test"]),
        {
            "in_domain_negative": _jsonl(files["navmr_id_neg_test"]),
            "out_of_domain_negative": _jsonl(files["navmr_ood_neg_test"]),
        },
        durations,
        args.test_videos,
    )
    validation = build_split(
        "validation",
        _sta(files["sta_train"]),
        {
            "in_domain_negative": _jsonl(files["navmr_id_neg_train"]),
            "out_of_domain_negative": _jsonl(files["navmr_ood_neg_train"]),
        },
        durations,
        args.validation_videos,
    )
    if set(test["videos"]) & set(validation["videos"]):
        raise SystemExit("test and validation videos overlap")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "test: Charades-STA test videos by sha256(video id), first N; validation: "
            "Charades-STA train videos by sha256(video id), first M; all STA positives and "
            "NA-VMR negatives of those videos"
        ),
        "sources": {
            name: {"path": path.as_posix(), "sha256": _sha(path)} for name, path in files.items()
        },
        "licences": {
            "charades_videos": "AI2 Charades licence: non-commercial academic research only, "
            "no redistribution",
            "charades_sta": "no licence stated (TALL, Gao et al. 2017); research evaluation only",
            "na_vmr": "no licence stated (keflanagan/MomentofUntruth); research evaluation only",
        },
        "splits": {"test": test, "validation": validation},
    }
    args.output.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for split, payload in manifest["splits"].items():
        print(split, len(payload["videos"]), "videos", payload["query_counts"],
              "skipped", payload["skipped_invalid_intervals"])


if __name__ == "__main__":
    main()
