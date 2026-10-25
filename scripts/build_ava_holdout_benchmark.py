"""Build a model-only Review-2 action benchmark from held-out AVA v2.2 windows.

Ground truth comes only from the official AVA v2.2 annotation CSVs, never from
system output.  The clips are the annotated 15-minute windows (900-1800 s) of
the AVA80 GNN's frozen validation and test videos, re-indexed from raw media by
the full model pipeline (no ``import-ava``, no demo manifest).

Fixed rules, decided before any retrieval was run:

* Split: the GNN's validation videos -> ``calibration``; its test videos ->
  ``test``.  Neither group was used for GNN weight training.
* Positives: per video, AVA labels present on 3+ keyframes and on at most
  ``--max-coverage`` of the window's keyframes (ubiquitous labels such as
  ``stand`` make P@1 trivial).  The ``--positives`` most frequent survivors.
* Negatives: labels absent from the video's *entire* annotation, excluding
  near-synonyms of present labels, ordered by global AVA frequency (plausible
  in-domain negatives first).  ``--negatives`` per video.
* Relevant moments: every maximal run of consecutive labelled keyframes, as
  ``[first - 0.5, last + 0.5]`` in clip time, peak at the run's middle.
* Query text: one fixed examiner-style phrase per label (``PHRASES``), written
  without consulting the parser.

AVA labels at most three person-object and three person-person actions per
person, so absence is near-exhaustive but not guaranteed.  That limitation is
recorded in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from vidquery.ava_labels import AVA_V22_ACTIONS

WINDOW_OFFSET_SECONDS = 900
SCHEMA_VERSION = "challenge-1.0"
BUILDER_VERSION = "ava-holdout-benchmark-v1"

PHRASES: dict[int, str] = {
    1: "someone bending over",
    2: "a person crawling",
    3: "someone kneeling down",
    4: "people dancing",
    5: "someone falling down",
    6: "a person getting up",
    7: "someone jumping",
    8: "a person lying down",
    9: "someone doing martial arts",
    10: "a person running",
    11: "a person sitting",
    12: "a person standing",
    13: "someone swimming",
    14: "a person walking",
    15: "someone answering the phone",
    16: "a person brushing their teeth",
    17: "a person holding something",
    18: "someone catching something",
    19: "someone chopping",
    20: "a person climbing",
    21: "people clinking glasses",
    22: "someone closing something",
    23: "a person cooking",
    24: "someone cutting something",
    25: "a person digging",
    26: "someone putting on clothes",
    27: "a person drinking",
    28: "someone driving",
    29: "a person eating",
    30: "someone entering the room",
    31: "a person leaving the room",
    32: "someone taking something out",
    33: "a person fishing",
    34: "someone hitting an object",
    35: "someone kicking something",
    36: "a person picking something up",
    37: "someone listening",
    38: "someone opening something",
    39: "a person painting",
    40: "people playing a board game",
    41: "someone playing a musical instrument",
    42: "a person playing with a pet",
    43: "someone pointing at something",
    44: "a person pressing a button",
    45: "someone pulling something",
    46: "a person pushing something",
    47: "someone putting something down",
    48: "a person reading",
    49: "someone riding",
    50: "a person rowing a boat",
    51: "someone sailing a boat",
    52: "someone shooting a gun",
    53: "a person shoveling",
    54: "someone smoking",
    55: "a person stirring",
    56: "someone taking a photo",
    57: "a person texting on a phone",
    58: "someone throwing something",
    59: "a person touching an object",
    60: "someone turning around",
    61: "a person watching something",
    62: "someone working on a computer",
    63: "a person writing",
    64: "people fighting",
    65: "someone handing something to another person",
    66: "a person grabbing someone",
    67: "someone clapping",
    68: "two people shaking hands",
    69: "a person waving",
    70: "two people hugging",
    71: "someone kicking a person",
    72: "two people kissing",
    73: "someone lifting another person",
    74: "a person listening to someone",
    75: "an adult playing with kids",
    76: "someone pushing another person",
    77: "a person singing to someone",
    78: "someone taking something from another person",
    79: "two people talking",
    80: "someone watching another person",
}

# Second, independently worded phrase set, written and committed on 2026-10-25
# *before* the parser action lexicon was extended.  It is reserved for fresh
# held-out videos and must never be used to develop parser rules.
PHRASES_V2: dict[int, str] = {
    1: "a person bowing",
    2: "someone crawling on the floor",
    3: "a person crouching",
    4: "someone dancing",
    5: "a person falls over",
    6: "someone stands up from a seat",
    7: "a person leaps",
    8: "someone sleeping",
    9: "people doing kung fu",
    10: "someone jogging",
    11: "people seated",
    12: "someone stands still",
    13: "a person swims in water",
    14: "people walking around",
    15: "a person on a phone call",
    16: "someone brushes his teeth",
    17: "a person carrying an object",
    18: "someone catches a ball",
    19: "a person chopping food",
    20: "someone climbing up",
    21: "a toast with glasses",
    22: "a person shuts the door",
    23: "someone preparing food on a stove",
    24: "a person cuts something with a knife",
    25: "someone digging a hole",
    26: "a person getting dressed",
    27: "someone takes a drink",
    28: "a person driving a car",
    29: "someone eats food",
    30: "a person comes in through the door",
    31: "someone walks out of the room",
    32: "a person pulls something out",
    33: "someone goes fishing",
    34: "a person strikes an object",
    35: "someone kicks a ball",
    36: "a person lifts something up",
    37: "someone listening to music",
    38: "a person opens a door",
    39: "someone painting a picture",
    40: "people playing chess",
    41: "someone playing the guitar",
    42: "a person playing with a dog",
    43: "someone points at an object",
    44: "a person presses an elevator button",
    45: "someone pulls an object",
    46: "a person pushing an object",
    47: "someone sets something down",
    48: "a person reading a book",
    49: "someone riding a bike",
    50: "people rowing",
    51: "a sailboat being sailed",
    52: "someone fires a gun",
    53: "a person shovels snow",
    54: "someone smoking a cigarette",
    55: "a person stirs a pot",
    56: "someone takes a picture with a camera",
    57: "a person looking at their cellphone",
    58: "someone throws an object",
    59: "a person touches something",
    60: "someone turns a screwdriver",
    61: "a person watching TV",
    62: "someone typing on a laptop",
    63: "a person writes notes",
    64: "someone punches another person",
    65: "a person gives an object to someone",
    66: "someone grabs a person",
    67: "people applauding",
    68: "two people shake hands",
    69: "someone waves hello",
    70: "two people embrace",
    71: "someone kicks another person",
    72: "a couple kissing",
    73: "someone picks up a child",
    74: "a person listens to someone speaking",
    75: "a person playing with children",
    76: "someone shoves another person",
    77: "a person singing",
    78: "someone takes an object from another person",
    79: "two people having a conversation",
    80: "a person looks at another person",
}
PHRASE_SETS = {"v1": PHRASES, "v2": PHRASES_V2}

# Near-synonym groups: if any member is present, none may serve as a negative.
SYNONYM_GROUPS: tuple[frozenset[int], ...] = (
    frozenset({37, 74}),
    frozenset({61, 80}),
    frozenset({35, 71}),
    frozenset({34, 64}),
    frozenset({36, 73}),
    frozenset({46, 76}),
    frozenset({42, 75}),
    frozenset({15, 57}),
    frozenset({17, 59}),
    frozenset({22, 38}),
    frozenset({30, 31}),
    frozenset({47, 36}),
)

DEFAULT_SPLITS = {
    "calibration": ("0f39OWEqJ24", "9mLYmkonWZQ", "covMYDBa5dk"),
    "test": ("hbYvDvJrpNk", "t0V4drbYDnc", "zR725veL-DI"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_annotations(paths: list[Path]) -> dict[str, dict[int, set[int]]]:
    """video -> label -> set of keyframe timestamps where any person has the label."""

    labels: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 7:
                    continue
                labels[row[0]][int(row[6])].add(int(float(row[1])))
    return labels


def _runs(timestamps: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in sorted(timestamps):
        if runs and value - runs[-1][1] <= 1:
            runs[-1] = (runs[-1][0], value)
        else:
            runs.append((value, value))
    return runs


def build_video_queries(
    video_id: str,
    source_video_id: str,
    video_labels: dict[int, set[int]],
    global_frequency: Counter[int],
    *,
    positives: int,
    negatives: int,
    max_coverage: float,
    min_keyframes: int = 3,
    phrases: dict[int, str] | None = None,
) -> tuple[list[dict], dict]:
    phrases = phrases or PHRASES
    keyframes = sorted({value for values in video_labels.values() for value in values})
    window = max(1, len(keyframes))
    present = {label for label, values in video_labels.items() if values}
    candidates = [
        (label, len(values))
        for label, values in video_labels.items()
        if len(values) >= min_keyframes and len(values) / window <= max_coverage
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    chosen_positive = [label for label, _ in candidates[:positives]]

    blocked = set(present)
    for group in SYNONYM_GROUPS:
        if group & present:
            blocked |= group
    negative_pool = [label for label in AVA_V22_ACTIONS if label not in blocked]
    negative_pool.sort(key=lambda label: (-global_frequency[label], label))
    chosen_negative = negative_pool[:negatives]

    queries: list[dict] = []
    for label in chosen_positive:
        relevant = []
        for first, last in _runs(list(video_labels[label])):
            start = max(0.0, first - WINDOW_OFFSET_SECONDS - 0.5)
            end = last - WINDOW_OFFSET_SECONDS + 0.5
            peak = round((first + last) / 2 - WINDOW_OFFSET_SECONDS, 1)
            relevant.append(
                {
                    "source_video_id": source_video_id,
                    "peak_time": min(max(peak, start), end),
                    "start_time": round(start, 1),
                    "end_time": round(end, 1),
                }
            )
        queries.append(
            {
                "id": f"ava-{video_id}-a{label:02d}",
                "query": phrases[label],
                "category": "action_relation",
                "source_video_id": source_video_id,
                "negative": False,
                "expected_intents": ["action_search"],
                "tolerance_seconds": 2.0,
                "relevant": relevant,
                "annotator": "ava_v2.2_official",
                "notes": (
                    f"AVA label {label} '{AVA_V22_ACTIONS[label]}' on "
                    f"{len(video_labels[label])}/{window} keyframes"
                ),
            }
        )
    for label in chosen_negative:
        queries.append(
            {
                "id": f"ava-{video_id}-n{label:02d}",
                "query": phrases[label],
                "category": "negative",
                "source_video_id": source_video_id,
                "negative": True,
                "expected_intents": ["action_search"],
                "tolerance_seconds": 2.0,
                "annotator": "ava_v2.2_official",
                "notes": (
                    f"in-domain negative: AVA label {label} '{AVA_V22_ACTIONS[label]}' "
                    "is absent from this video's full annotation"
                ),
            }
        )
    coverage = {
        "annotated_keyframes": window,
        "labels_present": len(present),
        "positive_candidates": len(candidates),
        "excluded_as_ubiquitous": sorted(
            AVA_V22_ACTIONS[label]
            for label, values in video_labels.items()
            if len(values) / window > max_coverage
        ),
        "positive_labels": [AVA_V22_ACTIONS[label] for label in chosen_positive],
        "negative_labels": [AVA_V22_ACTIONS[label] for label in chosen_negative],
    }
    return queries, coverage


def assert_no_training_overlap(video_ids: list[str], metadata_paths: list[Path]) -> dict:
    """Fail if any benchmark video appears in a checkpoint's training split."""

    report: dict[str, dict] = {}
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = payload.get("split") or {}
        train = set(split.get("train", []))
        overlap = sorted(set(video_ids) & train)
        report[path.as_posix()] = {
            "training_videos": len(train),
            "overlap_with_benchmark": overlap,
        }
        if overlap:
            raise SystemExit(f"benchmark videos {overlap} were used to train {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=[
            Path("data/ava/annotations/ava_train_v2.2.csv"),
            Path("data/ava/annotations/ava_val_v2.2.csv"),
        ],
    )
    parser.add_argument(
        "--checkpoint-metadata",
        type=Path,
        nargs="+",
        default=[
            Path(
                "data/app/models/ava80_scaling/30/ablations/"
                "ava80-gat-v5-scale30-roi-frame.metadata.json"
            ),
            Path("data/app/models/ava80_final/ava80-gat-v6-scale37-roi-frame-rare.metadata.json"),
        ],
    )
    parser.add_argument("--positives", type=int, default=15)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--max-coverage", type=float, default=0.25)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/queries_challenge_ava_holdout.json")
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    parser.add_argument("--phrase-set", choices=sorted(PHRASE_SETS), default="v1")
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="SPLIT=VIDEO[,VIDEO...]",
        help="Replace the default splits, e.g. --split test=abc,def (repeatable)",
    )
    args = parser.parse_args()
    splits: dict[str, tuple[str, ...]] = dict(DEFAULT_SPLITS)
    if args.split:
        splits = {}
        for item in args.split:
            name, _, videos = item.partition("=")
            if name not in ("calibration", "test") or not videos:
                raise SystemExit(f"invalid --split {item!r}")
            splits[name] = tuple(video for video in videos.split(",") if video)
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; refusing to overwrite without --force")

    labels = load_annotations(args.annotations)
    global_frequency: Counter[int] = Counter()
    for video_labels in labels.values():
        for label, values in video_labels.items():
            global_frequency[label] += len(values)

    all_ids = [video for videos in splits.values() for video in videos]
    leakage = assert_no_training_overlap(
        all_ids, [path for path in args.checkpoint_metadata if path.is_file()]
    )
    queries: list[dict] = []
    videos: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    for split, video_ids in splits.items():
        for video_id in video_ids:
            if video_id not in labels:
                raise SystemExit(f"no AVA annotations found for {video_id}")
            source = f"ava_{video_id}_w900"
            video_queries, video_coverage = build_video_queries(
                video_id,
                source,
                labels[video_id],
                global_frequency,
                positives=args.positives,
                negatives=args.negatives,
                max_coverage=args.max_coverage,
                phrases=PHRASE_SETS[args.phrase_set],
            )
            queries.extend(video_queries)
            videos[source] = {"split": split}
            coverage[source] = video_coverage

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Model-only action benchmark on held-out AVA v2.2 windows. Ground truth from "
            "official AVA annotations; indexes built from raw media by the full pipeline."
        ),
        "builder": BUILDER_VERSION,
        "phrase_set": args.phrase_set,
        "created_at": datetime.now(UTC).isoformat(),
        "window": {"source_start_seconds": WINDOW_OFFSET_SECONDS, "duration_seconds": 900},
        "selection_rules": {
            "positives_per_video": args.positives,
            "negatives_per_video": args.negatives,
            "max_keyframe_coverage_for_positive": args.max_coverage,
            "min_keyframes_for_positive": 3,
            "negative_order": (
                "global AVA frequency, descending, synonyms of present labels excluded"
            ),
        },
        "annotation_sources": {path.as_posix(): _sha256(path) for path in args.annotations},
        "training_overlap_audit": leakage,
        "limitations": [
            "Movie footage, not lectures or classrooms; speech, OCR, and ambiguity "
            "are not covered.",
            "AVA caps object/person interactions at three per person, so negatives "
            "are near-exhaustive.",
            "Query wording is fixed per label and was written by the benchmark author, "
            "not examiners.",
        ],
        "coverage": coverage,
        "videos": videos,
        "queries": queries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = Counter(item["category"] for item in queries)
    print(f"Wrote {len(queries)} queries ({dict(counts)}) to {args.output}")


if __name__ == "__main__":
    main()
