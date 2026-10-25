"""Paired comparison of frozen AVA action checkpoints on identical test persons.

Every checkpoint is scored on the same supervised persons with its own
validation-fitted thresholds. Metrics match ``classification_metrics``
(supported-class macro F1, micro F1, supported-class mAP). Uncertainty comes
from two paired bootstraps that reuse one set of resampling weights for all
models:

* ``keyframe``: keyframes are resampled with replacement inside each video
  (stratified), so every video keeps its size. This captures sampling noise
  within the test videos but not variation between videos.
* ``video``: whole videos are resampled. With only a handful of test videos
  this is coarse; read it as a conservative check, alongside per-video results.

Weighted metrics are exact for integer bootstrap weights: a person drawn k
times counts as k adjacent copies in the ranking used for average precision.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ava_gnn80 import classification_metrics, decode_ava_predictions


def predict_checkpoint(checkpoint: Path, graphs: Iterable[Any], device: Any) -> dict[str, Any]:
    """Per-person probabilities, targets, and grouping keys for one checkpoint."""

    import torch

    from .ava_ablation import StaticActionModel

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model: Any = StaticActionModel(
        int(payload["feature_dimension"]),
        use_edges=bool(payload.get("use_edges")),
        mlp_only=bool(payload.get("mlp_only")),
    )
    model = model.to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    attribute = str(payload["feature_attribute"])
    mean = payload["feature_mean"].to(device)
    std = payload["feature_std"].to(device)
    probabilities, targets, detector_only, videos, keyframes = [], [], [], [], []
    with torch.inference_mode():
        for graph in graphs:
            logits = model(
                (getattr(graph, attribute).to(device) - mean) / std,
                graph.class_ids.to(device),
                graph.edge_index.to(device),
                graph.edge_attr.to(device),
            )
            mask = graph.loss_mask.bool()
            count = int(mask.sum())
            if not count:
                continue
            probabilities.append(torch.sigmoid(logits).cpu()[mask].numpy())
            targets.append(graph.y[mask].numpy())
            detector_only.append((~graph.annotation_only_flag.bool())[mask].numpy())
            videos.extend([str(graph.video_id)] * count)
            keyframes.extend([f"{graph.video_id}@{float(graph.timestamp):.3f}"] * count)
    thresholds = payload["thresholds"].numpy()
    probability_array = np.concatenate(probabilities).astype(np.float32)
    return {
        "model_version": str(payload.get("model_version", checkpoint.stem)),
        "feature_attribute": attribute,
        "probabilities": probability_array,
        "predicted": decode_ava_predictions(probability_array, thresholds),
        "targets": np.concatenate(targets).astype(np.float32) == 1,
        "detector_only": np.concatenate(detector_only).astype(bool),
        "videos": np.asarray(videos),
        "keyframes": np.asarray(keyframes),
        "thresholds": thresholds,
    }


def _ap_terms(truth: np.ndarray, scores: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per class: stable descending order and the positive mask in that order."""

    terms = []
    for index in range(truth.shape[1]):
        order = np.argsort(-scores[:, index], kind="stable")
        terms.append((order, truth[order, index]))
    return terms


def weighted_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    ap_terms: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, float]:
    """``classification_metrics`` with integer person weights (bootstrap counts)."""

    from scipy.special import digamma  # type: ignore[import-untyped]

    weights = np.asarray(weights, dtype=np.float64)
    truth_f = truth.astype(np.float64)
    predicted_f = predicted.astype(np.float64)
    tp = weights @ (truth_f * predicted_f)
    predicted_count = weights @ predicted_f
    support = weights @ truth_f
    fp = predicted_count - tp
    fn = support - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    micro_denominator = 2 * tp.sum() + fp.sum() + fn.sum()
    terms = ap_terms if ap_terms is not None else _ap_terms(truth, scores)
    average_precision = np.zeros(truth.shape[1])
    for index, (order, positive) in enumerate(terms):
        if not supported[index]:
            continue
        ranked_weights = weights[order]
        cumulative = np.cumsum(ranked_weights)
        cumulative_positive = np.cumsum(ranked_weights * positive)
        keep = positive & (ranked_weights > 0)
        k = ranked_weights[keep]
        before = cumulative[keep] - k
        positive_before = cumulative_positive[keep] - k
        # Copies j = 1..k of one positive: sum (a + j) / (b + j)
        #   = k - (b - a) * (H(b + k) - H(b)), with H the harmonic number.
        contributions = k - (before - positive_before) * (
            digamma(before + k + 1) - digamma(before + 1)
        )
        average_precision[index] = contributions.sum() / support[index]
    count = max(1, int(supported.sum()))
    return {
        "macro_f1_supported_classes": float(f1[supported].sum() / count),
        "micro_f1": float(2 * tp.sum() / micro_denominator) if micro_denominator else 0.0,
        "mean_average_precision": float(average_precision[supported].sum() / count),
    }


METRICS = ("macro_f1_supported_classes", "mean_average_precision", "micro_f1")


def _summary(values: dict[str, Any]) -> dict[str, Any]:
    return {key: round(float(values[key]), 6) for key in METRICS}


def _resampling_weights(
    groups: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Integer weights per person from resampling ``groups`` (within ``strata``)."""

    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros(len(unique))
    if strata is None:
        counts += np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
    else:
        group_stratum = np.empty(len(unique), dtype=object)
        group_stratum[inverse] = strata
        for stratum in np.unique(group_stratum):
            members = np.flatnonzero(group_stratum == stratum)
            draws = rng.choice(members, size=len(members), replace=True)
            counts += np.bincount(draws, minlength=len(unique))
    return counts[inverse]


def compare_predictions(
    predictions: dict[str, dict[str, Any]],
    *,
    pairs: Sequence[tuple[str, str]],
    samples: int = 2000,
    seed: int = 2026,
    subset: str = "all_persons",
) -> dict[str, Any]:
    """Point metrics, per-video metrics, and paired bootstrap differences (b - a)."""

    names = list(predictions)
    reference = predictions[names[0]]
    for name in names[1:]:
        other = predictions[name]
        if not (
            np.array_equal(other["targets"], reference["targets"])
            and np.array_equal(other["keyframes"], reference["keyframes"])
        ):
            raise ValueError(f"{name} was not scored on the same test persons")
    keep = (
        reference["detector_only"]
        if subset == "detector_only"
        else np.ones(len(reference["targets"]), dtype=bool)
    )
    truth = reference["targets"][keep]
    videos = reference["videos"][keep]
    keyframes = reference["keyframes"][keep]
    data = {
        name: (item["predicted"][keep], item["probabilities"][keep])
        for name, item in predictions.items()
    }
    terms = {name: _ap_terms(truth, scores) for name, (_, scores) in data.items()}
    ones = np.ones(len(truth))
    point = {}
    for name, (predicted, scores) in data.items():
        exact = classification_metrics(
            truth.astype(np.float32), scores, predictions[name]["thresholds"]
        )
        weighted = weighted_metrics(truth, predicted, scores, ones, terms[name])
        for key in METRICS:
            if abs(weighted[key] - exact[key]) > 1e-5:
                raise AssertionError(f"weighted {key} disagrees with classification_metrics")
        point[name] = {**_summary(exact), "supported_classes": exact["supported_classes"]}
    per_video: dict[str, dict[str, Any]] = {}
    for video in sorted(set(videos)):
        rows = videos == video
        per_video[video] = {
            "persons": int(rows.sum()),
            **{
                name: _summary(
                    weighted_metrics(truth[rows], predicted[rows], scores[rows], ones[rows])
                )
                for name, (predicted, scores) in data.items()
            },
        }
    bootstraps: dict[str, Any] = {}
    for method, groups, strata in (
        ("keyframe_within_video", keyframes, videos),
        ("video", videos, None),
    ):
        rng = np.random.default_rng(seed)
        draws: dict[str, list[dict[str, float]]] = {name: [] for name in data}
        for _ in range(samples):
            weights = _resampling_weights(groups, strata, rng)
            for name, (predicted, scores) in data.items():
                draws[name].append(weighted_metrics(truth, predicted, scores, weights, terms[name]))
        result = {}
        for a, b in pairs:
            entry = {}
            for key in METRICS:
                differences = np.asarray(
                    [right[key] - left[key] for left, right in zip(draws[a], draws[b], strict=True)]
                )
                low, high = np.percentile(differences, [2.5, 97.5])
                entry[key] = {
                    "difference": round(point[b][key] - point[a][key], 6),
                    "ci95": [round(float(low), 6), round(float(high), 6)],
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            result[f"{b} - {a}"] = entry
        bootstraps[method] = result
    return {
        "subset": subset,
        "persons": int(len(truth)),
        "keyframes": int(len(set(keyframes))),
        "videos": sorted(set(videos)),
        "point": point,
        "per_video": per_video,
        "paired_bootstrap": bootstraps,
        "resampling": {"samples": samples, "seed": seed},
    }
