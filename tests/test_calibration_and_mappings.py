from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidquery.ava_gnn80 import COCO_CLASS_NAMES
from vidquery.challenge_evaluation import (
    MethodRun,
    RankedMoment,
    _query_metrics,
    _selective_metrics,
)
from vidquery.reliability import fit_abstention_thresholds, load_reliability_thresholds
from vidquery.search import (
    HashingTextEncoder,
    LocalHybridSearchEngine,
    configured_search_engine,
    require_semantic_encoder,
    resolve_sentence_cache_folder,
)
from vidquery.storage import SQLiteRepository
from vidquery.vidor_relation import (
    VIDOR_OBJECTS,
    YOLO_UNSUPPORTED_FOR_VIDOR,
    mapped_vidor_class,
)

# --------------------------------------------------------------------------
# COCO -> VidOR class mapping


def test_mouse_and_potted_plant_are_unsupported_not_relabelled():
    assert mapped_vidor_class("mouse") is None
    assert mapped_vidor_class("potted plant") is None
    assert mapped_vidor_class("potted_plant") is None


def test_every_coco_class_is_mapped_or_explicitly_unsupported():
    for name in COCO_CLASS_NAMES:
        if name == "unknown":
            continue
        mapped = mapped_vidor_class(name)
        assert mapped in VIDOR_OBJECTS or name in YOLO_UNSUPPORTED_FOR_VIDOR, name
        assert not (mapped is not None and name in YOLO_UNSUPPORTED_FOR_VIDOR), name


def test_renamed_equivalents_map_one_to_one():
    assert mapped_vidor_class("person") == "adult"
    assert mapped_vidor_class("couch") == "sofa"
    assert mapped_vidor_class("airplane") == "aircraft"
    assert mapped_vidor_class("cell phone") == "cellphone"


def test_relation_inference_skips_unsupported_detections(tmp_path):
    from PIL import Image
    from test_vidor_relation import _DeterministicVisualEncoder

    from vidquery.domain import BoundingBox, Detection
    from vidquery.vidor_improved import ImprovedVidORRelationPredictor

    checkpoint = Path(
        "data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt"
    )
    metadata = checkpoint.with_suffix(".metadata.json")
    if not checkpoint.is_file() or not metadata.is_file():
        pytest.skip("trained VidOR pair-visual GNN checkpoint is not present")

    def detection(identifier: str, label: str, class_id: int, x1: float) -> Detection:
        return Detection(
            detection_id=identifier,
            frame_id="frame-1",
            video_id="video-1",
            timestamp=1.0,
            class_id=class_id,
            class_label=label,
            confidence=0.9,
            pixel_bbox=BoundingBox(x1=x1 * 100, y1=10, x2=x1 * 100 + 30, y2=90),
            normalized_bbox=BoundingBox(x1=x1, y1=0.1, x2=x1 + 0.3, y2=0.9),
            centroid=(x1 * 100 + 15, 50.0),
            centroid_normalized=(x1 + 0.15, 0.5),
        )

    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100), "navy").save(frame)
    predictor = ImprovedVidORRelationPredictor(
        checkpoint,
        metadata,
        device="cpu",
        expected_model_version="vidor-relation-pair-visual-gat-v2",
        visual_encoder=_DeterministicVisualEncoder(),
    )
    person = detection("person-1", "person", 0, 0.1)
    mouse = detection("mouse-1", "mouse", 64, 0.45)
    chair = detection("chair-1", "chair", 56, 0.6)
    # Only one mappable detection: no pair, no crash, no relabelling.
    assert predictor.predict_detections([person, mouse], frame_path=frame) == []
    rows = predictor.predict_detections([person, mouse, chair], frame_path=frame)
    assert all("mouse-1" not in (row["source_id"], row["target_id"]) for row in rows)
    assert all(row["source_class"] != "hamster/rat" for row in rows)


# --------------------------------------------------------------------------
# Sentence-embedding cache resolution and the evaluation guard

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _fake_hub_cache(root: Path) -> Path:
    (root / ("models--" + MODEL.replace("/", "--")) / "snapshots").mkdir(parents=True)
    return root


def test_cache_resolution_prefers_configured_then_falls_back_to_hf_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    configured = tmp_path / "configured"
    configured.mkdir()
    default = _fake_hub_cache(tmp_path / "hub")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(default))
    assert resolve_sentence_cache_folder(MODEL, configured) == default
    _fake_hub_cache(configured)
    assert resolve_sentence_cache_folder(MODEL, configured) == configured
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    other = tmp_path / "other"
    assert resolve_sentence_cache_folder(MODEL, other) == other
    assert resolve_sentence_cache_folder(MODEL, None) is None


def test_evaluation_refuses_silent_hashing_fallback(tmp_path, settings_factory):
    engine = LocalHybridSearchEngine(SQLiteRepository(tmp_path / "x.sqlite3"))
    assert isinstance(engine.encoder, HashingTextEncoder)
    with pytest.raises(RuntimeError, match="fell back to hashing"):
        require_semantic_encoder(
            engine, settings_factory(semantic_retrieval_mode="sentence_transformer")
        )
    require_semantic_encoder(engine, settings_factory(semantic_retrieval_mode="hashing"))


# --------------------------------------------------------------------------
# Calibration-fitted thresholds


def test_thresholds_load_and_validate(tmp_path):
    path = tmp_path / "priors.json"
    path.write_text(json.dumps({"priors": {"gnn_action_model": 0.2}}), encoding="utf-8")
    assert load_reliability_thresholds(path) is None
    path.write_text(
        json.dumps(
            {
                "priors": {"gnn_action_model": 0.2},
                "thresholds": {"evidence_supported_threshold": 0.3, "evidence_weak_threshold": 0.1},
            }
        ),
        encoding="utf-8",
    )
    assert load_reliability_thresholds(path) == (0.3, 0.1)
    path.write_text(
        json.dumps(
            {"thresholds": {"evidence_supported_threshold": 0.1, "evidence_weak_threshold": 0.3}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="weak <= supported"):
        load_reliability_thresholds(path)


def test_priors_file_thresholds_reach_the_engine(tmp_path, settings_factory):
    path = tmp_path / "priors.json"
    path.write_text(
        json.dumps(
            {
                "priors": {"gnn_action_model": 0.2},
                "thresholds": {"evidence_supported_threshold": 0.3, "evidence_weak_threshold": 0.1},
            }
        ),
        encoding="utf-8",
    )
    repository = SQLiteRepository(tmp_path / "x.sqlite3")
    engine = configured_search_engine(repository, settings_factory(reliability_priors_path=path))
    assert (engine.evidence_supported_threshold, engine.evidence_weak_threshold) == (0.3, 0.1)
    assert engine.reliability_priors == {"gnn_action_model": 0.2}
    default = configured_search_engine(repository, settings_factory())
    assert default.evidence_weak_threshold == 0.15 and default.reliability_priors is None


def _decision(confidence, correct, negative=False):
    return {"max_confidence": confidence, "top1_correct": correct, "negative": negative}


def test_abstention_threshold_maximises_effective_reliability():
    rows = [
        _decision(0.9, True),
        _decision(0.8, True),
        _decision(0.5, False),
        _decision(0.4, False, negative=True),
        _decision(0.2, False),
        _decision(None, False),
    ]
    fitted = fit_abstention_thresholds(rows, supported_precision=0.8)
    # Answering only conf >= 0.8 scores +2; answering more adds only wrong answers.
    assert fitted["evidence_weak_threshold"] == 0.8
    assert fitted["evidence_supported_threshold"] == 0.8
    assert fitted["calibration_coverage"] == round(2 / 6, 4)


def test_threshold_abstains_on_everything_when_every_answer_is_wrong():
    rows = [_decision(0.3, False), _decision(0.2, False, negative=True)]
    fitted = fit_abstention_thresholds(rows)
    assert fitted["evidence_weak_threshold"] > 0.3
    assert fitted["evidence_supported_threshold"] >= fitted["evidence_weak_threshold"]
    assert fitted["calibration_coverage"] == 0.0
    with pytest.raises(ValueError):
        fit_abstention_thresholds([])


# --------------------------------------------------------------------------
# Evaluator semantics


def _moment(verdict, confidence, start=0.0, end=2.0, point=1.0):
    return RankedMoment(
        video_id="v",
        segment_start=0.0,
        segment_end=5.0,
        time_point=point,
        interval_start=start,
        interval_end=end,
        score=0.5,
        verdict=verdict,
        confidence=confidence,
        precision="frame",
    )


def _run(*moments, interpretation="single"):
    return MethodRun(
        ranked=list(moments),
        selected_intent=None,
        selected_hypothesis_id=None,
        interpretation=interpretation,
    )


POSITIVE = {
    "id": "p",
    "source_video_id": "src",
    "relevant": [{"source_video_id": "src", "peak_time": 1.0, "start_time": 0.5, "end_time": 1.5}],
}
NEGATIVE = {"id": "n", "source_video_id": "src", "negative": True}


def _metrics(item, run):
    return _query_metrics(item, run, {"v": "src"})


def test_correct_and_incorrect_positive_answers():
    correct = _metrics(POSITIVE, _run(_moment("supported", 0.8)))
    assert correct["precision_at_1"] == 1.0 and correct["confident_wrong"] is False
    wrong = _metrics(POSITIVE, _run(_moment("supported", 0.8, 10, 12, 11)))
    assert wrong["precision_at_1"] == 0.0 and wrong["confident_wrong"] is True


def test_unnecessary_abstention_is_not_a_correct_retrieval():
    run = _run(_moment("insufficient", 0.05), interpretation="insufficient_evidence")
    metrics = _metrics(POSITIVE, run)
    assert metrics["abstained"] is True
    assert metrics["precision_at_1"] == 0.0
    assert metrics["ranking_precision_at_1"] == 1.0
    assert metrics["false_abstention"] is True


def test_missing_results_count_as_abstention_without_credit():
    metrics = _metrics(POSITIVE, _run())
    assert metrics["abstained"] is True and metrics["precision_at_1"] == 0.0
    assert metrics["confidence"] == 0.0
    negative = _metrics(NEGATIVE, _run())
    assert negative["negative_rejected"] is True and negative["weak_answer"] is False


def test_strong_and_weak_answers_to_negatives_are_both_failures():
    strong = _metrics(NEGATIVE, _run(_moment("supported", 0.9)))
    weak = _metrics(NEGATIVE, _run(_moment("weak", 0.3)))
    assert strong["confident_wrong"] is True and strong["negative_rejected"] is False
    assert weak["weak_answer"] is True and weak["negative_rejected"] is False
    rows = [{"negative": True, "metrics": strong}, {"negative": True, "metrics": weak}]
    assert _selective_metrics(rows)["selective_risk"] == 1.0


def test_tied_confidences_use_expected_risk():
    right = {"negative": False, "metrics": _metrics(POSITIVE, _run(_moment("weak", 0.5)))}
    wrong = {"negative": True, "metrics": _metrics(NEGATIVE, _run(_moment("weak", 0.5)))}
    forward = _selective_metrics([right, wrong])
    backward = _selective_metrics([wrong, right])
    assert forward["aurc"] == backward["aurc"] == 0.5
    assert forward["answerability_auroc"] == 0.5
