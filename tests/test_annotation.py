from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_challenge_evaluation import _manifest, _repository
from test_media_pipeline_api import StubProcessor, register_sample

from vidquery.annotation import AnnotatedQuery, ChallengeManifestStore, QueryIntent
from vidquery.api import create_app
from vidquery.challenge_evaluation import (
    ChallengeManifestError,
    _bootstrap_interval,
    evaluate_challenge,
    load_manifest,
    validate_manifest,
)
from vidquery.domain import QueryPlan
from vidquery.query_planner import QueryPlanningService
from vidquery.storage import SQLiteRepository


def _query(**overrides) -> AnnotatedQuery:
    fields = {
        "id": "ts-meeting-01",
        "query": "  when is deployment mentioned  ",
        "category": "timestamp",
        "source_video_id": "meeting",
        "expected_intents": ["transcript_search"],
        "relevant": [{"peak_time": 6.3, "start_time": 6.2, "end_time": 6.9}],
        "annotator": "AK",
    }
    fields.update(overrides)
    return AnnotatedQuery(**fields)


def test_intent_vocabulary_matches_query_plan():
    assert set(QueryIntent.__args__) == set(QueryPlan.model_fields["intent"].annotation.__args__)


def test_annotated_query_validation():
    assert _query().query == "when is deployment mentioned"
    with pytest.raises(ValidationError, match="must not list relevant"):
        _query(category="negative")
    with pytest.raises(ValidationError, match="at least one relevant"):
        _query(relevant=[])
    with pytest.raises(ValidationError, match="peak_time must lie"):
        _query(relevant=[{"peak_time": 9.0, "start_time": 6.2, "end_time": 6.9}])
    with pytest.raises(ValidationError):
        _query(id="has spaces")
    with pytest.raises(ValidationError):
        _query(expected_hypothesis_ids=["invented"])


def test_store_round_trip_produces_a_loadable_manifest(tmp_path):
    store = ChallengeManifestStore(tmp_path / "nested" / "challenge.json")
    assert store.summary()["query_count"] == 0
    assert store.summary()["runnable"] is False

    first = store.upsert_query(_query())
    assert first["replaced"] is False
    assert first["query"]["negative"] is False
    assert first["query"]["relevant"][0]["source_video_id"] == "meeting"
    store.upsert_query(
        _query(id="neg-meeting-01", category="negative", relevant=[], query="riding a bicycle")
    )
    again = store.upsert_query(_query(query="when does deployment come up"))
    assert again["replaced"] is True
    store.set_video_split("meeting", "test")

    manifest = load_manifest(store.path)
    assert [item["id"] for item in manifest["queries"]] == ["neg-meeting-01", "ts-meeting-01"]
    assert manifest["videos"] == {"meeting": {"split": "test"}}
    summary = store.summary()
    assert summary["runnable"] is True
    assert summary["category_counts"]["timestamp"] == 1
    assert summary["category_counts"]["negative"] == 1
    assert summary["split_counts"] == {"test": 2}

    assert store.delete_query("ts-meeting-01") is True
    assert store.delete_query("ts-meeting-01") is False
    assert not list(store.path.parent.glob(".challenge-*"))


def test_manifest_split_validation():
    with pytest.raises(ChallengeManifestError, match="split"):
        validate_manifest(_manifest(videos={"meeting": {"split": "train"}}))
    warnings = validate_manifest(_manifest())
    assert any("without a calibration/test split: meeting" in item for item in warnings)
    assert any("action_relation" in item for item in warnings)
    assert not any(
        "without a calibration/test split" in item
        for item in validate_manifest(_manifest(videos={"meeting": {"split": "test"}}))
    )


def test_evaluation_filters_by_split_and_reports_intervals(tmp_path, settings_factory):
    repository = _repository(tmp_path)
    dataset = tmp_path / "challenge.json"
    dataset.write_text(
        json.dumps(_manifest(videos={"meeting": {"split": "test"}})), encoding="utf-8"
    )
    planner = QueryPlanningService(settings_factory(enable_query_planner=False))

    result = evaluate_challenge(repository, dataset, planner=planner, split="test")
    assert result["split"] == "test"
    assert result["query_count"] == 4
    intervals = result["summary"]["review2_full"]["overall"]["bootstrap_95_ci"]
    assert intervals["precision_at_1"] == [1.0, 1.0]
    assert intervals["negative_rejection_rate"] == [1.0, 1.0]

    with pytest.raises(ChallengeManifestError, match="no queries belong"):
        evaluate_challenge(repository, dataset, planner=planner, split="calibration")
    with pytest.raises(ChallengeManifestError, match="split must be one of"):
        evaluate_challenge(repository, dataset, planner=planner, split="train")


def test_bootstrap_interval_is_deterministic_and_bounded():
    values = [1.0] * 6 + [0.0] * 4
    low, high = _bootstrap_interval(values)
    assert 0.0 <= low < 0.6 < high <= 1.0
    assert _bootstrap_interval(values) == [low, high]
    assert _bootstrap_interval([1.0]) is None


def test_annotation_api(settings_factory, sample_video):
    settings = settings_factory()
    repository = SQLiteRepository(settings.database_path)
    video = register_sample(settings, repository, sample_video)
    client = TestClient(create_app(settings, repository, StubProcessor(repository)))

    listing = client.get("/api/annotations").json()
    assert listing["queries"] == []
    assert listing["videos"][0]["source_video_id"] == "sample"
    assert listing["videos"][0]["stream_url"] == f"/api/videos/{video.video_id}/stream"

    body = {
        "id": "ts-sample-01",
        "query": "when does the box move",
        "category": "timestamp",
        "source_video_id": "sample",
        "relevant": [{"peak_time": 0.5, "start_time": 0.2, "end_time": 1.0}],
    }
    saved = client.put("/api/annotations/queries/ts-sample-01", json=body)
    assert saved.status_code == 200
    assert saved.json()["summary"]["query_count"] == 1

    assert client.put("/api/annotations/queries/other-id", json=body).status_code == 422
    unknown = {**body, "source_video_id": "not-indexed"}
    assert client.put("/api/annotations/queries/ts-sample-01", json=unknown).status_code == 422
    too_late = {**body, "relevant": [{"peak_time": 50, "start_time": 49, "end_time": 60}]}
    assert client.put("/api/annotations/queries/ts-sample-01", json=too_late).status_code == 422

    split = client.put("/api/annotations/videos/sample/split", json={"split": "calibration"})
    assert split.status_code == 200
    assert client.get("/api/annotations").json()["videos"][0]["split"] == "calibration"
    assert (
        client.put("/api/annotations/videos/missing/split", json={"split": "test"}).status_code
        == 404
    )
    assert (
        client.put("/api/annotations/videos/sample/split", json={"split": "train"}).status_code
        == 422
    )

    assert client.delete("/api/annotations/queries/ts-sample-01").status_code == 200
    assert client.delete("/api/annotations/queries/ts-sample-01").status_code == 404
    assert json.loads(settings.challenge_manifest_path.read_text())["queries"] == []

    page = client.get("/").text
    assert 'id="annotate-view"' in page
    assert client.get("/assets/annotate.js").status_code == 200
