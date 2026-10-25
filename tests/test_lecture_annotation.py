from __future__ import annotations

import pytest
from pydantic import ValidationError

from vidquery.annotation import AnnotatedQuery, ChallengeManifestStore
from vidquery.challenge_evaluation import validate_manifest


def _query(**overrides) -> AnnotatedQuery:
    fields = {
        "id": "sp-lecture-01",
        "query": "when does she say binary search tree",
        "category": "speech",
        "source_video_id": "lecture_os_01",
        "relevant": [{"peak_time": 12.4, "start_time": 12.0, "end_time": 13.1}],
        "annotator": "AS",
        "verified_by": "AK",
    }
    fields.update(overrides)
    return AnnotatedQuery(**fields)


@pytest.mark.parametrize("category", ["speech", "ocr", "object", "timestamp", "action_relation"])
def test_lecture_categories_require_relevant_moments(category):
    assert _query(category=category).category == category
    with pytest.raises(ValidationError, match="at least one relevant"):
        _query(category=category, relevant=[])


def test_unanswerable_queries_use_the_negative_category():
    negative = _query(id="neg-lecture-01", category="negative", relevant=[])
    assert negative.to_manifest_entry()["negative"] is True
    with pytest.raises(ValidationError):
        _query(category="unanswerable", relevant=[])


def test_verification_and_category_survive_into_a_valid_manifest(tmp_path):
    store = ChallengeManifestStore(tmp_path / "challenge.json")
    store.upsert_query(_query())
    store.upsert_query(_query(id="ocr-lecture-01", category="ocr", query="slide that says Kubernetes"))
    store.set_video_split("lecture_os_01", "test")
    manifest = store.load()
    validate_manifest(manifest)
    by_id = {item["id"]: item for item in manifest["queries"]}
    assert by_id["sp-lecture-01"]["verified_by"] == "AK"
    assert by_id["ocr-lecture-01"]["category"] == "ocr"
    assert manifest["videos"]["lecture_os_01"]["split"] == "test"
