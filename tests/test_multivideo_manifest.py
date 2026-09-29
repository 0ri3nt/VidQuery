import json
from collections import Counter
from pathlib import Path


def test_multivideo_manifest_has_required_independent_coverage():
    manifest = json.loads(
        Path("evaluation/queries_multivideo.json").read_text(encoding="utf-8")
    )
    queries = manifest["queries"]
    required = {
        "id",
        "query",
        "category",
        "relevant_video_id",
        "ground_truth_start_time",
        "ground_truth_end_time",
        "timestamp_tolerance",
        "expected_entities",
        "expected_relation",
        "expected_action",
        "expected_spoken_concept",
        "expected_speaker",
        "negative",
        "relevant",
    }

    assert len(queries) >= 30
    assert len({item["relevant_video_id"] for item in queries}) >= 3
    assert all(required <= item.keys() for item in queries)
    assert all(count >= 5 for count in Counter(item["category"] for item in queries).values())
    assert any(item["negative"] for item in queries)
    assert len({item["id"] for item in queries}) == len(queries)
