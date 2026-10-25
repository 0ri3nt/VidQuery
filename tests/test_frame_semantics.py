from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from test_search_and_storage import evidence_segment, video_record

from vidquery.domain import QueryPlan, SearchRequest
from vidquery.frame_semantics import (
    FrameEmbeddingError,
    FrameSemanticIndex,
    build_frame_embeddings,
    embedding_path,
    load_frame_embeddings,
)
from vidquery.search import LocalHybridSearchEngine
from vidquery.storage import SQLiteRepository


class _ColourEncoder:
    """Embeds an image as its mean RGB colour; texts map to fixed colours."""

    model_version = "fake:test:frozen:appearance-schema-v1"

    def encode_images(self, images):
        output = []
        for image in images:
            vector = np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0)
            output.append((vector / np.linalg.norm(vector)).tolist())
        return output

    def encode_texts(self, texts):
        palette = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0]}
        return [palette.get(text.split()[-1], [0.0, 1.0, 0.0]) for text in texts]


def _frames(root, video_id, colours):
    frames = root / video_id / "frames"
    frames.mkdir(parents=True)
    ids = []
    for index, colour in enumerate(colours):
        frame_id = f"{video_id}_{index * 1000:012d}"
        Image.new("RGB", (8, 8), colour).save(frames / f"{frame_id}.jpg")
        ids.append(frame_id)
    return frames, ids


def test_embeddings_round_trip_and_pick_the_best_frame(tmp_path):
    frames, ids = _frames(tmp_path, "vid", ["red", "blue", "red"])
    encoder = _ColourEncoder()
    report = build_frame_embeddings("vid", frames, encoder, embedding_path(tmp_path, "vid"))
    assert report["frames"] == 3
    loaded = load_frame_embeddings(
        embedding_path(tmp_path, "vid"), expected_model_version=encoder.model_version
    )
    assert loaded["embeddings"].shape == (3, 3)
    index = FrameSemanticIndex(tmp_path, expected_model_version=encoder.model_version)
    similarity, frame = index.best_frame("vid", ids, np.asarray([0.0, 0.0, 1.0], np.float32))
    assert frame == ids[1] and similarity > 0.9
    assert index.best_frame("vid", ["unknown"], np.ones(3, np.float32)) is None
    assert index.best_frame("missing-video", ids, np.ones(3, np.float32)) is None


def test_mismatched_model_version_is_not_used(tmp_path):
    frames, _ = _frames(tmp_path, "vid", ["red"])
    build_frame_embeddings("vid", frames, _ColourEncoder(), embedding_path(tmp_path, "vid"))
    with pytest.raises(FrameEmbeddingError, match="was built with"):
        load_frame_embeddings(embedding_path(tmp_path, "vid"), expected_model_version="other")
    index = FrameSemanticIndex(tmp_path, expected_model_version="other")
    assert index.best_frame("vid", ["x"], np.ones(3, np.float32)) is None
    with pytest.raises(FrameEmbeddingError, match="no frames"):
        build_frame_embeddings("empty", tmp_path / "nothing", _ColourEncoder(), tmp_path / "o.npz")


class _FixedIndex:
    def __init__(self, similarity):
        self.similarity = similarity

    def best_frame(self, video_id, frame_ids, vector):
        return (self.similarity, frame_ids[0]) if frame_ids else None


def _repository(tmp_path):
    repository = SQLiteRepository(tmp_path / "index.sqlite3")
    repository.create_video(video_record())
    segment = evidence_segment()
    repository.replace_segments("video-1", [segment.model_copy(update={"frame_ids": ["f1"]})])
    return repository


def test_frame_similarity_is_off_by_default_and_tagged_when_enabled(tmp_path):
    repository = _repository(tmp_path)
    plan = QueryPlan(intent="visual_search", visual_entities=["bicycle"])
    request = SearchRequest(query="a bicycle", limit=5, ranking="reliability")
    assert LocalHybridSearchEngine(repository).search(request, plan_override=plan).results == []
    engine = LocalHybridSearchEngine(
        repository,
        appearance_encoder=_ColourEncoder(),
        frame_semantic_index=_FixedIndex(0.30),
        soft_constraints=True,
    )
    results = engine.search(request, plan_override=plan).results
    assert len(results) == 1
    sources = {item.source_method for item in results[0].evidence.contributions}
    assert "openclip_frame_similarity" in sources
    assert "visual-text similarity" in results[0].match_reason


def test_soft_constraints_down_weight_instead_of_rejecting(tmp_path):
    repository = _repository(tmp_path)
    plan = QueryPlan(intent="visual_search", visual_entities=["person", "bicycle"])
    request = SearchRequest(query="person with a bicycle", limit=5)
    strict = LocalHybridSearchEngine(repository).search(request, plan_override=plan)
    assert strict.results == []
    soft = LocalHybridSearchEngine(repository, soft_constraints=True).search(
        request, plan_override=plan
    )
    full = LocalHybridSearchEngine(repository).search(
        request, plan_override=QueryPlan(intent="visual_search", visual_entities=["person"])
    )
    assert len(soft.results) == 1
    assert soft.results[0].score < full.results[0].score


def test_linguistic_intents_skip_frame_similarity(tmp_path):
    repository = _repository(tmp_path)
    engine = LocalHybridSearchEngine(
        repository, appearance_encoder=_ColourEncoder(), frame_semantic_index=_FixedIndex(0.9)
    )
    plan = QueryPlan(intent="transcript_search", spoken_terms=["deployment"])
    results = engine.search(SearchRequest(query="deployment"), plan_override=plan).results
    assert results
    sources = {item.source_method for item in results[0].evidence.contributions}
    assert "openclip_frame_similarity" not in sources
    with pytest.raises(ValueError):
        LocalHybridSearchEngine(repository, frame_semantic_floor=0.5, frame_semantic_ceiling=0.4)
