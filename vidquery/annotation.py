"""Persistent store for the hand-labelled Review-2 challenge manifest.

The annotation page writes here; ``evaluate-challenge`` reads the same file.
Entries are validated on every write so the file is always loadable by
``challenge_evaluation.load_manifest`` once it has at least one query.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .challenge_evaluation import (
    MINIMUM_COUNTS,
    SCHEMA_VERSION,
    VIDEO_SPLITS,
    ChallengeManifestError,
    validate_manifest,
)
from .domain import StrictModel

# speech / ocr / object let a lecture benchmark report those query types separately;
# "timestamp" remains the generic precise-moment category.
ChallengeCategory = Literal[
    "ambiguous", "timestamp", "action_relation", "speech", "ocr", "object", "negative"
]
VideoSplit = Literal["calibration", "test"]
HypothesisId = Literal["visible_action", "spoken_mention", "visible_object", "visible_text"]
QueryIntent = Literal[
    "transcript_search",
    "visual_search",
    "action_search",
    "relationship_search",
    "speaker_search",
    "ocr_search",
    "multimodal_search",
]
TARGET_TOTAL_QUERIES = 60
MANIFEST_DESCRIPTION = (
    "Review-2 challenge benchmark, labelled with the VidQuery annotation page. "
    "Ground truth is a manually marked peak time and interval per query."
)


class AnnotatedMoment(StrictModel):
    peak_time: float = Field(ge=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> AnnotatedMoment:
        if not self.start_time <= self.peak_time <= self.end_time:
            raise ValueError("peak_time must lie between start_time and end_time")
        return self


class AnnotatedQuery(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    query: str = Field(min_length=1, max_length=500)
    category: ChallengeCategory
    source_video_id: str = Field(min_length=1, max_length=255)
    expected_intents: list[QueryIntent] = Field(default_factory=list)
    expected_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    tolerance_seconds: float = Field(default=2.0, ge=0.5, le=10.0)
    relevant: list[AnnotatedMoment] = Field(default_factory=list, max_length=20)
    annotator: str = Field(default="", max_length=32)
    # Second person who independently checked answerability and the interval.
    verified_by: str = Field(default="", max_length=32)
    notes: str = Field(default="", max_length=1000)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def consistent(self) -> AnnotatedQuery:
        if self.category == "negative" and self.relevant:
            raise ValueError("negative queries must not list relevant moments")
        if self.category != "negative" and not self.relevant:
            raise ValueError("positive queries need at least one relevant moment")
        return self

    def to_manifest_entry(self) -> dict[str, Any]:
        entry = self.model_dump()
        entry["negative"] = self.category == "negative"
        entry["relevant"] = [
            {"source_video_id": self.source_video_id, **moment.model_dump()}
            for moment in self.relevant
        ]
        return entry



class VideoSplitUpdate(StrictModel):
    split: VideoSplit


class ChallengeManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": SCHEMA_VERSION,
                "description": MANIFEST_DESCRIPTION,
                "videos": {},
                "queries": [],
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload.setdefault("videos", {})
        payload.setdefault("queries", [])
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".challenge-", suffix=".json"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def upsert_query(self, query: AnnotatedQuery) -> dict[str, Any]:
        with self._lock:
            payload = self.load()
            entry = query.to_manifest_entry()
            queries = [item for item in payload["queries"] if item.get("id") != query.id]
            replaced = len(queries) != len(payload["queries"])
            queries.append(entry)
            payload["queries"] = queries
            self._write(payload)
            return {"query": entry, "replaced": replaced}

    def delete_query(self, query_id: str) -> bool:
        with self._lock:
            payload = self.load()
            remaining = [item for item in payload["queries"] if item.get("id") != query_id]
            if len(remaining) == len(payload["queries"]):
                return False
            payload["queries"] = remaining
            self._write(payload)
            return True

    def set_video_split(self, source_id: str, split: VideoSplit) -> None:
        if split not in VIDEO_SPLITS:
            raise ValueError(f"split must be one of {VIDEO_SPLITS}")
        with self._lock:
            payload = self.load()
            payload["videos"][source_id] = {"split": split}
            self._write(payload)

    def summary(self) -> dict[str, Any]:
        payload = self.load()
        queries = payload["queries"]
        splits = payload["videos"]
        by_category = Counter(str(item.get("category")) for item in queries)
        by_split = Counter(
            (splits.get(str(item.get("source_video_id"))) or {}).get("split", "unassigned")
            for item in queries
        )
        warnings: list[str] = []
        runnable = False
        if queries:
            try:
                warnings = validate_manifest(payload)
                runnable = True
            except ChallengeManifestError as exc:
                warnings = [str(exc)]
        else:
            warnings = ["no queries labelled yet"]
        return {
            "manifest_path": str(self.path),
            "query_count": len(queries),
            "target_query_count": TARGET_TOTAL_QUERIES,
            "category_counts": {name: by_category.get(name, 0) for name in MINIMUM_COUNTS},
            "category_targets": dict(MINIMUM_COUNTS),
            "split_counts": dict(by_split),
            "runnable": runnable,
            "warnings": warnings,
        }
