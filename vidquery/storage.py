from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from .domain import (
    CanonicalSegment,
    EntityAppearance,
    ProcessingState,
    VideoRecord,
    utc_now,
)


class VideoNotFoundError(KeyError):
    pass


class SQLiteRepository:
    """Persistent local registry and canonical-segment store.

    SQLite is the reliable demo/system-of-record fallback.  Neo4j is an optional
    query/index mirror and its availability never determines whether uploads
    disappear after an API restart.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    duration REAL NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    has_audio INTEGER NOT NULL,
                    upload_time TEXT NOT NULL,
                    updated_time TEXT NOT NULL,
                    state TEXT NOT NULL,
                    current_stage TEXT,
                    failure_message TEXT,
                    failure_detail TEXT,
                    warnings_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    transcript TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS segments_video_time_idx
                ON segments(video_id, start_time, end_time);

                CREATE INDEX IF NOT EXISTS videos_state_idx
                ON videos(state);

                CREATE TABLE IF NOT EXISTS entity_appearances (
                    appearance_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    class_label TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    appearance_embedding_model TEXT NOT NULL,
                    appearance_embedding_version TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    source_method TEXT NOT NULL,
                    FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
                    UNIQUE(video_id, track_id)
                );

                CREATE INDEX IF NOT EXISTS entity_appearances_video_track_idx
                ON entity_appearances(video_id, track_id);

                CREATE INDEX IF NOT EXISTS entity_appearances_class_idx
                ON entity_appearances(class_label);
                """
            )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> VideoRecord:
        data = dict(row)
        data["has_audio"] = bool(data["has_audio"])
        data["warnings"] = json.loads(data.pop("warnings_json") or "[]")
        return VideoRecord.model_validate(data)

    def create_video(self, record: VideoRecord) -> VideoRecord:
        data = record.model_dump(mode="json")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (
                    video_id, display_name, original_filename, stored_path,
                    content_sha256, duration, width, height, fps, has_audio,
                    upload_time, updated_time, state, current_stage,
                    failure_message, failure_detail, warnings_json
                ) VALUES (
                    :video_id, :display_name, :original_filename, :stored_path,
                    :content_sha256, :duration, :width, :height, :fps, :has_audio,
                    :upload_time, :updated_time, :state, :current_stage,
                    :failure_message, :failure_detail, :warnings_json
                )
                """,
                {
                    **data,
                    "state": record.state.value,
                    "has_audio": int(record.has_audio),
                    "failure_detail": record.failure_detail,
                    "warnings_json": json.dumps(record.warnings),
                },
            )
        return record

    def find_by_hash(self, content_sha256: str) -> VideoRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE content_sha256 = ?", (content_sha256,)
            ).fetchone()
        return self._record_from_row(row) if row else None

    def get_video(self, video_id: str) -> VideoRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
        if row is None:
            raise VideoNotFoundError(video_id)
        return self._record_from_row(row)

    def list_videos(self) -> list[VideoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM videos ORDER BY upload_time DESC"
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def update_state(
        self,
        video_id: str,
        state: ProcessingState,
        *,
        current_stage: str | None = None,
        failure_message: str | None = None,
        failure_detail: str | None = None,
        warnings: list[str] | None = None,
    ) -> VideoRecord:
        existing = self.get_video(video_id)
        warning_values = existing.warnings if warnings is None else warnings
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE videos
                SET state = ?, current_stage = ?, failure_message = ?,
                    failure_detail = ?, warnings_json = ?, updated_time = ?
                WHERE video_id = ?
                """,
                (
                    state.value,
                    current_stage,
                    failure_message,
                    failure_detail,
                    json.dumps(warning_values),
                    utc_now().isoformat(),
                    video_id,
                ),
            )
            if cursor.rowcount != 1:
                raise VideoNotFoundError(video_id)
        return self.get_video(video_id)

    def replace_segments(self, video_id: str, segments: Iterable[CanonicalSegment]) -> int:
        payloads = list(segments)
        self.get_video(video_id)
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            connection.executemany(
                """
                INSERT INTO segments (
                    segment_id, video_id, start_time, end_time, transcript, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        segment.segment_id,
                        segment.video_id,
                        segment.start_time,
                        segment.end_time,
                        segment.transcript,
                        segment.model_dump_json(),
                    )
                    for segment in payloads
                ],
            )
        return len(payloads)

    def list_segments(self, video_ids: list[str] | None = None) -> list[CanonicalSegment]:
        query = "SELECT payload_json FROM segments"
        params: tuple[str, ...] = ()
        if video_ids:
            placeholders = ",".join("?" for _ in video_ids)
            query += f" WHERE video_id IN ({placeholders})"
            params = tuple(video_ids)
        query += " ORDER BY video_id, start_time"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [CanonicalSegment.model_validate_json(row["payload_json"]) for row in rows]

    def replace_entity_appearances(
        self, video_id: str, appearances: Iterable[EntityAppearance]
    ) -> int:
        payloads = list(appearances)
        self.get_video(video_id)
        if any(item.video_id != video_id for item in payloads):
            raise ValueError("appearance video_id does not match replacement video")
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM entity_appearances WHERE video_id = ?", (video_id,)
            )
            connection.executemany(
                """
                INSERT INTO entity_appearances (
                    appearance_id, video_id, track_id, class_label,
                    embedding_json, appearance_embedding_model,
                    appearance_embedding_version, observation_count,
                    attributes_json, source_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.appearance_id,
                        item.video_id,
                        item.track_id,
                        item.class_label,
                        json.dumps(item.embedding, separators=(",", ":")),
                        item.appearance_embedding_model,
                        item.appearance_embedding_version,
                        item.observation_count,
                        json.dumps(
                            {
                                key: value.model_dump(mode="json")
                                for key, value in item.attributes.items()
                            },
                            separators=(",", ":"),
                        ),
                        item.source_method.value,
                    )
                    for item in payloads
                ],
            )
        return len(payloads)

    def list_entity_appearances(
        self, video_ids: list[str] | None = None
    ) -> list[EntityAppearance]:
        query = "SELECT * FROM entity_appearances"
        params: tuple[str, ...] = ()
        if video_ids:
            placeholders = ",".join("?" for _ in video_ids)
            query += f" WHERE video_id IN ({placeholders})"
            params = tuple(video_ids)
        query += " ORDER BY video_id, track_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        output: list[EntityAppearance] = []
        for row in rows:
            data = dict(row)
            data["embedding"] = json.loads(data.pop("embedding_json"))
            data["attributes"] = json.loads(data.pop("attributes_json") or "{}")
            output.append(EntityAppearance.model_validate(data))
        return output

    def health_check(self) -> bool:
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row["ok"] == 1)
        except sqlite3.Error:
            return False
