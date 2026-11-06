"""
Neura Learning Memory
=====================

Persistent, SQLite-backed memory extracted from direct conversations.

Important:
- This does NOT modify model weights.
- It stores structured facts/signals for later retrieval.
- Low-confidence or empty information is ignored.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LearningMemory:
    def __init__(self, db_path: str | Path = "data/neura.db") -> None:
        self.db_path = Path(db_path)

        if self.db_path.parent:
            self.db_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _id() -> str:
        return f"learn_{uuid.uuid4().hex}"

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS learning_memory (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    source TEXT NOT NULL DEFAULT 'conversation',
                    conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(user_id, normalized)
                );

                CREATE INDEX IF NOT EXISTS
                    idx_learning_memory_user
                    ON learning_memory(user_id, updated_at);

                CREATE INDEX IF NOT EXISTS
                    idx_learning_memory_type
                    ON learning_memory(user_id, memory_type);
                """
            )

    @staticmethod
    def _normalize(value: str) -> str:
        value = str(value or "").strip().lower()
        value = re.sub(r"\s+", " ", value)
        return value

    def remember(
        self,
        *,
        user_id: str,
        memory_type: str,
        content: str,
        confidence: float = 0.5,
        source: str = "conversation",
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        content = str(content or "").strip()

        if not user_id or not content:
            return False

        confidence = max(0.0, min(1.0, float(confidence)))

        normalized = self._normalize(content)
        timestamp = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO learning_memory (
                    id,
                    user_id,
                    memory_type,
                    content,
                    normalized,
                    confidence,
                    source,
                    conversation_id,
                    created_at,
                    updated_at,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, normalized)
                DO UPDATE SET
                    confidence = MAX(
                        learning_memory.confidence,
                        excluded.confidence
                    ),
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    self._id(),
                    user_id,
                    memory_type,
                    content,
                    normalized,
                    confidence,
                    source,
                    conversation_id,
                    timestamp,
                    timestamp,
                    json.dumps(
                        metadata or {},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )

        return True

    def search(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        query = self._normalize(query)

        if not user_id or not query:
            return []

        terms = [
            term
            for term in re.findall(r"[\w\u0600-\u06FF]+", query)
            if len(term) >= 2
        ]

        if not terms:
            return []

        clauses = []
        params: list[Any] = [user_id]

        for term in terms[:12]:
            clauses.append("normalized LIKE ?")
            params.append(f"%{term}%")

        sql = f"""
            SELECT
                id,
                memory_type,
                content,
                confidence,
                source,
                conversation_id,
                created_at,
                updated_at
            FROM learning_memory
            WHERE user_id = ?
              AND ({' OR '.join(clauses)})
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
        """

        params.append(max(1, min(int(limit), 20)))

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                tuple(params),
            ).fetchall()

        return [dict(row) for row in rows]

    def list_user_memories(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    memory_type,
                    content,
                    confidence,
                    source,
                    conversation_id,
                    created_at,
                    updated_at
                FROM learning_memory
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (
                    user_id,
                    max(1, min(int(limit), 200)),
                ),
            ).fetchall()

        return [dict(row) for row in rows]

    def learn_from_message(
        self,
        user_id: str,
        text: str,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract explicit high-signal facts from a user message."""

        text = str(text or "").strip()
        if not text:
            return []

        patterns = (
            (
                "user_name",
                re.compile(
                    r"^(?:اسمي|انا اسمي|أنا اسمي)\s+(.+)$",
                    re.IGNORECASE,
                ),
                0.98,
            ),
            (
                "user_preference",
                re.compile(
                    r"^(?:احب|أحب|افضل|أفضل|أفضل أن|I prefer|I like)\s+(.+)$",
                    re.IGNORECASE,
                ),
                0.90,
            ),
            (
                "user_interest",
                re.compile(
                    r"^(?:اهتم بـ|أهتم بـ|مهتم بـ|I am interested in|I'm interested in)\s+(.+)$",
                    re.IGNORECASE,
                ),
                0.90,
            ),
            (
                "user_project",
                re.compile(
                    r"^(?:(?:اعمل على|أعمل على|مشروعي)\s+(?:هو\s+|هي\s+)?|(?:my project is|I am working on)\s+)(.+)$",
                    re.IGNORECASE,
                ),
                0.92,
            ),
        )

        learned: list[dict[str, Any]] = []

        for memory_type, pattern, confidence in patterns:
            match = pattern.match(text)
            if not match:
                continue

            value = match.group(1).strip(" .،,!?؟")
            if not value or len(value) > 300:
                continue

            if self.remember(
                user_id=user_id,
                memory_type=memory_type,
                content=value,
                confidence=confidence,
                source="conversation",
                conversation_id=conversation_id,
            ):
                learned.append(
                    {
                        "memory_type": memory_type,
                        "content": value,
                        "confidence": confidence,
                    }
                )

        return learned

