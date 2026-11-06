"""
Neura Conversation Store
========================

Persistent SQLite storage for conversations and messages.

This module is intentionally independent from the HTTP server.
It provides the single persistence layer used by future:

- chat sessions
- web UI
- agents
- memory
- dataset export
- training pipelines
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConversationStoreError(Exception):
    """Base exception for conversation storage errors."""


class ConversationNotFoundError(ConversationStoreError):
    """Raised when a conversation does not exist."""


class ConversationStore:
    """
    Persistent conversation storage backed by SQLite.

    Default database:

        data/neura.db

    The database is created automatically.
    """

    def __init__(
        self,
        db_path: str | Path = "data/neura.db",
    ) -> None:
        self.db_path = Path(db_path)

        if self.db_path.parent:
            self.db_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
        )

        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA foreign_keys = ON")

        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',

                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE,

                    UNIQUE (
                        conversation_id,
                        sequence
                    )
                );

                CREATE INDEX IF NOT EXISTS
                    idx_conversations_user_updated
                ON conversations (
                    user_id,
                    updated_at
                );

                CREATE INDEX IF NOT EXISTS
                    idx_messages_conversation_sequence
                ON messages (
                    conversation_id,
                    sequence
                );
                """
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    @staticmethod
    def _encode_metadata(
        metadata: dict[str, Any] | None,
    ) -> str:
        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise TypeError(
                "metadata must be a dictionary"
            )

        return json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_metadata(
        value: str,
    ) -> dict[str, Any]:
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}

        return (
            decoded
            if isinstance(decoded, dict)
            else {}
        )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        user_id: str = "default",
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(user_id, str):
            raise TypeError(
                "user_id must be a string"
            )

        if not user_id:
            raise ValueError(
                "user_id cannot be empty"
            )

        with self._connect() as connection:
            user = connection.execute(
                "SELECT 1 FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

        if user is None:
            raise ValueError("user_id does not exist")

        if not isinstance(title, str):
            raise TypeError(
                "title must be a string"
            )

        conversation_id = self._new_id(
            "conv"
        )

        timestamp = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id,
                    user_id,
                    title,
                    created_at,
                    updated_at,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                    title,
                    timestamp,
                    timestamp,
                    self._encode_metadata(metadata),
                ),
            )

        return conversation_id

    def conversation_exists(
        self,
        conversation_id: str,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()

        return row is not None

    def conversation_belongs_to_user(
        self,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        """Return True only when the conversation belongs to user_id."""
        conversation = self.get_conversation(conversation_id)

        if conversation is None:
            return False

        return conversation["user_id"] == user_id

    def get_conversation(
        self,
        conversation_id: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    title,
                    created_at,
                    updated_at,
                    metadata
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()

        if row is None:
            raise ConversationNotFoundError(
                conversation_id
            )

        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": self._decode_metadata(
                row["metadata"]
            ),
        }

    def list_conversations(
        self,
        user_id: str = "default",
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    user_id,
                    title,
                    created_at,
                    updated_at,
                    metadata
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": self._decode_metadata(
                    row["metadata"]
                ),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if role not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError(
                f"unsupported role: {role}"
            )

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string"
            )

        if not self.conversation_exists(
            conversation_id
        ):
            raise ConversationNotFoundError(
                conversation_id
            )

        message_id = self._new_id(
            "msg"
        )

        timestamp = self._now()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(
                    MAX(sequence),
                    -1
                ) + 1 AS next_sequence
                FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

            sequence = int(
                row["next_sequence"]
            )

            connection.execute(
                """
                INSERT INTO messages (
                    id,
                    conversation_id,
                    sequence,
                    role,
                    content,
                    created_at,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    sequence,
                    role,
                    content,
                    timestamp,
                    self._encode_metadata(metadata),
                ),
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    timestamp,
                    conversation_id,
                ),
            )

        return message_id

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        if not self.conversation_exists(
            conversation_id
        ):
            raise ConversationNotFoundError(
                conversation_id
            )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    sequence,
                    role,
                    content,
                    created_at,
                    metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY sequence ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "conversation_id": row[
                    "conversation_id"
                ],
                "sequence": row["sequence"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "metadata": self._decode_metadata(
                    row["metadata"]
                ),
            }
            for row in rows
        ]

    def message_count(
        self,
        conversation_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

        return int(row["count"])

    # ------------------------------------------------------------------
    # Combined conversation
    # ------------------------------------------------------------------

    def get_conversation_with_messages(
        self,
        conversation_id: str,
    ) -> dict[str, Any]:
        conversation = self.get_conversation(
            conversation_id
        )

        conversation["messages"] = (
            self.get_messages(
                conversation_id
            )
        )

        return conversation

    # ------------------------------------------------------------------
    # Deletion / clearing
    # ------------------------------------------------------------------

    def clear_messages(
        self,
        conversation_id: str,
    ) -> None:
        if not self.conversation_exists(
            conversation_id
        ):
            raise ConversationNotFoundError(
                conversation_id
            )

        timestamp = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    timestamp,
                    conversation_id,
                ),
            )

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            )

        if cursor.rowcount == 0:
            raise ConversationNotFoundError(
                conversation_id
            )

    # ------------------------------------------------------------------
    # Dataset export foundation
    # ------------------------------------------------------------------

    def export_messages(
        self,
        conversation_id: str,
    ) -> list[dict[str, str]]:
        """
        Return training-friendly role/content pairs.

        IDs, timestamps and database-specific metadata
        are intentionally excluded from this representation.
        """

        messages = self.get_messages(
            conversation_id
        )

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]


__all__ = [
    "ConversationStore",
    "ConversationStoreError",
    "ConversationNotFoundError",
]
