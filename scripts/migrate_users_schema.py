from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB = Path("data/neura.db")
BACKUP = Path("data/neura.db.backup_before_auth_schema")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_default_user_id() -> str:
    # ثابت حتى تكون migration قابلة لإعادة التشغيل بأمان.
    return hashlib.sha256(
        b"neura-default-user-v1"
    ).hexdigest()


def main() -> None:
    print("=" * 68)
    print("NEURA USERS SCHEMA MIGRATION")
    print("=" * 68)

    if not DB.exists():
        raise SystemExit(f"Database not found: {DB}")

    if not BACKUP.exists():
        raise SystemExit(
            f"Backup missing: {BACKUP}\n"
            "Migration refused."
        )

    print(f"Database : {DB}")
    print(f"Backup   : {BACKUP}")
    print()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON")

        print("Before migration:")

        old_conversations = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        old_messages = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        print(f"  Conversations : {old_conversations}")
        print(f"  Messages      : {old_messages}")

        print()
        print("Creating users table...")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

        print("  users: OK")

        print()
        print("Creating sessions table...")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,

                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        print("  sessions: OK")

        print()
        print("Creating indexes...")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_sessions_user_id
            ON sessions(user_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_sessions_expires_at
            ON sessions(expires_at)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_conversations_user_id
            ON conversations(user_id)
            """
        )

        print("  indexes: OK")

        print()
        print("Creating default user...")

        default_id = stable_default_user_id()
        timestamp = now_iso()

        conn.execute(
            """
            INSERT OR IGNORE INTO users (
                id,
                username,
                password_hash,
                display_name,
                created_at,
                updated_at,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                default_id,
                "default",
                "MIGRATION_NO_PASSWORD",
                "Default User",
                timestamp,
                timestamp,
                "{}",
            ),
        )

        print(f"  id       : {default_id}")
        print("  username : default")
        print("  status   : OK")

        conn.commit()

        print()
        print("Verification:")

        user_count = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]

        default_user = conn.execute(
            """
            SELECT id, username
            FROM users
            WHERE username = 'default'
            """
        ).fetchone()

        new_conversations = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        new_messages = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        print(f"  Users             : {user_count}")
        print(f"  Sessions          : {session_count}")
        print(
            f"  Default user      : "
            f"{default_user['username'] if default_user else 'MISSING'}"
        )
        print(f"  Conversations     : {new_conversations}")
        print(f"  Messages          : {new_messages}")
        print(f"  Integrity         : {integrity}")

        print()
        print("Checking data preservation...")

        if new_conversations != old_conversations:
            raise SystemExit(
                "FAILED: conversation count changed."
            )

        if new_messages != old_messages:
            raise SystemExit(
                "FAILED: message count changed."
            )

        if default_user is None:
            raise SystemExit(
                "FAILED: default user was not created."
            )

        if integrity != "ok":
            raise SystemExit(
                f"FAILED: SQLite integrity check = {integrity}"
            )

        print("  Conversation count : PRESERVED")
        print("  Message count      : PRESERVED")
        print("  SQLite integrity   : PASS")

        print()
        print("=" * 68)
        print("USERS SCHEMA MIGRATION: PASS")
        print("=" * 68)
        print()
        print("Database structure was extended safely.")
        print("Existing conversations/messages were preserved.")
        print("NO password has been assigned to the default user.")
        print("NO server authentication code was changed.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
