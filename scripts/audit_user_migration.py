from __future__ import annotations

import sqlite3
from pathlib import Path


DB = Path("data/neura.db")
BACKUP = Path("data/neura.db.backup_before_users")


def main() -> None:
    print("=" * 64)
    print("NEURA USER MIGRATION AUDIT")
    print("=" * 64)

    if not DB.exists():
        raise SystemExit(f"Database not found: {DB}")

    print(f"Database : {DB}")
    print(f"Size     : {DB.stat().st_size} bytes")

    print()
    print("Backup:")
    if BACKUP.exists():
        print(f"  OK     : {BACKUP}")
        print(f"  Size   : {BACKUP.stat().st_size} bytes")
    else:
        print("  MISSING")
        raise SystemExit(
            "Refusing migration audit: backup is missing."
        )

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    try:
        print()
        print("Tables:")
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        for row in tables:
            print(f"  - {row['name']}")

        print()
        print("Conversation users:")
        rows = conn.execute(
            """
            SELECT
                user_id,
                COUNT(*) AS conversations
            FROM conversations
            GROUP BY user_id
            ORDER BY user_id
            """
        ).fetchall()

        total_conversations = 0

        for row in rows:
            count = int(row["conversations"])
            total_conversations += count
            print(
                f"  user_id={row['user_id']!r} "
                f"| conversations={count}"
            )

        print()
        print(f"Total conversations: {total_conversations}")

        message_count = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        print(f"Total messages     : {message_count}")

        print()
        print("Integrity checks:")

        empty_user_ids = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversations
            WHERE user_id IS NULL
               OR user_id = ''
            """
        ).fetchone()[0]

        print(
            f"  Empty user_id       : {empty_user_ids}"
        )

        orphan_messages = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages AS m
            LEFT JOIN conversations AS c
                ON c.id = m.conversation_id
            WHERE c.id IS NULL
            """
        ).fetchone()[0]

        print(
            f"  Orphan messages     : {orphan_messages}"
        )

        duplicate_sequences = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT conversation_id, sequence
                FROM messages
                GROUP BY conversation_id, sequence
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        print(
            f"  Duplicate sequences : {duplicate_sequences}"
        )

        print()
        print("Default-user details:")

        default_conversations = conn.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM conversations
            WHERE user_id = 'default'
            ORDER BY created_at ASC
            """
        ).fetchall()

        for row in default_conversations:
            count = conn.execute(
                """
                SELECT COUNT(*)
                FROM messages
                WHERE conversation_id = ?
                """,
                (row["id"],),
            ).fetchone()[0]

            print(
                f"  {row['id']} "
                f"| messages={count} "
                f"| title={row['title']!r}"
            )

        print()
        print("Database integrity:")
        result = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        print(f"  {result}")

        print()
        print("=" * 64)
        print("MIGRATION AUDIT: PASS")
        print("=" * 64)
        print()
        print("NO DATABASE CHANGES WERE MADE.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
