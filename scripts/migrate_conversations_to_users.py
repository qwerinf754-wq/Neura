from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


DB = Path("data/neura.db")
BACKUP = Path("data/neura.db.backup_before_conversation_link")


def default_user_id() -> str:
    return hashlib.sha256(
        b"neura-default-user-v1"
    ).hexdigest()


def main() -> None:
    print("=" * 68)
    print("NEURA CONVERSATION USER LINK MIGRATION")
    print("=" * 68)

    if not DB.exists():
        raise SystemExit(f"Database not found: {DB}")

    if not BACKUP.exists():
        raise SystemExit(
            f"Backup missing: {BACKUP}\n"
            "Migration refused."
        )

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON")

        print(f"Database : {DB}")
        print(f"Backup   : {BACKUP}")
        print()

        # --------------------------------------------------------
        # Baseline
        # --------------------------------------------------------

        conversations_before = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        messages_before = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        print("Before:")
        print(f"  Conversations : {conversations_before}")
        print(f"  Messages      : {messages_before}")

        # --------------------------------------------------------
        # Verify users table
        # --------------------------------------------------------

        user = conn.execute(
            """
            SELECT id, username
            FROM users
            WHERE username = 'default'
            """
        ).fetchone()

        if user is None:
            raise SystemExit(
                "Default user missing. Migration refused."
            )

        expected_user_id = default_user_id()

        if user["id"] != expected_user_id:
            raise SystemExit(
                "Default user ID mismatch. "
                "Migration refused."
            )

        print()
        print("Default user:")
        print(f"  username : {user['username']}")
        print(f"  id       : {user['id']}")

        # --------------------------------------------------------
        # Verify current conversation state
        # --------------------------------------------------------

        unknown_users = conn.execute(
            """
            SELECT DISTINCT c.user_id
            FROM conversations AS c
            LEFT JOIN users AS u
                ON u.username = c.user_id
            WHERE u.id IS NULL
            """
        ).fetchall()

        if unknown_users:
            values = [row["user_id"] for row in unknown_users]
            raise SystemExit(
                "Unknown conversation user IDs found: "
                f"{values}"
            )

        default_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversations
            WHERE user_id = 'default'
            """
        ).fetchone()[0]

        print()
        print("Current conversation ownership:")
        print(f"  default : {default_count}")

        # --------------------------------------------------------
        # Migrate user_id from username -> immutable user ID
        # --------------------------------------------------------

        print()
        print("Updating conversation ownership...")

        conn.execute(
            """
            UPDATE conversations
            SET user_id = ?
            WHERE user_id = 'default'
            """,
            (expected_user_id,),
        )

        changed = conn.execute(
            "SELECT changes()"
        ).fetchone()[0]

        print(f"  Rows updated : {changed}")

        conn.commit()

        # --------------------------------------------------------
        # Verification
        # --------------------------------------------------------

        print()
        print("Verification:")

        conversations_after = conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        messages_after = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        linked_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversations AS c
            JOIN users AS u
                ON u.id = c.user_id
            """
        ).fetchone()[0]

        invalid_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversations AS c
            LEFT JOIN users AS u
                ON u.id = c.user_id
            WHERE u.id IS NULL
            """
        ).fetchone()[0]

        integrity = conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        print(f"  Conversations       : {conversations_after}")
        print(f"  Messages            : {messages_after}")
        print(f"  Linked conversations: {linked_count}")
        print(f"  Invalid owners      : {invalid_count}")
        print(f"  SQLite integrity    : {integrity}")

        # --------------------------------------------------------
        # Hard safety checks
        # --------------------------------------------------------

        if conversations_after != conversations_before:
            raise SystemExit(
                "FAILED: conversation count changed."
            )

        if messages_after != messages_before:
            raise SystemExit(
                "FAILED: message count changed."
            )

        if linked_count != conversations_after:
            raise SystemExit(
                "FAILED: not every conversation has a valid user."
            )

        if invalid_count != 0:
            raise SystemExit(
                "FAILED: invalid conversation owners detected."
            )

        if integrity != "ok":
            raise SystemExit(
                f"FAILED: SQLite integrity check = {integrity}"
            )

        # Ensure all existing conversations now belong
        # to the migrated default account.
        migrated_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM conversations
            WHERE user_id = ?
            """,
            (expected_user_id,),
        ).fetchone()[0]

        if migrated_count != conversations_before:
            raise SystemExit(
                "FAILED: not all existing conversations "
                "were migrated to the default account."
            )

        print()
        print("Data preservation:")
        print("  Conversations : PRESERVED")
        print("  Messages      : PRESERVED")
        print("  Ownership     : VALID")
        print("  Integrity     : PASS")

        print()
        print("=" * 68)
        print("CONVERSATION USER LINK: PASS")
        print("=" * 68)
        print()
        print("All existing conversations now use the immutable")
        print("default user's ID.")
        print()
        print("NO chat server code was changed.")
        print("NO authentication was enabled yet.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
