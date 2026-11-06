from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


class AuthStoreError(Exception):
    """Base exception for authentication storage errors."""


class AuthStore:
    """
    SQLite-backed authentication/session storage.

    Security rules:
    - Passwords are never stored in plaintext.
    - Password hashing uses PBKDF2-HMAC-SHA256.
    - Session tokens are stored as SHA-256 hashes.
    - Session validation rejects revoked and expired sessions.
    """

    PASSWORD_SCHEME = "pbkdf2_sha256"
    PASSWORD_ITERATIONS = 310_000
    SALT_BYTES = 32
    SESSION_TOKEN_BYTES = 32
    DEFAULT_SESSION_DAYS = 30

    def __init__(self, db_path: str | Path = "data/neura.db") -> None:
        self.db_path = Path(db_path)

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
    def _hash_session_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def _hash_password(cls, password: str) -> str:
        if not isinstance(password, str):
            raise TypeError("password must be a string")

        if not password:
            raise ValueError("password must not be empty")

        salt = secrets.token_bytes(cls.SALT_BYTES)

        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            cls.PASSWORD_ITERATIONS,
        )

        return (
            f"{cls.PASSWORD_SCHEME}$"
            f"{cls.PASSWORD_ITERATIONS}$"
            f"{salt.hex()}$"
            f"{derived.hex()}"
        )

    @classmethod
    def _verify_password_hash(
        cls,
        password: str,
        stored_hash: str,
    ) -> bool:
        if not isinstance(password, str):
            return False

        if not isinstance(stored_hash, str):
            return False

        if stored_hash == "MIGRATION_NO_PASSWORD":
            return False

        try:
            scheme, iterations_text, salt_hex, digest_hex = (
                stored_hash.split("$", 3)
            )

            if scheme != cls.PASSWORD_SCHEME:
                return False

            iterations = int(iterations_text)

            salt = bytes.fromhex(salt_hex)
            expected_digest = bytes.fromhex(digest_hex)

            actual_digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                iterations,
            )

            return hmac.compare_digest(
                actual_digest,
                expected_digest,
            )

        except (ValueError, TypeError):
            return False

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str = "",
        metadata: str = "{}",
    ) -> str:
        username = str(username).strip()

        if not username:
            raise ValueError("username must not be empty")

        password_hash = self._hash_password(password)
        user_id = secrets.token_hex(32)
        now = self._now()

        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO users (
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
                        user_id,
                        username,
                        password_hash,
                        display_name,
                        now,
                        now,
                        metadata,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthStoreError(
                    "Username already exists."
                ) from exc

        return user_id

    def get_user_by_username(
        self,
        username: str,
    ) -> dict | None:
        username = str(username).strip()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    username,
                    password_hash,
                    display_name,
                    created_at,
                    updated_at,
                    metadata
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def verify_password(
        self,
        username: str,
        password: str,
    ) -> dict | None:
        user = self.get_user_by_username(username)

        if user is None:
            return None

        if not self._verify_password_hash(
            password,
            user["password_hash"],
        ):
            return None

        return user

    def set_password(
        self,
        user_id: str,
        password: str,
    ) -> bool:
        password_hash = self._hash_password(password)
        now = self._now()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET password_hash = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    password_hash,
                    now,
                    user_id,
                ),
            )

        return cursor.rowcount == 1

    def create_session(
        self,
        user_id: str,
        days: int = DEFAULT_SESSION_DAYS,
    ) -> tuple[str, str]:
        if days <= 0:
            raise ValueError("days must be greater than zero")

        session_token = secrets.token_urlsafe(
            self.SESSION_TOKEN_BYTES
        )

        # Never store the raw session token.
        # The database primary key is the SHA-256 token hash.
        session_id = self._hash_session_token(
            session_token
        )

        now_dt = datetime.now(timezone.utc)
        created_at = now_dt.isoformat()
        expires_at = (
            now_dt + timedelta(days=days)
        ).isoformat()

        token_hash = self._hash_session_token(
            session_token
        )

        with self._connect() as connection:
            user = connection.execute(
                """
                SELECT id
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if user is None:
                raise AuthStoreError(
                    "User does not exist."
                )

            connection.execute(
                """
                INSERT INTO sessions (
                    id,
                    user_id,
                    created_at,
                    expires_at,
                    revoked_at
                )
                VALUES (?, ?, ?, ?, NULL)
                """,
                (
                    session_id,
                    user_id,
                    created_at,
                    expires_at,
                ),
            )

        return session_token, session_id

    def validate_session(
        self,
        session_token: str,
    ) -> dict | None:
        if not session_token:
            return None

        token_hash = self._hash_session_token(
            session_token
        )
        now = self._now()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    s.id AS session_id,
                    s.user_id AS user_id,
                    s.created_at AS session_created_at,
                    s.expires_at AS expires_at,
                    s.revoked_at AS revoked_at,
                    u.username AS username,
                    u.display_name AS display_name
                FROM sessions AS s
                INNER JOIN users AS u
                    ON u.id = s.user_id
                WHERE s.id = ?
                  AND s.revoked_at IS NULL
                  AND s.expires_at > ?
                """,
                (
                    token_hash,
                    now,
                ),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def revoke_session(
        self,
        session_token: str,
    ) -> bool:
        if not session_token:
            return False

        token_hash = self._hash_session_token(
            session_token
        )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET revoked_at = ?
                WHERE id = ?
                  AND revoked_at IS NULL
                """,
                (
                    self._now(),
                    token_hash,
                ),
            )

        return cursor.rowcount == 1
