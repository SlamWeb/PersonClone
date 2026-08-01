"""Local user accounts and revocable browser sessions."""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

UserRole = Literal["admin", "member"]
USERNAME_PATTERN = re.compile(r"^[\w.-]{2,32}$", re.UNICODE)
SESSION_TOKEN_BYTES = 32
DEFAULT_SESSION_DAYS = 30
PASSWORD_HASHER = PasswordHasher()


@dataclass(frozen=True, slots=True)
class AuthUser:
    id: str
    username: str
    display_name: str
    role: UserRole
    created_at: str

    def to_api(self) -> dict[str, str]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
        }


class AuthStore:
    """SQLite-backed identity store sharing PersonaForge's system database."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.path = self.data_dir / "system" / "personaforge.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.delete_expired_sessions()

    def has_users(self) -> bool:
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        return count > 0

    def create_user(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
        role: UserRole = "member",
        claim_local_data: bool = False,
        require_empty: bool = False,
    ) -> AuthUser:
        username = validate_username(username)
        validate_password(password)
        if role not in {"admin", "member"}:
            raise ValueError("role must be admin or member")
        now = utc_now_iso()
        user = AuthUser(
            id=f"user-{uuid4().hex}",
            username=username,
            display_name=(display_name or username).strip()[:80] or username,
            role=role,
            created_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if require_empty and int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]) > 0:
                raise ValueError("PersonaForge has already been initialized.")
            try:
                connection.execute(
                    """
                    INSERT INTO users (
                        id, username, username_key, display_name, password_hash,
                        role, created_at, updated_at, last_login_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        user.id,
                        user.username,
                        username.casefold(),
                        user.display_name,
                        PASSWORD_HASHER.hash(password),
                        user.role,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Username already exists.") from exc
            if claim_local_data:
                connection.execute(
                    "UPDATE conversations SET owner_id = ? WHERE owner_id = 'local-user'",
                    (user.id,),
                )
                table_names = {
                    str(row["name"])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if "user_memories" in table_names:
                    connection.execute(
                        "UPDATE user_memories SET owner_id = ? WHERE owner_id = 'local-user'",
                        (user.id,),
                    )
                if "user_memory_settings" in table_names:
                    connection.execute(
                        "UPDATE OR IGNORE user_memory_settings SET owner_id = ? WHERE owner_id = 'local-user'",
                        (user.id,),
                    )
        return user

    def bootstrap_admin(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
    ) -> AuthUser:
        return self.create_user(
            username=username,
            password=password,
            display_name=display_name,
            role="admin",
            claim_local_data=True,
            require_empty=True,
        )

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        username_key = username.strip().casefold()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username_key = ?",
                (username_key,),
            ).fetchone()
        if row is None:
            return None
        password_hash = str(row["password_hash"])
        try:
            PASSWORD_HASHER.verify(password_hash, password)
        except (VerifyMismatchError, InvalidHashError):
            return None
        now = utc_now_iso()
        with self._connect() as connection:
            if PASSWORD_HASHER.check_needs_rehash(password_hash):
                connection.execute(
                    "UPDATE users SET password_hash = ?, updated_at = ?, last_login_at = ? WHERE id = ?",
                    (PASSWORD_HASHER.hash(password), now, now, str(row["id"])),
                )
            else:
                connection.execute(
                    "UPDATE users SET last_login_at = ? WHERE id = ?",
                    (now, str(row["id"])),
                )
        return _row_to_user(row)

    def create_session(self, user_id: str, *, days: int = DEFAULT_SESSION_DAYS) -> str:
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (
                    id, user_id, token_hash, created_at, expires_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"session-{uuid4().hex}",
                    user_id,
                    hash_session_token(token),
                    now.isoformat(),
                    (now + timedelta(days=max(1, days))).isoformat(),
                    now.isoformat(),
                ),
            )
        return token

    def resolve_session(self, token: str | None) -> AuthUser | None:
        if not token:
            return None
        token_hash = hash_session_token(token)
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT u.*, s.id AS session_id
                FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?",
                (now, str(row["session_id"])),
            )
        return _row_to_user(row)

    def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (hash_session_token(token),),
            )

    def delete_expired_sessions(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?",
                (utc_now_iso(),),
            )
        return int(cursor.rowcount)

    def list_users(self) -> list[AuthUser]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
        return [_row_to_user(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_key TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                    ON auth_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires
                    ON auth_sessions(expires_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def validate_username(username: str) -> str:
    value = username.strip()
    if not USERNAME_PATTERN.fullmatch(value):
        raise ValueError("Username must be 2-32 letters, numbers, dots, dashes, or underscores.")
    return value


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    if len(password) > 256:
        raise ValueError("Password is too long.")


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_user(row: sqlite3.Row) -> AuthUser:
    return AuthUser(
        id=str(row["id"]),
        username=str(row["username"]),
        display_name=str(row["display_name"]),
        role=str(row["role"]),  # type: ignore[arg-type]
        created_at=str(row["created_at"]),
    )
