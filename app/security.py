"""Server-issued anonymous capabilities; not account login or enterprise auth.

The bearer token grants access to one anonymous owner's resources. Only its SHA-256
digest is persisted. Losing the browser token loses access; theft grants access.
CLI use of InterviewService remains independent of this HTTP boundary.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import re
import secrets
import sqlite3
import uuid

from app.config import settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}\Z")


class AnonymousIdentityStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else settings.data_root / "anonymous_auth.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS anonymous_identities (
                    token_hash TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS anonymous_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO anonymous_metadata(key, value) VALUES ('server_id', ?)",
                (uuid.uuid4().hex,),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def issue(self) -> str:
        # No client-supplied owner, bearer value, or file name is accepted.
        token = secrets.token_urlsafe(32)
        owner = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO anonymous_identities(token_hash, owner_id) VALUES (?, ?)",
                (hashlib.sha256(token.encode("ascii")).hexdigest(), owner),
            )
        return token

    def server_id(self) -> str:
        """A persistent, non-secret namespace; never usable as a bearer token."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM anonymous_metadata WHERE key = 'server_id'").fetchone()
        if row is None:
            raise sqlite3.DatabaseError("Anonymous identity namespace unavailable")
        return str(row[0])

    def resolve(self, token: str) -> str | None:
        if not _TOKEN_RE.fullmatch(token):
            return None
        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT owner_id FROM anonymous_identities WHERE token_hash = ?", (digest,)
            ).fetchone()
        return str(row[0]) if row else None


def identity_store() -> AnonymousIdentityStore:
    # No process-only credentials: a fresh process consults the same database.
    return AnonymousIdentityStore()
