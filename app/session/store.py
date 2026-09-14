from __future__ import annotations

import json
import re
import sqlite3
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import contextmanager

from app.config import settings
from app.errors import ServiceError

# 0 = 永久保留（历史会话不再因为长时间未访问而消失）
STALE_DAYS = 0


def safe_owner(owner: str) -> str:
    value = str(owner or "local")
    if not re.fullmatch(r"[0-9A-Za-z_\-]{1,40}", value):
        raise ValueError("invalid_owner_format")
    return value


def owner_db_path(owner: str) -> Path:
    root = settings.data_root
    return root / f"sessions_{safe_owner(owner)}.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    def __init__(self, db_path: str | Path | None = None, owner: str = ""):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = owner_db_path(owner) if owner else settings.session_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_db(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    jd_text TEXT NOT NULL,
                    resume_text TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    report_json TEXT,
                    summary TEXT,
                    total_tokens INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)"
            )
            columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            ]
            if "name" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            if "question_version" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN question_version INTEGER NOT NULL DEFAULT 0")
            if "active_operation" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN active_operation TEXT")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS answer_requests (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    question_version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('processing','succeeded','recovery_required')),
                    response_json TEXT,
                    checkpoint_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, request_id)
                )
            """)
            connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_one_pending_answer
                ON answer_requests(session_id) WHERE status IN ('processing','recovery_required')""")
            from app.telemetry.ledger import initialize_ledger
            initialize_ledger(connection)

    def create(
        self,
        session_id: str,
        jd_text: str,
        resume_text: str,
        state: dict,
        name: str = "",
    ) -> dict:
        if name.strip() and self.name_exists(name):
            raise ValueError("该会话名称已存在，请更换一个名称")
        now = _now()
        row = {
            "id": session_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "status": "interviewing",
            "jd_text": jd_text,
            "resume_text": resume_text,
            "state_json": json.dumps(state, ensure_ascii=False),
            "report_json": None,
            "summary": "",
            "total_tokens": 0,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, name, created_at, updated_at, status, jd_text, resume_text,
                    state_json, report_json, summary, total_tokens, active_operation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row.values()) + ("start:" + session_id if state.get("session_phase") == "generating" else None,),
            )
        return row

    def update(self, session_id: str, state: dict, connection=None):
        report = state.get("evaluation_report")
        status = "failed" if state.get("error") else ("completed" if report else "interviewing")
        from app.telemetry.summary import state_usage_records
        usage_records = state_usage_records(state)
        total_tokens = sum(
            int(item.get("total_tokens", 0) or 0) for item in usage_records
        )
        now = _now()
        values = (
            now, status, json.dumps(state, ensure_ascii=False),
            json.dumps(report, ensure_ascii=False) if report else None,
            total_tokens, int(state.get("question_version", 0)), session_id,
        )
        sql = """UPDATE sessions SET updated_at=?, status=?, state_json=?,
            report_json=?, total_tokens=?, question_version=? WHERE id=?"""
        if connection is not None:
            connection.execute(sql, values)
        else:
            with self._connect() as conn:
                conn.execute(sql, values)

    def get_answer_request(self, session_id: str, request_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM answer_requests WHERE session_id=? AND request_id=?",
                               (session_id, request_id)).fetchone()
        return dict(row) if row else None

    def project_if_idle(self, session_id: str, state: dict, expected_state_json: str) -> bool:
        """Reconcile a read snapshot only if no mutator changed/claimed it.

        Compare the full snapshot, not second-resolution updated_at. Checking
        before a separate UPDATE would leave a cross-worker race window.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT active_operation,state_json FROM sessions WHERE id=?",
                               (session_id,)).fetchone()
            if not row or row["active_operation"] or row["state_json"] != expected_state_json:
                return False
            self.update(session_id, state, connection=conn)
            return True

    def claim_answer(self, session_id: str, request_id: str, version: int, answer: str) -> dict:
        fingerprint = hashlib.sha256(json.dumps([version, answer], ensure_ascii=False).encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not record:
                raise ServiceError("session_not_found", 404)
            if record["status"] == "deleting":
                raise ServiceError("session_deleting", 410)
            existing = conn.execute("SELECT * FROM answer_requests WHERE session_id=? AND request_id=?",
                                    (session_id, request_id)).fetchone()
            if existing:
                if existing["request_hash"] != fingerprint:
                    raise ServiceError("idempotency_conflict")
                return dict(existing)
            if record["status"] != "interviewing":
                raise ServiceError("session_not_answerable")
            if record["question_version"] != version:
                raise ServiceError("stale_question_version")
            updated = conn.execute("""UPDATE sessions SET active_operation=? WHERE id=?
                AND question_version=? AND active_operation IS NULL AND status='interviewing'""",
                ("answer:" + request_id, session_id, version))
            if updated.rowcount != 1:
                raise ServiceError("session_busy")
            conn.execute("""INSERT INTO answer_requests
                (session_id,request_id,request_hash,question_version,status,created_at,updated_at)
                VALUES (?,?,?,?,'processing',?,?)""",
                (session_id, request_id, fingerprint, version, _now(), _now()))
        return {"status": "claimed", "request_id": request_id}

    def finish_answer(self, session_id: str, request_id: str, state: dict, checkpoint_id: str) -> None:
        # Business projection and replay cache share ONE transaction. The graph
        # is a separate database; completion markers permit safe projection repair.
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT active_operation FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row or row[0] != "answer:" + request_id:
                existing = conn.execute("SELECT status FROM answer_requests WHERE session_id=? AND request_id=?",
                                        (session_id, request_id)).fetchone()
                if existing and existing[0] == "succeeded":
                    return
                raise ServiceError("operation_fence_mismatch")
            self.update(session_id, state, connection=conn)
            conn.execute("""UPDATE answer_requests SET status='succeeded',response_json=?,
                checkpoint_id=?,updated_at=? WHERE session_id=? AND request_id=?""",
                (json.dumps(state, ensure_ascii=False), checkpoint_id, _now(), session_id, request_id))
            conn.execute("UPDATE sessions SET active_operation=NULL WHERE id=?", (session_id,))

    def mark_recovery_required(self, session_id: str, request_id: str) -> None:
        with self._connect() as conn:
            conn.execute("""UPDATE answer_requests SET status='recovery_required', updated_at=?
                WHERE session_id=? AND request_id=? AND status='processing'""",
                (_now(), session_id, request_id))

    def claim_operation(self, session_id: str, operation: str) -> None:
        with self._connect() as conn:
            updated = conn.execute("UPDATE sessions SET active_operation=? WHERE id=? AND active_operation IS NULL AND status!='deleting'",
                                   (operation, session_id))
            if updated.rowcount != 1:
                if not conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone():
                    raise ServiceError("session_not_found", 404)
                raise ServiceError("session_busy")

    def release_operation(self, session_id: str, operation: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET active_operation=NULL WHERE id=? AND active_operation=?",
                         (session_id, operation))

    def get(self, session_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def rename(self, session_id: str, name: str):
        cleaned = str(name or "").strip()
        if not cleaned:
            raise ValueError("会话名称不能为空")
        if self.name_exists(cleaned, exclude_id=session_id):
            raise ValueError("该会话名称已存在，请更换一个名称")
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (cleaned[:60], _now(), session_id),
            )

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def name_exists(self, name: str, exclude_id: str | None = None) -> bool:
        target = str(name or "").strip().lower()
        if not target:
            return False
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name FROM sessions"
            ).fetchall()
        for row in rows:
            if exclude_id and row["id"] == exclude_id:
                continue
            if str(row["name"] or "").strip().lower() == target:
                return True
        return False

    def delete(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    def begin_delete(self, session_id: str) -> bool:
        """Durable fence: interrupted cross-database cleanup can be retried."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status,active_operation FROM sessions WHERE id=?",
                               (session_id,)).fetchone()
            if not row:
                return False
            if row["status"] == "deleting" and row["active_operation"] == "delete":
                return True
            if row["active_operation"]:
                raise ServiceError("session_busy")
            conn.execute("UPDATE sessions SET status='deleting',active_operation='delete' WHERE id=?",
                         (session_id,))
            return True

    def finish_delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status,active_operation FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row:
                return
            if row["status"] != "deleting" or row["active_operation"] != "delete":
                raise ServiceError("operation_fence_mismatch")
            # Wipe private columns while retaining a retryable, non-content fence.
            conn.execute("""UPDATE sessions SET name='',jd_text='',resume_text='',state_json='{}',
                report_json=NULL,summary='',total_tokens=0 WHERE id=?""", (session_id,))
            conn.execute("DELETE FROM answer_requests WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM llm_calls WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM llm_session_budgets WHERE session_id=?", (session_id,))
        with self._connect() as conn:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result and result[0]:
                raise RuntimeError("business_cleanup_busy")
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id=? AND status='deleting' AND active_operation='delete'",
                         (session_id,))

    def load_state(self, session_id: str) -> dict:
        record = self.get(session_id)
        if not record or not record.get("state_json"):
            return {}
        try:
            return json.loads(record["state_json"])
        except (json.JSONDecodeError, TypeError):
            return {}

    def cleanup_stale(self):
        """Return expiry candidates; only InterviewService may delete all stores."""
        days = int(getattr(settings, "session_stale_days", STALE_DAYS) or 0)
        if days <= 0:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM sessions WHERE updated_at < ? AND active_operation IS NULL",
                (cutoff.isoformat(timespec="seconds"),),
            ).fetchall()
        return [row["id"] for row in rows]
