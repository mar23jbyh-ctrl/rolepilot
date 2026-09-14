from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import settings


def get_checkpointer(db_path=None):
    """Return a LangGraph SQLite checkpointer (session breakpoints)."""
    target = Path(db_path) if db_path else settings.checkpoint_db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - depends on requirements
        raise RuntimeError(
            "langgraph-checkpoint-sqlite is not installed. Run: "
            "pip install -r requirements.txt"
        ) from exc
    connection = sqlite3.connect(
        str(target),
        check_same_thread=False,
    )
    connection.execute("PRAGMA secure_delete=ON")
    connection.execute("PRAGMA busy_timeout=10000")
    return SqliteSaver(connection)


def delete_checkpoints(checkpointer, session_id: str) -> None:
    """Delete every namespace/history and pending write, then truncate local WAL."""
    checkpointer.delete_thread(session_id)
    with checkpointer.lock:
        result = checkpointer.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result and result[0]:
            raise RuntimeError("checkpoint_cleanup_busy")
