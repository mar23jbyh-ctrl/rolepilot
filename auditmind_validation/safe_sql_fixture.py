"""Unmerged AuditMind review fixture. Never import or execute this file."""

def lookup_user(connection, user_id: str):
    return connection.execute(
        "SELECT id FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
