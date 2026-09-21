"""Unmerged AuditMind review fixture. Never import or execute this file."""

def lookup_user(connection, user_id: str):
    query = f"SELECT id FROM users WHERE id = '{user_id}'"
    return connection.execute(query).fetchone()
