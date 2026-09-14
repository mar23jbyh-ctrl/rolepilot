"""Offline anonymous capability boundary tests; all data is temporary/synthetic."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import sqlite3

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

import api.deps as deps
from api.routers import auth
from app.security import AnonymousIdentityStore


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    store = AnonymousIdentityStore(tmp_path / "auth.db")
    monkeypatch.setattr(auth, "identity_store", lambda: store)
    monkeypatch.setattr(deps, "identity_store", lambda: store)
    monkeypatch.setattr(deps, "_services", {})

    class Service:
        def __init__(self, owner):
            self.owner = owner

    monkeypatch.setattr(deps, "InterviewService", Service)
    app = FastAPI()
    app.include_router(auth.router)

    @app.get("/protected")
    def protected(owner: str = Depends(deps.resolve_owner)):
        return {"owner": owner}

    @app.get("/service")
    def service(service=Depends(deps.service_dep)):
        return {"owner": service.owner, "instance": id(service)}

    return TestClient(app), store


def test_issued_tokens_are_hashed_and_owners_are_server_generated(boundary):
    client, store = boundary
    response = client.post("/api/auth/anonymous", json={"owner": "attacker-selected-owner"})
    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    credential = response.json()
    assert credential["token_type"] == "Bearer"
    assert bool(re.fullmatch(r"[A-Za-z0-9_-]{43}", credential["token"]))
    owner = store.resolve(credential["token"])
    assert bool(re.fullmatch(r"[0-9a-f]{32}", owner or ""))
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT token_hash, owner_id FROM anonymous_identities").fetchall()
    assert len(rows) == 1
    assert bool(rows[0][0] == hashlib.sha256(credential["token"].encode("ascii")).hexdigest())
    assert all(credential["token"] not in value for value in rows[0])


def test_reopening_database_preserves_credential_resolution(boundary):
    client, store = boundary
    credential = client.post("/api/auth/anonymous").json()
    owner = store.resolve(credential["token"])
    reopened = AnonymousIdentityStore(store.db_path)
    assert bool(reopened.resolve(credential["token"]) == owner)


def test_context_is_persistent_non_secret_and_does_not_issue_identity(boundary):
    client, store = boundary
    response = client.get("/api/auth/context")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert set(response.json()) == {"server_id"}
    server_id = response.json()["server_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", server_id)
    assert AnonymousIdentityStore(store.db_path).server_id() == server_id
    assert store.resolve(server_id) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM anonymous_identities").fetchone()[0] == 0


def test_distinct_data_environments_have_distinct_namespaces(tmp_path):
    first = AnonymousIdentityStore(tmp_path / "first.db")
    second = AnonymousIdentityStore(tmp_path / "second.db")
    assert first.server_id() != second.server_id()
    token = first.issue()
    assert second.resolve(token) is None


def test_existing_identity_database_upgrade_preserves_token_and_owner(tmp_path):
    path = tmp_path / "legacy.db"
    token = "a" * 43
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE anonymous_identities (token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL UNIQUE, created_at TEXT)")
        conn.execute("INSERT INTO anonymous_identities VALUES (?, ?, CURRENT_TIMESTAMP)",
                     (hashlib.sha256(token.encode("ascii")).hexdigest(), "b" * 32))
    upgraded = AnonymousIdentityStore(path)
    assert upgraded.resolve(token) == "b" * 32
    assert re.fullmatch(r"[0-9a-f]{32}", upgraded.server_id())


def test_concurrent_database_open_uses_one_persistent_namespace(tmp_path):
    path = tmp_path / "parallel-context.db"
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: AnonymousIdentityStore(path).server_id(), range(16)))
    assert len(set(ids)) == 1


def test_bootstrap_reuses_valid_identity_without_creating_another_owner(boundary):
    client, store = boundary
    first = client.post("/api/auth/anonymous").json()
    response = client.post("/api/auth/anonymous", headers={"Authorization": "Bearer " + first["token"]})
    assert response.status_code == 200
    again = response.json()
    assert again["reused"] is True
    for field in ("token", "server_id", "identity_id"):
        assert again[field] == first[field]
    assert again["identity_id"] == store.resolve(first["token"])
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM anonymous_identities").fetchone()[0] == 1


@pytest.mark.parametrize("authorization", ["Bearer " + "a" * 43, "Bearer malformed", "Basic invalid"])
def test_bootstrap_replaces_only_unresolvable_capabilities(boundary, authorization):
    client, store = boundary
    response = client.post("/api/auth/anonymous", headers={"Authorization": authorization})
    assert response.status_code == 201
    credential = response.json()
    assert credential["reused"] is False
    assert store.resolve(credential["token"]) == credential["identity_id"]


def test_context_failure_is_redacted_and_does_not_issue_token(boundary, monkeypatch):
    client, store = boundary

    def unavailable():
        raise sqlite3.OperationalError("synthetic private storage path")

    monkeypatch.setattr(store, "server_id", unavailable)
    response = client.get("/api/auth/context")
    assert response.status_code == 503
    assert "synthetic private" not in response.text


def test_failed_validation_never_silently_creates_new_identity(boundary, monkeypatch):
    client, store = boundary
    token = store.issue()

    def unavailable(*args):
        raise sqlite3.OperationalError("synthetic private connection")

    def forbidden():
        raise AssertionError("Storage errors must not create a replacement identity")

    monkeypatch.setattr(store, "resolve", unavailable)
    monkeypatch.setattr(store, "issue", forbidden)
    response = client.post("/api/auth/anonymous", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 503
    assert "synthetic private" not in response.text


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer not-issued"}, {"Authorization": "Basic invalid"},
    {"Authorization": "Bearer"}, {"Authorization": "Bearer a b"},
])
def test_missing_or_invalid_credentials_are_401(boundary, headers):
    client, _ = boundary
    response = client.get("/protected", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("query, extra_headers", [
    ("", {"X-Owner-Id": "runtime/audit"}),
    ("?owner=runtime_audit", {}),
    ("?owner=", {}),
    ("", {"X-Owner-Id": ""}),
])
def test_client_owner_override_is_rejected_even_with_valid_bearer(boundary, query, extra_headers):
    client, _ = boundary
    credential = client.post("/api/auth/anonymous").json()
    headers = {"Authorization": "Bearer " + credential["token"], **extra_headers}
    assert client.get("/protected" + query, headers=headers).status_code == 400


def test_same_identity_uses_one_cache_key_and_two_identities_are_distinct(boundary):
    client, _ = boundary
    first = client.post("/api/auth/anonymous").json()
    second = client.post("/api/auth/anonymous").json()
    a = client.get("/service", headers={"Authorization": "Bearer " + first["token"]}).json()
    a_again = client.get("/service", headers={"Authorization": "bearer " + first["token"]}).json()
    b = client.get("/service", headers={"Authorization": "Bearer " + second["token"]}).json()
    assert a == a_again
    assert a["owner"] != b["owner"]
    assert a["instance"] != b["instance"]
    assert len(deps._services) == 2


def test_unissued_well_formed_token_is_not_an_identity(boundary):
    client, _ = boundary
    assert client.get("/protected", headers={"Authorization": "Bearer " + "a" * 43}).status_code == 401


def test_concurrent_issuance_creates_distinct_server_owners(tmp_path):
    store = AnonymousIdentityStore(tmp_path / "parallel.db")
    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = list(pool.map(lambda _: store.issue(), range(32)))
    assert len(set(tokens)) == 32
    assert len({store.resolve(token) for token in tokens}) == 32


def test_storage_failure_has_redacted_response(boundary, monkeypatch):
    client, store = boundary

    def unavailable(*args):
        raise sqlite3.OperationalError("synthetic sensitive connection detail")

    monkeypatch.setattr(store, "issue", unavailable)
    response = client.post("/api/auth/anonymous")
    assert response.status_code == 503
    assert "synthetic sensitive" not in response.text
    monkeypatch.setattr(store, "resolve", unavailable)
    response = client.get("/protected", headers={"Authorization": "Bearer " + "a" * 43})
    assert response.status_code == 503
    assert "synthetic sensitive" not in response.text


def test_actual_main_guards_sessions_uploads_and_self_check(boundary, monkeypatch):
    from api.main import app

    def forbidden(*args, **kwargs):
        raise AssertionError("Unauthorized request must not initialize a graph/service")

    monkeypatch.setattr(deps, "InterviewService", forbidden)
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/self-check").status_code == 401
    assert client.post("/api/uploads/resume", files={"file": ("synthetic.txt", b"test")}).status_code == 401


def test_self_check_only_exposes_safe_summary(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "self_check", lambda: {
        "base_url": "https://synthetic-user:synthetic-secret@example.test/v1?token=synthetic#fragment",
        "project_root": "synthetic-private-path", "api_key_set": True,
        "model": "deepseek-chat",
    })
    summary = main.public_self_check()
    assert summary["model_host"] == "example.test"
    assert summary["api_key_set"] is True
    assert "base_url" not in summary and "project_root" not in summary
    assert summary["model"] == "deepseek-chat"
    assert "synthetic-secret" not in str(summary)


def test_actual_sessions_route_isolates_two_authenticated_sqlite_owners(boundary, tmp_path, monkeypatch):
    from api.main import app
    from app.errors import ServiceError
    from app.session.store import SessionStore

    client, identities = boundary
    a_token = client.post("/api/auth/anonymous").json()["token"]
    b_token = client.post("/api/auth/anonymous").json()["token"]
    a_owner = identities.resolve(a_token)

    class SQLiteService:
        def __init__(self, owner):
            self.owner = owner
            self.store = SessionStore(tmp_path / (owner + ".db"))

        def restore(self, session_id):
            if not self.store.get(session_id):
                raise ServiceError("session_not_found", 404)
            return {"state": self.store.load_state(session_id)}

    monkeypatch.setattr(deps, "InterviewService", SQLiteService)
    a_service = deps.get_service(a_owner)
    a_service.store.create("synthetic-private-session", "Synthetic JD", "Synthetic resume", {
        "current_question": "Synthetic private question", "question_version": 1,
        "conversation_history": [], "assessments": [],
    })
    client = TestClient(app)
    own = client.get("/api/sessions/synthetic-private-session", headers={"Authorization": "Bearer " + a_token})
    other = client.get("/api/sessions/synthetic-private-session", headers={"Authorization": "Bearer " + b_token})
    forged = client.get("/api/sessions/synthetic-private-session", headers={
        "Authorization": "Bearer " + b_token, "X-Owner-Id": a_owner,
    })
    assert own.status_code == 200
    assert own.json()["question"] == "Synthetic private question"
    assert other.status_code == 404
    assert forged.status_code == 400
