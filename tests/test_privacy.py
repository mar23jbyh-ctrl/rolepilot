"""Synthetic-only privacy lifecycle tests: real SQLite/graph, no paid providers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from PIL import Image
import pytest

import api.deps as deps
from api.main import app
from api.routers import auth, uploads
from app.config import settings
from app.errors import ServiceError
from app.privacy import MASK, redact_messages, redact_text
from app.security import AnonymousIdentityStore
from app.service import InterviewService
from app.session.store import SessionStore


@pytest.fixture
def private_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    identities = AnonymousIdentityStore(tmp_path / "auth.db")
    monkeypatch.setattr(auth, "identity_store", lambda: identities)
    monkeypatch.setattr(deps, "identity_store", lambda: identities)
    monkeypatch.setattr(deps, "_services", {})
    services = []

    def service(owner="synthetic-owner"):
        result = InterviewService(owner, SessionStore(tmp_path / f"sessions_{owner}.db"))
        services.append(result)
        return result

    yield service, identities, tmp_path
    for item in services:
        item._checkpointer.conn.close()


def seed(service, sid="synthetic-session"):
    state = {"current_question": "Explain Python cancellation", "question_version": 1,
             "jd_text": "Public Python role summary", "resume_text": "Synthetic Python project",
             "conversation_history": [], "usage_records": []}
    service.store.create(sid, state["jd_text"], state["resume_text"], state)
    service.store.update(sid, state)
    config = service._config(sid)
    service.graph.update_state(config, state, as_node="ask")
    service.graph.update_state(config, {"current_answer": "Synthetic answer"}, as_node="ask")
    snapshot = service.graph.get_state(config)
    service._checkpointer.put_writes(snapshot.config, [("current_answer", "Synthetic answer")], "synthetic-task")
    # Exercise deletion of *all* namespaces, not only the root snapshot.
    with service._checkpointer.cursor() as cursor:
        cursor.execute("""INSERT INTO checkpoints
            SELECT thread_id,'synthetic-nested',checkpoint_id,parent_checkpoint_id,type,checkpoint,metadata
            FROM checkpoints WHERE thread_id=? AND checkpoint_ns=''""", (sid,))
    return sid


def counts(service, sid):
    with service.store._connect() as conn:
        business = {table: conn.execute(f"SELECT COUNT(*) FROM {table} WHERE " +
                    ("id=?" if table == "sessions" else "session_id=?"), (sid,)).fetchone()[0]
                    for table in ("sessions", "answer_requests")}
    with service._checkpointer.cursor() as cursor:
        graph = {table: cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE thread_id=?", (sid,)).fetchone()[0]
                 for table in ("checkpoints", "writes")}
    return {**business, **graph}


def test_delete_removes_ledger_every_checkpoint_and_write_but_preserves_other_sessions(private_runtime):
    make, _, _ = private_runtime
    service = make()
    sid = seed(service)
    keep = seed(service, "synthetic-keep")
    rid = str(uuid.uuid4())
    service.store.claim_answer(sid, rid, 1, "Synthetic answer")
    service.store.finish_answer(sid, rid, service.store.load_state(sid), "synthetic-checkpoint")
    before = counts(service, sid)
    keep_before = counts(service, keep)
    assert before["checkpoints"] >= 4 and before["writes"] >= 1 and before["answer_requests"] == 1
    assert service.delete_session(sid)
    assert counts(service, sid) == dict.fromkeys(before, 0)
    assert counts(service, keep) == keep_before
    assert not service.graph.get_state(service._config(sid)).values
    assert service._checkpointer.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    with service.store._connect() as conn:
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    with pytest.raises(ServiceError, match="session_not_found"):
        service.submit_answer(sid, "Synthetic answer", rid, 1)


def test_delete_owner_boundary_uses_actual_http_and_sqlite(private_runtime, monkeypatch):
    make, identities, _ = private_runtime
    a, b = identities.issue(), identities.issue()
    service_a, service_b = make(identities.resolve(a)), make(identities.resolve(b))
    sid = seed(service_a)
    seed(service_b)
    deps._services.update({service_a.owner: service_a, service_b.owner: service_b})
    client = TestClient(app)
    assert client.delete(f"/api/sessions/{sid}").status_code == 401
    assert client.delete(f"/api/sessions/{sid}", headers={"Authorization": "Bearer " + a}).status_code == 200
    assert counts(service_a, sid) == dict.fromkeys(counts(service_a, sid), 0)
    assert counts(service_b, sid)["checkpoints"] > 0
    assert client.get(f"/api/sessions/{sid}", headers={"Authorization": "Bearer " + a}).status_code == 404
    assert client.get(f"/api/sessions/{sid}", headers={"Authorization": "Bearer " + b}).status_code == 200
    assert client.delete(f"/api/sessions/{sid}", headers={"Authorization": "Bearer " + b, "X-Owner-Id": service_a.owner}).status_code == 400


@pytest.mark.parametrize("operation", ["answer:synthetic", "stop:synthetic", "job_title:synthetic", "start:synthetic"])
def test_delete_cannot_race_an_active_mutation(private_runtime, operation):
    make, _, _ = private_runtime
    first = make()
    sid = seed(first)
    second = make()
    first.store.claim_operation(sid, operation)
    before = counts(first, sid)
    with pytest.raises(ServiceError, match="session_busy"):
        second.delete_session(sid)
    assert counts(first, sid) == before
    first.store.release_operation(sid, operation)
    assert second.delete_session(sid)


@pytest.mark.parametrize("after_checkpoint_delete", [False, True])
def test_partial_delete_is_inaccessible_and_retriable_after_service_reopen(private_runtime, monkeypatch, after_checkpoint_delete):
    import app.service as module
    make, _, _ = private_runtime
    service = make()
    sid = seed(service)
    rid = str(uuid.uuid4())
    service.store.claim_answer(sid, rid, 1, "Synthetic answer")
    service.store.finish_answer(sid, rid, service.store.load_state(sid), "synthetic-checkpoint")
    real_delete = module.delete_checkpoints

    def fail(checkpointer, session_id):
        if after_checkpoint_delete:
            real_delete(checkpointer, session_id)
        raise OSError("synthetic private filesystem detail")

    monkeypatch.setattr(module, "delete_checkpoints", fail)
    with pytest.raises(ServiceError, match="session_cleanup_failed"):
        service.delete_session(sid)
    assert service.store.get(sid)["status"] == "deleting"
    for action in (lambda: service.restore(sid), lambda: service.answer_request_status(sid, rid),
                   lambda: service.submit_answer(sid, "Synthetic answer", rid, 1)):
        with pytest.raises(ServiceError, match="session_deleting") as error:
            action()
        assert error.value.status == 410
    monkeypatch.setattr(module, "delete_checkpoints", real_delete)
    restarted = make()
    assert restarted.delete_session(sid)
    assert not any(counts(restarted, sid).values())


def test_parallel_duplicate_deletes_do_not_resurrect_state(private_runtime):
    make, _, _ = private_runtime
    services = [make() for _ in range(4)]
    sid = seed(services[0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda service: service.delete_session(sid), services))
    assert any(results)
    assert not any(counts(services[0], sid).values())


def test_delete_lock_is_shared_by_service_instances(private_runtime):
    """The checkpoint cleanup phase must not race across service objects."""
    make, _, _ = private_runtime
    services = [make() for _ in range(2)]
    assert services[0].store.db_path == services[1].store.db_path
    sid = seed(services[0])
    import app.service as service_module
    first = service_module._delete_lock(services[0].store, sid)
    second = service_module._delete_lock(services[1].store, sid)
    assert first is second
    assert services[1].delete_session(sid) is True
    assert services[0].delete_session(sid) is False


def test_initial_generation_has_cross_process_deletion_fence(private_runtime):
    make, _, _ = private_runtime
    first = make()
    first.store.create("synthetic-generating", "Synthetic JD", "Synthetic resume", {"session_phase": "generating"})
    second = make()
    assert first.store.get("synthetic-generating")["active_operation"] == "start:synthetic-generating"
    with pytest.raises(ServiceError, match="session_busy"):
        second.delete_session("synthetic-generating")


def test_business_wal_busy_keeps_wiped_retryable_fence(private_runtime, monkeypatch):
    make, _, _ = private_runtime
    service = make()
    sid = seed(service)
    with service.store._connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    reader = sqlite3.connect(service.store.db_path)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM sessions").fetchone()
    original = service.store._connect
    @contextmanager
    def short_timeout():
        with original() as conn:
            conn.execute("PRAGMA busy_timeout=1")
            yield conn
    monkeypatch.setattr(service.store, "_connect", short_timeout)
    try:
        with pytest.raises(ServiceError, match="session_cleanup_failed"):
            service.delete_session(sid)
        row = service.store.get(sid)
        assert row["status"] == "deleting" and row["resume_text"] == "" and row["state_json"] == "{}"
    finally:
        reader.close()
    assert service.delete_session(sid)
    assert not any(counts(service, sid).values())


def test_checkpoint_wal_busy_does_not_expose_partially_deleted_session(private_runtime):
    make, _, _ = private_runtime
    service = make()
    sid = seed(service)
    path = service._checkpointer.conn.execute("PRAGMA database_list").fetchone()[2]
    service._checkpointer.conn.execute("PRAGMA busy_timeout=1")
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
    try:
        with pytest.raises(ServiceError, match="session_cleanup_failed"):
            service.delete_session(sid)
        with pytest.raises(ServiceError, match="session_deleting"):
            service.restore(sid)
    finally:
        reader.close()
    assert service.delete_session(sid)
    assert not any(counts(service, sid).values())


def test_configured_expiry_uses_full_cleanup_not_business_only(private_runtime, monkeypatch):
    make, _, _ = private_runtime
    first = make()
    sid = seed(first)
    with first.store._connect() as conn:
        conn.execute("UPDATE sessions SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (sid,))
    monkeypatch.setattr(settings, "session_stale_days", 1)
    restarted = make()
    assert not any(counts(restarted, sid).values())


@pytest.mark.parametrize("text", ["synthetic@example.test", "13800138000", "+86 138 0013 8000", "110101199001010011",
                                       "姓名：合成候选人", "Name: Synthetic Candidate", "Address: Synthetic street"])
def test_recognizable_private_contacts_are_masked(text):
    assert MASK in redact_text(text)


def test_privacy_filter_preserves_skills_json_protocol_and_input():
    source = [{"role": "assistant", "content": 'Python, SQLite; email: synthetic@example.test',
               "tool_calls": [{"id": "synthetic-id", "type": "function", "function": {
                   "name": "WebSearch", "arguments": json.dumps({"query": "synthetic@example.test Python"})}}]}]
    original = json.dumps(source)
    result = redact_messages(source)
    assert json.dumps(source) == original
    assert result[0]["tool_calls"][0]["function"]["name"] == "WebSearch"
    assert result[0]["tool_calls"][0]["id"] == "synthetic-id"
    assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"])["query"] == MASK + " Python"
    assert "Python, SQLite" in result[0]["content"]
    assert redact_text("Use Python asyncio cancellation and SQLite transactions.") == "Use Python asyncio cancellation and SQLite transactions."


@pytest.mark.parametrize("with_tools", [False, True])
def test_actual_llm_exits_mask_request_before_sdk_and_budget(monkeypatch, with_tools):
    import app.llm.client as llm
    sent = []
    response = SimpleNamespace(model="synthetic-model", usage=None, choices=[SimpleNamespace(
        message=SimpleNamespace(content="Synthetic result", tool_calls=[]))])
    def capture(**kwargs):
        sent.append(kwargs)
        return response
    monkeypatch.setattr(llm.client.chat.completions, "create", capture)
    source = [{"role": "user", "content": "姓名：合成候选人\nPython project\nsynthetic@example.test 13800138000"}]
    if with_tools:
        llm.chat_with_tools(source, [], max_tokens=100)
    else:
        llm.chat_with_usage(source, max_tokens=100)
    assert len(sent) == 1
    payload = json.dumps(sent[0]["messages"], ensure_ascii=False)
    assert all(value not in payload for value in ("合成候选人", "synthetic@example.test", "13800138000"))
    assert "Python project" in payload


def test_function_call_arguments_also_masked_at_actual_llm_exit(monkeypatch):
    import app.llm.client as llm
    seen = []
    monkeypatch.setattr(llm, "_create_with_retry", lambda **kw: seen.append(kw) or SimpleNamespace(
        model="synthetic", usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))]))
    llm.chat_with_tools([AIMessage(content="", tool_calls=[{"id": "synthetic-id", "name": "WebSearch",
                         "args": {"query": "synthetic@example.test Python"}}])], [])
    assert "synthetic@example.test" not in json.dumps(seen)
    assert json.loads(seen[0]["messages"][0]["tool_calls"][0]["function"]["arguments"])["query"] == MASK + " Python"


def test_search_sensitive_query_is_rejected_before_network(monkeypatch):
    from app.tools import executor
    calls = []
    monkeypatch.setattr(executor, "_tavily_search", lambda *a, **kw: calls.append(a) or [])
    result = executor.web_search_handler({"query": "synthetic@example.test Python role"})
    assert result == {"success": False, "content": "privacy_query_rejected"}
    assert not calls


def test_direct_search_helper_cannot_bypass_filter():
    from app.tools.executor import _tavily_search
    with pytest.raises(ValueError, match="privacy_query_rejected"):
        _tavily_search("13800138000 Python role")


@pytest.mark.parametrize("failure", [None, "http", "empty"])
def test_image_original_always_removed_without_touching_other_requests(private_runtime, monkeypatch, failure):
    _, identities, root = private_runtime
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer " + identities.issue()
    folder = root / "uploads"
    folder.mkdir()
    other = folder / "synthetic-other-request.txt"
    other.write_text("synthetic in-flight upload", encoding="utf-8")
    def extract(path):
        assert path.exists()
        if failure == "http":
            from fastapi import HTTPException
            raise HTTPException(422, "synthetic_parse_error")
        return "" if failure == "empty" else "Synthetic Python role"
    monkeypatch.setattr(uploads, "ocr_image_file", extract)
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    response = client.post("/api/uploads/jd-image", files={"file": ("synthetic.png", image.getvalue())})
    assert response.status_code == {None: 200, "http": 422, "empty": 400}[failure]
    assert list(folder.iterdir()) == [other]


def test_cleanup_failure_does_not_report_upload_success(private_runtime, monkeypatch):
    from pathlib import Path
    _, identities, _ = private_runtime
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer " + identities.issue()
    monkeypatch.setattr(Path, "unlink", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("synthetic private path")))
    response = client.post("/api/uploads/resume", files={"file": ("synthetic.txt", b"Synthetic Python project")})
    assert response.status_code == 503
    assert response.json()["detail"] == "upload_cleanup_failed"
    assert "private path" not in response.text
