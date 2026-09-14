"""No-network tests: original service, state schema, saver and SQLite ledger."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading
import uuid
from contextlib import nullcontext

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.graph import START, END, StateGraph

from app.errors import ServiceError, NodeExecutionError
from app.graph.runtime import wrap_node
from app.graph.state import InterviewState
from app.session.store import SessionStore, safe_owner
from app import service as module


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # Historical rig invents node usage, not SDK calls. Preserve legacy-accounting tests;
    # the real metering boundary is exercised separately in test_cost_control.py.
    monkeypatch.setattr(module, "metering_scope", lambda *args, **kwargs: nullcontext())
    counters = {"assess": 0, "ask": 0}
    flags = {"fail_ask": False}
    def ask(state):
        counters["ask"] += 1
        if flags["fail_ask"] and state.get("global_question_counter", 0):
            raise RuntimeError("synthetic-fatal")
        return {"current_question": "Synthetic question", "current_answer": "",
                "question_version": state.get("question_version", 0) + 1}
    def assess(state):
        counters["assess"] += 1
        n = state.get("global_question_counter", 0) + 1
        return {"global_question_counter": n,
                "assessments": [{"score": 6, "question_id": n}],
                "usage_records": [{"node": "assess", "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}]}
    def finish(state):
        return {"evaluation_report": {"overall_score": 6, "grade": "B"},
                "usage_records": [{"node": "self_check", "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}]}
    def build(checkpointer):
        g = StateGraph(InterviewState)
        g.add_node("ask", wrap_node(ask, "ask"))
        g.add_node("assess", wrap_node(assess, "assess"))
        g.add_node("self_check", wrap_node(finish, "self_check"))
        g.add_edge(START, "ask")
        g.add_edge("ask", "assess")
        g.add_conditional_edges("assess", lambda s: "self_check" if s.global_question_counter >= 2 else "ask")
        g.add_edge("self_check", END)
        return g.compile(checkpointer=checkpointer, interrupt_before=["assess"])
    monkeypatch.setattr(module, "build_graph", build)
    monkeypatch.setattr(module.settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    store = SessionStore(tmp_path / "sessions.db")
    service = module.InterviewService("release", store)
    sid = service.start_session("synthetic JD", "synthetic CV")["session_id"]
    return service, sid, counters, flags


def test_100_sequential_replays(rig):
    service, sid, counts, _ = rig
    rid = str(uuid.uuid4())
    first = service.submit_answer(sid, "synthetic answer", rid, 1)
    for _ in range(99):
        result = service.submit_answer(sid, "synthetic answer", rid, 1)
        assert result["replayed"]
        assert result["state"] == first["state"]
    assert counts["assess"] == 1
    assert first["state"]["question_version"] == 2
    assert first["state"]["current_answer"] == ""


def test_100_concurrent_same_id(rig):
    service, sid, counts, _ = rig
    rid = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda _: service.submit_answer(sid, "same", rid, 1), range(100)))
    assert counts["assess"] == 1
    assert all(r.get("processing") or r["state"]["question_version"] == 2 for r in results)
    assert service.answer_request_status(sid, rid)["replayed"]


def test_20_different_ids_one_version(rig):
    service, sid, counts, _ = rig
    def send(_):
        try:
            return service.submit_answer(sid, "same", str(uuid.uuid4()), 1)
        except ServiceError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(send, range(20)))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert counts["assess"] == 1


def test_id_conflict_and_old_version(rig):
    service, sid, counts, _ = rig
    rid = str(uuid.uuid4())
    service.submit_answer(sid, "same", rid, 1)
    with pytest.raises(ServiceError, match="idempotency_conflict"):
        service.submit_answer(sid, "changed", rid, 1)
    with pytest.raises(ServiceError, match="stale_question_version"):
        service.submit_answer(sid, "same", str(uuid.uuid4()), 1)
    assert counts["assess"] == 1


def test_completed_replay_and_full_cost(rig):
    service, sid, counts, _ = rig
    first_id, final_id = str(uuid.uuid4()), str(uuid.uuid4())
    service.submit_answer(sid, "one", first_id, 1)
    final = service.submit_answer(sid, "two", final_id, 2)
    assert final["done"]
    assert final["report"]["token_totals"]["total"] == 12
    assert service.store.get(sid)["total_tokens"] == 12
    assert service.submit_answer(sid, "one", first_id, 1)["replayed"]
    assert service.submit_answer(sid, "two", final_id, 2)["done"]
    with pytest.raises(ServiceError, match="session_not_answerable"):
        service.submit_answer(sid, "new", str(uuid.uuid4()), 2)
    assert counts["assess"] == 2


def test_restart_replay(rig):
    service, sid, counts, _ = rig
    rid = str(uuid.uuid4())
    first = service.submit_answer(sid, "same", rid, 1)
    restarted = module.InterviewService("release", SessionStore(service.store.db_path))
    assert restarted.submit_answer(sid, "same", rid, 1)["state"] == first["state"]
    assert counts["assess"] == 1
    assert restarted.submit_answer(sid, "next", str(uuid.uuid4()), 2)["done"]


def test_business_write_failure_repaired_without_graph_execution(rig, monkeypatch):
    service, sid, counts, _ = rig
    original = service.store.update
    def failing(*args, **kwargs):
        raise OSError("synthetic DB failure")
    monkeypatch.setattr(service.store, "update", failing)
    rid = str(uuid.uuid4())
    with pytest.raises(ServiceError, match="recovery_required"):
        service.submit_answer(sid, "same", rid, 1)
    assert service.store.load_state(sid)["question_version"] == 1
    assert service.graph.get_state(service._config(sid)).values["question_version"] == 2
    monkeypatch.setattr(service.store, "update", original)
    restarted = module.InterviewService("release", SessionStore(service.store.db_path))
    assert restarted.submit_answer(sid, "same", rid, 1)["replayed"]
    assert restarted.store.load_state(sid)["question_version"] == 2
    assert counts["assess"] == 1


def test_uncertain_failure_never_automatically_reexecutes(rig):
    service, sid, counts, flags = rig
    flags["fail_ask"] = True
    rid = str(uuid.uuid4())
    with pytest.raises(ServiceError, match="node_failed"):
        service.submit_answer(sid, "same", rid, 1)
    assert service.store.get(sid)["status"] == "failed"
    with pytest.raises(ServiceError, match="recovery_required"):
        service.submit_answer(sid, "same", rid, 1)
    assert counts["assess"] == 1
    assert service.store.get(sid)["active_operation"] == "answer:" + rid


def test_node_validation_fails_closed():
    with pytest.raises(NodeExecutionError, match="PermissionError"):
        wrap_node(lambda _: {"resume_text": "forbidden"}, "plan")(InterviewState())


def test_fatal_analysis_does_not_execute_downstream(tmp_path, monkeypatch):
    from app.graph.builder import build_graph
    from app.nodes import analyze, research_job
    from app.graph.checkpointer import get_checkpointer
    executed = []
    def fatal(_):
        raise RuntimeError("synthetic failure - must not reach research")
    monkeypatch.setattr(analyze, "analyze", fatal)
    monkeypatch.setattr(research_job, "research_job_node", lambda _: executed.append("research") or {})
    graph = build_graph(get_checkpointer(tmp_path / "fatal.db"))
    with pytest.raises(NodeExecutionError):
        graph.invoke({"jd_text": "synthetic", "resume_text": "synthetic"},
                     {"configurable": {"thread_id": "fatal"}})
    assert executed == []


def test_abandoned_followup_routes_to_next_topic():
    from app.graph.edges import route_after_follow_up
    assert route_after_follow_up({"follow_up_abandoned": True, "current_question_index": 0,
                                  "question_plan": [{}, {}]}) == "advance"
    assert route_after_follow_up({"follow_up_abandoned": True, "current_question_index": 1,
                                  "question_plan": [{}, {}]}) == "evaluate"
    assert route_after_follow_up({"follow_up_abandoned": False}) == "assess"


@pytest.mark.parametrize("owner", ["runtime/audit", "x" * 41, " a "])
def test_owner_alias_rejected(owner):
    with pytest.raises(ValueError):
        safe_owner(owner)


def test_http_protocol(rig):
    from api.routers.sessions import router
    from api.deps import service_dep
    service, sid, counts, _ = rig
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[service_dep] = lambda: service
    with TestClient(app) as client:
        path = f"/api/sessions/{sid}/answer"
        assert client.post(path, json={"answer": "old client"}).status_code == 422
        body = {"answer": "same", "answer_request_id": str(uuid.uuid4()), "expected_question_version": 1}
        assert client.post(path, json=body).status_code == 200
        replay = client.post(path, json=body)
        assert replay.status_code == 200 and replay.json()["replayed"]
        body["answer_request_id"] = str(uuid.uuid4())
        assert client.post(path, json=body).status_code == 409
        assert counts["assess"] == 1


def _process_claim(path, rid, ready, output):
    store = SessionStore(path)
    ready.wait(10)
    try:
        output.put(store.claim_answer("sid", rid, 1, "same")["status"])
    except ServiceError as exc:
        output.put(exc.code)


def test_multi_process_sqlite_claim(tmp_path):
    path = str(tmp_path / "multi.db")
    store = SessionStore(path)
    store.create("sid", "JD", "CV", {})
    store.update("sid", {"question_version": 1})
    ctx = multiprocessing.get_context("spawn")
    ready, output = ctx.Event(), ctx.Queue()
    rid = str(uuid.uuid4())
    workers = [ctx.Process(target=_process_claim, args=(path, rid, ready, output)) for _ in range(4)]
    for worker in workers:
        worker.start()
    ready.set()
    results = [output.get(timeout=30) for _ in workers]
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    assert results.count("claimed") == 1
    assert results.count("processing") == 3
