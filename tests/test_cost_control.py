"""Synthetic, network-denied tests of the actual SDK boundary, ledger and graph recovery."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import uuid

import pytest
from langgraph.graph import StateGraph, START, END

from app.config import Settings, settings
from app.context.budget import ContextBudgetExceeded, prepare_request
from app.graph.runtime import wrap_node
from app.graph.state import InterviewState
from app.llm import client as llm
from app.telemetry.ledger import CallLedger, SessionBudgetExceeded
from app.telemetry.meter import metering_scope, node_scope, submit_with_context
from app.telemetry.pricing import snapshot_price
from app.telemetry.summary import summarize_usage, merge_usage, state_usage_records
from app.session.store import SessionStore


def response(input_tokens=10, output_tokens=5, *, model="synthetic-model", missing=False):
    return SimpleNamespace(id="synthetic-completion", model=model,
                           usage=None if missing else SimpleNamespace(prompt_tokens=input_tokens,completion_tokens=output_tokens,
                                                                     total_tokens=input_tokens+output_tokens),
                           choices=[SimpleNamespace(finish_reason="stop",message=SimpleNamespace(
                               content=json.dumps({"text_analysis":"Synthetic feedback","status":"ok","findings":[],"suggestions":[]}),tool_calls=[]))])


@pytest.fixture
def metered(tmp_path, monkeypatch):
    monkeypatch.setattr(settings,"context_budget",4000)
    monkeypatch.setattr(settings,"max_output_tokens",512)
    monkeypatch.setattr(settings,"session_token_budget",10000)
    monkeypatch.setattr(settings,"final_report_token_reserve",2000)
    monkeypatch.setattr(settings,"llm_output_limit_param","max_tokens")
    monkeypatch.setattr(settings,"llm_send_temperature",True)
    monkeypatch.setattr(settings,"cost_input_per_1m",None)
    monkeypatch.setattr(settings,"cost_output_per_1m",None)
    store = SessionStore(tmp_path / "meter.db")
    sid = uuid.uuid4().hex
    store.create(sid,"synthetic JD","synthetic CV",{"usage_schema_version":"usage-v1"})
    store.release_operation(sid,"start:"+sid)
    calls=[]
    def create(**kw):
        calls.append(kw)
        return response()
    monkeypatch.setattr(llm.client.chat.completions,"create",create)
    return store,sid,calls


def call(store,sid,*,node="ask",messages=None,tools=None,max_tokens=None):
    with metering_scope(store,sid,node):
        messages=messages or [{"role":"user","content":"Synthetic question"}]
        if tools is None:
            return llm.chat_with_usage(messages,max_tokens=max_tokens)
        return llm.chat_with_tools(messages,tools,max_tokens=max_tokens)


@pytest.mark.parametrize("limit_param",["max_tokens","max_completion_tokens"])
@pytest.mark.parametrize("send_temperature",[False,True])
def test_provider_parameter_policy(metered,monkeypatch,limit_param,send_temperature):
    store,sid,calls=metered
    monkeypatch.setattr(settings,"llm_output_limit_param",limit_param)
    monkeypatch.setattr(settings,"llm_send_temperature",send_temperature)
    call(store,sid)
    assert limit_param in calls[0] and ("temperature" in calls[0]) == send_temperature
    assert not any(key.startswith("_") for key in calls[0])


@pytest.mark.parametrize("tools",[None,[]])
def test_context_rejection_zero_sdk_and_answer_unchanged(metered,tools):
    store,sid,calls=metered
    messages=[{"role":"user","content":"合成完整回答 TAIL "*10000}]
    original=deepcopy(messages)
    with pytest.raises(ContextBudgetExceeded):
        call(store,sid,messages=messages,tools=tools)
    records=CallLedger(store,sid).records()
    assert calls == [] and messages == original
    assert len(records)==1 and records[0]["usage_status"]=="not_called"
    assert summarize_usage(records)["unknown_calls"]==0


def test_output_reduction_reaches_real_sdk_boundary(metered):
    store,sid,calls=metered
    _,usage=call(store,sid,max_tokens=20000)
    assert usage["budget"]["action"]=="output_cap_reduced"
    assert calls[0]["max_tokens"] < 20000
    assert CallLedger(store,sid).budget_summary()["actions"]["output_cap_reduced"]==1


def test_supplier_usage_and_no_implicit_money(metered):
    store,sid,_=metered
    call(store,sid)
    summary=summarize_usage(CallLedger(store,sid).records())
    assert summary["total"]==15 and summary["known_calls"]==1
    assert summary["cost"] is None and summary["currency"] is None
    assert summary["provider_billed_cost"] is None


def test_missing_usage_holds_reservation(metered,monkeypatch):
    store,sid,_=metered
    monkeypatch.setattr(llm.client.chat.completions,"create",lambda **k:response(missing=True))
    call(store,sid)
    ledger=CallLedger(store,sid)
    assert ledger.budget_summary()["held_tokens"] > 0
    assert summarize_usage(ledger.records())["total"] is None


def test_retry_attempts_and_unknown_debt_are_not_zero(metered,monkeypatch):
    import httpx
    from openai import APITimeoutError
    store,sid,calls=metered
    def create(**kw):
        calls.append(kw)
        if len(calls)==1:
            raise APITimeoutError(request=httpx.Request("POST","https://provider.example/v1/chat/completions"))
        return response()
    monkeypatch.setattr(llm.client.chat.completions,"create",create)
    monkeypatch.setattr(llm.time,"sleep",lambda seconds:None)
    call(store,sid)
    ledger=CallLedger(store,sid); records=ledger.records()
    assert len(records)==len(calls)==2 and len({row["logical_call_id"] for row in records})==1
    totals=summarize_usage(records)
    assert totals["known_total"]==15 and totals["unknown_calls"]==1 and totals["total"] is None
    assert ledger.budget_summary()["held_tokens"] > 0


def test_audit_commit_failure_does_not_resend_sdk(metered,monkeypatch):
    store,sid,calls=metered
    monkeypatch.setattr(CallLedger,"finish",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("synthetic write failure")))
    with pytest.raises(RuntimeError): call(store,sid)
    assert len(calls)==1 and CallLedger(store,sid).budget_summary()["held_tokens"] > 0


def _reservation(metadata=None):
    if metadata is None:
        _,metadata=prepare_request([{"role":"user","content":"Synthetic question"}],[],512,4000)
    return metadata


def test_report_reserve_and_normal_block(metered,monkeypatch):
    store,sid,calls=metered
    metadata=_reservation()
    amount=sum(metadata[k] for k in ("estimated_input_tokens","safety_margin_tokens","reserved_output_tokens"))
    monkeypatch.setattr(settings,"session_token_budget",amount*2)
    monkeypatch.setattr(settings,"final_report_token_reserve",amount)
    monkeypatch.setattr(llm.client.chat.completions,"create",lambda **k:response(amount-5,5))
    call(store,sid)
    with pytest.raises(SessionBudgetExceeded): call(store,sid)
    call(store,sid,node="evaluate")
    summary=CallLedger(store,sid).budget_summary()
    assert summary["known_used_tokens"]==amount*2 and summary["exhausted"]
    assert summary["blocked_calls"]==1 and not summary["overrun"]


def test_concurrent_requests_cannot_over_reserve(metered,monkeypatch):
    store,sid,calls=metered
    release=threading.Event(); barrier=threading.Barrier(20); lock=threading.Lock()
    metadata=_reservation()
    amount=sum(metadata[k] for k in ("estimated_input_tokens","safety_margin_tokens","reserved_output_tokens"))
    expected=(settings.session_token_budget-settings.final_report_token_reserve)//amount
    def create(**kw):
        with lock: calls.append(kw)
        assert release.wait(10)
        return response()
    monkeypatch.setattr(llm.client.chat.completions,"create",create)
    def send(_):
        barrier.wait(timeout=10)
        try: call(store,sid); return "accepted"
        except SessionBudgetExceeded: return "blocked"
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures=[pool.submit(send,index) for index in range(20)]
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            if sum(f.done() for f in futures)+len(calls)==20: break
            time.sleep(.01)
        before=CallLedger(store,sid).budget_summary()
        release.set()
        results=[future.result() for future in futures]
    assert len(calls)==expected and results.count("accepted")==expected
    assert before["held_tokens"]==expected*amount
    assert before["held_tokens"]<=before["token_limit"]-before["report_reserve"]


def test_analyze_worker_context_is_propagated(metered):
    store,sid,calls=metered
    with metering_scope(store,sid,"analyze"):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[submit_with_context(pool,llm.chat_with_usage,[{"role":"user","content":"synthetic"}]) for _ in range(3)]
            [future.result() for future in futures]
    assert len(calls)==3 and {row["node"] for row in CallLedger(store,sid).records()}=={"analyze"}


def test_parallel_tool_and_compression_calls_are_counted(metered):
    from app.tools.executor import execute_tool_calls
    store,sid,calls=metered
    with metering_scope(store,sid,"ask"):
        results=execute_tool_calls([{"type":"tool_call","name":"CodeExplainer","args":{"code":"print(1)"}},
                                    {"type":"tool_call","name":"CodeExplainer","args":{"code":"print(2)"}}],allowed_tools={"CodeExplainer"})
    call(store,sid,node="maybe_compress")
    records=CallLedger(store,sid).records()
    assert all(result.success for result in results)
    assert len(calls)==len(records)==3
    assert {row["node"] for row in records}=={"ask:tool:CodeExplainer","maybe_compress"}


def _restart_worker(db_path,sid,queue):
    # Spawned process never calls a provider. It attempts only a SQLite reservation.
    store=SessionStore(db_path)
    ledger=CallLedger(store,sid);ledger.ensure_budget()
    before=ledger.budget_summary()
    try:
        ledger.reserve(logical_id=uuid.uuid4().hex,attempt=1,node="ask",provider_host="synthetic.example",
                       requested_model="synthetic",input_hash="0"*64,metadata=_reservation())
        outcome="accepted"
    except SessionBudgetExceeded:
        outcome="blocked"
    queue.put({"before":before,"outcome":outcome})


def test_real_process_restart_keeps_pending_reservation(metered,monkeypatch):
    store,sid,_=metered; metadata=_reservation()
    amount=sum(metadata[k] for k in ("estimated_input_tokens","safety_margin_tokens","reserved_output_tokens"))
    monkeypatch.setattr(settings,"session_token_budget",amount+2000)
    with metering_scope(store,sid):
        ledger=CallLedger(store,sid)
        ledger.reserve(logical_id="pending",attempt=1,node="ask",provider_host="synthetic.example",
                       requested_model="synthetic",input_hash="0"*64,metadata=metadata)
    ctx=multiprocessing.get_context("spawn"); queue=ctx.Queue()
    process=ctx.Process(target=_restart_worker,args=(str(store.db_path),sid,queue))
    process.start(); result=queue.get(timeout=20); process.join(timeout=20)
    assert process.exitcode==0 and result["outcome"]=="blocked"
    assert result["before"]["held_tokens"]==amount
    assert result["before"]["token_limit"]==amount+2000


def test_session_deletion_cascades_ledger_without_affecting_other_session(metered):
    store,sid,_=metered;call(store,sid)
    other=uuid.uuid4().hex;store.create(other,"synthetic","synthetic",{})
    store.release_operation(other,"start:"+other);call(store,other)
    assert store.begin_delete(sid);store.finish_delete(sid)
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM llm_calls WHERE session_id=?",(sid,)).fetchone()[0]==0
        assert conn.execute("SELECT COUNT(*) FROM llm_session_budgets WHERE session_id=?",(sid,)).fetchone()[0]==0
        assert conn.execute("SELECT COUNT(*) FROM llm_calls WHERE session_id=?",(other,)).fetchone()[0]==1


def test_ledger_contains_no_prompt_or_resume(metered):
    store,sid,_=metered
    call(store,sid,messages=[{"role":"user","content":"SYNTHETIC_PRIVATE_BODY_LONG_MARKER"}])
    with store._connect() as conn:
        raw=json.dumps([dict(row) for row in conn.execute("SELECT * FROM llm_calls")])
    assert "SYNTHETIC_PRIVATE_BODY_LONG_MARKER" not in raw and "synthetic CV" not in raw
    assert "input_hash" in raw and "provider_host" in raw


def _price_policy(monkeypatch):
    for key,value in {"cost_model":"synthetic-model","cost_currency":"USD","cost_input_per_1m":1.,
                      "cost_output_per_1m":2.,"cost_price_source":"https://pricing.example/rates",
                      "cost_price_effective_date":"2026-09-13"}.items():
        monkeypatch.setattr(settings,key,value)


def test_price_snapshot_immutable_after_config_change(metered,monkeypatch):
    store,sid,_=metered;_price_policy(monkeypatch);call(store,sid)
    before=summarize_usage(CallLedger(store,sid).records())
    monkeypatch.setattr(settings,"cost_input_per_1m",999.)
    after=summarize_usage(CallLedger(store,sid).records())
    assert before["cost"]==after["cost"]==.00002 and after["currency"]=="USD"
    assert before["price_snapshots"]==after["price_snapshots"]


@pytest.mark.parametrize("actual",["different-model",None])
def test_wrong_model_is_not_priced(monkeypatch,actual):
    _price_policy(monkeypatch)
    assert snapshot_price(actual)["status"]=="model_mismatch"


@pytest.mark.parametrize("field,value",[("cost_currency",""),("cost_price_effective_date","wrong"),
                                        ("cost_price_source","https://u:p@pricing.example/rates"),
                                        ("cost_price_source","https://pricing.example/rates?key=secret")])
def test_incomplete_or_sensitive_pricing_disabled(monkeypatch,field,value):
    _price_policy(monkeypatch);monkeypatch.setattr(settings,field,value)
    assert snapshot_price("synthetic-model")["status"] != "priced"


def test_price_survives_node_usage_merge(monkeypatch):
    _price_policy(monkeypatch);merged={}
    for _ in range(2):
        merge_usage(merged,{"input_tokens":10,"output_tokens":5,"total_tokens":15,"pricing":snapshot_price("synthetic-model")})
    summary=summarize_usage([merged])
    assert summary["total"]==30 and summary["cost"]==.00004


def test_multiple_currencies_are_not_summed_as_money(monkeypatch):
    _price_policy(monkeypatch);usd=snapshot_price("synthetic-model")
    monkeypatch.setattr(settings,"cost_currency","CNY");cny=snapshot_price("synthetic-model")
    rows=[{"input_tokens":10,"output_tokens":5,"total_tokens":15,"pricing":policy} for policy in [usd,cny]]
    result=summarize_usage(rows)
    assert result["cost"] is None and result["currency"] is None and result["cost_status"]=="mixed_currencies"


def test_config_optional_blank_rates_and_invalid_budget():
    cfg=Settings(_env_file=None,api_key="synthetic",cost_input_per_1m="",cost_output_per_1m="")
    assert cfg.cost_input_per_1m is None and cfg.cost_output_per_1m is None
    with pytest.raises(ValueError): Settings(_env_file=None,session_token_budget=100,final_report_token_reserve=100)


def test_legacy_token_baseline_not_reset(metered):
    store,sid,_=metered
    store.update(sid,{"usage_schema_version":"usage-v1","usage_records":[{"input_tokens":70,"output_tokens":30,"total_tokens":100}]})
    call(store,sid)
    summary=summarize_usage(CallLedger(store,sid).records())
    assert summary["known_total"]==115 and summary["known_input"]==80 and summary["known_output"]==35


def graph_service(tmp_path,monkeypatch,*,token_limit=0,report_reserve=0):
    """Real graph/service/SQLite; synthetic nodes, SDK usage and replies explicitly labelled."""
    from app import service as module
    from app.nodes import evaluate,self_check
    from app.graph.edges import route_after_ask
    from app.tools.executor import execute_tool_call
    monkeypatch.setattr(settings,"session_token_budget",token_limit)
    monkeypatch.setattr(settings,"final_report_token_reserve",report_reserve)
    monkeypatch.setattr(settings,"checkpoint_db",str(tmp_path / "checkpoints.db"))
    def ask(state):
        llm.chat_with_usage([{"role":"user","content":"Synthetic question"}],max_tokens=128)
        return {"current_question":"如何验证回滚？","current_answer":"","question_version":state.get("question_version",0)+1}
    def assess(state):
        llm.chat_with_usage([{"role":"user","content":state["current_answer"]}],max_tokens=128)
        execute_tool_call("CodeExplainer",{"code":"print(1)"},allowed_tools={"CodeExplainer"})
        n=state.get("global_question_counter",0)+1
        record={"question_id":n,"question":state["current_question"],"plan_question_index":n-1,
                "score":6,"scoreable":True,"dimensions":{"d1":6},"next_action":"next_question"}
        return {"global_question_counter":n,"assessments":[record],
                "turn_records":[{"question_id":n,"question":state["current_question"],"answer":state["current_answer"],"assessment":record}]}
    def build(checkpointer):
        graph=StateGraph(InterviewState)
        for name,func in [("ask",ask),("assess",assess),("evaluate",evaluate.evaluate),("self_check",self_check.self_check)]:
            graph.add_node(name,wrap_node(func,name))
        graph.add_edge(START,"ask")
        graph.add_conditional_edges("ask",lambda s:route_after_ask({"current_question":s.current_question,"budget_exhausted":s.budget_exhausted}),{"assess":"assess","evaluate":"evaluate"})
        graph.add_conditional_edges("assess",lambda s:"evaluate" if s.budget_exhausted or s.global_question_counter>=2 or s.end_requested else "ask",{"ask":"ask","evaluate":"evaluate"})
        graph.add_edge("evaluate","self_check");graph.add_edge("self_check",END)
        return graph.compile(checkpointer=checkpointer,interrupt_before=["assess"])
    monkeypatch.setattr(module,"build_graph",build)
    return module.InterviewService("cost",SessionStore(tmp_path / "graph.db"))


def test_graph_normal_pause_resume_report_replay_and_restart(metered,tmp_path,monkeypatch):
    _,_,calls=metered
    service=graph_service(tmp_path,monkeypatch)
    first=service.start_session("synthetic JD","synthetic CV");sid=first["session_id"]
    assert service.graph.get_state(service._config(sid)).next == ("assess",)
    rid=str(uuid.uuid4());one=service.submit_answer(sid,"先验证备份恢复。",rid,1)
    count=len(calls); assert service.submit_answer(sid,"先验证备份恢复。",rid,1)["replayed"] and len(calls)==count
    final=service.submit_answer(sid,"灰度验证回滚。",str(uuid.uuid4()),2)
    assert final["done"] and final["report"]["overall_score"]==6
    assert final["report"]["token_totals"]["total"]==len(calls)*15
    assert service.store.get(sid)["total_tokens"]==len(calls)*15
    from app.service import InterviewService
    recovered=InterviewService("cost",SessionStore(service.store.db_path)).restore(sid)
    assert recovered["report"]==final["report"]
    assert service.delete_session(sid)
    assert not service.graph.get_state(service._config(sid)).values


def test_budget_closes_graph_and_keeps_report(metered,tmp_path,monkeypatch):
    _,_,calls=metered
    def create(**kw): calls.append(kw); return response(1000,200)
    monkeypatch.setattr(llm.client.chat.completions,"create",create)
    # Reservation total follows actual prompt size. Allow the ending report room.
    service=graph_service(tmp_path,monkeypatch,token_limit=7000,report_reserve=3500)
    first=service.start_session("synthetic JD","synthetic CV");sid=first["session_id"]
    final=service.submit_answer(sid,"合成回答：验证备份与回滚。",str(uuid.uuid4()),1)
    assert final["done"] and final["state"]["budget_exhausted"]
    assert final["report"]["overall_score"]==6 and final["state"]["budget_summary"]["blocked_calls"]>=1
    assert final["report"]["token_totals"]["total"]==len(calls)*1200


def test_budget_skipped_current_answer_is_retained(metered,tmp_path,monkeypatch):
    _,_,calls=metered
    monkeypatch.setattr(llm.client.chat.completions,"create",lambda **kw:response(1000,200))
    service=graph_service(tmp_path,monkeypatch,token_limit=3000,report_reserve=2300)
    first=service.start_session("synthetic JD","synthetic CV")
    answer="合成完整回答，不允许丢失 TAIL_CURRENT_ANSWER"
    final=service.submit_answer(first["session_id"],answer,str(uuid.uuid4()),1)
    assert final["done"] and final["state"]["turn_records"][0]["answer"]==answer
    assert final["state"]["assessments"][0]["score"] is None
    assert final["state"]["assessments"][0]["budget_skipped"]


def test_http_answers_and_history_keep_backend_usage(metered,tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.deps import service_dep
    from api.routers.sessions import router
    _,_,calls=metered
    service=graph_service(tmp_path,monkeypatch)
    app=FastAPI();app.include_router(router)
    # Isolate HTTP serialization, not authentication (covered in release_auth).
    app.dependency_overrides[service_dep]=lambda:service
    with TestClient(app) as client:
        first=client.post("/api/sessions",json={"jd_text":"synthetic JD","resume_text":"synthetic CV"})
        assert first.status_code==200
        sid=first.json()["session_id"]
        for version in (1,2):
            result=client.post(f"/api/sessions/{sid}/answer",json={"answer":"先验证回滚。","answer_request_id":str(uuid.uuid4()),"expected_question_version":version})
            assert result.status_code==200
        payload=client.get(f"/api/sessions/{sid}").json()
        assert payload["done"] and payload["usage_total"]==len(calls)*15
        assert payload["report"]["token_totals"]["total"]==payload["usage_total"]
        assert payload["report"]["cost"] is None


@pytest.mark.parametrize("bad_usage",[
    SimpleNamespace(prompt_tokens=10,completion_tokens=5,total_tokens=999),
    SimpleNamespace(prompt_tokens=True,completion_tokens=5,total_tokens=6),
    SimpleNamespace(prompt_tokens=10,completion_tokens=-1,total_tokens=9),
])
def test_invalid_supplier_usage_is_unknown_and_keeps_reservation(metered,monkeypatch,bad_usage):
    store,sid,_=metered;result=response();result.usage=bad_usage
    monkeypatch.setattr(llm.client.chat.completions,"create",lambda **kw:result)
    call(store,sid)
    ledger=CallLedger(store,sid)
    assert summarize_usage(ledger.records())["total"] is None
    assert ledger.budget_summary()["held_tokens"] > 0


def test_generic_base_url_and_alias_precedence(monkeypatch):
    cfg=SimpleNamespace(base_url="https://default.example",model_fields_set=set())
    monkeypatch.setattr(llm,"settings",cfg)
    monkeypatch.setenv("OPENAI_BASE_URL","https://alias.example/v1")
    assert llm.resolve_base_url()=="https://alias.example/v1"
    cfg.model_fields_set={"base_url"};cfg.base_url="https://explicit.example/v1"
    assert llm.resolve_base_url()=="https://explicit.example/v1"


def test_node_totals_do_not_zero_fill_unknown_or_blocked():
    from app.nodes.evaluate import _token_by_node
    totals=_token_by_node([{"node":"known_zero","usage_status":"known","total_tokens":0},
                           {"node":"missing","usage_status":"missing","total_tokens":None},
                           {"node":"blocked","usage_status":"not_called","total_tokens":None}])
    assert totals=={"known_zero":0,"missing":None}
