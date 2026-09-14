r"""Opt-in release observation server; never replaces business inputs/outputs.

Review before running. Example (does start the real application):
  .venv\Scripts\python scripts\release_server.py --run-id my-provider-check --execute
Default address: 127.0.0.1:8001. No imports of the production app on --help.
Only use with this release's synthetic fixtures. A run refuses existing trace files.
"""
from __future__ import annotations

import argparse
import contextvars
from datetime import datetime, timezone
import functools
import inspect
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
CTX = contextvars.ContextVar("release_observation", default={})
STATE_KEYS = (
    "question_version", "current_question_index", "global_question_counter",
    "follow_up_count", "total_follow_ups", "end_requested", "explanation_pending",
    "active_answer_request_id", "last_completed_answer_request_id", "difficulty",
)
COUNT_KEYS = (
    "question_plan", "conversation_history", "turn_records", "assessments",
    "usage_records", "tool_records", "difficulty_events", "summaries", "summarized_ids",
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--execute", action="store_true", help="Explicitly permit server startup")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("run-id must contain only letters, digits, hyphens or underscores")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    return args


def state_summary(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    result = {"field_keys": sorted(map(str, value))}
    for key in STATE_KEYS:
        if key in value:
            result[key] = value[key]
    result["counts"] = {key: len(value.get(key) or []) for key in COUNT_KEYS if key in value}
    # Full resumes, prompts, search webpage content and message history are not traced.
    for key in ("current_question", "current_answer"):
        if key in value:
            result[key] = str(value[key] or "")[:12000]
    if "error" in value:
        result["has_error"] = bool(value["error"])
    report = value.get("evaluation_report")
    if isinstance(report, dict):
        result["report"] = {key: report.get(key) for key in (
            "overall_score", "grade", "effective_sample_count", "token_totals", "cost")}
    assessments = value.get("assessments") or []
    if assessments and isinstance(assessments[-1], dict):
        result["latest_assessment"] = {key: assessments[-1].get(key) for key in (
            "question_id", "score", "is_follow_up", "next_action", "candidate_intent",
            "should_follow_up", "stop_suggested", "score_error")}
    return result


def main(argv=None):
    args = arguments(argv)
    if not args.execute:
        print("Prepared only; review and add --execute to start the observation server.")
        return 0
    folder = ROOT / "docs" / "release" / "evidence" / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    trace_path = folder / "trace.jsonl"
    trace = trace_path.open("x", encoding="utf-8")  # Never overwrite an earlier run.
    lock = threading.Lock()
    sys.path.insert(0, str(ROOT))

    from dotenv import dotenv_values
    # Values exist only in memory to scrub accidental text; never emit env data.
    secret_values = {str(v) for k, v in dotenv_values(ROOT / ".env").items()
                     if v and any(w in k.upper() for w in ("KEY", "TOKEN", "PASSWORD", "SECRET"))}
    secret_values.update(str(v) for k, v in os.environ.items()
                         if v and any(w in k.upper() for w in ("KEY", "TOKEN", "PASSWORD", "SECRET")))

    def scrub(value):
        if isinstance(value, dict):
            return {str(k): "[REDACTED]" if any(w in str(k).lower() for w in (
                "authorization", "password", "api_key", "access_token", "refresh_token", "cookie"
            )) or str(k).lower() == "token" else scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            for secret in secret_values:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
            return value
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return type(value).__name__

    def emit(kind, **data):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "kind": kind,
                  "pid": os.getpid(), **CTX.get(), **data}
        with lock:
            trace.write(json.dumps(scrub(record), ensure_ascii=False) + "\n")
            trace.flush()

    class SafeLogFilter(logging.Filter):
        def filter(self, record):
            record.msg = scrub(record.getMessage())
            record.args = ()
            return True

    logging.getLogger("uvicorn.error").addFilter(SafeLogFilter())
    original_submit = ThreadPoolExecutor.submit

    def submit(self, fn, /, *call_args, **kwargs):
        context = contextvars.copy_context()
        return original_submit(self, context.run, fn, *call_args, **kwargs)

    ThreadPoolExecutor.submit = submit
    from app.llm import client as lc
    from app.nodes import analyze, research_job, plan, ask, assess, advance, evaluate, maybe_compress, self_check
    from app.context import manager
    from app import guard, validation
    from app.tools import executor
    from app.graph import edges
    from app.config import settings
    from app.session.store import SessionStore

    def observable(kind, fn, name=None):
        label = name or fn.__name__
        source = str(Path(inspect.getsourcefile(fn)).relative_to(ROOT)).replace("\\", "/")
        line = inspect.getsourcelines(fn)[1]

        @functools.wraps(fn)
        def wrapped(*call_args, **kwargs):
            started = time.perf_counter()
            span = uuid.uuid4().hex
            parent_span = CTX.get().get("span_id")
            context_token = CTX.set({**CTX.get(), "span_id": span, "function": label})
            details = {"file": source, "line": line, "parent_span_id": parent_span}
            if kind in ("node", "route") and call_args:
                details["input_state"] = state_summary(call_args[0])
            if kind == "llm":
                messages = call_args[0] if call_args else kwargs.get("messages", [])
                details.update(requested_model=settings.model, message_count=len(messages),
                               context_characters=sum(len(str(getattr(m, "content", m))) for m in messages))
            if kind == "search":
                details.update(query=str(call_args[0] if call_args else kwargs.get("query", "")),
                               max_results=kwargs.get("max_results", 5))
            if kind == "tool":
                details["tool_name"] = str(call_args[0] if call_args else kwargs.get("name", ""))
            emit(kind + "_start", **details)
            try:
                result = fn(*call_args, **kwargs)
                details = {"seconds": round(time.perf_counter() - started, 6)}
                if kind == "node":
                    details["output_state"] = state_summary(result)
                elif kind == "route":
                    details["next_node"] = str(result)
                elif kind == "llm":
                    details.update(usage=result[-1] if isinstance(result, tuple) else {},
                                   response_characters=len(str(result[0])) if isinstance(result, tuple) else 0)
                elif kind == "search":
                    details["result_count"] = len(result)
                    details["results"] = [{k: item.get(k) for k in (
                        "title", "url", "domain", "published_date")} for item in result if isinstance(item, dict)]
                elif kind == "tool":
                    details.update(tool_name=getattr(result, "tool", ""),
                                   success=getattr(result, "success", None),
                                   degraded=getattr(result, "degraded", False))
                emit(kind + "_end", **details)
                return result
            except Exception as exc:
                emit(kind + "_error", seconds=round(time.perf_counter() - started, 6),
                     error_type=type(exc).__name__)  # Never stringify provider exceptions.
                raise
            finally:
                CTX.reset(context_token)
        return wrapped

    # Capture actual provider model/usage per SDK attempt, without bodies or headers.
    original_create = lc.client.chat.completions.create

    @functools.wraps(original_create)
    def create(*call_args, **kwargs):
        start = time.perf_counter()
        attempt_id = uuid.uuid4().hex
        emit("llm_provider_attempt_start", attempt_id=attempt_id, requested_model=kwargs.get("model"))
        try:
            response = original_create(*call_args, **kwargs)
            emit("llm_provider_attempt_end", attempt_id=attempt_id,
                 seconds=round(time.perf_counter() - start, 6), actual_model=getattr(response, "model", None),
                 usage=lc._usage_dict(response))
            return response
        except Exception as exc:
            emit("llm_provider_attempt_error", attempt_id=attempt_id,
                 seconds=round(time.perf_counter() - start, 6), error_type=type(exc).__name__)
            raise
    lc.client.chat.completions.create = create
    for name in ("chat_with_usage", "chat_with_tools"):
        old = getattr(lc, name)
        new = observable("llm", old)
        for module in list(sys.modules.values()):
            if module and getattr(module, "__name__", "").startswith("app.") and getattr(module, name, None) is old:
                setattr(module, name, new)

    def on_request(request):
        request.extensions["release_started"] = time.perf_counter()
        emit("llm_http_attempt_start", method=request.method, host=request.url.host, path=request.url.path)

    def on_response(response):
        start = response.request.extensions.get("release_started", time.perf_counter())
        emit("llm_http_attempt_end", status=response.status_code,
             seconds=round(time.perf_counter() - start, 6))

    lc.client._client.event_hooks.setdefault("request", []).append(on_request)
    lc.client._client.event_hooks.setdefault("response", []).append(on_response)
    executor._tavily_search = observable("search", executor._tavily_search)
    executor.execute_tool_call = observable("tool", executor.execute_tool_call)
    for module, name in ((analyze, "analyze"), (research_job, "research_job_node"), (plan, "plan"),
                         (ask, "ask"), (ask, "ask_follow_up"), (ask, "ask_explain"), (assess, "assess"),
                         (maybe_compress, "maybe_compress"), (advance, "advance"),
                         (evaluate, "evaluate"), (self_check, "self_check")):
        setattr(module, name, observable("node", getattr(module, name), name))
    for name in ("route_after_ask", "route_after_assessment", "after_explain"):
        setattr(edges, name, observable("route", getattr(edges, name), name))

    from langgraph.pregel import Pregel
    original_invoke = Pregel.invoke

    @functools.wraps(original_invoke)
    def invoke(self, input, config=None, **kwargs):
        config = config or {}
        context_token = CTX.set({**CTX.get(), "thread_id": config.get("configurable", {}).get("thread_id")})
        start = time.perf_counter()
        emit("graph_start", resume=input is None, input_state=state_summary(input))
        try:
            result = original_invoke(self, input, config=config, **kwargs)
            snapshot = self.get_state(config)
            next_nodes = list(snapshot.next)
            emit("graph_end", seconds=round(time.perf_counter() - start, 6), next_nodes=next_nodes,
                 paused_before_assess=next_nodes == ["assess"], natural_end=not next_nodes,
                 checkpoint_id=(snapshot.config or {}).get("configurable", {}).get("checkpoint_id"),
                 state=state_summary(snapshot.values),
                 usage_records=[{key: row.get(key) for key in (
                     "node", "input_tokens", "output_tokens", "total_tokens", "finish_reason")}
                     for row in (snapshot.values.get("usage_records") or []) if isinstance(row, dict)])
            return result
        except Exception as exc:
            emit("graph_error", error_type=type(exc).__name__, seconds=round(time.perf_counter() - start, 6))
            raise
        finally:
            CTX.reset(context_token)
    Pregel.invoke = invoke

    def observe_write(fn):
        @functools.wraps(fn)
        def wrapped(self, *call_args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(self, *call_args, **kwargs)
                emit("business_db_write", operation=fn.__name__, database_path=str(self.db_path),
                     session_id=str(call_args[0]) if call_args else kwargs.get("session_id"),
                     seconds=round(time.perf_counter() - start, 6))
                return result
            except Exception as exc:
                emit("business_db_write_error", operation=fn.__name__, error_type=type(exc).__name__)
                raise
        return wrapped
    for name in ("create", "update", "finish_answer", "claim_answer", "mark_recovery_required"):
        if hasattr(SessionStore, name):
            setattr(SessionStore, name, observe_write(getattr(SessionStore, name)))

    from app.service import InterviewService
    original_service_init = InterviewService.__init__

    @functools.wraps(original_service_init)
    def service_init(self, *call_args, **kwargs):
        original_service_init(self, *call_args, **kwargs)
        checkpoint_paths = []
        try:
            connection = getattr(self._checkpointer, "conn", None)
            if connection is not None:
                checkpoint_paths = [row[2] for row in connection.execute("PRAGMA database_list") if row[2]]
        except Exception as exc:
            emit("observation_metadata_error", error_type=type(exc).__name__)
        emit("service_ready", owner_id=self.owner, business_database_path=str(self.store.db_path),
             checkpoint_database_paths=checkpoint_paths)
    InterviewService.__init__ = service_init

    from api.main import app

    @app.middleware("http")
    async def observe_http(request, call_next):
        path = request.url.path
        metadata = {"path": path, "method": request.method,
                    "http_id": request.headers.get("X-Release-Http-Id", "")[:80],
                    "case_id": request.headers.get("X-Release-Case", "")[:80],
                    "answer_request_id": request.headers.get("X-Release-Answer-Request-Id", "")[:80],
                    "expected_question_version": request.headers.get("X-Release-Question-Version", "")[:20]}
        match = re.match(r"/api/sessions/([a-f0-9]{32})(?:/|$)", path)
        if match:
            metadata["session_id"] = match.group(1)
        context_token = CTX.set(metadata)
        start = time.perf_counter()
        emit("http_start")
        try:
            response = await call_next(request)
            emit("http_end", status=response.status_code, seconds=round(time.perf_counter() - start, 6))
            return response
        except Exception as exc:
            emit("http_error", error_type=type(exc).__name__, seconds=round(time.perf_counter() - start, 6))
            raise
        finally:
            CTX.reset(context_token)

    import uvicorn
    emit("server_start", requested_model=settings.model,
         candidate_simulation_observed_here=False, port=args.port,
         cost_estimate_currency="CNY", cost_input_per_1m=settings.cost_input_per_1m,
         cost_output_per_1m=settings.cost_output_per_1m,
         billing_verified=False, synthetic_only=True)
    print(f"Observation server: 127.0.0.1:{args.port}; trace: {trace_path}")
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    finally:
        emit("server_stop")
        trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
