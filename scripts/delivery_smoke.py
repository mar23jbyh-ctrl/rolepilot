"""Opt-in local TCP smoke: real API/graph/SQLite, explicit model/search fixtures.

Never uses a cloud credential, user's database or personal resume. This proves
integration behavior, not model quality, supplier usage, cost or Tavily access.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def serve(port, database_root):
    database_root = database_root.resolve()
    if (not database_root.is_relative_to(Path(tempfile.gettempdir()).resolve())
            or not database_root.name.startswith("rolepilot-smoke-")):
        raise ValueError("Smoke server only accepts its isolated temporary database directory")
    os.environ.update(API_KEY="synthetic-no-cloud", BASE_URL="http://127.0.0.1:1/v1",
                      MODEL="synthetic-smoke", TAVILY_API_KEY="", DATA_DIR=str(database_root),
                      SESSION_DB=str(database_root / "sessions.db"), CHECKPOINT_DB=str(database_root / "checkpoints.db"),
                      ENABLE_EXTERNAL_TRACING="false", SESSION_TOKEN_BUDGET="500000", FINAL_REPORT_TOKEN_RESERVE="50000")
    import uvicorn
    from app.llm import client as llm
    from app.graph import builder
    from app.nodes import research_job
    from app.tools import executor
    from api.main import app

    lock = threading.Lock()
    rubric = [{"key": f"d{i}", "label": f"合成维度{i}", "weight": .25,
               "definition": "数据分析工作中的验证与表达", "threshold": "能具体说明处理与验证步骤"} for i in range(1, 5)]
    counter = {"assess": 0, "ask": 0}
    events = database_root / "events.jsonl"

    def event(record):
        with lock:
            with events.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fixture_search(query, max_results=5, days=365):
        event({"kind": "search_fixture", "query_hash": digest(query)})
        return [{"title": "Synthetic role reference", "url": "https://example.invalid/role",
                 "content": "业务分析员需确认数据口径，用 SQL 聚合并核对结果，解释业务假设。",
                 "domain": "example.invalid", "published_date": "2026-09-13"}]

    executor._tavily_search = fixture_search
    research_job.web_search_handler = executor.web_search_handler

    def create(**kwargs):
        messages = llm._convert_messages(kwargs["messages"])
        system = str(messages[0].get("content", ""))
        text = "\n".join(str(message.get("content", "")) for message in messages)
        if "extract resume facts" in system:
            content = {"summary": "合成业务分析候选人", "skills": ["SQL", "数据核对"], "education": [],
                       "experience": [], "projects": ["合成校园业务分析项目：用 SQL 聚合数据，抽样核对结果。"], "sections": [], "concerns": []}
        elif "extract job requirements" in system:
            content = {"requirements": [{"category": "tech", "text": "用 SQL 聚合数据并抽样核对结果", "skills": ["SQL"], "weight": 1}]}
        elif "identify the job domain" in system:
            content = {"job_title": "合成业务分析员", "domain": "互联网-业务分析", "domain_slug": "business-analysis",
                       "industry": "互联网", "industry_known": True, "confidence": .95,
                       "assessment_focus": [{"key": "f1", "name": "数据核对实务", "subtopics": ["数据口径确认", "SQL聚合核对", "业务假设解释"]}],
                       "key_skills": ["SQL", "数据核对"], "core_responsibilities": ["确认数据口径", "核对分析结果"],
                       "out_of_scope_topics": ["临床护理操作", "诉讼流程实务", "模型预训练"], "needs_web_research": False}
        elif "skill gap analysis" in system:
            content = {"matches": [{"status": "mastered", "skill": "SQL"}], "missing_skills": [], "weak_skills": [], "strong_skills": ["SQL"]}
        elif "job-specific interview rubric" in system:
            content = {"dimensions": rubric}
        elif "senior interviewer. Output JSON only" in system:
            content = {"questions": [{"id": i, "category": "scenario", "question_type": "professional",
                        "difficulty": "medium", "depth_level": "application", "focus_key": "f1", "jd_ref": "0",
                        "project_ref": "", "research_ref": "research:job:1", "skills": ["SQL"],
                        "content": f"在第{i}个合成业务场景中，你如何用 SQL 聚合并核对数据？"} for i in range(1, 13)]}
        elif "dimension_levels" in text and "candidate_intent" in text and "复盘本次模拟面试" not in text:
            with lock:
                counter["assess"] += 1
                follow = counter["assess"] == 1
            content = {"scoreable": True, "dimension_levels": {item["key"]: 3 for item in rubric},
                       "evidence": {item["key"]: ["SQL"] for item in rubric}, "missing_points": ["抽样核对细节"] if follow else [],
                       "hallucination_or_conflict": False, "next_action": "follow_up" if follow else "next_question",
                       "confidence": "high", "candidate_intent": "answer", "follow_up_reason": "抽样核对细节"}
        elif "项目清单" in text and '"conflict"' in text:
            content = {"conflict": False, "claims": [], "explanation": "Synthetic fixture"}
        elif "复盘本次模拟面试" in text:
            content = {"status": "ok", "findings": [], "suggestions": []}
        elif "逐题评分" in text and "平均分" in text:
            content = {"text_analysis": "Synthetic integration feedback, not a real model evaluation.",
                       "strengths": ["合成样例表达了数据核对步骤"], "weaknesses": [], "learning_path": ["练习结果核对"], "resources": []}
        else:
            with lock:
                counter["ask"] += 1
                index = counter["ask"]
            content = f"在合成场景{index}中，你会如何用 SQL 核对数据口径与结果？"
        result = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        event({"kind": "sdk_fixture", "input_hash": digest(messages), "output_hash": digest(result),
               "requested_model": kwargs["model"], "fixture_usage": {"input": 10, "output": 5, "total": 15}})
        time.sleep(.03)
        return SimpleNamespace(id="synthetic-response", model="synthetic-smoke",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=result, tool_calls=[]))])

    llm.client.chat.completions.create = create
    original_wrap = builder.wrap_node

    def trace_wrap(func, node_name=None):
        wrapped = original_wrap(func, node_name)
        def traced(state):
            start = time.perf_counter()
            result = wrapped(state)
            event({"kind": "node", "name": node_name, "function": func.__name__,
                   "question_version": result.get("question_version"), "assessments_added": len(result.get("assessments") or []),
                   "elapsed_seconds": round(time.perf_counter() - start, 6)})
            return result
        return traced

    builder.wrap_node = trace_wrap
    # Last line of defense: a missing fixture must never reach a cloud/proxy URL.
    import httpx
    import requests
    def forbidden(*args, **kwargs):
        raise RuntimeError("External traffic forbidden in the delivery smoke server")
    httpx.Client.send = forbidden
    requests.Session.request = forbidden
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error", access_log=False)


def run(output, hold_seconds=0):
    if output.exists() or not output.is_relative_to((ROOT / "docs/release/evidence").resolve()):
        raise ValueError("Output must be a NEW folder inside docs/release/evidence")
    output.mkdir(parents=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="rolepilot-smoke-") as folder:
        database_root = Path(folder)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        child = None
        token = None
        command = [sys.executable, str(Path(__file__).resolve()), "--serve", str(port), str(database_root)]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(method, path, body=None, *, raw=None, content_type=None):
            headers = {"Authorization": "Bearer " + token} if token else {}
            data = raw if raw is not None else (json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
            if content_type or body is not None:
                headers["Content-Type"] = content_type or "application/json"
            req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
            started = time.perf_counter()
            try:
                response = opener.open(req, timeout=30)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                payload = response.read()
                try:
                    value = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    value = payload.decode("utf-8", errors="replace")
                with records_lock:
                    records.append({"method": method, "path": path, "status": response.code,
                                    "elapsed_seconds": round(time.perf_counter()-started, 6)})
                return response.code, value

        records_lock = threading.Lock()
        def start():
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Smoke server exited before health check")
                try:
                    if request("GET", "/api/health")[0] == 200:
                        return process
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(.1)
            process.terminate()
            process.communicate(timeout=10)
            raise RuntimeError("Smoke server health timeout")

        def stop(process):
            process.terminate()
            process.communicate(timeout=10)

        try:
            child = start()
            status, html = request("GET", "/")
            assert status == 200 and "<title>RolePilot" in html
            _, openapi = request("GET", "/openapi.json")
            assert openapi["info"]["title"] == "RolePilot API"
            boundary = "rolepilot-synthetic-upload"
            resume = "合成候选人：业务分析方向在校生。技能 SQL、数据核对。合成校园分析项目用 SQL 聚合数据并抽样检查结果。"
            raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="synthetic.txt"\r\n'
                   'Content-Type: text/plain\r\n\r\n' + resume + f'\r\n--{boundary}--\r\n').encode()
            status, context = request("GET", "/api/auth/context")
            assert status == 200 and set(context) == {"server_id"}
            token = "unissued-synthetic-token"  # Simulate a browser carrying another copy's token.
            assert request("POST", "/api/uploads/resume", raw=raw, content_type="multipart/form-data; boundary="+boundary)[0] == 401
            status, credential = request("POST", "/api/auth/anonymous")
            assert status == 201 and credential["reused"] is False
            assert credential["server_id"] == context["server_id"]
            token = credential["token"]  # NEVER exported or logged.
            status, reused = request("POST", "/api/auth/anonymous")
            assert status == 200 and reused["reused"] is True and reused["token"] == token
            status, extracted = request("POST", "/api/uploads/resume", raw=raw, content_type="multipart/form-data; boundary="+boundary)
            assert status == 200 and extracted["text"] == resume
            status, first = request("POST", "/api/sessions", {"resume_text": extracted["text"],
                                    "jd_text": "合成业务分析员：确认数据口径，使用 SQL 聚合数据，抽样核对结果并解释业务假设。", "session_name": "Synthetic TCP smoke"})
            assert status == 200 and first["question_version"] == 1 and first["question"] and not first["done"]
            sid = first["session_id"]
            route = "/api/sessions/"+sid
            body = {"answer": "我先确认数据口径，再用 SQL 聚合并抽样核对差异，最后记录检查结果和业务假设。",
                    "answer_request_id": "00000000-0000-4000-8000-000000000001", "expected_question_version": 1}
            status, second = request("POST", route+"/answer", body)
            assert status == 200 and second["question_version"] == 2
            calls = second["usage_summary"]["known_calls"]
            status, replay = request("POST", route+"/answer", body)
            assert status == 200 and replay["replayed"] and replay["usage_summary"]["known_calls"] == calls
            stale = {**body, "answer_request_id": "00000000-0000-4000-8000-000000000002"}
            assert request("POST", route+"/answer", stale)[0] == 409
            follow = {**body, "answer_request_id": "00000000-0000-4000-8000-000000000003", "expected_question_version": 2}
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent = list(pool.map(lambda _: request("POST", route+"/answer", follow), range(4)))
            for _ in range(100):
                status, third = request("GET", route+"/answer-requests/"+follow["answer_request_id"])
                if status == 200:
                    break
                time.sleep(.05)
            assert status == 200 and third["question_version"] == 3
            stop(child); child = None
            child = start()
            assert request("GET", "/api/auth/context")[1] == context
            status, restored = request("GET", route)
            assert status == 200 and restored["question_version"] == 3 and digest(restored["messages"]) == digest(third["messages"])
            status, finished = request("POST", route+"/stop")
            assert status == 200 and finished["done"] and finished["report"]["overall_score"] == 6
            assert finished["report"]["cost"] is None
            assert request("GET", route)[1]["report"] == finished["report"]
            assert request("DELETE", route)[0] == 200 and request("GET", route)[0] == 404
            assert request("GET", "/api/sessions")[1] == []
            trace = [json.loads(line) for line in (database_root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            node_order = [item["name"] for item in trace if item["kind"] == "node"]
            assert node_order.count("assess") == 3  # two answers plus unscored stop.
            assert sum(item["assessments_added"] for item in trace if item["kind"] == "node") == 2
            assert not list((database_root / "uploads").glob("*"))
            result = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
                      "integration": "Real localhost TCP, production API/service/graph/SQLite/checkpointer with SDK and search fixtures",
                      "real_llm_calls": 0, "real_search_calls": 0, "ocr": "not_run (TXT extraction only)",
                      "session_id": sid, "thread_id": sid, "base_url": base, "node_order": node_order,
                      "sdk_fixture_calls": sum(item["kind"] == "sdk_fixture" for item in trace),
                      "search_fixture_calls": sum(item["kind"] == "search_fixture" for item in trace),
                      "concurrent_http_statuses": [item[0] for item in concurrent],
                      "answer_assessments": 2, "versions": [first["question_version"],second["question_version"],third["question_version"]],
                      "replay_added_calls": 0, "stale_http_status": 409, "restart_restored": True,
                      "authentication": {"stale_upload_status": 401, "recovered_upload_status": 200,
                                         "valid_identity_reused": True, "namespace_persists_after_restart": True},
                      "score": finished["report"]["overall_score"], "grade": finished["report"]["grade"],
                      "usage_is_fixture_not_provider_measurement": True, "report_cost": None,
                      "delete_get_status": 404, "originals_remaining": 0, "requests": records, "trace": trace,
                      "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in [*sorted((ROOT/"app").rglob("*.py")),*sorted((ROOT/"api").rglob("*.py")),Path(__file__).resolve()]}}
            (output / "runtime.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: result[key] for key in ("status","base_url","node_order","sdk_fixture_calls","real_llm_calls","score","restart_restored")},ensure_ascii=False),flush=True)
            if hold_seconds:
                print("Browser inspection window (fresh synthetic identity only): "+base, flush=True)
                time.sleep(hold_seconds)
        finally:
            if child is not None:
                stop(child)
    return 0


def browser_session(output):
    """Keep a synthetic-only server alive until explicit exit; export safe metadata."""
    if output.exists() or not output.is_relative_to((ROOT / "docs/release/evidence").resolve()):
        raise ValueError("Output must be a NEW folder inside docs/release/evidence")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="rolepilot-smoke-") as folder:
        database_root = Path(folder).resolve()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [sys.executable, str(Path(__file__).resolve()), "--serve", str(port), str(database_root)]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        child = None
        lifecycle = []

        def launch():
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Browser smoke server exited before health check")
                try:
                    with opener.open(base + "/api/health", timeout=1) as response:
                        if response.status == 200:
                            lifecycle.append({"event": "start", "at": datetime.now(timezone.utc).isoformat()})
                            return process
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(.1)
            process.terminate(); process.communicate(timeout=10)
            raise RuntimeError("Browser smoke server health timeout")

        def stop_child():
            nonlocal child
            if child is not None:
                child.terminate(); child.communicate(timeout=10); child = None
                lifecycle.append({"event": "stop", "at": datetime.now(timezone.utc).isoformat()})

        def snapshot():
            events = database_root / "events.jsonl"
            trace = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()] if events.exists() else []
            sessions = []
            for db in sorted(database_root.glob("sessions*.db")):
                with closing(sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)) as connection:
                    connection.row_factory = sqlite3.Row
                    for row in connection.execute("SELECT id,status,question_version,state_json,report_json FROM sessions"):
                        state = json.loads(row["state_json"])
                        report = json.loads(row["report_json"]) if row["report_json"] else None
                        sessions.append({"session_id": row["id"], "thread_id": row["id"], "status": row["status"],
                            "question_version": row["question_version"], "messages_hash": digest(state.get("conversation_history", [])),
                            "message_count": len(state.get("conversation_history", [])),
                            "user_message_count": sum(message.get("role") in {"user", "human"} for message in state.get("conversation_history", [])),
                            "assessment_count": len(state.get("assessments", [])),
                            "score": report.get("overall_score") if report else None,
                            "grade": report.get("grade") if report else None,
                            "self_check_status": (state.get("self_check_report") or {}).get("status"),
                            "answer_requests": dict(connection.execute("SELECT status,count(*) FROM answer_requests WHERE session_id=? GROUP BY status", (row["id"],))),
                            "jd_text_hash": digest(state.get("jd_text")), "resume_text_hash": digest(state.get("resume_text"))})
            result = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": "browser_isolated_fixture_server",
                      "base_url": base, "real_llm_calls": 0, "real_search_calls": 0,
                      "sdk_fixture_calls": sum(item["kind"] == "sdk_fixture" for item in trace),
                      "search_fixture_calls": sum(item["kind"] == "search_fixture" for item in trace),
                      "node_order": [item["name"] for item in trace if item["kind"] == "node"],
                      "sessions": sessions, "originals_remaining": len(list((database_root / "uploads").glob("*"))),
                      "lifecycle": lifecycle, "trace": trace,
                      "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in [*sorted((ROOT/"app").rglob("*.py")),*sorted((ROOT/"api").rglob("*.py")),Path(__file__).resolve()]}}
            (output / "server-runtime.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"snapshot": str(output.relative_to(ROOT)), "sdk_fixture_calls": result["sdk_fixture_calls"],
                              "sessions": sessions}, ensure_ascii=False), flush=True)

        try:
            child = launch()
            print("Browser isolated server: " + base, flush=True)
            print("Commands: snapshot / restart / exit. No automatic cloud calls or user database access.", flush=True)
            while True:
                try:
                    instruction = input().strip()
                except EOFError:
                    break
                if instruction == "exit":
                    break
                if instruction == "restart":
                    snapshot(); stop_child(); child = launch()
                    print("Isolated process restarted; database preserved: " + base, flush=True)
                elif instruction == "snapshot":
                    snapshot()
        finally:
            try:
                snapshot()
            finally:
                stop_child()
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--serve":
        return serve(int(sys.argv[2]), Path(sys.argv[3]).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", default="docs/release/evidence/my-delivery-smoke")
    parser.add_argument("--hold-seconds", type=int, default=0, choices=range(0,301))
    parser.add_argument("--browser-session", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Prepared only; no service, network, database or file changes.")
        return 0
    output = (ROOT/args.output).resolve()
    return browser_session(output) if args.browser_session else run(output, args.hold_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
