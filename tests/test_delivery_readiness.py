"""Delivery regressions: synthetic inputs, real local code, no cloud calls."""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.graph.schema import record_to_message
from app.llm import client as llm
from app.nodes.research_job import _domain_tier
from app.privacy import MASK, redact_messages
from app.scoring.protocol import normalize_assessment
from app.telemetry.pricing import snapshot_price
from app.utils.tokens import estimate_tokens, truncate_text

ROOT = Path(__file__).resolve().parents[1]


def test_cli_report_with_unconfigured_cost_does_not_crash(monkeypatch, capsys):
    import main

    report = {"overall_score": 6, "text_analysis": "Synthetic feedback", "cost": None,
              "token_totals": {"total": None}, "learning_path": [], "resources": []}

    class Service:
        def start_session(self, **kwargs):
            return {"session_id": "synthetic", "question": "Synthetic question", "state": {}}

        def submit_answer(self, *args):
            return {"report": report}

    inputs = iter(["Synthetic JD", "Synthetic resume", "Synthetic answer"])
    monkeypatch.setattr(main, "InterviewService", Service)
    monkeypatch.setattr("builtins.input", lambda *args: next(inputs))
    main.run_interview()
    output = capsys.readouterr().out
    assert "Synthetic feedback" in output and "6" in output
    assert "CNY" not in output


def test_cli_report_returned_at_start_does_not_ask_for_an_answer(monkeypatch, capsys):
    import main

    class Service:
        def start_session(self, **kwargs):
            return {"session_id": "synthetic", "question": "", "state": {
                "evaluation_report": {"overall_score": 0, "text_analysis": "Synthetic early report"}}}

        def submit_answer(self, *args):
            pytest.fail("An already-finished interview must not submit an answer")

    inputs = iter(["Synthetic JD", "Synthetic resume"])
    monkeypatch.setattr(main, "InterviewService", Service)
    monkeypatch.setattr("builtins.input", lambda *args: next(inputs))
    main.run_interview()
    assert "Synthetic early report" in capsys.readouterr().out


@pytest.mark.parametrize("call", [
    {"id": "synthetic-call", "type": "function", "function": {
        "name": "WebSearch", "arguments": '{"query":"test@example.invalid Python"}'}},
    {"id": "synthetic-call", "type": "tool_call", "name": "WebSearch",
     "args": {"query": "test@example.invalid Python"}},
])
def test_dict_tool_calls_survive_conversion_and_contact_redaction(call):
    messages = [{"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": "Synthetic result", "tool_call_id": "synthetic-call"}]
    converted = redact_messages(llm._convert_messages(messages))
    tool_call = converted[0]["tool_calls"][0]
    assert tool_call["id"] == converted[1]["tool_call_id"] == "synthetic-call"
    assert tool_call["function"]["name"] == "WebSearch"
    assert json.loads(tool_call["function"]["arguments"])["query"] == MASK + " Python"


@pytest.mark.parametrize("role, expected", [
    ("AIMessage", AIMessage), ("HumanMessage", HumanMessage),
    ("SystemMessage", SystemMessage), ("ToolMessage", ToolMessage),
])
def test_message_class_aliases_preserve_role(role, expected):
    message = record_to_message({"type": role, "content": "Synthetic text", "tool_call_id": "synthetic-call"})
    assert isinstance(message, expected)
    sdk_role = {AIMessage: "assistant", HumanMessage: "user", SystemMessage: "system", ToolMessage: "tool"}[expected]
    assert llm._convert_messages([{"type": role, "content": "Synthetic text"}])[0]["role"] == sdk_role
    if expected is ToolMessage:
        assert message.tool_call_id == "synthetic-call"


@pytest.mark.parametrize("field", ["confidence", "next_action"])
@pytest.mark.parametrize("invalid", [[], {}])
def test_structured_enum_output_is_rejected_without_crashing(field, invalid):
    parsed = {"scoreable": True, "dimension_levels": {"d1": 4},
              "evidence": {"d1": ["SQLite"]}, "missing_points": [],
              "hallucination_or_conflict": False, "next_action": "next_question", "confidence": "high"}
    parsed[field] = invalid
    result = normalize_assessment(parsed, "I used SQLite", [{"key": "d1", "weight": 1}])
    assert not result["scoreable"] and result["dimension_levels"] == {}
    assert "invalid_" + field in result["protocol_errors"]


def test_invalid_price_url_never_breaks_call_accounting(monkeypatch):
    for key, value in {"cost_input_per_1m": 1, "cost_output_per_1m": 2,
                       "cost_model": "synthetic", "cost_currency": "USD",
                       "cost_price_source": "https://[invalid", "cost_price_effective_date": "2026-09-13"}.items():
        monkeypatch.setattr(settings, key, value)
    assert snapshot_price("synthetic")["status"] == "invalid_price_source"


@pytest.mark.parametrize("domain,tier", [
    ("agency.gov.cn", "high"), ("gov.cn", "high"), ("docs.python.org", "high"),
    ("sub.github.com", "medium"), ("github.com", "medium"),
    ("docs.python.org.attacker.invalid", "low"), ("agency.gov.cn.attacker.invalid", "low"),
    ("notgithub.com", "low"), ("github.com.attacker.invalid", "low"),
])
def test_source_trust_uses_hostname_boundaries(domain, tier):
    assert _domain_tier(domain) == tier


@pytest.mark.parametrize("text", ["Synthetic <|endoftext|> answer", "Synthetic <|fim_prefix|> JD"])
def test_special_token_literals_are_untrusted_text_not_tokenizer_errors(text):
    assert estimate_tokens(text) > 0
    assert truncate_text(text, 1000) == text


@pytest.mark.parametrize("parsed", [
    {}, {"status": []}, {"status": "ok", "findings": "bad", "suggestions": []},
])
def test_invalid_self_check_output_is_not_reported_as_success(monkeypatch, parsed):
    from app.nodes import self_check as module

    monkeypatch.setattr(module, "chat_with_usage", lambda *args, **kwargs: (json.dumps(parsed), {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }))
    result = module.self_check({"assessments": [], "summaries": [], "conversation_history": []})
    assert result["self_check_report"]["status"] == "warning"
    assert result["self_check_report"]["reason"] == "invalid_self_check_output"
    assert result["usage_records"][0]["total_tokens"] == 15


def test_public_regression_fixture_has_no_private_history_and_is_not_ignored():
    fixture = ROOT / "tests/fixtures/intent/replay-metadata.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    assert data["total"] == sum(row["count"] for row in data["cohorts"]) == 208
    assert len(data["source_sha256"]) == 64
    allowed = {"count", "old_explain", "old_stop", "new_stop_command", "genuine_explain", "new_intent"}
    assert all(set(row) == allowed for row in data["cohorts"])
    assert not (set(data) & {"resume", "question", "answer", "session", "db"})


def test_public_readme_and_package_metadata_use_current_brand():
    import api.main

    package = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "frontend/package-lock.json").read_text(encoding="utf-8"))
    assert package["name"] == lock["name"] == lock["packages"][""]["name"] == "rolepilot-frontend"
    assert api.main.app.title == "RolePilot API"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# RolePilot")
    assert "C:\\AI_Projects" not in readme
    assert "<title>RolePilot" in (ROOT / "frontend/index.html").read_text(encoding="utf-8")


def test_repeated_quote_is_not_counted_as_multiple_answer_evidence_points():
    parsed = {"scoreable": True, "dimension_levels": {"d1": 3, "d2": 3},
              "evidence": {"d1": ["SQL"], "d2": ["SQL"]}, "missing_points": [],
              "hallucination_or_conflict": False, "next_action": "next_question", "confidence": "high"}
    result = normalize_assessment(parsed, "I used SQL", [{"key": "d1", "weight": .5}, {"key": "d2", "weight": .5}])
    assert result["scoreable"] and result["dimension_levels"] == {"d1": 3, "d2": 3}
    assert result["evidence"] == parsed["evidence"]
    assert result["covered_points"] == ["SQL"]


def test_self_check_receives_complete_json_without_claiming_unseen_messages(monkeypatch):
    from app.nodes import self_check as module
    captured = []
    def reply(messages, **kwargs):
        captured.append(messages[-1].content)
        return '{"status":"ok","findings":[],"suggestions":[]}', {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    monkeypatch.setattr(module, "chat_with_usage", reply)
    assessments = [{"comment": "Synthetic " * 1000, "end_marker": "ASSESSMENT_END"}]
    result = module.self_check({"assessments": assessments, "summaries": [],
                                "conversation_history": [{"role": "user", "content": "not supplied"}]})
    payload = captured[0].split("逐题评估：\n", 1)[1].split("\n\n只输出", 1)[0]
    assert json.loads(payload) == assessments
    report = result["self_check_report"]
    assert report["checked_messages"] == 0 and report["checked_assessments"] == 1
    assert report["input_scope"] == "summaries_and_assessments"


def test_summary_receives_complete_assessment_json(monkeypatch):
    from app.context import manager
    captured = []
    def reply(messages, **kwargs):
        captured.append(messages[-1].content)
        return '{"summary":"Synthetic summary","score":6}', {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    monkeypatch.setattr(manager, "chat_with_usage", reply)
    assessment = {"score": 6, "comment": "Synthetic " * 500, "end_marker": "ASSESSMENT_END"}
    summary, usage = manager.summarize_turn("Synthetic question", "Synthetic answer", assessment, 1)
    assert summary is not None and usage["total_tokens"] == 15
    payload = captured[0].split("本轮评分：", 1)[1].strip()
    assert json.loads(payload) == assessment


def test_oversized_self_check_is_rejected_without_provider_call(monkeypatch):
    from app.nodes import self_check as module
    monkeypatch.setattr(settings, "context_budget", 1000)
    monkeypatch.setattr(llm.client.chat.completions, "create", lambda **kwargs: pytest.fail("Must reject before SDK"))
    state = {"assessments": [{"comment": "Synthetic " * 10000}], "summaries": [],
             "evaluation_report": {"overall_score": 6}}
    result = module.self_check(state)
    assert result["self_check_report"]["status"] == "error"
    assert result["usage_records"][0]["usage_status"] == "not_called"
    assert state["evaluation_report"]["overall_score"] == 6


def test_fallback_assessment_does_not_silently_cut_current_answer(monkeypatch):
    from app.nodes import assess as module
    captured = []
    def reply(messages, **kwargs):
        captured.append(messages[-1].content)
        return '{"overall_level":3}', {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    monkeypatch.setattr(module, "chat_with_usage", reply)
    answer = "Synthetic " * 400 + "CRITICAL_ANSWER_END"
    dimensions, usage = module._fallback_overall_level({}, "Synthetic question", answer, [], [{"key": "d1", "weight": 1}])
    assert answer in captured[0]
    assert dimensions == {"d1": 6} and usage["total_tokens"] == 15


@pytest.mark.parametrize("snapshot_fails", [False, True])
def test_browser_smoke_closes_sqlite_and_stops_child_even_if_snapshot_fails(monkeypatch, tmp_path, snapshot_fails):
    from contextlib import contextmanager
    import sqlite3
    from types import SimpleNamespace
    from scripts import delivery_smoke as module

    root = tmp_path / "database"
    root.mkdir()
    original_connect = sqlite3.connect
    connection = original_connect(root / "sessions_synthetic.db")
    connection.execute("CREATE TABLE sessions(id,status,question_version,state_json,report_json)")
    connection.execute("CREATE TABLE answer_requests(session_id,status)")
    connection.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (
        "synthetic-session", "completed", 3, json.dumps({"conversation_history": [], "assessments": []}),
        json.dumps({"overall_score": 6, "grade": "B"})))
    connection.commit()
    connection.close()
    opened, children = [], []

    class TrackedConnection(sqlite3.Connection):
        closed = False
        def close(self):
            self.closed = True
            super().close()

    def tracked_connect(*args, **kwargs):
        result = original_connect(*args, factory=TrackedConnection, **kwargs)
        opened.append(result)
        return result

    @contextmanager
    def temporary_directory(**kwargs):
        try:
            yield str(root)
        finally:
            assert opened and all(item.closed for item in opened), "SQLite context manager did not close the handle"
            assert children and all(item.stopped for item in children), "Snapshot failure left the child alive"

    class Child:
        stopped = False
        def poll(self):
            return None
        def terminate(self):
            self.stopped = True
        def communicate(self, **kwargs):
            return b"", b""

    def launch(*args, **kwargs):
        child = Child()
        children.append(child)
        return child

    @contextmanager
    def health(*args, **kwargs):
        yield SimpleNamespace(status=200)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    synthetic_script = tmp_path / "synthetic_smoke.py"
    synthetic_script.write_text("# Synthetic subprocess fixture\n", encoding="utf-8")
    monkeypatch.setattr(module, "__file__", str(synthetic_script))
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=health))
    monkeypatch.setattr(module.sqlite3, "connect", tracked_connect)
    instructions = iter(["snapshot", "restart", "exit"])
    monkeypatch.setattr("builtins.input", lambda: next(instructions))
    if snapshot_fails:
        def fail_digest(value):
            raise RuntimeError("synthetic snapshot failure")
        monkeypatch.setattr(module, "digest", fail_digest)
    try:
        if snapshot_fails:
            with pytest.raises(RuntimeError, match="synthetic snapshot failure"):
                module.browser_session(tmp_path / "docs/release/evidence/synthetic-exit")
        else:
            assert module.browser_session(tmp_path / "docs/release/evidence/synthetic-exit") == 0
    finally:
        for item in opened:
            item.close()
