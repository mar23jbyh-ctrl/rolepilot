"""Research-only planning regressions. External providers are mocked, not billed."""

import os
from pathlib import Path
import subprocess
import sys
import uuid
import json

import pytest

from app.config import settings
from app.nodes import ask, plan
from app.prompts.references import research_references_block
from app.tools.executor import HANDLERS, execute_tool_call
from app.tools.schemas import ARG_MODELS, get_tool_schemas

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"WebSearch", "CodeExplainer", "DynamicQuestion"}


def test_research_reference_formatter_preserves_source_and_limits_excerpt():
    block = research_references_block([
        {"source_id": "research:job:1", "category": "scenario",
         "difficulty": "medium", "answer": "x" * 1300},
    ])
    assert "source_id=research:job:1" in block
    assert "A: " + "x" * 1200 in block
    assert "x" * 1201 not in block


def test_research_reference_formatter_caps_fragments_and_handles_empty_input():
    assert research_references_block([]) == ""
    items = [{"source_id": f"research:job:{i}", "answer": str(i)}
             for i in range(1, 10)]
    block = research_references_block(items)
    assert "source_id=research:job:8" in block
    assert "source_id=research:job:9" not in block


@pytest.mark.parametrize("source_id,expected", [
    ("research:job:1", "web"), ("research:job", "web"),
    ("research:job:missing", "llm"), ("llm:generated", "llm"), ("", "llm"),
])
def test_question_source_type_is_derived_from_actual_research_ids(source_id, expected):
    result = plan._normalize_plan(
        [{"content": "请说明需求分析的方法", "source_id": source_id,
          "source_type": "question_bank"}],
        [{"source_id": "research:job:1"}, {"source_id": "research:job"}], {},
    )
    assert result[0]["source_type"] == expected
    assert result[0]["source_id"]


def test_only_current_tools_are_registered_and_exposed():
    assert set(HANDLERS) == set(ARG_MODELS) == TOOLS
    assert {s["function"]["name"] for s in get_tool_schemas()} == TOOLS


def test_removed_tool_is_refused_without_execution():
    result = execute_tool_call("KbSearch", {"query": "synthetic"})
    assert not result.success and "Unknown tool" in result.error
    assert result.source == "" and not result.degraded


@pytest.mark.parametrize("domain,expected", [
    ("技术-后端开发", TOOLS),
    ("法律-企业法务", {"WebSearch"}),
    ("采购-供应链", {"WebSearch"}),
])
def test_tool_allowlist_remains_role_scoped(domain, expected):
    state = {"role_profile": {"domain": domain}}
    assert ask._allowed_tool_names(state) == expected
    assert {s["function"]["name"] for s in ask._tool_schemas(state)} == expected


def test_deleted_settings_are_not_runtime_features():
    deleted = {"kb_enabled", "question_bank_dir", "question_bank_file",
               "question_bank_exclude", "chroma_dir", "embedding_model",
               "embedding_api_key", "embedding_base_url"}
    assert deleted.isdisjoint(type(settings).model_fields)
    assert not (ROOT / "app/rag").exists()
    assert not (ROOT / "data/question_bank").exists()
    assert not (ROOT / "data/chroma").exists()
    dependencies = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "chromadb" not in dependencies and "rank-bm25" not in dependencies


def test_fresh_process_imports_without_removed_modules_and_ignores_old_env():
    child = r"""
import importlib.abc
import sys
import socket

class BlockRemoved(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'chromadb', 'rank_bm25'} or fullname.startswith('app.rag'):
            raise ImportError('Removed retrieval dependency must not be imported: ' + fullname)

sys.meta_path.insert(0, BlockRemoved())
def deny(*args, **kwargs):
    raise RuntimeError('No network allowed in startup regression')
socket.socket.connect = deny
import app.service
import api.main
from app.config import settings
assert 'kb_enabled' not in type(settings).model_fields
api.main.app.openapi()
print('startup_without_local_retrieval_ok')
"""
    environment = os.environ.copy()
    environment.update(
        API_KEY="synthetic-test-key", OPENAI_API_KEY="synthetic-test-key",
        DEEPSEEK_API_KEY="synthetic-test-key", ENABLE_EXTERNAL_TRACING="false",
        LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false",
        LANGCHAIN_TRACING="false", KB_ENABLED="true",
        QUESTION_BANK_DIR="synthetic-do-not-read",
        CHROMA_DIR="synthetic-do-not-create",
        EMBEDDING_API_KEY="synthetic-do-not-use",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", child], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    # Do not echo provider configuration or arbitrary child stderr.
    assert result.returncode == 0, "Fresh process import failed"
    assert result.stdout.strip() == "startup_without_local_retrieval_ok"
    assert not (ROOT / "synthetic-do-not-create").exists()


@pytest.mark.parametrize("job,domain,slug,topic", [
    ("合同法务", "法律-企业法务", "legal-corporate", "合同审查"),
    ("采购专员", "采购-供应链", "procurement", "供应商评估"),
    ("Python后端工程师", "技术-后端开发", "tech-backend", "接口设计"),
])
def test_real_graph_and_sqlite_still_pause_answer_report_and_restore(
    tmp_path, monkeypatch, job, domain, slug, topic
):
    from app import service as service_mod
    from app.nodes import analyze, assess, evaluate, research_job, self_check
    from app.session.store import SessionStore

    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
             "finish_reason": "stop"}
    answer = "我先梳理验收标准，再根据数据比较方案，最后复核风险与结果。"
    seen = []
    assessment_calls = []
    searches = []
    plan_prompts = []
    dimensions = [
        {"key": f"d{i}", "label": f"能力{i}", "weight": 0.25,
         "definition": topic, "threshold": "具体步骤与依据"}
        for i in range(1, 5)
    ]

    def as_json(value):
        return json.dumps(value, ensure_ascii=False), dict(usage)

    def analyze_chat(messages, **kwargs):
        system = messages[0].content
        if "resume facts" in system:
            return as_json({"skills": [topic, "沟通", "分析", "交付"],
                            "projects": ["合成练习项目：" + topic]})
        if "job requirements" in system:
            return as_json({"requirements": [
                {"category": "experience", "text": topic, "skills": [topic], "weight": 1}
            ]})
        if "job domain" in system:
            return as_json({
                "job_title": job, "domain": domain, "domain_slug": slug,
                "industry": "合成测试行业", "industry_known": True,
                "needs_web_research": False, "confidence": 0.95,
                "key_skills": [topic],
                "assessment_focus": [{"key": "f1", "name": topic,
                                      "keywords": [topic], "depth": "application"}],
                "out_of_scope_topics": [], "forbidden_topics": [],
            })
        if "skill gap" in system:
            return as_json({"matches": [
                {"skill": skill, "status": "mastered"} for skill in (topic, "沟通", "分析")
            ], "missing_skills": [], "weak_skills": [], "strong_skills": [topic]})
        if "rubric" in system:
            return as_json({"dimensions": dimensions})
        pytest.fail("Unexpected analysis provider request")

    def fake_search(arguments):
        searches.append(arguments)
        return {"success": True, "content": topic + "：梳理需求、比较方案、复核风险。",
                "source": "https://example.test/synthetic-reference", "raw": []}

    def plan_chat(messages, **kwargs):
        plan_prompts.append(messages[-1].content)
        return as_json({"questions": [
            {"id": index, "content": f"请说明{topic}的{angle}",
             "category": "scenario", "difficulty": "medium",
             "depth_level": "application", "focus_key": "f1", "jd_ref": "0",
             "research_ref": "research:job:1", "source_id": "research:job:1",
             "source_type": "web", "skills": [topic]}
            for index, angle in enumerate(("需求确认步骤", "风险复核方法"), 1)
        ]})

    def assessment_chat(messages, **kwargs):
        assessment_calls.append(messages)
        return as_json({
            "scoreable": True, "dimension_levels": {d["key"]: 4 for d in dimensions},
            "evidence": {d["key"]: ["梳理验收标准"] for d in dimensions},
            "missing_points": [], "hallucination_or_conflict": False,
            "next_action": "next_question", "confidence": "high",
            "candidate_intent": "answer", "is_relevant": True,
        })

    monkeypatch.setattr(analyze, "chat_with_usage", analyze_chat)
    monkeypatch.setattr(research_job, "web_search_handler", fake_search)
    monkeypatch.setattr(plan, "chat_with_usage", plan_chat)
    monkeypatch.setattr(ask, "chat_with_usage", lambda *a, **k: (
        f"请结合合成经历说明{topic}的处理方法？", dict(usage)))
    monkeypatch.setattr(ask, "chat_with_tools", lambda *a, **k: (
        f"请结合合成经历说明{topic}的处理方法？", [], dict(usage)))
    monkeypatch.setattr(assess, "chat_with_usage", assessment_chat)
    monkeypatch.setattr(evaluate, "chat_with_usage", lambda *a, **k: as_json(
        {"overall_score": 1, "grade": "D", "text_analysis": "合成测试评语"}))
    monkeypatch.setattr(self_check, "chat_with_usage", lambda *a, **k: as_json(
        {"status": "ok", "findings": [], "suggestions": []}))
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "target_questions", 2)
    monkeypatch.setattr(settings, "min_questions", 2)
    monkeypatch.setattr(settings, "max_questions", 2)
    monkeypatch.setattr(plan, "MIN_QUESTIONS", 2)
    monkeypatch.setattr(plan, "MAX_QUESTIONS", 2)

    # Observe real node functions; do not replace the graph builder or node logic.
    for module, function in (
        (analyze, "analyze"), (research_job, "research_job_node"), (plan, "plan"),
        (ask, "ask"), (assess, "assess"), (evaluate, "evaluate"),
        (self_check, "self_check"),
    ):
        original = getattr(module, function)
        def observe(state, _original=original, _name=function):
            seen.append(_name)
            return _original(state)
        monkeypatch.setattr(module, function, observe)

    services = []
    store = SessionStore(tmp_path / "sessions.db")
    try:
        initial = service_mod.InterviewService("synthetic-removal", store)
        services.append(initial)
        started = initial.start_session(
            job + "，岗位要求：" + topic,
            ("合成候选人，练习项目：" + topic + "，完成需求梳理、方案比较与验收。") * 12,
        )
        sid = started["session_id"]
        config = initial._config(sid)
        assert seen == ["analyze", "research_job_node", "plan", "ask"]
        assert len(searches) == 5
        assert started["state"]["references_status"] == "ok"
        assert started["state"]["research_status"] == "ok"
        assert len(started["state"]["question_plan"]) == 2
        assert all(q["source_type"] == "web" for q in started["state"]["question_plan"])
        assert "联网岗位调研参考" in plan_prompts[0]
        assert topic in plan_prompts[0] and "source_id=research:job:1" in plan_prompts[0]
        assert initial.graph.get_state(config).next == ("assess",)
        assert config["configurable"]["thread_id"] == sid
        assert assessment_calls == []

        request_id = str(uuid.uuid4())
        first = initial.submit_answer(sid, answer, request_id, 1)
        assert first["state"]["assessments"][0]["score"] == 8
        assert not first["state"]["assessments"][0].get("protocol_errors")
        assert first["state"]["question_version"] == 2
        assert first["state"]["current_answer"] == ""
        assert initial.graph.get_state(config).next == ("assess",)
        assert len(assessment_calls) == 1
        initial._checkpointer.conn.close()

        restarted = service_mod.InterviewService(
            "synthetic-removal", SessionStore(store.db_path)
        )
        services.append(restarted)
        assert restarted.restore(sid)["state"]["question_version"] == 2
        replay = restarted.submit_answer(sid, answer, request_id, 1)
        assert replay["replayed"] and replay["state"] == first["state"]
        assert len(assessment_calls) == 1

        final = restarted.stop_session(sid)
        assert final["done"]
        assert final["report"]["overall_score"] == 8
        assert final["report"]["grade"] == "A"
        assert final["report"]["text_analysis"] == "合成测试评语"
        assert seen[-2:] == ["evaluate", "self_check"]
        assert restarted.store.get(sid)["status"] == "completed"
        assert restarted.restore(sid)["report"] == final["report"]
        assert restarted.graph.get_state(config).next == ()
        assert not (tmp_path / "chroma").exists()
    finally:
        for service in services:
            service._checkpointer.conn.close()
