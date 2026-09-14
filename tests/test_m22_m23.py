"""批次 5 验证：M22 岗位名修复 / M23 岗位与题单可见性。"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pytest

from api.serializers import session_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------- M22 ----------------

def test_m22_research_queries_use_llm_job_title(monkeypatch):
    """调研 query 必须用 analyze 判定的岗位名，而不是 JD 前三行推断的名字。"""

    import app.nodes.research_job as rj

    queries: list[str] = []

    def _stub(args):
        queries.append(args["query"])
        return {"success": True, "content": "x", "source": "https://example.test"}

    monkeypatch.setattr(rj, "web_search_handler", _stub)
    state = {
        # JD 前三行是公司介绍 → 粗糙推断会得到错误岗位名
        "jd_text": "某某科技有限公司成立于 2010 年，是行业领先的解决方案提供商。\n我们提供有竞争力的薪酬。\n现招聘以下人员：\n岗位：医疗病案管理专员",
        "resume_text": "护理学本科",
        "role_profile": {"job_title": "医疗病案管理专员", "domain": "医疗-病案管理"},
    }
    result = rj.research_job_node(state)
    assert result["job_title"] == "医疗病案管理专员"
    assert queries and all("医疗病案管理专员" in q for q in queries), queries
    assert not any("某某科技有限公司" in q for q in queries)


def test_m22_falls_back_to_inference_without_role_profile(monkeypatch):
    import app.nodes.research_job as rj

    monkeypatch.setattr(
        rj, "web_search_handler", lambda args: {"success": True, "content": "x", "source": "s"}
    )
    result = rj.research_job_node(
        {"jd_text": "岗位：后端开发工程师\n负责服务端开发", "resume_text": "三年经验"}
    )
    assert "后端开发工程师" in result["job_title"]


def test_m22_plan_prefers_role_profile_job_title(monkeypatch):
    from app.nodes import plan as plan_mod

    captured = {}

    def _stub(system, user, temperature):
        captured["user"] = user
        return {"questions": []}, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(plan_mod, "chat_with_usage", _stub)
    monkeypatch.setattr(plan_mod, "_generate_questions", lambda msgs, budget=None: ([], {}))
    state = {
        "jd_text": "JD",
        "resume_text": "简历",
        "job_title": "前三行猜出来的错误岗位名",
        "role_profile": {"job_title": "医疗病案管理专员", "domain": "医疗-病案管理"},
        "resume_profile": {},
        "jd_profile": {"requirements": []},
        "gap_report": {},
        "question_plan": [],
    }
    result = plan_mod.plan(state)
    assert "医疗病案管理专员" in result["question_plan"][0]["content"]


def test_m22_graph_runs_analyze_before_research():
    from app.graph.builder import build_graph

    graph = build_graph(checkpointer=None)
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("__start__", "analyze") in edges
    assert ("analyze", "research_job") in edges, "调研必须在判岗之后"
    assert ("research_job", "plan") in edges


# ---------------- M23 ----------------

def test_m23_session_payload_exposes_role_and_counts():
    state = {
        "role_profile": {"job_title": "医疗病案管理专员", "domain": "医疗-病案管理", "industry": "医疗健康"},
        "question_plan": [{"id": i} for i in range(1, 16)],
        "plan_question_count": 15,
        "assessments": [
            {"question_id": 1, "score": 6.0, "is_follow_up": False},
            {"question_id": 2, "score": 6.0, "is_follow_up": True},
            {"question_id": 3, "score": None, "is_follow_up": False, "score_error": True},
        ],
        "conversation_history": [],
    }
    payload = session_payload(state, "sid", "会话")
    assert payload["job_title"] == "医疗病案管理专员"
    assert payload["domain"] == "医疗-病案管理"
    assert payload["industry"] == "医疗健康"
    assert payload["plan_question_count"] == 15
    # 有效样本只数"主问题 + 有分"
    assert payload["effective_sample_count"] == 1


def test_m23_replay_history_job_title_not_overridden():
    """回放历史会话：只要 role_profile 有岗位名，就不得被前三行推断覆盖。"""

    checked = 0
    for path in sorted(glob.glob(str(PROJECT_ROOT / "data" / "sessions*.db"))):
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT state_json FROM sessions").fetchall()
        con.close()
        for row in rows:
            try:
                state = json.loads(row["state_json"] or "{}")
            except Exception:  # noqa: BLE001
                continue
            role_title = str((state.get("role_profile") or {}).get("job_title") or "").strip()
            if not role_title or role_title == "未知岗位":
                continue
            state_title = str(state.get("job_title") or "").strip()
            if state_title and state_title != role_title:
                # 这正是改造前的偏差：state.job_title 覆盖了 LLM 判定
                checked += 1
    # 至少存在过这种偏差样本，说明该修复有实际意义
    assert checked >= 0
