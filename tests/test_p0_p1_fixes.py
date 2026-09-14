"""P0/P1 四项修复的验证：B1+B2 会话级追问预算 / B3 评分兜底 / A1 岗位名修正。"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.graph.edges import route_after_assessment
from app.nodes import assess as assess_mod
from app.nodes import ask as ask_mod
from app.nodes.assess import LEVEL_TO_SCORE

USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "finish_reason": "stop"}
RUBRIC = [{"key": "d1", "label": "维度一", "weight": 1.0, "definition": "x", "threshold": "t"}]


def _state(**overrides) -> dict:
    state = {
        "assessments": [
            {"question_id": 1, "score": 4.0, "is_follow_up": False, "next_action": "follow_up",
             "should_follow_up": True}
        ],
        "current_question_index": 0,
        "question_plan": [{"id": 1}, {"id": 2}],
        "follow_up_count": 1,
        "max_follow_ups": 3,
        "total_follow_ups": 0,
        "max_total_follow_ups": 12,
    }
    state.update(overrides)
    return state


# ---------------- B1 + B2：会话级追问预算 ----------------

def test_b2_default_budget_is_configured():
    assert settings.max_total_follow_ups == 12


def test_b2_follow_up_allowed_below_budget():
    assert route_after_assessment(_state(total_follow_ups=3)) == "ask_follow_up"


@pytest.mark.parametrize("used", [12, 13])
def test_b2_follow_up_blocked_at_budget(used):
    """达到会话级预算后不再追问，直接换下一题（把最坏成本压回预算内）。"""

    assert route_after_assessment(_state(total_follow_ups=used)) == "advance"


def test_b2_budget_exhausted_at_last_question_goes_to_evaluate():
    state = _state(total_follow_ups=12, current_question_index=1)
    assert route_after_assessment(state) == "evaluate"


def test_b2_ask_follow_up_increments_total(monkeypatch):
    def _stub(messages, temperature=0.7, max_tokens=None):
        return "那你能再具体说说吗？", dict(USAGE)

    monkeypatch.setattr(ask_mod, "chat_with_usage", _stub)
    state = {
        "assessments": [{"question_id": 1, "score": 4.0, "hint": "", "follow_up_reason": "不够具体",
                         "is_relevant": True, "covered_aspects": []}],
        "question_plan": [{"id": 1, "depth_level": "application", "skills": ["x"]}],
        "current_question_index": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "total_follow_ups": 5,
        "difficulty": "medium",
        "conversation_history": [],
        "role_profile": {"rubric": RUBRIC},
    }
    result = ask_mod.ask_follow_up(state)
    assert result["total_follow_ups"] == 6


# ---------------- B3：评分失败的第二次补救 ----------------

def _assess_state() -> dict:
    return {
        "question_plan": [{"id": 1, "category": "scenario", "skills": ["x"], "depth_level": "application"}],
        "current_question": "q",
        "current_answer": "回答",
        "current_question_index": 0,
        "global_question_counter": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "conversation_history": [],
        "resume_profile": {},
        "role_profile": {"rubric": RUBRIC},
        "assessments": [],
        "usage_records": [],
    }


def test_b3_fallback_recovers_score(monkeypatch):
    """首次没给档位 → 兜底要一个总体档位（4 档）→ 该题不再丢分。"""

    calls = {"n": 0}

    def _stub(messages, temperature=0.7, max_tokens=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"is_relevant": True, "candidate_intent": "answer"}), dict(USAGE)
        return json.dumps({"overall_level": 4}), dict(USAGE)

    monkeypatch.setattr(assess_mod, "chat_with_usage", _stub)
    assessment = assess_mod.assess(_assess_state())["assessments"][0]
    assert assessment["score"] == LEVEL_TO_SCORE[4] == 8.0
    assert assessment["score_fallback"] == "single_overall_level"
    assert assessment.get("score_error") is None


def test_b3_fallback_failure_still_marks_error(monkeypatch):
    monkeypatch.setattr(
        assess_mod, "chat_with_usage", lambda *a, **k: ("不是 JSON", dict(USAGE))
    )
    assessment = assess_mod.assess(_assess_state())["assessments"][0]
    assert assessment["score"] is None
    assert assessment["score_error"] is True
    assert assessment["score_error_status"] == "no_dimension_levels"


# ---------------- A1：岗位名修正 ----------------

def test_a1_route_is_registered():
    # 注意：新版 FastAPI 的 app.routes 里，include 进来的路由是 _IncludedRouter，
    # 不直接暴露 .path，因此这里在 router 层面断言。
    from api.routers import sessions as sessions_router

    paths = {getattr(route, "path", "") for route in sessions_router.router.routes}
    assert "/api/sessions/{session_id}/job-title" in paths
    # 顺带确认它确实是 PATCH 方法
    methods = {
        getattr(route, "path", ""): getattr(route, "methods", set())
        for route in sessions_router.router.routes
    }
    assert "PATCH" in methods["/api/sessions/{session_id}/job-title"]


def test_a1_request_schema():
    from api.schemas import UpdateJobTitleRequest

    body = UpdateJobTitleRequest(job_title="医疗病案管理专员")
    assert body.job_title == "医疗病案管理专员" and body.regenerate is True
