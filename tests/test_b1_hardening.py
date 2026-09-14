"""B1 加固批的验证用例（M1 / M5 / M6 / M11）。

对应 docs/optimization_plan.md §6.1 的 B1 验证方式：
- M1：提示词含 candidate_intent；评分调用次数不增加
- M5：注入脏数据 → 不崩、不写假分、score_errors 计数、报告可生成
- M6：截断时重试 1 次且传了 max_tokens
- M11：早停可达且门槛随 target 变化，只数主问题
"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.graph.edges import (
    early_stop_threshold,
    recent_scores,
    route_after_assessment,
)
from app.prompts.templates import ASSESS_PROMPT, ASSESS_SYSTEM

USAGE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150,
    "finish_reason": "stop",
}


def _state(**overrides) -> dict:
    state = {
        "question_plan": [
            {
                "id": 1,
                "category": "scenario",
                "skills": ["RAG"],
                "depth_level": "application",
                "project_ref": "",
                "jd_ref": "",
                "content": "介绍一下你的RAG项目",
            }
        ],
        "current_question": "介绍一下你的RAG项目",
        "current_answer": "我负责把检索链路从关键词改成向量召回。",
        "current_question_index": 0,
        "global_question_counter": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "difficulty": "medium",
        "conversation_history": [],
        "resume_profile": {"skills": ["RAG"], "projects": ["问答系统"]},
        "role_profile": {
            "domain": "技术-AI应用",
            "industry": "互联网",
            "industry_known": True,
            # M18：单题分由 rubric 维度的档位换算得到，测试需要一份最小 rubric
            "rubric": [{"key": "d1", "label": "工程能力", "weight": 1.0, "definition": "工程实践", "threshold": "能结合项目讲清做法"}],
        },
        "assessments": [],
        "turn_records": [],
        "summaries": [],
        "summarized_ids": [],
        "usage_records": [],
    }
    state.update(overrides)
    return state


def _valid_payload(**overrides) -> str:
    payload = {
        "question_id": 1,
        # M18：评分契约 = 模型给维档位，分数由代码算
        "dimension_levels": {"d1": 4},
        "is_relevant": True,
        "should_follow_up": False,
        "follow_up_reason": "",
        "hint": "",
        "missed_points": ["缺少量化结果"],
        "covered_points": ["讲了检索链路改造"],
        "covered_aspects": ["流程"],
        "next_action": "next_question",
        "needs_external_knowledge": False,
        "candidate_intent": "answer",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


# ---------------- M1 ----------------

def test_m1_prompt_declares_candidate_intent():
    assert "candidate_intent" in ASSESS_SYSTEM
    assert "candidate_intent" in ASSESS_PROMPT
    for token in ("answer", "request_explanation", "request_stop", "off_topic"):
        assert token in ASSESS_SYSTEM


def test_m1_assess_keeps_single_llm_call(monkeypatch):
    from app.nodes import assess as assess_mod

    calls = []

    def _stub(messages, temperature=0.7, max_tokens=None):
        calls.append(max_tokens)
        return _valid_payload(), dict(USAGE)

    monkeypatch.setattr(assess_mod, "chat_with_usage", _stub)
    result = assess_mod.assess(_state())
    assert len(calls) == 1, "M1 不得引入额外的 LLM 调用"
    assert result["assessments"][0]["candidate_intent"] == "answer"


def test_m1_intent_whitelist_falls_back_to_answer(monkeypatch):
    from app.nodes import assess as assess_mod

    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (_valid_payload(candidate_intent="随便写点什么"), dict(USAGE)),
    )
    result = assess_mod.assess(_state())
    assert result["assessments"][0]["candidate_intent"] == "answer"


# ---------------- M5 ----------------

def test_m5_unparseable_reply_gets_no_fake_score(monkeypatch):
    from app.nodes import assess as assess_mod

    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: ("这不是 JSON，只是一段解释文字", dict(USAGE)),
    )
    result = assess_mod.assess(_state())
    assessment = result["assessments"][0]
    assert assessment["score"] is None, "解析失败不得回退成 5.0"
    assert assessment["score_error"] is True
    # M18：分数改由维度档位推导，档位缺失即视为拿不到分数
    assert assessment["score_error_status"] == "no_dimension_levels"
    # 分数拿不到时不再追问同一题
    assert assessment["next_action"] == "next_question"


@pytest.mark.parametrize(
    "bad_level",
    [
        "比较一般",
        None,
        {"value": 7},
        True,
        float("inf"),
    ],
)
def test_m5_dirty_level_values_are_rejected(monkeypatch, bad_level):
    """M5 + M18：档位不是合法数字时，不得产出任何分数（宁可无记录分值）。"""

    from app.nodes import assess as assess_mod

    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (_valid_payload(dimension_levels={"d1": bad_level}), dict(USAGE)),
    )
    assessment = assess_mod.assess(_state())["assessments"][0]
    assert assessment["score"] is None
    assert assessment["score_error"] is True
    assert assessment["score_error_status"] == "no_dimension_levels"


def test_m5_out_of_range_level_is_rejected(monkeypatch):
    """Invalid levels must not be upgraded into high scores."""

    from app.nodes import assess as assess_mod

    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (_valid_payload(dimension_levels={"d1": 7}), dict(USAGE)),
    )
    assessment = assess_mod.assess(_state())["assessments"][0]
    assert assessment['score'] is None and assessment['dimensions'] == {}
    assert assessment['score_error'] is True


def test_m5_recent_scores_skips_failed_records():
    result = recent_scores([{"score": None}, {"score": 8.0}, {"score": "6"}], 4)
    assert result == [8.0, 6.0], "失败的评分不能被当成 0 分参与难度/早停判断"


def test_m5_evaluate_excludes_score_errors_from_total(monkeypatch):
    from app.nodes import evaluate as eval_mod

    monkeypatch.setattr(
        eval_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps({"overall_score": 9.9}), dict(USAGE)),
    )
    assessments = [
        {"question_id": 1, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
        {"question_id": 2, "score": 4.0, "dimensions": {"d1": 4.0}, "is_follow_up": False},
        {
            "question_id": 3,
            "score": None,
            "score_error": True,
            "score_error_status": "invalid",
            "dimensions": {"d1": 5.0},
            "is_follow_up": False,
        },
    ]
    state = _state(
        assessments=assessments,
        role_profile={"rubric": [{"key": "d1", "label": "岗位基础", "weight": 1.0}]},
    )
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["score_errors"] == 1
    assert report["dimension_scores"]["岗位基础"] == 5.0, "失败记录不得拉低维度均值"
    assert report["per_question"][2]["score_error"] is True


# ---------------- M6 ----------------

def test_m6_retries_once_on_truncated_output(monkeypatch):
    from app.nodes import assess as assess_mod

    calls = []

    def _stub(messages, temperature=0.7, max_tokens=None):
        calls.append(max_tokens)
        if len(calls) == 1:
            usage = dict(USAGE)
            usage["finish_reason"] = "length"
            return '{"score": 7, "dimensi', usage
        return _valid_payload(), dict(USAGE)

    monkeypatch.setattr(assess_mod, "chat_with_usage", _stub)
    result = assess_mod.assess(_state())
    assert len(calls) == 2, "截断后必须重试 1 次"
    assert calls[0] == settings.max_output_tokens
    assert calls[1] == settings.max_output_tokens * 2, "重试应放宽 max_tokens"
    # M18：4 档 → 8.0 分
    assert result["assessments"][0]["score"] == 8.0
    # 重试的 token 必须累加进成本
    usage_record = result["usage_records"][0]
    assert usage_record["total_tokens"] == 300


def test_m6_gives_up_after_max_attempts(monkeypatch):
    from app.nodes import assess as assess_mod

    calls = []

    def _stub(messages, temperature=0.7, max_tokens=None):
        calls.append(max_tokens)
        usage = dict(USAGE)
        usage["finish_reason"] = "length"
        return '{"score": 7, "dimensi', usage

    monkeypatch.setattr(assess_mod, "chat_with_usage", _stub)
    assessment = assess_mod.assess(_state())["assessments"][0]
    # P0/B3：主调用 2 次（首次 + 1 次重试）之后，还会有 1 次"总体档位"兜底调用；
    # 由于桩始终返回不可解析内容，兜底也失败 → 仍然记 score_error。
    assert len(calls) == assess_mod.ASSESS_MAX_ATTEMPTS + 1 == 3
    assert assessment["score_error"] is True


# ---------------- M11 ----------------

def _next_question_index_state(main_count: int, index: int = 11) -> dict:
    plan = [{"id": i, "content": f"q{i}"} for i in range(1, 16)]
    assessments = [
        {"question_id": i, "score": 8.5, "is_follow_up": False, "next_action": "next_question"}
        for i in range(1, main_count + 1)
    ]
    return _state(
        question_plan=plan,
        current_question_index=index,
        assessments=assessments,
    )


def test_m11_early_stop_fires_through_the_normal_path():
    state = _next_question_index_state(main_count=12)
    assert route_after_assessment(state) == "evaluate"


def test_m11_early_stop_does_not_fire_below_threshold():
    state = _next_question_index_state(main_count=11)
    assert route_after_assessment(state) == "advance"


def test_m11_early_stop_ignores_follow_ups():
    state = _next_question_index_state(main_count=12)
    # 再挂 30 条追问：如果不排除，会把主问题数算成 42
    state["assessments"] = state["assessments"] + [
        {"question_id": 100 + i, "score": 8.5, "is_follow_up": True, "next_action": "next_question"}
        for i in range(30)
    ]
    assert route_after_assessment(state) == "evaluate"

    thin = _next_question_index_state(main_count=11)
    thin["assessments"] = thin["assessments"] + [
        {"question_id": 200 + i, "score": 8.5, "is_follow_up": True, "next_action": "next_question"}
        for i in range(30)
    ]
    assert route_after_assessment(thin) == "advance", "追问不能把主问题数凑够门槛"


def test_m11_threshold_follows_target(monkeypatch):
    monkeypatch.setattr(settings, "target_questions", 15)
    monkeypatch.setattr(settings, "min_questions", 12)
    assert early_stop_threshold() == 12  # max(12, ceil(9))

    monkeypatch.setattr(settings, "target_questions", 25)
    assert early_stop_threshold() == 15  # max(12, ceil(15))

    monkeypatch.setattr(settings, "target_questions", 5)
    assert early_stop_threshold() == 12  # 下限保护


def test_m11_no_early_stop_on_last_question():
    state = _next_question_index_state(main_count=14, index=14)
    assert route_after_assessment(state) == "evaluate"
