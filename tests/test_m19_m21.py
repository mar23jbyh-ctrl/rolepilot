"""批次 4 验证：M19 动机题独立量规 / M21 难度参与加权。"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.nodes import assess as assess_mod
from app.nodes import evaluate as eval_mod
from app.nodes.assess import LEVEL_TO_SCORE, MOTIVATION_RUBRIC

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
_ALL_KEYS = ("d1", "job1", "m1", "m2")


def _state(meta: dict, monkeypatch=None) -> dict:
    return {
        "question_plan": [{"id": 1, "category": "scenario", "skills": ["x"], "depth_level": "application", **meta}],
        "current_question": "q",
        "current_answer": "a",
        "current_question_index": 0,
        "global_question_counter": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "conversation_history": [],
        "resume_profile": {},
        "role_profile": {"rubric": [{"key": "d1", "label": "专业维度", "weight": 1.0, "definition": "x", "threshold": "t"}]},
        "assessments": [],
        "usage_records": [],
    }


def _levels(monkeypatch, level=4):
    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (
            json.dumps(
                {"dimension_levels": {k: level for k in _ALL_KEYS}, "is_relevant": True, "candidate_intent": "answer"},
                ensure_ascii=False,
            ),
            dict(USAGE),
        ),
    )


def test_m19_motivation_rubric_has_two_dimensions():
    assert [d["key"] for d in MOTIVATION_RUBRIC] == ["m1", "m2"]
    assert abs(sum(d["weight"] for d in MOTIVATION_RUBRIC) - 1.0) < 1e-9


def test_m19_motivation_question_uses_own_rubric(monkeypatch):
    _levels(monkeypatch, level=4)
    state = _state({"question_kind": "motivation", "focus_key": "", "question_type": "behavioral"})
    assessment = assess_mod.assess(state)["assessments"][0]
    assert set(assessment["dimensions"]) == {"m1", "m2"}, "动机题不得用专业维度评分"
    assert assessment["question_kind"] == "motivation"
    assert assessment["score"] == LEVEL_TO_SCORE[4]


def test_m19_professional_question_uses_role_rubric(monkeypatch):
    _levels(monkeypatch, level=4)
    state = _state({"question_kind": "professional", "focus_key": "f1", "question_type": "foundation"})
    assessment = assess_mod.assess(state)["assessments"][0]
    assert set(assessment["dimensions"]) == {"d1"}


def test_m19_motivation_dimensions_appear_in_report(monkeypatch):
    monkeypatch.setattr(
        eval_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"overall_score": 8}), dict(USAGE))
    )
    state = {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 8.0, "is_follow_up": False,
             "dimensions": {"m1": 8.0, "m2": 8.0}, "question_kind": "motivation", "difficulty": "easy"}
        ],
        "question_plan": [{"id": 1}],
        "plan_question_count": 1,
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": [{"key": "d1", "label": "专业维度", "weight": 1.0}]},
        "summaries": [], "turn_records": [], "usage_records": [], "difficulty_events": [],
    }
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert set(report["per_question"][0]["dimensions"]) == {"m1", "m2"}


# ---------------- M21：难度加权 ----------------

def test_m21_hard_question_weighs_more(monkeypatch):
    monkeypatch.setattr(
        eval_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"overall_score": 6}), dict(USAGE))
    )
    state = {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 10.0, "is_follow_up": False,
             "dimensions": {}, "difficulty": "hard"},
            {"question_id": 2, "plan_question_index": 1, "score": 0.0, "is_follow_up": False,
             "dimensions": {}, "difficulty": "easy"},
        ],
        "question_plan": [{"id": 1}, {"id": 2}],
        "plan_question_count": 2,
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": []},
        "summaries": [], "turn_records": [], "usage_records": [], "difficulty_events": [],
    }
    # 1.2*10 + 0.8*0 = 12 / 2.0 = 6.0；不加权则是 (10+0)/2 = 5.0
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["overall_score"] == 6.0


def test_m21_weighting_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "difficulty_weighting", False)
    monkeypatch.setattr(
        eval_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"overall_score": 6}), dict(USAGE))
    )
    state = {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 10.0, "is_follow_up": False,
             "dimensions": {}, "difficulty": "hard"},
            {"question_id": 2, "plan_question_index": 1, "score": 0.0, "is_follow_up": False,
             "dimensions": {}, "difficulty": "easy"},
        ],
        "question_plan": [{"id": 1}, {"id": 2}],
        "plan_question_count": 2,
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": []},
        "summaries": [], "turn_records": [], "usage_records": [], "difficulty_events": [],
    }
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["overall_score"] == 5.0
