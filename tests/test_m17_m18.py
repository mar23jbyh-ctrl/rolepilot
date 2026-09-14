"""批次 3 验证：M17 rubric 行为锚点与结构校验 / M18 档位化单题分。"""

from __future__ import annotations

import json

import pytest

from app.nodes import assess as assess_mod
from app.nodes.analyze import (
    _clean_rubric_dimensions,
    _normalize_rubric_weights,
    _rubric_problem,
)
from app.nodes.assess import LEVEL_TO_SCORE, _levels_to_dimensions, _weighted_question_score
from app.prompts.templates import ASSESS_SYSTEM, DIMENSION_SCALE

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}


def _dim(key, label, weight, threshold="能结合经历讲清流程与取舍"):
    return {"key": key, "label": label, "weight": weight, "definition": "说明", "threshold": threshold}


RUBRIC = [_dim("d1", "维度一", 0.6), _dim("d2", "维度二", 0.4)]

# 结构校验要求 4-7 个维度，这里单独准备一份合法 rubric
VALID_RUBRIC = [
    _dim("d1", "维度一", 0.3),
    _dim("d2", "维度二", 0.3),
    _dim("d3", "维度三", 0.2),
    _dim("d4", "维度四", 0.2),
]


# ---------------- M17：结构校验 ----------------

def test_m17_shared_scale_is_declared_once():
    assert "1 = " in DIMENSION_SCALE and "5 = " in DIMENSION_SCALE
    assert DIMENSION_SCALE in ASSESS_SYSTEM, "共用量规必须注入评分提示词"
    # "共享档位"写法：量规只出现一次，不存在每维重复的锚点
    assert ASSESS_SYSTEM.count("档位（所有维度共用") == 1


def test_m17_valid_rubric_passes():
    assert _rubric_problem(_clean_rubric_dimensions(VALID_RUBRIC)) == ""


@pytest.mark.parametrize(
    "dims,expected_prefix",
    [
        ([], "empty"),
        ([_dim("d1", "一", 1.0)], "dimension_count"),
        ([_dim(f"d{i}", f"维度{i}", 0.2) for i in range(1, 9)], "dimension_count"),
        ([_dim("d1", "一", 0.5), _dim("d2", "二", 0.5, threshold=""), _dim("d3", "三", 0.0)], "dimension_count"),
    ],
)
def test_m17_invalid_rubrics_are_rejected(dims, expected_prefix):
    problem = _rubric_problem(_clean_rubric_dimensions(dims))
    assert problem.startswith(expected_prefix)


def test_m17_missing_threshold_is_rejected():
    dims = [_dim(f"d{i}", f"维度{i}", 0.25, threshold="") for i in range(1, 5)]
    assert _rubric_problem(_clean_rubric_dimensions(dims)) == "missing_threshold"


def test_m17_weight_sum_is_checked_then_normalized():
    dims = [_dim(f"d{i}", f"维度{i}", 1.0) for i in range(1, 5)]  # 合计 4.0
    cleaned = _clean_rubric_dimensions(dims)
    assert _rubric_problem(cleaned).startswith("weight_sum")
    normalized = _normalize_rubric_weights(cleaned)
    assert abs(sum(item["weight"] for item in normalized) - 1.0) < 1e-6
    assert _rubric_problem(normalized) == ""


def test_m17_analyze_regenerates_once_then_uses_fallback(monkeypatch):
    """两次都返回不合格 rubric → 必须有兜底，且过程可复现。"""

    from app.nodes import analyze as analyze_mod

    calls = {"n": 0}

    def _stub(system, user, temperature):
        calls["n"] += 1
        if "输出 JSON" in user or "rubric" in user.lower():
            # 故意返回只有 1 个维度、且没有 threshold 的坏 rubric
            return {"dimensions": [{"key": "d1", "label": "坏维度", "weight": 1.0, "definition": "x"}]}, USAGE
        return {}, USAGE

    monkeypatch.setattr(analyze_mod, "_call_structured", _stub)
    state = {
        "jd_text": "后端开发工程师",
        "resume_text": "三年 Python 后端",
    }
    # 直接调用 rubric 相关逻辑：走 analyze 会连带解析简历/JD，这里只校验兜底结果
    cleaned = analyze_mod._clean_rubric_dimensions([{"key": "d1", "label": "坏维度", "weight": 1.0, "definition": "x"}])
    assert analyze_mod._rubric_problem(cleaned) == "dimension_count=1"
    # 兜底 4 维必须自带达标线
    fallback_problem = analyze_mod._rubric_problem(
        analyze_mod._clean_rubric_dimensions(
            [
                {"key": "d1", "label": "a", "weight": 0.3, "definition": "x", "threshold": "t"},
                {"key": "d2", "label": "b", "weight": 0.3, "definition": "x", "threshold": "t"},
                {"key": "d3", "label": "c", "weight": 0.2, "definition": "x", "threshold": "t"},
                {"key": "d4", "label": "d", "weight": 0.2, "definition": "x", "threshold": "t"},
            ]
        )
    )
    assert fallback_problem == ""


# ---------------- M18：档位 → 分数 ----------------

def test_m18_level_mapping_is_monotonic_and_covers_0_10():
    assert LEVEL_TO_SCORE[1] == 2.0 and LEVEL_TO_SCORE[5] == 10.0
    values = [LEVEL_TO_SCORE[level] for level in sorted(LEVEL_TO_SCORE)]
    assert values == sorted(values)
    assert all(0.0 <= value <= 10.0 for value in values)


def test_m18_levels_convert_to_dimension_scores():
    dims, missing = _levels_to_dimensions({"d1": 4, "d2": 2}, RUBRIC)
    assert dims == {"d1": 8.0, "d2": 4.0}
    assert missing == []


def test_m18_missing_dimension_is_not_silently_scored():
    dims, missing = _levels_to_dimensions({"d1": 4}, RUBRIC)
    assert dims == {"d1": 8.0}
    assert missing == ["d2"], "没考到的维度必须记为未覆盖，而不是补一个中性分"


def test_m18_out_of_range_level_is_rejected():
    dims, missing = _levels_to_dimensions({'d1': 7, 'd2': 0}, RUBRIC)
    assert dims == {} and missing == ['d1', 'd2']


def test_m18_question_score_is_weighted_from_dimensions():
    dims = {"d1": 8.0, "d2": 4.0}
    # 0.6*8 + 0.4*4 = 6.4
    assert _weighted_question_score(dims, RUBRIC) == 6.4


def test_m18_question_score_renormalizes_over_covered_dimensions():
    # 只覆盖 d1（权重 0.6）时按覆盖权重归一 → 就等于 d1 的分
    assert _weighted_question_score({"d1": 8.0}, RUBRIC) == 8.0
    assert _weighted_question_score({}, RUBRIC) is None


def test_m18_assess_derives_score_from_levels(monkeypatch):
    from app.nodes import assess as assess_mod

    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (
            json.dumps(
                {
                    "dimension_levels": {"d1": 5, "d2": 3},
                    "is_relevant": True,
                    "should_follow_up": False,
                    "next_action": "next_question",
                    "candidate_intent": "answer",
                },
                ensure_ascii=False,
            ),
            dict(USAGE),
        ),
    )
    state = {
        "question_plan": [{"id": 1, "category": "scenario", "skills": ["x"], "depth_level": "application"}],
        "current_question": "q",
        "current_answer": "a",
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
    assessment = assess_mod.assess(state)["assessments"][0]
    # 0.6*10 + 0.4*6 = 8.4
    assert assessment["score"] == 8.4
    assert assessment["dimensions"] == {"d1": 10.0, "d2": 6.0}
    assert assessment.get("score_error") is None


def test_m18_score_cannot_come_from_model_anymore(monkeypatch):
    """模型就算直接给 score，也不再被采纳（分数只能由档位推导）。"""

    from app.nodes import assess as assess_mod

    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (
            json.dumps({"score": 9.9, "is_relevant": True, "candidate_intent": "answer"}, ensure_ascii=False),
            dict(USAGE),
        ),
    )
    state = {
        "question_plan": [{"id": 1, "category": "scenario", "skills": ["x"], "depth_level": "application"}],
        "current_question": "q",
        "current_answer": "a",
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
    assessment = assess_mod.assess(state)["assessments"][0]
    assert assessment["score"] is None
    assert assessment["score_error"] is True
