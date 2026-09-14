"""D12 rubric selection regressions; synthetic fixtures, no DB/network calls."""

import json
import ast
import inspect
from copy import deepcopy

import pytest

from app.nodes import assess as assess_mod

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
PROFESSIONAL_RUBRIC = [
    {"key": "d_sql", "label": "合成数据查询", "weight": 0.75},
    {"key": "d_analysis", "label": "合成业务分析", "weight": 0.25},
]


@pytest.fixture(autouse=True)
def forbid_real_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("D12 tests must not invoke a real model or resume validation")

    monkeypatch.setattr(assess_mod, "chat_with_usage", forbidden)
    monkeypatch.setattr(assess_mod, "validate_resume_claims", lambda **kwargs: {})


def _state(kind="professional", **metadata):
    return {
        "question_plan": [{"id": 1, "category": "scenario", "depth_level": "application", "question_kind": kind, **metadata}],
        "current_question": "请结合合成经历说明你为什么选择这个岗位？",
        "current_answer": "在合成校园业务分析项目中，我用 SQL 验证结论，希望继续做业务分析并补齐行业知识。",
        "current_question_index": 0,
        "global_question_counter": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "role_profile": {"job_title": "合成业务分析员", "rubric": deepcopy(PROFESSIONAL_RUBRIC)},
        "resume_profile": {}, "conversation_history": [], "assessments": [],
    }


def _rubric_from_messages(messages):
    content = messages[1].content
    marker = "\n[{"
    start = content.index(marker) + 1
    rubric, _ = json.JSONDecoder().raw_decode(content[start:])
    return rubric


def _stub_sequence(monkeypatch, replies):
    pending, captured = iter(replies), []

    def fake(messages, **kwargs):
        captured.append(messages)
        reply = next(pending)
        if isinstance(reply, Exception):
            raise reply
        return (json.dumps(reply, ensure_ascii=False) if isinstance(reply, dict) else reply), dict(USAGE)

    monkeypatch.setattr(assess_mod, "chat_with_usage", fake)
    return captured


def _record(state):
    return assess_mod.assess(state)["assessments"][0]


def test_motivation_prompt_contract_drives_correct_dimension_keys(monkeypatch):
    captured = []

    def obey_prompt(messages, **kwargs):
        rubric = _rubric_from_messages(messages)
        captured.append(rubric)
        levels = {item["key"]: level for item, level in zip(rubric, (4, 3))}
        return json.dumps({"dimension_levels": levels, "candidate_intent": "answer"}), dict(USAGE)

    monkeypatch.setattr(assess_mod, "chat_with_usage", obey_prompt)
    assessment = _record(_state("motivation"))
    assert captured == [assess_mod.MOTIVATION_RUBRIC]
    assert assessment["dimensions"] == {"m1": 8, "m2": 6}
    assert assessment["score"] == 7 and not assessment.get("score_error")
    assert not assessment.get("score_fallback")


def test_professional_prompt_and_score_still_use_role_weights(monkeypatch):
    captured = _stub_sequence(monkeypatch, [{"dimension_levels": {"d_sql": 4, "d_analysis": 3, "m1": 5}}])
    assessment = _record(_state())
    assert _rubric_from_messages(captured[0]) == PROFESSIONAL_RUBRIC
    assert assessment["dimensions"] == {"d_sql": 8, "d_analysis": 6}
    assert assessment["score"] == 7.5 and assessment["question_kind"] == "professional"
    assert len(captured) == 1


@pytest.mark.parametrize("metadata,expected_kind", [
    ({"question_kind": "", "question_type": "behavioral", "focus_key": ""}, "motivation"),
    ({"question_kind": "", "question_type": "behavioral", "focus_key": "d_sql"}, "professional"),
    ({"question_kind": "", "question_type": "behavioral", "jd_ref": "0"}, "professional"),
])
def test_legacy_kind_detection_selects_rubric_before_prompt(monkeypatch, metadata, expected_kind):
    expected = assess_mod.MOTIVATION_RUBRIC if expected_kind == "motivation" else PROFESSIONAL_RUBRIC
    levels = {item["key"]: 4 for item in expected}
    captured = _stub_sequence(monkeypatch, [{"dimension_levels": levels}])
    assessment = _record(_state(**metadata))
    assert _rubric_from_messages(captured[0]) == expected
    assert assessment["question_kind"] == expected_kind and assessment["score"] == 8


@pytest.mark.parametrize("kind", ["professional", "motivation"])
def test_fallback_prompt_uses_exact_same_rubric_and_is_labeled(monkeypatch, kind):
    captured = _stub_sequence(monkeypatch, [{"dimension_levels": {}}, {"overall_level": 4}])
    state = _state(kind)
    original = deepcopy(state)
    assessment = _record(state)
    expected = assess_mod.MOTIVATION_RUBRIC if kind == "motivation" else PROFESSIONAL_RUBRIC
    assert _rubric_from_messages(captured[0]) == expected
    assert _rubric_from_messages(captured[1]) == expected
    assert set(assessment["dimensions"]) == {item["key"] for item in expected}
    assert assessment["score"] == 8 and not assessment.get("score_error")
    assert assessment["score_fallback"] == "single_overall_level"
    assert assessment["dimensions_not_covered"] == [item["key"] for item in expected]
    assert state == original


@pytest.mark.parametrize("kind,key,missing", [("professional", "d_sql", "d_analysis"), ("motivation", "m1", "m2")])
def test_partially_missing_dimensions_are_not_filled_or_sent_to_fallback(monkeypatch, kind, key, missing):
    captured = _stub_sequence(monkeypatch, [{"dimension_levels": {key: 3}}])
    assessment = _record(_state(kind))
    assert assessment["dimensions"] == {key: 6} and assessment["score"] == 6
    assert assessment["dimensions_not_covered"] == [missing]
    assert not assessment.get("score_fallback") and len(captured) == 1


@pytest.mark.parametrize("kind,wrong_key", [("professional", "m1"), ("motivation", "d_sql")])
def test_wrong_rubric_keys_and_model_score_do_not_create_fake_scores(monkeypatch, kind, wrong_key):
    captured = _stub_sequence(monkeypatch, [{"dimension_levels": {wrong_key: 5}, "score": 10}, {"overall_level": "invalid"}])
    assessment = _record(_state(kind))
    assert assessment["dimensions"] == {} and assessment["score"] is None
    assert assessment["score_error"] and assessment["next_action"] == "next_question"
    assert not assessment.get("score_fallback") and len(captured) == 2


@pytest.mark.parametrize("raw", ["not JSON", "[]", "null"])
def test_invalid_main_and_fallback_outputs_have_bounded_calls_and_no_score(monkeypatch, raw):
    captured = _stub_sequence(monkeypatch, [raw, raw, raw])
    result = assess_mod.assess(_state("motivation"))
    assessment = result["assessments"][0]
    assert assessment["score"] is None and assessment["dimensions"] == {}
    assert assessment["score_error_status"] == "no_dimension_levels"
    assert assessment["dimensions_not_covered"] == ["m1", "m2"]
    assert len(captured) == 3
    assert [usage["total_tokens"] for usage in result["usage_records"]] == [30, 15]


@pytest.mark.parametrize("level", [None, True, {}, "not a level", float("inf")])
def test_invalid_fallback_level_keeps_dimensions_missing(monkeypatch, level):
    _stub_sequence(monkeypatch, [{"dimension_levels": {}}, {"overall_level": level}])
    assessment = _record(_state("motivation"))
    assert assessment["score"] is None and assessment["dimensions"] == {}
    assert assessment["score_error"] and not assessment.get("score_fallback")


def test_fallback_exception_preserves_no_score_semantics(monkeypatch):
    _stub_sequence(monkeypatch, [{}, RuntimeError("synthetic provider error")])
    assessment = _record(_state("motivation"))
    assert assessment["score"] is None and assessment["score_error"]


def test_fallback_legacy_four_argument_signature_still_uses_role_rubric(monkeypatch):
    captured = _stub_sequence(monkeypatch, [{"overall_level": 3}])
    dimensions, usage = assess_mod._fallback_overall_level(_state(), "q", "a", [])
    assert dimensions == {"d_sql": 6, "d_analysis": 6}
    assert _rubric_from_messages(captured[0]) == PROFESSIONAL_RUBRIC
    assert usage["node"] == "assess_fallback"


def test_explicit_empty_fallback_rubric_does_not_revert_to_role_rubric(monkeypatch):
    _stub_sequence(monkeypatch, [{"overall_level": 4}])
    dimensions, _ = assess_mod._fallback_overall_level(_state(), "q", "a", [], rubric=[])
    assert dimensions == {}


def test_request_explanation_still_bypasses_scoring_and_fallback(monkeypatch):
    captured = _stub_sequence(monkeypatch, [{"candidate_intent": "request_explanation"}])
    result = assess_mod.assess(_state("motivation"))
    assert result["explanation_pending"] and "assessments" not in result
    assert len(captured) == 1


def test_fallback_has_no_dead_top_level_statements_after_return():
    function = ast.parse(inspect.getsource(assess_mod._fallback_overall_level)).body[0]
    returns = [index for index, statement in enumerate(function.body) if isinstance(statement, ast.Return)]
    assert returns == [len(function.body) - 1]


def test_unattributed_rate_claims_are_removed_from_assessment_comments():
    source = inspect.getsource(assess_mod)
    assert "估算重试率" not in source
    assert "1/201" not in source
