"""B5（M12）的验证用例：讲解后补同维度替代题 + 有效样本数标注。"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pytest

from app.nodes import ask as ask_mod
from app.nodes import evaluate as eval_mod
from app.nodes.ask import MAX_REPLACEMENTS_PER_SESSION

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _plan(n: int) -> list[dict]:
    return [
        {
            "id": i + 1,
            "category": "scenario",
            "difficulty": "medium",
            "depth_level": "application",
            "focus_key": f"f{(i % 2) + 1}",
            "project_ref": "",
            "jd_ref": "",
            "skills": [f"技能{i}"],
            "content": f"第{i + 1}题",
        }
        for i in range(n)
    ]


def _explain_state(**overrides) -> dict:
    state = {
        "question_plan": _plan(3),
        "plan_question_count": 3,
        "current_question_index": 0,
        "current_question": "第1题",
        "difficulty": "medium",
        "conversation_history": [
            {"role": "assistant", "content": "第1题"},
            {"role": "user", "content": "请讲解一下这道题涉及的知识点"},
        ],
        "role_profile": {"domain": "技术-AI应用", "industry": "互联网", "industry_known": True},
        "skipped_questions": [
            {
                "question_index": 0,
                "focus_key": "f1",
                "skills": ["技能0"],
                "category": "scenario",
                "depth_level": "application",
                "reason": "request_explanation",
                "replacement_index": None,
            }
        ],
        "replacement_count": 0,
        "assessments": [],
        "summaries": [],
        "turn_records": [],
        "usage_records": [],
    }
    state.update(overrides)
    return state


def _stub_two_calls(monkeypatch, replacement: dict | None):
    """第一次调用给讲解文本，第二次（出题）给替代题 JSON。"""

    def _stub(messages, temperature=0.7, max_tokens=None):
        text = " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))
            for m in messages
        )
        if "现场出一道面试题" in text:
            if replacement is None:
                return "{}", dict(USAGE)
            return json.dumps(replacement, ensure_ascii=False), dict(USAGE)
        return "这道题考察的是检索链路的召回与重排，通常先做切分再混合检索。", dict(USAGE)

    monkeypatch.setattr(ask_mod, "chat_with_usage", _stub)


# ---------------- 1. 讲解后补同维度替代题 ----------------

def test_b5_explanation_adds_same_dimension_replacement(monkeypatch):
    _stub_two_calls(
        monkeypatch,
        {"content": "那换成另一个场景：如果召回质量差，你会先调哪一环？", "skills": ["技能0"]},
    )
    result = ask_mod.ask_explain(_explain_state())
    plan = result["question_plan"]
    assert len(plan) == 4, "应追加一道替代题"
    replacement = plan[-1]
    assert replacement["focus_key"] == "f1", "替代题必须与被跳过题目同维度"
    assert replacement["replacement_for"] == 0
    assert replacement["content"].startswith("那换成另一个场景")
    assert result["skipped_questions"][-1]["replacement_index"] == 3
    assert result["replacement_count"] == 1
    assert any(r["node"] == "ask_replacement" for r in result["usage_records"])


def test_b5_replacement_failure_is_recorded(monkeypatch):
    _stub_two_calls(monkeypatch, None)
    result = ask_mod.ask_explain(_explain_state())
    assert "question_plan" not in result
    assert result["skipped_questions"][-1]["replacement_skipped_reason"] == (
        "generation_failed_or_off_domain"
    )


def test_b5_replacement_cap_is_enforced(monkeypatch):
    _stub_two_calls(monkeypatch, {"content": "替代题", "skills": ["技能0"]})
    state = _explain_state(replacement_count=MAX_REPLACEMENTS_PER_SESSION)
    result = ask_mod.ask_explain(state)
    assert "question_plan" not in result
    assert result["skipped_questions"][-1]["replacement_skipped_reason"] == "replacement_cap_reached"


def test_b5_off_domain_replacement_is_rejected(monkeypatch):
    # 岗位政策里显式禁止的话题出现在替代题里 → 必须拒绝
    _stub_two_calls(monkeypatch, {"content": "请讲讲临床护理的静脉输液操作要点", "skills": ["技能0"]})
    state = _explain_state(
        role_profile={
            "domain": "技术-AI应用",
            "industry": "互联网",
            "industry_known": True,
            "out_of_scope_topics": ["临床护理操作"],
            "forbidden_topics": ["临床护理操作"],
        }
    )
    result = ask_mod.ask_explain(state)
    assert "question_plan" not in result, "命中禁止话题的替代题不得进入题单"
    assert result["skipped_questions"][-1]["replacement_skipped_reason"] == (
        "generation_failed_or_off_domain"
    )


# ---------------- 2 & 3. 有效样本数标注 ----------------

def _report_state(**overrides) -> dict:
    state = {
        "assessments": [
            {"question_id": 1, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
            {"question_id": 2, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
        ],
        "question_plan": _plan(3),
        "plan_question_count": 3,
        "skipped_questions": [
            {"question_index": 2, "reason": "request_explanation", "replacement_index": None}
        ],
        "conversation_history": [{"role": "assistant", "content": "问题"}],
        "role_profile": {"rubric": [{"key": "d1", "label": "岗位基础", "weight": 1.0}]},
        "summaries": [],
        "turn_records": [],
        "usage_records": [],
        "difficulty_events": [],
    }
    state.update(overrides)
    return state


def _stub_eval(monkeypatch, payload: dict | None = None):
    monkeypatch.setattr(
        eval_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps(payload or {"overall_score": 6.0}), dict(USAGE)),
    )


def test_b5_missing_replacement_triggers_note(monkeypatch):
    _stub_eval(monkeypatch)
    report = eval_mod.evaluate(_report_state())["evaluation_report"]
    assert report["effective_sample_count"] == 2
    assert report["plan_question_count"] == 3
    assert report["sample_note"] == "有效样本数 2 / 题单题数 3"
    assert report["skipped_questions"], "被跳过的题目必须留痕"


def test_b5_compensated_skip_has_no_note(monkeypatch):
    _stub_eval(monkeypatch)
    state = _report_state(
        assessments=[
            {"question_id": 1, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
            {"question_id": 2, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
            {"question_id": 3, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
        ],
        question_plan=_plan(3) + [{"id": 4, "content": "替代题", "focus_key": "f1", "replacement_for": 2}],
        skipped_questions=[
            {"question_index": 2, "reason": "request_explanation", "replacement_index": 3}
        ],
    )
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["effective_sample_count"] == 3
    assert report["plan_question_count"] == 3, "替代题不改变题单题数"
    assert report["sample_note"] == ""


def test_b5_follow_ups_do_not_count_as_samples(monkeypatch):
    _stub_eval(monkeypatch)
    state = _report_state(
        assessments=[
            {"question_id": 1, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
            {"question_id": 2, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": True},
            {"question_id": 3, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": True},
        ],
    )
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["effective_sample_count"] == 1, "追问不算有效样本"


def test_b5_zero_assessment_branch_also_notes(monkeypatch):
    _stub_eval(monkeypatch)
    state = _report_state(assessments=[])
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["effective_sample_count"] == 0
    assert report["sample_note"] == "有效样本数 0 / 题单题数 3"


# ---------------- 3. 历史回放：不再出现"无标注的样本缩水" ----------------

def _load_report_states() -> list[dict]:
    states: list[dict] = []
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
            if state.get("evaluation_report"):
                states.append(state)
    return states


def test_b5_replay_no_unannotated_shortfall(monkeypatch):
    states = _load_report_states()
    if not states:
        pytest.skip("本地没有历史会话库（data/sessions*.db）")

    _stub_eval(monkeypatch)
    shortfall = 0
    for state in states:
        report = eval_mod.evaluate(state)["evaluation_report"]
        if report["effective_sample_count"] < report["plan_question_count"]:
            shortfall += 1
            assert report["sample_note"], "样本缩水必须带标注"
            assert report["sample_note"].startswith("有效样本数 ")
        else:
            assert report["sample_note"] == ""
    assert shortfall > 0, "历史样本里应存在样本缩水的会话，否则回归无意义"
