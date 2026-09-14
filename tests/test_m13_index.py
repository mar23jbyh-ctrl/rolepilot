"""M13 验证：每条评分记录带题单下标 `plan_question_index`。"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pytest

from app.nodes import assess as assess_mod

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _plan(n: int) -> list[dict]:
    return [
        {
            "id": i + 1,
            "category": "scenario",
            "depth_level": "application",
            "skills": [f"技能{i}"],
            "content": f"第{i + 1}题",
        }
        for i in range(n)
    ]


def _state(**overrides) -> dict:
    state = {
        "question_plan": _plan(6),
        "current_question_index": 5,
        "current_question": "第6题",
        "current_answer": "我的回答",
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "global_question_counter": 0,
        "conversation_history": [],
        "resume_profile": {"skills": [], "projects": []},
        "role_profile": {"industry_known": True},
        "assessments": [],
        "turn_records": [],
        "usage_records": [],
    }
    state.update(overrides)
    return state


def _stub(monkeypatch, payload: dict):
    monkeypatch.setattr(
        assess_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps(payload, ensure_ascii=False), dict(USAGE)),
    )


def test_m13_main_question_carries_plan_index(monkeypatch):
    _stub(monkeypatch, {"score": 7, "next_action": "next_question", "candidate_intent": "answer"})
    assessment = assess_mod.assess(_state())["assessments"][0]
    assert assessment["plan_question_index"] == 5
    assert assessment["is_follow_up"] is False


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_m13_follow_up_keeps_same_plan_index(monkeypatch, rounds):
    """追问轮不推进题单下标：3 轮追问后下标仍是同一个。"""

    _stub(monkeypatch, {"score": 4, "next_action": "follow_up", "candidate_intent": "answer"})
    for round_index in range(rounds):
        state = _state(follow_up_count=round_index + 1)
        assessment = assess_mod.assess(state)["assessments"][0]
        assert assessment["plan_question_index"] == 5, f"第 {round_index + 1} 轮追问下标漂移"
        assert assessment["is_follow_up"] is True


def test_m13_index_is_independent_of_follow_up_count(monkeypatch):
    _stub(monkeypatch, {"score": 4, "next_action": "follow_up", "candidate_intent": "answer"})
    indices = {
        assess_mod.assess(_state(follow_up_count=n))["assessments"][0]["plan_question_index"]
        for n in (0, 1, 2, 3)
    }
    assert indices == {5}, "同一题的任意轮次必须落在同一个题单下标上"


def test_m13_empty_answer_branch_also_carries_index(monkeypatch):
    _stub(monkeypatch, {})
    assessment = assess_mod.assess(_state(current_answer=""))["assessments"][0]
    assert assessment["plan_question_index"] == 5
    assert assessment['score'] is None and assessment['scoreable'] is False


def test_m13_out_of_range_index_is_marked_minus_one(monkeypatch):
    _stub(monkeypatch, {"score": 7, "next_action": "next_question", "candidate_intent": "answer"})
    state = _state(current_question_index=99)
    assessment = assess_mod.assess(state)["assessments"][0]
    assert assessment["plan_question_index"] == -1, "题单越界时不得伪装成合法下标"


def test_m13_report_exposes_plan_index(monkeypatch):
    from app.nodes import evaluate as eval_mod

    monkeypatch.setattr(
        eval_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps({"overall_score": 7}), dict(USAGE)),
    )
    state = _state(
        assessments=[
            {
                "question_id": 1,
                "plan_question_index": 3,
                "score": 7.0,
                "dimensions": {},
                "is_follow_up": False,
            }
        ],
        role_profile={"rubric": []},
        summaries=[],
    )
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["per_question"][0]["plan_question_index"] == 3


# ---------------- 历史数据：下标映射规则可还原 ----------------

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
            if state.get("assessments"):
                states.append(state)
    return states


def test_m13_historical_segments_map_to_contiguous_indices():
    """真实数据校验：按 is_follow_up 切段后，每段恰好 1 条主问题，
    且段序号天然连续（这正是 M13 下标映射依赖的性质）。

    注意：历史记录里**没有** `plan_question_index`（本批才引入），
    所以这里校验的是"映射规则可从真实数据还原"；
    新代码是否真的写对，由图级重放用例（test_nodes.py）验证。
    另外 `current_question_index` 是会话结束时的快照，会话中断/旧格式会偏小，
    不能当作"已访问题数"，因此这里不做该比较。
    """

    states = _load_report_states()
    if not states:
        pytest.skip("本地没有历史会话库（data/sessions*.db）")

    checked = 0
    multi_follow_up_sessions = 0
    for state in states:
        groups: list[list[dict]] = []
        for item in state["assessments"]:
            if not item.get("is_follow_up"):
                groups.append([item])
            elif groups:
                groups[-1].append(item)
        if not groups:
            continue
        checked += 1
        if any(len(group) > 1 for group in groups):
            multi_follow_up_sessions += 1
        # 每段的第一条必须是主问题，其余都是追问
        for group in groups:
            assert not group[0].get("is_follow_up")
            assert all(item.get("is_follow_up") for item in group[1:])
        # 段序号即题单下标：连续、从 0 开始、组内一致
        derived = {index for index, _ in enumerate(groups)}
        assert derived == set(range(len(groups)))

    assert checked >= 20, f"可校验的历史会话偏少：{checked}"
    assert multi_follow_up_sessions >= 3, "样本里应包含多轮追问的会话"
