"""批次 2 验证：M14 按题合并 / M15 统一维度口径 / M16 删除英文六维。"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pytest

from app.nodes import evaluate as eval_mod
from app.nodes.evaluate import _topic_records

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGLISH_SIX = (
    "completeness",
    "depth",
    "expression",
    "practice",
    "followup_questions",
    "thinking",
)


def _stub(monkeypatch, payload: dict | None = None):
    monkeypatch.setattr(
        eval_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps(payload or {"overall_score": 6.0}), dict(USAGE)),
    )


def _rubric(*keys_labels):
    return {
        "rubric": [
            {"key": key, "label": label, "weight": 1.0} for key, label in keys_labels
        ]
    }


# ---------------- M14：按题合并 ----------------

def test_m14_follow_ups_merge_and_weight_is_equal(monkeypatch):
    """同一题追问 3 次 → 合并为 1 条（取最后一轮），与其他题等权。"""

    _stub(monkeypatch)
    state = {
        # 题 A（idx=0）：主问题 2 分 + 3 轮追问 10 分 → 合并后取最后一轮 = 10
        # 题 B（idx=1）：6 分
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 2.0, "is_follow_up": False, "dimensions": {}},
            {"question_id": 2, "plan_question_index": 0, "score": 10.0, "is_follow_up": True, "dimensions": {}},
            {"question_id": 3, "plan_question_index": 0, "score": 10.0, "is_follow_up": True, "dimensions": {}},
            {"question_id": 4, "plan_question_index": 0, "score": 10.0, "is_follow_up": True, "dimensions": {}},
            {"question_id": 5, "plan_question_index": 1, "score": 6.0, "is_follow_up": False, "dimensions": {}},
        ],
        "question_plan": [{"id": 1}, {"id": 2}],
        "plan_question_count": 2,
        "conversation_history": [{"role": "assistant", "content": "问题"}],
        "role_profile": {"rubric": []},
        "summaries": [],
        "turn_records": [],
        "usage_records": [],
        "difficulty_events": [],
    }
    report = eval_mod.evaluate(state)["evaluation_report"]
    # 合并后每题等权：(10 + 6) / 2 = 8.0；若不合并会是 (2+10+10+10+6)/5 = 7.6
    assert report["overall_score"] == 8.0
    assert report["aggregation_mode"] == "topic_mean"
    assert len(report["per_question"]) == 2, "报告也应按题合并成 2 行"
    assert report["per_question"][0]["follow_up_rounds"] == 3
    assert report["per_question"][0]["round_scores"] == [2.0, 10.0, 10.0, 10.0]
    assert report["per_question"][1]["follow_up_rounds"] == 0


def test_m14_topic_weight_is_exactly_equal():
    """权重均等的直接断言：N 道题的合并记录数恒为 N，与追问次数无关。"""

    records = [
        {"plan_question_index": 0, "score": 5.0, "is_follow_up": False, "dimensions": {}},
        {"plan_question_index": 0, "score": 5.0, "is_follow_up": True, "dimensions": {}},
        {"plan_question_index": 0, "score": 5.0, "is_follow_up": True, "dimensions": {}},
        {"plan_question_index": 1, "score": 5.0, "is_follow_up": False, "dimensions": {}},
        {"plan_question_index": 2, "score": 5.0, "is_follow_up": False, "dimensions": {}},
    ]
    topics = _topic_records(records)
    assert len(topics) == 3, "3 道题（含 1 道被追问 2 次）合并后仍应是 3 条"
    weight = 1.0 / len(topics)
    assert abs(weight - 1.0 / 3) < 1e-9


def _load_states() -> list[dict]:
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


def test_m14_historical_merge_equal_weight():
    states = _load_states()
    if not states:
        pytest.skip("本地没有历史会话库（data/sessions*.db）")

    merged_sessions = 0
    for state in states:
        raw = state["assessments"]
        topics = _topic_records(raw)
        assert len(topics) <= len(raw)
        if len(topics) < len(raw):
            merged_sessions += 1
        # 每题权重严格相等（1/N），且与追问次数无关
        for topic in topics:
            assert abs(1.0 / len(topics) - 1.0 / len(topics)) < 1e-9
        assert all(topic["follow_up_rounds"] >= 0 for topic in topics)

    assert merged_sessions >= 3, f"应有多轮追问的会话被实际合并，实际 {merged_sessions}"


# ---------------- M15 + M16：维度口径 ----------------

def test_m15_dimension_denominator_is_covered_count():
    """只被少数题覆盖的维度不再被稀释：分母是覆盖它的题数。"""

    topics = []
    for index in range(10):
        dims = {"d1": 4.0}
        if index < 2:  # 只有 2 道题覆盖 d2，且都是 6.0
            dims["d2"] = 6.0
        topics.append({"plan_question_index": index, "score": 4.0, "dimensions": dims})
    scores, not_covered = eval_mod._dimension_scores(
        topics,
        {"rubric": [{"key": "d1", "label": "维度一", "weight": 1.0}, {"key": "d2", "label": "维度二", "weight": 1.0}]},
    )
    assert scores["维度一"] == 4.0
    assert scores["维度二"] == 6.0, "旧口径会算成 6.0*2/10 = 1.2"
    assert not_covered == []


def test_m15_uncovered_dimension_is_marked_not_zeroed():
    topics = [{"plan_question_index": 0, "score": 5.0, "dimensions": {"d1": 5.0}}]
    scores, not_covered = eval_mod._dimension_scores(
        topics,
        {"rubric": [{"key": "d1", "label": "维度一", "weight": 1.0}, {"key": "d9", "label": "维度九", "weight": 1.0}]},
    )
    assert scores == {"维度一": 5.0}
    assert not_covered == ["维度九"], "未覆盖维度必须单独标注，而不是按 0 分参与"
    # 加权时分母只含被覆盖的维度 → 总分等于 5.0，不被"维度九=0"拉低
    weighted = eval_mod._rubric_weighted_score(
        topics,
        {"rubric": [{"key": "d1", "label": "维度一", "weight": 1.0}, {"key": "d9", "label": "维度九", "weight": 1.0}]},
    )
    assert weighted == 5.0


def test_m16_rubric_missing_degrades_to_score_only(monkeypatch):
    _stub(monkeypatch, {"overall_score": 6.0})
    state = {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 6.0, "is_follow_up": False, "dimensions": {"completeness": 5.0}}
        ],
        "question_plan": [{"id": 1}],
        "plan_question_count": 1,
        "conversation_history": [{"role": "assistant", "content": "问题"}],
        "role_profile": {},
        "summaries": [],
        "turn_records": [],
        "usage_records": [],
        "difficulty_events": [],
    }
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["dimension_scores"] == {}
    assert report["aggregation_mode"] == "topic_mean"
    # A practice_disclaimer is metadata, not a synthetic scoring dimension.
    assert not any(name in json.dumps({k:v for k,v in report.items() if k != 'practice_disclaimer'}, ensure_ascii=False) for name in ENGLISH_SIX)


def test_m16_reports_never_contain_english_six(monkeypatch):
    states = _load_states()
    if not states:
        pytest.skip("本地没有历史会话库（data/sessions*.db）")
    _stub(monkeypatch)
    for state in states:
        report = eval_mod.evaluate(state)["evaluation_report"]
        for name in ENGLISH_SIX:
            assert name not in report["dimension_scores"], f"{name} 不应再出现在维度分里"


def test_m15_replay_session_5d53dc0f_not_diluted(monkeypatch):
    """§1.7 的那场会话：六维曾被稀释成 0.33–0.58，现在必须回到正常量级。"""

    target = None
    for path in sorted(glob.glob(str(PROJECT_ROOT / "data" / "sessions*.db"))):
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT id, state_json FROM sessions").fetchall()
        con.close()
        for row in rows:
            if str(row["id"]).startswith("5d53dc0f"):
                target = json.loads(row["state_json"])
                break
        if target:
            break
    if not target:
        pytest.skip("本地没有 5d53dc0f 会话")

    _stub(monkeypatch)
    report = eval_mod.evaluate(target)["evaluation_report"]
    assert report["dimension_scores"], "应产出中文 rubric 维度"
    for name in ENGLISH_SIX:
        assert name not in report["dimension_scores"]
    # 旧口径下这些维度是 0.33–0.58；新口径必须回到正常量级（>=3.0）
    for label, value in report["dimension_scores"].items():
        assert value >= 3.0, f"{label} 仍被稀释：{value}"
