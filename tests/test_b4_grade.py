"""B4（M9）的验证用例：等级由代码推导 + 阈值入配置 + LLM 等级意见留档。

其中 test_b4_replay_grade_self_consistent 会回放本地历史会话库
（`data/sessions*.db`）；若目录缺失则跳过。
"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import DEFAULT_GRADE_CUTOFFS, settings
from app.models.schemas import EvaluationReport
from app.nodes import evaluate as eval_mod
from app.nodes.evaluate import _grade_from_score, _safe_grade

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_state(**overrides) -> dict:
    state = {
        "assessments": [
            {"question_id": 1, "score": 6.5, "dimensions": {"d1": 6.5}, "is_follow_up": False},
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


def _stub_llm(monkeypatch, payload: dict):
    monkeypatch.setattr(
        eval_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps(payload, ensure_ascii=False), dict(USAGE)),
    )


# ---------------- M9：等级完全由代码推导 ----------------

@pytest.mark.parametrize(
    "score,expected",
    [(10.0, "A"), (8.0, "A"), (7.99, "B"), (6.0, "B"), (5.99, "C"), (4.0, "C"), (3.99, "D"), (0.0, "D")],
)
def test_m9_grade_mapping(score, expected):
    assert _grade_from_score(score, DEFAULT_GRADE_CUTOFFS) == expected


def test_m9_grade_ignores_llm_value(monkeypatch):
    """模型说 A，但代码算出的总分只到 C —— 报告必须是 C，且模型意见留档。"""

    _stub_llm(monkeypatch, {"overall_score": 9.9, "grade": "A"})
    state = _base_state(
        assessments=[{"question_id": 1, "score": 5.0, "dimensions": {"d1": 5.0}, "is_follow_up": False}]
    )
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["overall_score"] == 5.0
    assert report["grade"] == "C", "等级必须由代码总分推导"
    assert report["llm_grade_opinion"] == "A", "模型等级意见要留档"


def test_m9_grade_without_llm_opinion(monkeypatch):
    _stub_llm(monkeypatch, {"overall_score": 7.0})
    report = eval_mod.evaluate(_base_state())["evaluation_report"]
    assert report["grade"] == "B"
    assert report["llm_grade_opinion"] is None


# ---------------- 阈值可配置 ----------------

def test_m9_thresholds_come_from_config(monkeypatch):
    monkeypatch.setattr(settings, "grade_thresholds", "9,7,5")
    assert settings.grade_cutoffs == (9.0, 7.0, 5.0)
    assert _grade_from_score(8.5) == "B"  # 默认阈值下是 A
    assert _grade_from_score(6.0) == "C"
    assert _grade_from_score(4.9) == "D"


def test_m9_grade_changes_with_config_end_to_end(monkeypatch):
    _stub_llm(monkeypatch, {"overall_score": 6.5})
    state = _base_state(
        assessments=[{"question_id": 1, "score": 6.5, "dimensions": {"d1": 6.5}, "is_follow_up": False}]
    )
    monkeypatch.setattr(settings, "grade_thresholds", "8,6,4")
    assert eval_mod.evaluate(state)["evaluation_report"]["grade"] == "B"  # 6.5 >= 6
    monkeypatch.setattr(settings, "grade_thresholds", "8,7,4")
    assert eval_mod.evaluate(state)["evaluation_report"]["grade"] == "C"  # 6.5 < 7 但 >= 4


def test_m9_malformed_threshold_config_falls_back(monkeypatch):
    for bad in ("", "abc", "8", "8,6"):
        monkeypatch.setattr(settings, "grade_thresholds", bad)
        assert settings.grade_cutoffs == DEFAULT_GRADE_CUTOFFS, bad


# ---------------- 枚举校验 ----------------

def test_m9_grade_enum_guard():
    assert _safe_grade("优秀") == "—"
    assert _safe_grade("") == "—"
    assert _safe_grade("A") == "A"


def test_m9_schema_rejects_invalid_grade():
    with pytest.raises(ValidationError):
        EvaluationReport(overall_score=5.0, grade="优秀")
    # 零评估分支使用的长破折号必须合法
    assert EvaluationReport(overall_score=0.0, grade="—").grade == "—"


# ---------------- 历史回放：等级与总分 100% 自洽 ----------------

def _load_reports() -> list[dict]:
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


def test_b4_replay_grade_self_consistent(monkeypatch):
    states = _load_reports()
    if not states:
        pytest.skip("本地没有历史会话库（data/sessions*.db）")

    _stub_llm(monkeypatch, {"overall_score": 9.9, "grade": "A"})

    old_inconsistent = 0
    new_inconsistent = 0
    for state in states:
        stored = state["evaluation_report"]

        # 改造前：等级取 LLM 值，用阈值校验它是否自洽
        old_score = stored.get("overall_score")
        old_grade = str(stored.get("grade"))
        if isinstance(old_score, (int, float)) and old_grade in {"A", "B", "C", "D"}:
            if old_grade != _grade_from_score(float(old_score), DEFAULT_GRADE_CUTOFFS):
                old_inconsistent += 1

        # 改造后：等级必须由总分推导（零评估分支的"—"是合法特例，不参与该断言）
        report = eval_mod.evaluate(state)["evaluation_report"]
        if report["grade"] == "—":
            assert report["overall_score"] == 0.0, "只有零评估分支才允许—"
            continue
        if report["grade"] != _grade_from_score(report["overall_score"], DEFAULT_GRADE_CUTOFFS):
            new_inconsistent += 1

    assert new_inconsistent == 0, f"改造后仍有 {new_inconsistent} 份不自洽"
    assert old_inconsistent > 0, "样本里应当存在改造前的矛盾报告，否则回归无意义"
