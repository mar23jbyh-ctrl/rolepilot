"""B3（M7 / M8 / M10）的验证用例：字段规范化、总分归属反转、范围校验。

其中 test_b3_replay_overall_equals_code 会回放本地历史会话库
（`data/sessions*.db`）；若目录缺失则跳过。
"""

from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pytest

from app.nodes import evaluate as eval_mod

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_state(**overrides) -> dict:
    state = {
        "assessments": [
            {"question_id": 1, "score": 6.0, "dimensions": {"d1": 6.0}, "is_follow_up": False},
            {"question_id": 2, "score": 4.0, "dimensions": {"d1": 4.0}, "is_follow_up": False},
        ],
        "conversation_history": [{"role": "assistant", "content": "问题"}],
        "role_profile": {"rubric": [{"key": "d1", "label": "岗位基础", "weight": 1.0}]},
        "summaries": [],
        "turn_records": [],
        "usage_records": [],
        "difficulty_events": [],
        "current_question_index": 0,
        "question_plan": [],
    }
    state.update(overrides)
    return state


def _stub_llm(monkeypatch, payload: dict):
    monkeypatch.setattr(
        eval_mod,
        "chat_with_usage",
        lambda *a, **k: (json.dumps(payload, ensure_ascii=False), dict(USAGE)),
    )


# ---------------- M7：字段规范化 ----------------

def test_m7_null_and_string_fields_are_normalized(monkeypatch):
    _stub_llm(
        monkeypatch,
        {
            "overall_score": 5.0,
            "grade": None,
            "text_analysis": None,
            "strengths": "沟通能力强",
            "weaknesses": None,
            "learning_path": ["先补检索", "再补评估"],
            "resources": "官方文档",
        },
    )
    report = eval_mod.evaluate(_base_state())["evaluation_report"]

    # 关键断言 1：null 不再变成字面量 "None"
    assert report["text_analysis"] == ""
    assert "None" not in json.dumps(report, ensure_ascii=False)
    # 关键断言 2：字符串必须变成单元素列表，而不是被拆成单字符
    assert report["strengths"] == ["沟通能力强"]
    assert report["resources"] == ["官方文档"]
    assert report["weaknesses"] == []
    assert report["learning_path"] == ["先补检索", "再补评估"]
    assert report["grade"] not in {"None", ""}


def test_m7_string_is_not_split_into_characters(monkeypatch):
    _stub_llm(monkeypatch, {"overall_score": 5.0, "strengths": "沟通能力强"})
    report = eval_mod.evaluate(_base_state())["evaluation_report"]
    assert len(report["strengths"]) == 1, "5 字字符串不得变成 5 个元素"


def test_m7_unexpected_structures_do_not_leak_into_report(monkeypatch):
    _stub_llm(
        monkeypatch,
        {"overall_score": 5.0, "strengths": {"a": 1}, "learning_path": 42},
    )
    report = eval_mod.evaluate(_base_state())["evaluation_report"]
    assert report["strengths"] == []
    assert report["learning_path"] == ["42"]


# ---------------- M8：总分归属反转 ----------------

def test_m8_report_ignores_llm_overall(monkeypatch):
    _stub_llm(monkeypatch, {"overall_score": 9.9, "grade": "A"})
    report = eval_mod.evaluate(_base_state())["evaluation_report"]
    # 代码加权：(6+4)/2 = 5.0；模型说 9.9，必须被忽略
    assert report["overall_score"] == 5.0
    assert report["llm_overall_opinion"] == 9.9, "模型意见要留档，便于审计偏离"


def test_m8_missing_llm_overall_still_works(monkeypatch):
    _stub_llm(monkeypatch, {"grade": "B"})
    report = eval_mod.evaluate(_base_state())["evaluation_report"]
    assert report["overall_score"] == 5.0
    assert report["llm_overall_opinion"] is None


# ---------------- M10：范围校验 ----------------

def test_m10_out_of_range_total_is_clamped_and_flagged(monkeypatch):
    _stub_llm(monkeypatch, {"overall_score": 9.9})
    # 注入一个"超过 10 分的总分"：rubric 里没有该维度 → 走题均分路径 → overall = 85
    # （M15 之后维度值本身会被夹取，所以越界总分只能从 score 侧进入）
    state = _base_state(
        assessments=[
            {"question_id": 1, "score": 85, "dimensions": {}, "is_follow_up": False},
        ]
    )
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["overall_score"] == 10.0, "85 必须被夹到 10.0"
    assert report["out_of_range"] is True


def test_m10_in_range_total_is_not_flagged(monkeypatch):
    _stub_llm(monkeypatch, {"overall_score": 5.0})
    report = eval_mod.evaluate(_base_state())["evaluation_report"]
    assert report["overall_score"] == 5.0
    assert report["out_of_range"] is False


# ---------------- 26 场回放：overall_score == overall_code ----------------

def _code_overall(state: dict) -> float:
    """独立复算代码口径的总分（不调用生产函数，避免自证）。

    口径已随 M14 / M15 更新，与生产实现保持一致：
      1. M14 先按题合并（plan_question_index，缺失时按 is_follow_up 切段还原）；
      2. M15 维度只认 rubric，分母是**覆盖该维度的题数**；
      3. 未被覆盖的维度不参与加权；rubric 完全不可用时退化为题均分。
    """

    # 与生产一致：**先按题合并**，再排除"该题最后一轮评分失败"的主题
    raw = list(state.get("assessments") or [])
    # ---- M14：按题合并 ----
    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    fallback = -1
    for item in raw:
        if not item.get("is_follow_up"):
            fallback += 1
        index = item.get("plan_question_index")
        key = index if isinstance(index, int) and index >= 0 else fallback
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    topics: list[dict] = []
    for key in order:
        items = groups[key]
        record = dict(items[-1])
        dims: dict[str, float] = {}
        for item in items:
            for dim_key, dim_value in (item.get("dimensions") or {}).items():
                try:
                    dims[str(dim_key)] = min(10.0, max(0.0, float(dim_value)))
                except (TypeError, ValueError):
                    continue
        record["dimensions"] = dims
        # 生产口径：score_error 取自合并后的最后一条记录
        if not record.get("score_error"):
            topics.append(record)

    rubric = (state.get("role_profile") or {}).get("rubric") or []
    weights: dict[str, float] = {}
    # M21：难度权重（回放口径必须与生产一致）
    difficulty_weight = {"easy": 0.8, "medium": 1.0, "hard": 1.2}

    def _w(record: dict) -> float:
        return difficulty_weight.get(str(record.get("difficulty") or "medium").lower(), 1.0)

    for item in rubric:
        if isinstance(item, dict) and item.get("key"):
            try:
                weights[str(item["key"])] = float(item.get("weight", 0.0))
            except (TypeError, ValueError):
                continue
    if weights and topics:
        available: list[tuple[float, float]] = []
        for key, weight in weights.items():
            values = []
            for record in topics:
                value = (record.get("dimensions") or {}).get(key)
                if value is None:
                    continue
                try:
                    values.append((float(value), _w(record)))
                except (TypeError, ValueError):
                    continue
            if values:
                local = sum(w for _, w in values)
                available.append((sum(v * w for v, w in values) / (local or 1), weight))
        weight_sum = sum(weight for _, weight in available)
        if available and weight_sum > 0:
            return round(sum(mean * weight for mean, weight in available) / weight_sum, 2)
    scores: list[tuple[float, float]] = []
    for record in topics:
        try:
            scores.append((float(record.get("score")), _w(record)))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    total_weight = sum(w for _, w in scores)
    return round(sum(v * w for v, w in scores) / (total_weight or 1), 2)


def _load_replay_states() -> list[dict]:
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
            if state.get("evaluation_report") and state.get("assessments"):
                states.append(state)
    return states


def test_b3_replay_overall_equals_code(monkeypatch):
    states = _load_replay_states()
    if not states:
        pytest.skip("本地没有可回放的历史会话库（data/sessions*.db）")

    # 模型故意给出一个夸张的总分，验证它不再影响报告
    _stub_llm(monkeypatch, {"overall_score": 9.9, "grade": "A"})

    mismatches = []
    for state in states:
        expected = _code_overall(state)
        report = eval_mod.evaluate(state)["evaluation_report"]
        if abs(report["overall_score"] - expected) > 1e-9:
            mismatches.append((report["overall_score"], expected))
    assert not mismatches, f"报告总分与代码口径不一致：{mismatches[:5]}"
    assert len(states) >= 20, f"可回放样本偏少：{len(states)}"
