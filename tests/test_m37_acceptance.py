"""批次 11 / M37：四类验收夹具（分数健壮性 / 早停可达性 / 意图回归 / 聚合口径）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.graph.edges import early_stop_threshold, route_after_assessment
from app.nodes import assess as assess_mod
from app.nodes import evaluate as eval_mod
from app.nodes.evaluate import _topic_records
from app.service import _is_explicit_stop_command

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "finish_reason": "stop"}
RUBRIC = [{"key": "d1", "label": "维度一", "weight": 1.0, "definition": "x", "threshold": "t"}]


def _assess_state(answer: str = "正常回答", answer_meta: dict | None = None) -> dict:
    meta = {"id": 1, "category": "scenario", "skills": ["x"], "depth_level": "application"}
    meta.update(answer_meta or {})
    return {
        "question_plan": [meta],
        "current_question": "q",
        "current_answer": answer,
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


# ---------------- 1. 分数健壮性（四类脏数据）----------------

@pytest.mark.parametrize(
    "reply,label",
    [
        ("这不是 JSON", "非数字/非 JSON"),
        (json.dumps({"dimension_levels": {"d1": None}}), "null 档位"),
        (json.dumps({"dimension_levels": {"d1": "比较一般"}}), "非数字档位"),
        (json.dumps({"dimension_levels": {"d1": 99}}), "越界档位"),
        ("", "空回复（截断）"),
    ],
)
def test_m37_dirty_data_never_crashes_or_fakes_score(monkeypatch, reply, label):
    monkeypatch.setattr(assess_mod, "chat_with_usage", lambda *a, **k: (reply, dict(USAGE)))
    result = assess_mod.assess(_assess_state())
    assert "assessments" in result, label
    record = result["assessments"][0]
    # 不崩
    assert isinstance(record, dict), label
    # 不产生假分：要么是合法分，要么显式标记失败
    if record.get("score") is None:
        assert record.get("score_error") is True, label
    else:
        assert 0.0 <= record["score"] <= 10.0, label


def test_m37_dirty_data_still_produces_a_report(monkeypatch):
    monkeypatch.setattr(
        eval_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"overall_score": 6}), dict(USAGE))
    )
    state = {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": None, "score_error": True,
             "dimensions": {}, "is_follow_up": False}
        ],
        "question_plan": [{"id": 1}],
        "plan_question_count": 1,
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": RUBRIC},
        "summaries": [], "turn_records": [], "usage_records": [], "difficulty_events": [],
    }
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["overall_score"] == 0.0
    assert report["score_errors"] == 1


# ---------------- 2. 早停可达性 ----------------

def _stop_state(main_count: int, index: int = 11) -> dict:
    plan = [{"id": i} for i in range(1, 16)]
    assessments = [
        {"question_id": i, "score": 8.5, "is_follow_up": False, "next_action": "next_question"}
        for i in range(1, main_count + 1)
    ]
    return {
        "assessments": assessments,
        "current_question_index": index,
        "question_plan": plan,
        "follow_up_count": 0,
        "max_follow_ups": 3,
    }


def test_m37_early_stop_is_reachable():
    assert route_after_assessment(_stop_state(12)) == "evaluate"
    assert route_after_assessment(_stop_state(11)) == "advance"
    assert early_stop_threshold() == 12


# ---------------- 3. 意图回归夹具（历史回放的无正文元数据）----------------

def _recorded_replay_metadata():
    replay = PROJECT_ROOT / "tests" / "fixtures" / "intent" / "replay-metadata.json"
    data = json.loads(replay.read_text(encoding="utf-8"))
    # These frozen labels do not rerun or validate a current cloud model.
    rows = [
        {key: value for key, value in cohort.items() if key != "count"}
        for cohort in data["cohorts"] for _ in range(cohort["count"])
    ]
    return data, rows

def test_m37_intent_regression_matches_recorded_replay():
    """Verify frozen historical labels, without exporting private Q/A or using data/."""

    data, rows = _recorded_replay_metadata()
    assert data["total"] == len(rows) == 208

    false_explain = [
        row for row in rows
        if row["new_intent"] == "request_explanation" and not row["genuine_explain"]
    ]
    missed = [
        row for row in rows
        if row["genuine_explain"] and row["new_intent"] != "request_explanation"
    ]
    false_stop = [row for row in rows if row["new_intent"] == "request_stop"]

    # The original replay contains four explanation false positives against
    # heuristic labels. Preserve that fact; this is not current model evaluation.
    assert len(false_explain) == 4
    assert missed == []
    assert false_stop == [], "样本中没有真实结束请求，不得判出 request_stop"
    # Recorded old-rule hit counts, not a claim of current model accuracy.
    assert sum(1 for row in rows if row["old_explain"]) == 10
    assert sum(1 for row in rows if row["old_stop"]) == 6


def test_m37_new_stop_rule_clears_all_six_false_positives():
    _, rows = _recorded_replay_metadata()
    old_hits = [row for row in rows if row["old_stop"]]
    assert len(old_hits) == 6
    assert all(not row["new_stop_command"] for row in old_hits)
    # Current command behavior is separately exercised on synthetic text in test_b2_intent.py.


# ---------------- 4. 聚合口径（每题等权）----------------

def test_m37_topics_are_equally_weighted():
    records = [
        {"plan_question_index": 0, "score": 4.0, "is_follow_up": False, "dimensions": {}},
        {"plan_question_index": 0, "score": 4.0, "is_follow_up": True, "dimensions": {}},
        {"plan_question_index": 0, "score": 4.0, "is_follow_up": True, "dimensions": {}},
        {"plan_question_index": 1, "score": 8.0, "is_follow_up": False, "dimensions": {}},
    ]
    topics = _topic_records(records)
    assert len(topics) == 2
    weight = 1.0 / len(topics)
    assert abs(weight - 0.5) < 1e-9
    # 每题一票：(4 + 8) / 2 = 6.0
    assert sum(item["score"] for item in topics) / len(topics) == 6.0
