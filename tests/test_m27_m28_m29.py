"""批次 8 验证：M27 终评输入补全 / M28 结构化截断 / M29 压缩失败不吞轮次。"""

from __future__ import annotations

import json

from app.context import manager as mgr
from app.nodes import evaluate as eval_mod
from app.nodes.evaluate import _trim_to_budget

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}


def _capture(monkeypatch):
    captured = {}

    def _stub(messages, temperature=0.7, max_tokens=None):
        captured["messages"] = messages
        return json.dumps({"overall_score": 6, "grade": "B"}), dict(USAGE)

    monkeypatch.setattr(eval_mod, "chat_with_usage", _stub)
    return captured


def _state(rounds: int = 3) -> dict:
    assessments = []
    turns = []
    for index in range(rounds):
        assessments.append(
            {
                "question_id": index + 1,
                "plan_question_index": index,
                "follow_up_rounds": 2,
                "is_follow_up": False,
                "difficulty": "hard" if index % 2 else "easy",
                "question_kind": "motivation" if index == 1 else "professional",
                "score": 6.0,
                "dimensions": {"d1": 6.0},
                "covered_points": ["点"],
                "missed_points": [],
                "follow_up_reason": "补充",
            }
        )
        turns.append({"question": f"问题{index}", "answer": f"回答{index}"})
    return {
        "assessments": assessments,
        "question_plan": [{"id": i + 1} for i in range(rounds)],
        "plan_question_count": rounds,
        "turn_records": turns,
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": [{"key": "d1", "label": "专业维度", "weight": 1.0}]},
        "summaries": [],
        "usage_records": [],
        "difficulty_events": [],
        "guard_summary": {"guard_dropped_count": 3, "guard_uncertain_count": 1},
        "research_status": "failed",
        "references_status": "empty",
    }


def _text_of(messages) -> str:
    return " ".join(
        str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))
        for m in messages
    )


# ---------------- M27 ----------------

def test_m27_final_input_contains_aggregates(monkeypatch):
    captured = _capture(monkeypatch)
    eval_mod.evaluate(_state())
    text = _text_of(captured["messages"])
    assert "本轮统计" in text
    for token in (
        '"dimension_scores"',
        '"plan_question_count"',
        '"matched_question_count"',
        '"follow_up_rounds_total"',
        '"guard_dropped_count"',
        '"guard_uncertain_count"',
        '"research_status": "failed"',
        '"aggregation_mode"',
    ):
        assert token in text, token


def test_m27_assessment_text_has_flags(monkeypatch):
    captured = _capture(monkeypatch)
    eval_mod.evaluate(_state())
    text = _text_of(captured["messages"])
    assert '"is_follow_up"' in text
    assert '"difficulty"' in text
    assert '"question_kind"' in text
    assert '"difficulty": "hard"' in text
    assert '"question_kind": "motivation"' in text


# ---------------- M28 ----------------

def test_m28_trim_keeps_valid_json_and_head_tail():
    records = [{"q": index, "payload": "x" * 200} for index in range(40)]
    trimmed, was_trimmed = _trim_to_budget(records, budget_chars=2000)
    assert was_trimmed is True
    assert len(trimmed) < len(records)
    # 关键：裁剪后仍是**合法 JSON**
    parsed = json.loads(json.dumps(trimmed, ensure_ascii=False))
    assert isinstance(parsed, list) and parsed
    # 首尾都保留
    assert parsed[0]["q"] == 0
    assert parsed[-1]["q"] == 39


def test_m28_long_session_produces_two_valid_json_blobs(monkeypatch):
    captured = _capture(monkeypatch)
    state = _state(rounds=40)
    state["turn_records"] = [
        {"question": f"问题{index}" + "长" * 300, "answer": f"回答{index}" + "长" * 300}
        for index in range(40)
    ]
    eval_mod.evaluate(state)
    text = _text_of(captured["messages"])
    # 从 user_content 里把两块 JSON 抠出来，必须都能解析
    start = text.index("逐题评分：")
    body = text[start + len("逐题评分：") :]
    assessment_json = body[: body.index("\n\n完整问答原文：")].strip()
    qa_json = body[body.index("完整问答原文：") + len("完整问答原文：") :]
    qa_json = qa_json[: qa_json.index("\n\n平均分")].strip()
    assert isinstance(json.loads(assessment_json), list)
    assert isinstance(json.loads(qa_json), list)
    assert json.loads(qa_json)[0]["question"].startswith("问题0")
    assert json.loads(qa_json)[-1]["question"].startswith("问题39")


def test_m28_short_session_is_not_trimmed():
    records = [{"q": index} for index in range(3)]
    trimmed, was_trimmed = _trim_to_budget(records, budget_chars=8000)
    assert trimmed == records and was_trimmed is False


# ---------------- M29 ----------------

def test_m29_failed_summary_does_not_swallow_round(monkeypatch):
    """压缩失败时：不写 summarized_ids、原文保留、下一轮还能重试。"""

    monkeypatch.setattr(
        mgr,
        "summarize_turn",
        lambda **kwargs: (None, dict(USAGE)),
    )
    # 让 history_tokens 超过阈值，触发压缩
    from app.config import settings

    monkeypatch.setattr(settings, "compress_threshold", 0)
    monkeypatch.setattr(settings, "keep_recent_messages", 12)

    state = {
        "conversation_history": [{"role": "user", "content": "x" * 100}],
        "summaries": [],
        "summarized_ids": [],
        "turn_records": [
            {"question_id": 1, "question": "q1", "answer": "a1", "assessment": {}},
            {"question_id": 2, "question": "q2", "answer": "a2", "assessment": {}},
        ],
    }
    result = mgr.compress_history(state)
    assert result["summarized_ids"] == [], "失败的轮次不得被标记为已压缩"
    assert result["summaries"] == []
    assert result["conversation_history"], "原文不得被裁掉"
    assert len(result["conversation_history"]) == 1


def test_m29_successful_summary_still_marks_round(monkeypatch):
    class _Summary:
        def model_dump(self):
            return {"question_id": 1, "summary": "ok"}

    monkeypatch.setattr(mgr, "summarize_turn", lambda **kwargs: (_Summary(), dict(USAGE)))
    from app.config import settings

    monkeypatch.setattr(settings, "compress_threshold", 0)
    state = {
        "conversation_history": [{"role": "user", "content": "x" * 100}],
        "summaries": [],
        "summarized_ids": [],
        "turn_records": [{"question_id": 1, "question": "q1", "answer": "a1", "assessment": {}}],
    }
    result = mgr.compress_history(state)
    assert result["summarized_ids"] == [1]
    assert len(result["summaries"]) == 1
