"""批次 7 验证：M25 题量真实生效 / M26 项目按相关度排序取前 N。"""

from __future__ import annotations

from app.config import settings
from app.nodes.plan import (
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    PROJECT_LIMIT,
    _improve_quality,
    _rank_projects,
)


def _questions(count: int) -> list[dict]:
    return [
        {
            "id": index + 1,
            "category": "foundation",
            "question_type": "foundation",
            "difficulty": "medium",
            "depth_level": "concept",
            "project_ref": "",
            "jd_ref": "",
            "focus_key": "f1",
            "research_ref": "",
            "intent": "",
            "content": f"第 {index + 1} 道题的内容，用于测试题量上下限。",
            # 每题唯一技能：避免命中"同一技能最多 2 题"的去重规则干扰计数
            "skills": [f"技能{index}"],
            "source_id": f"llm:q{index}",
            "source_type": "llm",
        }
        for index in range(count)
    ]


def _state() -> dict:
    return {"jd_profile": {"requirements": []}}


# ---------------- M25 ----------------

def test_m25_config_is_now_enforced():
    assert MIN_QUESTIONS == int(settings.min_questions) == 12
    assert MAX_QUESTIONS == int(settings.max_questions) == 18


def test_m25_twenty_questions_are_truncated_to_max():
    merged = _improve_quality(_questions(20), _state(), {}, MAX_QUESTIONS)
    assert len(merged) == MAX_QUESTIONS == 18
    assert [item["id"] for item in merged] == list(range(1, 19))


def test_m25_eighteen_questions_kept_as_is():
    merged = _improve_quality(_questions(18), _state(), {}, MAX_QUESTIONS)
    assert len(merged) == MAX_QUESTIONS == 18


def test_m25_fifteen_questions_kept_as_is():
    """12-18 之间原样保留（可能因补齐 2 道动机题而略增，但绝不超上限）。"""

    merged = _improve_quality(_questions(15), _state(), {}, MAX_QUESTIONS)
    assert 15 <= len(merged) <= MAX_QUESTIONS


def test_m25_short_plan_needs_topup():
    """6 道题会被补齐到 min_questions（补题逻辑在 plan() 里走一次额外出题调用）。"""

    merged = _improve_quality(_questions(6), _state(), {}, MAX_QUESTIONS)
    assert len(merged) < MIN_QUESTIONS, "补题应在 plan() 阶段完成，而不是 _improve_quality"
    assert len(merged) <= 8  # 最多再补 2 道动机题


def test_m25_topup_is_wired_into_plan():
    import inspect

    from app.nodes import plan as plan_mod

    source = inspect.getsource(plan_mod.plan)
    assert "MIN_QUESTIONS" in source and "plan_questions_topup" in source


# ---------------- M26 ----------------

def test_m26_ranks_by_relevance_and_recency():
    projects = [
        "2020 年：企业官网改版（前端页面）",
        "2024 年：RAG 检索问答系统（向量检索、重排）",
        "2023 年：数据报表平台（ETL、指标）",
        "2019 年：内部工具脚本",
    ]
    selected, stats = _rank_projects(projects, ["RAG", "向量检索"], limit=2)
    assert selected[0].startswith("2024"), "与 JD 相关的项目必须排第一"
    assert stats == {"total": 4, "used": 2, "truncated": 2}


def test_m26_twelve_projects_capped_and_counted():
    projects = [f"20{20 + index % 5} 年：项目 {index}（技术栈 X）" for index in range(12)]
    selected, stats = _rank_projects(projects, ["不存在的关键词"], limit=PROJECT_LIMIT)
    assert len(selected) == PROJECT_LIMIT == 8
    assert stats["total"] == 12
    assert stats["used"] == 8
    assert stats["truncated"] == 4, "必须记录被截断的项目数量"


def test_m26_empty_projects():
    selected, stats = _rank_projects([], ["X"], limit=8)
    assert selected == []
    assert stats == {"total": 0, "used": 0, "truncated": 0}


def test_m26_report_exposes_project_stats(monkeypatch):
    import json

    from app.nodes import evaluate as eval_mod

    monkeypatch.setattr(
        eval_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"overall_score": 6}), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "finish_reason": "stop"})
    )
    state = {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 6.0, "is_follow_up": False, "dimensions": {}}
        ],
        "question_plan": [{"id": 1}],
        "plan_question_count": 1,
        "resume_project_stats": {"total": 12, "used": 8, "truncated": 4},
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": []},
        "summaries": [], "turn_records": [], "usage_records": [], "difficulty_events": [],
    }
    report = eval_mod.evaluate(state)["evaluation_report"]
    assert report["resume_project_stats"]["truncated"] == 4
