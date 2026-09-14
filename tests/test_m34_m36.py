"""批次 10 验证：M34 scoring_meta / M35 overall_code / M36 文档一致性。"""

from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import SCHEMA_VERSION
from app.nodes import evaluate as eval_mod
from app.prompts.templates import PROMPT_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}


def _state() -> dict:
    return {
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": 7.0, "is_follow_up": False, "dimensions": {}}
        ],
        "question_plan": [{"id": 1}],
        "plan_question_count": 1,
        "conversation_history": [{"role": "assistant", "content": "q"}],
        "role_profile": {"rubric": []},
        "summaries": [], "turn_records": [], "usage_records": [], "difficulty_events": [],
    }


def _report(monkeypatch) -> dict:
    monkeypatch.setattr(
        eval_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"overall_score": 9.9}), dict(USAGE))
    )
    return eval_mod.evaluate(_state())["evaluation_report"]


# ---------------- M34 ----------------

def test_m34_scoring_meta_has_six_keys(monkeypatch):
    meta = _report(monkeypatch)["scoring_meta"]
    for key in (
        "model",
        "temperature",
        "prompt_version",
        "rubric_version",
        "schema_version",
        "generated_at",
    ):
        assert key in meta, key
    assert meta["schema_version"] == SCHEMA_VERSION and meta["schema_version"]
    assert meta["prompt_version"] == PROMPT_VERSION
    assert meta["generated_at"], "生成时间不得为空"


# ---------------- M35 ----------------

def test_m35_overall_code_equals_overall_score(monkeypatch):
    report = _report(monkeypatch)
    assert report["overall_code"] == report["overall_score"] == 7.0
    # 模型说的 9.9 仍只进审计字段
    assert report["llm_overall_opinion"] == 9.9


# ---------------- M36 ----------------

def test_m36_docs_no_longer_claim_python_backend_is_kept():
    targets = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "app" / "config.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "python_backend" not in line:
                continue
            # 不得再出现"保留在磁盘/原样保留"这类描述
            assert "原样保留" not in line, (path.name, line)
            assert "保留在磁盘" not in line, (path.name, line)


def test_m36_config_comments_say_enforced():
    text = (PROJECT_ROOT / "app" / "config.py").read_text(encoding="utf-8")
    assert "题量上下限由**代码强制**" in text
    assert "max_questions" in text and "min_questions" in text
