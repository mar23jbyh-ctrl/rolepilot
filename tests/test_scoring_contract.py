"""Reader-facing scoring contract stays aligned with the implementation."""

from __future__ import annotations

import glob
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "docs" / "scoring_contract.md"

# This metadata is intentionally kept as an implementation guard: it must not
# become a hidden scoring input merely because an old payload still contains it.
DEAD_FIELDS = ("source_type",)


def test_contract_document_exists_with_reader_sections():
    assert CONTRACT.exists(), "docs/scoring_contract.md must exist"
    text = CONTRACT.read_text(encoding="utf-8")
    for token in (
        "practice-v1",
        "评分输入",
        "模型输出",
        "程序校验",
        "分数换算与聚合",
        "评分的适用范围",
        "相关实现",
    ):
        assert token in text, token


def test_contract_lists_scoring_fields():
    text = CONTRACT.read_text(encoding="utf-8")
    for field in ("content", "depth_level", "project_ref", "difficulty", "question_kind",
                  "plan_question_index", "rubric"):
        assert f"`{field}`" in text, f"scoring field {field} must be documented"


def test_legacy_metadata_is_not_used_for_scoring():
    """Legacy metadata must not become an implicit scoring input."""

    offenders: list[str] = []
    for path in glob.glob(str(PROJECT_ROOT / "app" / "**" / "*.py"), recursive=True):
        text = Path(path).read_text(encoding="utf-8")
        for field in DEAD_FIELDS:
            # 读取形态：（question|item|raw|data|meta）.get("field", ...) 或 ["field"]
            pattern = rf"\.get\(\"{field}\"|\[\"{field}\"\]"
            if re.search(pattern, text):
                offenders.append(f"{Path(path).name}:{field}")
    assert offenders == [], f"legacy metadata is read by app code: {offenders}"


def test_documented_fields_are_consumed_by_scoring_nodes():
    """Documented scoring metadata must remain connected to the scoring code."""

    assess_src = (PROJECT_ROOT / "app" / "nodes" / "assess.py").read_text(encoding="utf-8")
    eval_src = (PROJECT_ROOT / "app" / "nodes" / "evaluate.py").read_text(encoding="utf-8")
    for field in ("depth_level", "project_ref", "question_kind", "difficulty"):
        assert f'"{field}"' in assess_src, f"{field} should be consumed by assess"
    for field in ("plan_question_index", "difficulty", "question_kind"):
        assert f'"{field}"' in eval_src, f"{field} should be consumed by evaluate"
