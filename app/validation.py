"""Validate that the candidate's spoken projects exist in the parsed resume."""

from __future__ import annotations


from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm.client import chat_with_usage
from app.utils.json_utils import parse_json_with_retries

PROJECT_KEYWORDS = ("项目", "系统", "平台", "做过", "开发", "搭建", "上线", "负责", "实现了")


def _resume_project_corpus(resume_profile: dict) -> list[str]:
    profile = resume_profile or {}
    corpus = []
    for field in ("projects", "experience", "summary"):
        values = profile.get(field, []) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            text = str(value).strip()
            if text and text not in corpus:
                corpus.append(text)
    return corpus


def should_check_resume(answer: str, question_category: str) -> bool:
    text = str(answer or "").strip()
    if question_category == "project":
        return len(text) >= 8
    if len(text) < 20:
        return False
    return any(keyword in text for keyword in PROJECT_KEYWORDS)


def validate_resume_claims(
    answer: str,
    resume_profile: dict,
    question_category: str,
) -> dict | None:
    """Return a conflict report when the candidate claims a resume-less project."""
    if not should_check_resume(answer, question_category):
        return None
    resume_projects = _resume_project_corpus(resume_profile)
    prompt = (
        "候选人正在面试中回答项目类问题。请核对回答中提到的项目是否出现在“简历项目清单”中。\n"
        "规则：\n"
        "1. 只判断被当作本人经历口述的项目，不判断通用技术名词。\n"
        "2. 若项目名称/技术主题与简历清单中任意项目可对应（允许不同叫法），视为在简历中。\n"
        "3. 若候选人明确口述了简历里不存在的项目，conflict 为 true，并解释为何不属于简历项目。\n\n"
        "简历项目清单：\n"
        + "\n".join(f"- {project}" for project in resume_projects)
        + "\n\n"
        f"候选人回答：\n{answer[:4000]}\n\n"
        "只输出 JSON：\n"
        '{"claims": [{"name": "项目名/主题", "in_resume": true或false, "reason": "依据"}], '
        '"conflict": true或false, "explanation": "为何冲突/为何不冲突"}'
    )
    messages = [
        SystemMessage(content="你是面试流程校验模块，只输出 JSON。"),
        HumanMessage(content=prompt),
    ]
    raw, usage = chat_with_usage(messages, temperature=settings.temp_fact_check)
    parsed = parse_json_with_retries(raw) or {}
    conflict = parsed.get("conflict") is True
    if not conflict:
        return {"conflict": False, "claims": [], "explanation": "", "usage": usage}
    claims = parsed.get("claims", []) or []
    missing = [
        claim
        for claim in claims
        if isinstance(claim, dict) and claim.get("in_resume") is False
    ]
    if not missing:
        return {"conflict": False, "claims": [], "explanation": "", "usage": usage}
    return {
        "conflict": True,
        "claims": missing,
        "explanation": str(parsed.get("explanation", "")),
        "usage": usage,
    }
