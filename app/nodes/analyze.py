from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.telemetry.summary import usage_record
from app.telemetry.meter import submit_with_context
from app.llm.client import chat_with_usage
from app.parsers.skill_utils import normalize_skills
from app.prompts.templates import (
    GAP_PROMPT,
    GLOBAL_ROLE_POLICY,
    JD_PROMPT,
    ROLE_PROFILE_PROMPT,
    RESUME_PROMPT,
    RUBRIC_PROMPT,
)
from app.role import build_role_profile, role_policy_text
from app.utils.json_utils import parse_json_with_retries
from app.utils.numeric import parse_number


def _call_structured(system: str, user: str, temperature: float):
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    raw, usage = chat_with_usage(messages, temperature=temperature)
    parsed = parse_json_with_retries(raw) or {}
    return parsed, usage


def _fallback_resume(resume_text: str) -> dict:
    return {
        "summary": resume_text[:500],
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "sections": [],
        "concerns": [],
    }


def _fallback_jd(jd_text: str) -> dict:
    return {
        "requirements": [
            {
                "category": "tech",
                "text": jd_text[:500],
                "skills": [],
                "weight": 0.5,
            }
        ]
    }


def _usage_record(node: str, usage: dict) -> dict:
    return usage_record(node, usage)


# M17：rubric 结构校验参数
RUBRIC_MIN_DIMENSIONS = 4
RUBRIC_MAX_DIMENSIONS = 7
RUBRIC_WEIGHT_TOLERANCE = 0.15


def _clean_rubric_dimensions(raw) -> list[dict]:
    """M17：把 LLM 返回的维度规范成合法结构（字段完整、权重为正）。"""

    if not isinstance(raw, list):
        return []
    cleaned: list[dict] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or f"d{index}").strip()
        label = str(item.get("label") or "").strip()
        definition = str(item.get("definition") or "").strip()
        threshold = str(item.get("threshold") or "").strip()
        if not label or not definition or key in seen_keys:
            continue
        parsed_weight = parse_number(item.get("weight", 0.0), low=-1e9, high=1e9)
        if parsed_weight.value is None or parsed_weight.value <= 0:
            continue
        seen_keys.add(key)
        cleaned.append(
            {
                "key": key,
                "label": label,
                "weight": parsed_weight.value,
                "definition": definition,
                "threshold": threshold,
            }
        )
    return cleaned


def _rubric_problem(dimensions: list[dict]) -> str:
    """M17：返回不合格原因（空串表示合格）。

    校验维度数、字段完整性（含达标线）与权重和。
    """

    if not dimensions:
        return "empty"
    if not RUBRIC_MIN_DIMENSIONS <= len(dimensions) <= RUBRIC_MAX_DIMENSIONS:
        return f"dimension_count={len(dimensions)}"
    if any(not item.get("threshold") for item in dimensions):
        return "missing_threshold"
    total = sum(item["weight"] for item in dimensions)
    if abs(total - 1.0) > RUBRIC_WEIGHT_TOLERANCE:
        return f"weight_sum={round(total, 3)}"
    return ""


def _normalize_rubric_weights(dimensions: list[dict]) -> list[dict]:
    """M17：权重和偏离时按比例归一化（比对整份 rubric 弃用更实用）。"""

    total = sum(item["weight"] for item in dimensions)
    if total <= 0:
        return dimensions
    for item in dimensions:
        item["weight"] = round(item["weight"] / total, 4)
    return dimensions


def _match_level(gap_report: dict, resume_profile: dict) -> str:
    mastered = [
        match
        for match in gap_report.get("matches", []) or []
        if match.get("status") == "mastered"
    ]
    project_count = len(resume_profile.get("projects", []) or [])
    skill_count = len(resume_profile.get("skills", []) or [])
    if len(mastered) >= 3 and (project_count >= 1 or skill_count >= 4):
        return "HIGH"
    if mastered or (project_count >= 1 and skill_count >= 2):
        return "MEDIUM"
    return "LOW"


def _experience_level(resume_text: str, resume_profile: dict) -> str:
    text = f"{resume_text} {' '.join(str(x) for x in resume_profile.get('experience', []) or [])}"
    lowered = str(text).lower()
    if any(marker in lowered for marker in ("实习", "在校", "应届", "intern")):
        return "intern"
    years = 0
    import re

    matches = re.findall(r"(\d+)\s*年", lowered)
    if matches:
        years = max(int(value) for value in matches)
    if years >= 3 or any(marker in lowered for marker in ("资深", "高级", "senior")):
        return "senior"
    return "junior"


def analyze(state) -> dict:
    jd_text = str(state.get("jd_text", ""))
    resume_text = str(state.get("resume_text", ""))

    with ThreadPoolExecutor(max_workers=3) as executor:
        resume_future = submit_with_context(executor,
            _call_structured,
            "You extract resume facts into JSON. Output JSON only.",
            RESUME_PROMPT.format(role_policy=GLOBAL_ROLE_POLICY, resume=resume_text),
            settings.temp_parse,
        )
        jd_future = submit_with_context(executor,
            _call_structured,
            "You extract job requirements into JSON. Output JSON only.",
            JD_PROMPT.format(role_policy=GLOBAL_ROLE_POLICY, jd=jd_text),
            settings.temp_parse,
        )
        # 岗位画像与简历/JD 解析并行（依赖同一份原文，互不等待）。
        profile_future = submit_with_context(executor,
            _call_structured,
            "You identify the job domain and assessment focus. Output JSON only.",
            ROLE_PROFILE_PROMPT.format(jd=jd_text, resume=resume_text),
            settings.temp_parse,
        )
        resume_profile, usage1 = resume_future.result()
        jd_profile, usage2 = jd_future.result()
        llm_profile, usage_profile = profile_future.result()

    if not resume_profile:
        resume_profile = _fallback_resume(resume_text)
    resume_profile["skills"] = normalize_skills(resume_profile.get("skills") or [])
    if not jd_profile:
        jd_profile = _fallback_jd(jd_text)
    for requirement in jd_profile.get("requirements", []):
        requirement["skills"] = normalize_skills(requirement.get("skills") or [])

    gap_user = GAP_PROMPT.format(
        role_policy=GLOBAL_ROLE_POLICY,
        jd=jd_text,
        resume_profile=resume_profile,
        resume_text=resume_text[:6000],
    )
    gap_report, usage3 = _call_structured(
        "You produce a skill gap analysis. Output JSON only.",
        gap_user,
        settings.temp_parse,
    )
    if not gap_report:
        gap_report = {
            "requirements": jd_profile.get("requirements", []),
            "matches": [],
            "missing_skills": [],
            "weak_skills": [],
            "strong_skills": resume_profile.get("skills", []),
            "summary": "Fallback gap summary: use resume skills as baseline.",
        }
    match_level = _match_level(gap_report, resume_profile)
    gap_report["resume_jd_match_level"] = match_level

    role_profile = build_role_profile(jd_text, resume_text, llm_profile)
    role_profile["experience_level"] = _experience_level(resume_text, resume_profile)
    role_label = str(role_profile.get("job_title", "") or "目标岗位")
    rubric_raw, usage4 = _call_structured(
        "You generate a job-specific interview rubric. Output JSON only.",
        RUBRIC_PROMPT.format(
            role_policy=role_policy_text(role_profile),
            role_label=role_label,
            jd=jd_text[:5000],
            resume=resume_text[:5000],
        ),
        settings.temp_parse,
    )
    # M17：rubric 结构校验 + 最多重生成一次
    rubric_dims = _clean_rubric_dimensions(
        rubric_raw.get("dimensions", []) if isinstance(rubric_raw, dict) else []
    )
    problem = _rubric_problem(rubric_dims)
    rubric_retry_usage = None
    if problem:
        retry_raw, rubric_retry_usage = _call_structured(
            "You generate a job-specific interview rubric. Output JSON only.",
            RUBRIC_PROMPT.format(
                role_policy=role_policy_text(role_profile),
                role_label=role_label,
                jd=jd_text[:5000],
                resume=resume_text[:5000],
            )
            + f"\n\n上一次输出不合格（{problem}）。请严格按要求重新输出：维度 4-7 项、"
            "权重合计 1.0、每维必须带 threshold 达标线。",
            settings.temp_parse,
        )
        retry_dims = _clean_rubric_dimensions(
            retry_raw.get("dimensions", []) if isinstance(retry_raw, dict) else []
        )
        if not _rubric_problem(retry_dims):
            rubric_dims, problem = retry_dims, ""
    if problem:
        # 仍不合格 → 能修则修（权重归一化），修不了才回退到通用 4 维
        if rubric_dims and problem.startswith("weight_sum"):
            rubric_dims = _normalize_rubric_weights(rubric_dims)
            problem = _rubric_problem(rubric_dims)
    if problem or not rubric_dims:
        rubric_dims = [
            {
                "key": "d1",
                "label": "岗位专业基础",
                "weight": 0.3,
                "definition": "本岗位核心知识掌握程度",
                "threshold": "能结合岗位场景说清核心概念与常见做法，并给出具体步骤",
            },
            {
                "key": "d2",
                "label": "实务/业务处理",
                "weight": 0.3,
                "definition": "面对真实工作场景的分析与处理能力",
                "threshold": "能结合真实经历讲清处理流程、判断依据与取舍",
            },
            {
                "key": "d3",
                "label": "沟通与职业素养",
                "weight": 0.2,
                "definition": "表达、协作、抗压与职业态度",
                "threshold": "表达有条理、能说明协作方式与压力应对的具体做法",
            },
            {
                "key": "d4",
                "label": "学习与成长潜力",
                "weight": 0.2,
                "definition": "补足岗位能力的学习规划与可塑性",
                "threshold": "能说出具体的学习路径与验证方式，而不是泛泛表态",
            },
        ]
    role_profile["core_duties"] = [
        str(r.get("text", ""))
        for r in (jd_profile.get("requirements", []) or [])
        if r.get("category") in {"experience", "tech", "plus"}
    ][:5]
    role_profile["rubric"] = rubric_dims
    has_specific_content = bool(
        (resume_profile.get("projects") or [])
        or (resume_profile.get("experience") or [])
    )
    resume_vague = (
        len(str(resume_text).strip()) < 200
        or len(resume_profile.get("skills", []) or []) < 2
        or not has_specific_content
    )
    role_profile["resume_vague"] = resume_vague
    role_profile["resume_jd_match_level"] = match_level
    if resume_vague:
        role_profile["needs_web_research"] = True
    usage_records = [
        _usage_record(node, usage)
        for node, usage in (
            ("analyze_resume", usage1),
            ("analyze_jd", usage2),
            ("role_profile", usage_profile),
        ("gap_analysis", usage3),
        ("rubric_generation", usage4),
    )
    ]
    if rubric_retry_usage is not None:
        usage_records.append(_usage_record("rubric_regeneration", rubric_retry_usage))
    return {
        "resume_profile": resume_profile,
        "jd_profile": jd_profile,
        "gap_report": gap_report,
        "role_profile": role_profile,
        "conversation_history": [],
        "summaries": [],
        "summarized_ids": [],
        "current_question_index": 0,
        "follow_up_count": 0,
        "difficulty": "medium",
        "max_follow_ups": settings.max_follow_ups,
        # P0/B2：会话级追问预算（跨题累计上限）
        "max_total_follow_ups": settings.max_total_follow_ups,
        "total_follow_ups": 0,
        "usage_records": usage_records,
    }
