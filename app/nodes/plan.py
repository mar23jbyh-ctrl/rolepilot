from __future__ import annotations

import json
import logging
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.telemetry.summary import usage_record, merge_usage
from app.guard import BatchArbiter, audit_questions, summarize_audit
from app.llm.client import chat_with_usage
from app.prompts.templates import PLAN_PROMPT
from app.prompts.references import research_references_block
from app.role import (
    MOTIVATION_QUOTA,
    build_role_profile,
    is_motivation_question,
    render_focus_whitelist,
    role_policy_text,
)
from app.utils.json_utils import parse_json_with_retries

logger = logging.getLogger(__name__)

# 出题后守卫：命中禁止话题的题被丢弃；丢弃率超过该阈值时重生成一次（最多一次）。
GUARD_DROP_THRESHOLD = 0.3
# 重生成门槛：只有"丢弃率高"且"保留题数不足目标×该比例"时才重生成，
# 避免"掉了 5 道还剩 12 道"也去重生成。
GUARD_RETRY_MIN_KEEP_RATIO = 0.6
# 单次 plan 节点执行的出题调用上限：截断重试、补题与重生成共享预算。
MAX_PLAN_LLM_CALLS = 2
# 写进图状态的 AuditRecord.raw 截断长度；日志仅记录数量。
GUARD_RAW_MAX_CHARS = 2000

# 由程序执行的题量上下限。
MIN_QUESTIONS = int(settings.min_questions)
MAX_QUESTIONS = int(settings.max_questions)
# 单次规划最多纳入的简历项目数。
PROJECT_LIMIT = 8


class _CallBudget:
    """出题调用的共享预算。

    每次 plan 节点执行共享一个预算，覆盖截断重试、补题与守卫重生成。
    预算用尽后使用当前题单或兜底题。重新规划会创建新的节点预算；
    整场 Token 额度由统一模型调用出口另行控制。
    """

    def __init__(self, limit: int = MAX_PLAN_LLM_CALLS) -> None:
        self.limit = max(0, int(limit))
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def take(self) -> bool:
        """占用一个调用额度；额度用尽返回 False。"""
        if self.exhausted:
            return False
        self.used += 1
        return True


def _project_relevance(project: str, jd_terms: list[str]) -> int:
    text = str(project or "").lower()
    return sum(1 for term in jd_terms if term and str(term).lower() in text)


def _project_year(project: str) -> int:
    years = [int(value) for value in re.findall(r"(20\d{2})", str(project or ""))]
    return max(years) if years else 0


def _rank_projects(
    projects: list[str], jd_terms: list[str], limit: int = 8
) -> tuple[list[str], dict]:
    """项目排序：按词面相关度和时间新近度取前 N，并记录截断数量。

    - 相关度：项目文本与 JD 要求/技能的词面重合度
    - 新近度：项目文本里出现的最新年份（没有年份记 0，不额外加分）
    - 返回 ``(选中的项目, {total, used, truncated})``
    """

    indexed = _rank_project_indices(projects, jd_terms, limit)
    total = sum(bool(str(project).strip()) for project in (projects or []))
    selected = [str(projects[index]) for index in indexed]
    return selected, {"total": total, "used": len(selected), "truncated": total - len(selected)}


def _rank_project_indices(projects: list[str], jd_terms: list[str], limit: int) -> list[int]:
    """Keep original resume indices stable through ranking and prompt selection."""
    indexed = [(index, str(project)) for index, project in enumerate(projects or []) if str(project).strip()]
    indexed.sort(
        key=lambda pair: (
            -_project_relevance(pair[1], jd_terms),
            -_project_year(pair[1]),
            pair[0],
        )
    )
    return [index for index, _ in indexed[: max(1, int(limit))]]


def _usage_entry(node: str, usage: dict) -> dict:
    return usage_record(node, usage)


def _generate_questions(
    messages: list,
    budget: _CallBudget | None = None,
) -> tuple[list, dict]:
    """出题调用（含截断重试），返回原始题单与 usage。

    budget 不为空时与"守卫重生成"共享同一额度：额度用尽立即停止调用，
    返回已经拿到的题单（可能为空，由调用方决定是否用兜底题）。
    """
    parsed: dict = {}
    usage: dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    current = list(messages)
    accumulated = {}
    for attempt in range(2):
        if budget is not None and not budget.take():
            break
        raw, usage = chat_with_usage(
            current,
            temperature=settings.temp_plan,
            max_tokens=12000 if attempt else 8000,
        )
        merge_usage(accumulated, usage)
        parsed = parse_json_with_retries(raw) or {}
        if usage.get("finish_reason") != "length" and parsed.get("questions"):
            break
        current = current + [
            {
                "role": "user",
                "content": "上一次输出因长度限制被截断。请重新输出完整的 JSON 题单，不要再中断。",
            }
        ]
    return list(parsed.get("questions", []) or []), accumulated or usage


def _guard_dropped_records(audit_result: dict) -> list[dict]:
    """把审计结果映射成兼容既有结构的丢弃记录（question/category/hits/reason/round）。"""
    by_id = {
        int(record.get("question_id", 0)): record
        for record in (audit_result.get("audit") or [])
    }
    records: list[dict] = []
    for question in audit_result.get("dropped") or []:
        record = by_id.get(int(question.get("id", 0)), {})
        hits = [
            item
            for item in str((record.get("vote_evidence") or {}).get("D2", "") or "").split("、")
            if item
        ]
        records.append(
            {
                "question": str(question.get("content", ""))[:120],
                "category": str(question.get("category", "")),
                "hits": hits,
                "reason": str(record.get("reason", "")),
                "round": int(record.get("round", 1)),
            }
        )
    return records


def _truncate_audit_records(records: list[dict]) -> list[dict]:
    """写进图状态前截断 AuditRecord.raw。

    取舍：raw（仲裁原始返回）是离线复现的关键证据，但单条可达数千字符，而 state 会在
    每次 submit 时整份序列化落盘（SQLite）。因此状态里只保留前 GUARD_RAW_MAX_CHARS
    个字符并标记 raw_truncated=True；不将原始模型内容写入应用日志。
    """
    trimmed: list[dict] = []
    for record in records or []:
        item = dict(record)
        raw = item.get("raw")
        if isinstance(raw, dict) and raw:
            text = json.dumps(raw, ensure_ascii=False)
            if len(text) > GUARD_RAW_MAX_CHARS:
                item["raw"] = {"truncated": True, "text": text[:GUARD_RAW_MAX_CHARS]}
                item["raw_truncated"] = True
        trimmed.append(item)
    return trimmed


def _fallback_question(job_title: str) -> dict:
    """兜底题：通用动机题（不再写死技术题，避免非技术岗被塞进 Python 题）。"""
    return {
        "id": 1,
        "category": "scenario",
        "question_type": "behavioral",
        "difficulty": "easy",
        "depth_level": "application",
        "project_ref": "",
        "jd_ref": "",
        "focus_key": "",  # 通用动机/经历题：显式标空（3.2.0 的豁免标记）
        "research_ref": "",
        "intent": "求职动机与岗位认知",
        "content": f"请结合你的经历，谈谈你为什么选择应聘{job_title}这个岗位？",
        "skills": ["求职动机"],
        "source_id": "llm:fallback",
        "source_type": "llm",
        "required": True,
    }


def _normalize_plan(
    raw_questions: list,
    references: list[dict],
    role_profile: dict,
) -> list[dict]:
    valid_categories = {"foundation", "project", "scenario", "algorithm"}
    valid_difficulty = {"easy", "medium", "hard"}
    valid_depth = {"concept", "application", "deep"}
    reference_ids = {str(item.get("source_id", "")) for item in references}
    questions = []
    for index, raw in enumerate(raw_questions, 1):
        if not isinstance(raw, dict) or not str(raw.get("content", "")).strip():
            continue
        category = str(raw.get("category", "scenario"))
        if category not in valid_categories:
            category = "scenario"
        question_type = str(raw.get("question_type", "") or "")
        if question_type not in {"foundation", "scenario", "behavioral"}:
            question_type = {
                "foundation": "foundation",
                "project": "scenario",
                "scenario": "behavioral",
                "algorithm": "foundation",
            }.get(category, "foundation")
        difficulty = str(raw.get("difficulty", "medium"))
        if difficulty not in valid_difficulty:
            difficulty = "medium"
        depth_level = str(raw.get("depth_level", "")).strip().lower()
        if depth_level not in valid_depth:
            # 基础与算法类问题默认只要求概念级理解，其余按应用级处理。
            depth_level = (
                "concept" if category in {"foundation", "algorithm"} else "application"
            )
        source_id = str(raw.get("source_id", "")).strip()
        source_type = "web" if source_id and source_id in reference_ids else "llm"
        if not source_id:
            source_id = "llm:" + uuid.uuid4().hex[:8]
        focus_key = str(raw.get("focus_key", "") or "").strip().lower()
        question = {
                "id": index,
                "category": category,
                "difficulty": difficulty,
                "question_type": question_type,
                "required": bool(raw.get("required", True)),
                "depth_level": depth_level,
                "project_ref": str(raw.get("project_ref", "") or ""),
                "jd_ref": str(raw.get("jd_ref", "") or ""),
                "focus_key": focus_key,
                "research_ref": str(raw.get("research_ref", "") or "").strip(),
                "intent": str(raw.get("intent", "") or ""),
                "content": str(raw.get("content", "")).strip(),
                "skills": list(raw.get("skills", []) or []),
                "source_id": source_id,
                "source_type": source_type,
        }
        # M19：动机/经历题打独立标记（复用守卫已有的判定函数，避免把所有
        # scenario 题都误判成动机题——_normalize_plan 会把 scenario 统一映射成 behavioral）。
        question["question_kind"] = (
            "motivation" if is_motivation_question(question) else "professional"
        )
        questions.append(question)
    return questions


def _improve_quality(
    questions: list[dict],
    state,
    role_profile: dict,
    limit: int,
) -> list[dict]:
    """去重、限制同一技能出现次数，并补足求职动机/岗位认知题。"""
    seen_content: set[str] = set()
    skill_counts: dict[str, int] = {}
    result: list[dict] = []
    for question in questions:
        content_key = " ".join(str(question.get("content", "")).split()).lower()
        if not content_key or content_key in seen_content:
            continue
        skill = (question.get("skills") or [""])[0]
        if skill and skill_counts.get(skill, 0) >= 2:
            continue
        seen_content.add(content_key)
        if skill:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        result.append(question)

    job_title = str(
        # M22：LLM 读全文判定的岗位名优先；前三行粗糙推断只作兜底
        role_profile.get("job_title") or state.get("job_title") or "目标岗位"
    )
    requirement = "岗位要求的核心能力"
    for item in (state.get("jd_profile", {}) or {}).get("requirements", []) or []:
        text = str(item.get("text", "")).strip()
        if text:
            requirement = text[:40]
            break

    behavioral = [q for q in result if q.get("question_type") == "behavioral"]
    fillers: list[dict] = []
    if len(behavioral) < 2:
        fillers.append(
            {
                "category": "scenario",
                "question_type": "behavioral",
                "question_kind": "motivation",
                "difficulty": "easy",
                "depth_level": "application",
                "project_ref": "",
                "jd_ref": "",
                "focus_key": "",  # 通用动机/经历题：显式标空（3.2.0 的豁免标记）
                "research_ref": "",
                "intent": "求职动机与岗位认知",
                "content": f"你为什么选择应聘{job_title}这个岗位？结合你的经历谈谈你的动机。",
                "skills": ["求职动机"],
                "source_id": "llm:motivation",
                "source_type": "llm",
                "required": True,
            }
        )
    if len(behavioral) + len(fillers) < 2:
        fillers.append(
            {
                "category": "scenario",
                "question_type": "behavioral",
                "question_kind": "motivation",
                "difficulty": "easy",
                "depth_level": "application",
                "project_ref": "",
                "jd_ref": "",
                "intent": "岗位认知与学习规划",
                "content": f"如果入职，你打算如何快速补齐“{requirement}”方面的能力？",
                "skills": ["学习规划"],
                "source_id": "llm:growth-plan",
                "source_type": "llm",
                "required": True,
            }
        )

    # M25：截断上限改为 config 的 max_questions（原先用的是 target=15）
    # M25：上限就是 config 的 max_questions；LLM 给 12-18 之间的题量时原样保留
    cap = int(limit or MAX_QUESTIONS)
    keep = max(0, cap - len(fillers))
    merged = result[:keep] + fillers
    for index, question in enumerate(merged, 1):
        question["id"] = index
        question.setdefault("intent", "")
    return merged


def plan(state) -> dict:
    gap_report = state.get("gap_report", {})
    role_profile = state.get("role_profile") or build_role_profile(
        str(state.get("jd_text", "") or ""),
        str(state.get("resume_text", "") or ""),
    )
    resume_profile = state.get("resume_profile", {}) or {}
    resume_projects = resume_profile.get("projects", []) or []
    concerns = resume_profile.get("concerns", []) or []
    jd_summary = str(state.get("jd_text", ""))[:1000]
    jd_requirements = ""
    jd_items: list[str] = []
    for index, requirement in enumerate((state.get("jd_profile", {}) or {}).get("requirements", []) or []):
        jd_items.append(str(requirement.get("text", "")))
        jd_requirements += f"[{index}] {requirement.get('text', '')}\n"
    # M26：项目按"与 JD 的相关度 + 时间新近度"排序后取前 N，并记录截断数量
    jd_terms: list[str] = []
    for requirement in (state.get("jd_profile", {}) or {}).get("requirements", []) or []:
        jd_terms.extend(str(skill) for skill in (requirement.get("skills") or []))
        jd_terms.append(str(requirement.get("text", "")))
    jd_terms.extend(str(skill) for skill in (resume_profile.get("skills") or []))
    resume_project_list, project_stats = _rank_projects(
        resume_projects, jd_terms, limit=PROJECT_LIMIT
    )
    project_indices = _rank_project_indices(resume_projects, jd_terms, PROJECT_LIMIT)
    project_lines = [f"[{index}] {resume_projects[index]}" for index in project_indices]
    if concerns:
        project_lines.append("简历存疑点：" + "；".join(str(c) for c in concerns))
    target = int(settings.target_questions)
    job_title = str(
        # M22：与上面同口径——role_profile 优先
        role_profile.get("job_title") or state.get("job_title") or "目标岗位"
    )
    references = []
    job_research = state.get("job_research", {}) or {}
    fragments = job_research.get("fragments") or []
    if fragments:
        # 逐片段注入（每段截 1200 字符，总量与 raw_notes[:6000] 相当），
        # 让 research_ref 有真实取值空间：research:job:1..5。
        for index, fragment in enumerate(fragments, 1):
            fragment_id = str(fragment.get("id") or f"research:job:{index}")
            references.append(
                {
                    "answer": str(fragment.get("content", "") or "")[:1200],
                    "category": "scenario",
                    "difficulty": "medium",
                    "source_id": fragment_id,
                }
            )
    else:
        # 兼容模块 3 之前的历史会话：只有 raw_notes，没有 fragments
        research_notes = str(job_research.get("raw_notes", "") or "")[:6000]
        if research_notes:
            references.append(
                {
                    "answer": research_notes,
                    "category": "scenario",
                    "difficulty": "medium",
                    "source_id": "research:job",
                }
            )
    references_status = "ok" if references else "empty"
    reference_ids = [
        str(ref.get("source_id", "")) for ref in references if ref.get("source_id")
    ]
    # M24：把调研片段的原文一起带进守卫，用于 research_ref 的内容相关性校验
    reference_texts = {
        str(ref.get("source_id", "")): str(ref.get("answer", "") or "")
        for ref in references
        if ref.get("source_id")
    }
    if fragments and "research:job" not in reference_ids:
        # 兼容：允许题目沿用旧的 research:job（历史会话/旧提示词习惯）
        reference_ids.append("research:job")

    user_content = PLAN_PROMPT.format(
        job_title=job_title,
        experience_level=str(role_profile.get("experience_level", "junior")),
        target=target,
        min_q=settings.min_questions,
        max_q=settings.max_questions,
        motivation_quota=MOTIVATION_QUOTA,
        focus_whitelist=render_focus_whitelist(role_profile),
        source_ids="、".join(reference_ids) or "（无）",
        jd_ref_range=f"0..{len(jd_items) - 1}" if jd_items else "（无）",
        project_ref_range="、".join(str(index) for index in project_indices) or "（无）",
        gap_report=json.dumps(gap_report, ensure_ascii=False),
        jd_summary=jd_summary,
        jd_requirements=jd_requirements.strip() or "（未提取到结构化 JD 要求）",
        resume_projects="\n".join(project_lines) if project_lines else "（简历未提取到具体项目，题目以 JD 差距与通用实践为主）",
        role_policy=role_policy_text(role_profile),
        references=research_references_block(references) if references else "（无可用联网调研参考）",
    )
    messages = [
        SystemMessage(content="You are a senior interviewer. Output JSON only."),
        HumanMessage(content=user_content),
    ]
    budget = _CallBudget()
    # 单会话共用同一个仲裁器：初版 + 重生成共享"1 次仲裁调用"的预算
    arbiter = BatchArbiter()
    raw_questions, usage = _generate_questions(messages, budget)
    questions = _normalize_plan(raw_questions, references, role_profile)
    # M25：上限用 max_questions（12-18 之间原样保留），下限在下面用 min_questions 强制
    questions = _improve_quality(questions, state, role_profile, MAX_QUESTIONS)
    usage_records = [_usage_entry("plan_questions", usage)]

    # M25：题量下限由代码强制——不足 min_questions 时补题。
    # 成本：仅在不足时 +1 次出题调用（与截断重试、守卫重生成共享同一预算）。
    if len(questions) < MIN_QUESTIONS and not budget.exhausted:
        missing = MIN_QUESTIONS - len(questions)
        topup_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"题单还差 {missing} 道。请再补 {missing} 道**不重复**的题，"
                    "优先覆盖尚未涉及的考察维度与来源，只输出同样格式的 JSON。"
                ),
            }
        ]
        topup_raw, topup_usage = _generate_questions(topup_messages, budget)
        usage_records.append(_usage_entry("plan_questions_topup", topup_usage))
        extra_questions = _normalize_plan(topup_raw, references, role_profile)
        existing_contents = {question["content"] for question in questions}
        for item in extra_questions:
            if item["content"] in existing_contents:
                continue
            questions.append(item)
            existing_contents.add(item["content"])
            if len(questions) >= MIN_QUESTIONS:
                break
        for index, question in enumerate(questions, 1):
            question["id"] = index

    # 统一审计入口（guard_mode 在 guard.audit_questions 内分流）：
    # ⚠️ 删除语义差异：legacy 画像命中 4 组词表即丢（单票，与模块 5 一致）；
    #    whitelist 画像需要 D1/D2/D3 中至少 2 票反对才丢。两者用 stage 区分。
    audit_result = audit_questions(
        questions,
        role_profile,
        jd_items=jd_items,
        resume_projects=resume_projects,
        reference_ids=reference_ids,
        reference_texts=reference_texts,
        arbiter=arbiter,
        round_index=1,
    )
    questions = list(audit_result["kept"])
    guard_dropped: list[dict] = _guard_dropped_records(audit_result)
    all_audit: list[dict] = list(audit_result["audit"])
    all_dropped: list[dict] = list(audit_result["dropped"])
    statuses: set[str] = {str(audit_result["status"])}

    kept_floor = int(target * GUARD_RETRY_MIN_KEEP_RATIO)
    if (
        audit_result["dropped"]
        and audit_result["drop_rate"] > GUARD_DROP_THRESHOLD
        and len(questions) < kept_floor
        and not budget.exhausted
    ):
        # 丢弃率高且保留题数不足目标×0.6，才重生成一次（共享调用预算）。
        hit_topics = sorted(
            {hit for item in guard_dropped for hit in item["hits"]}
        )
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"上一次题单里有 {len(audit_result['dropped'])} 道题涉及禁止话题"
                    f"（{'、'.join(hit_topics)}）。请重新输出完整题单，"
                    "这些话题一律不得出现在题目与技能里；每题仍要填 focus_key 与来源。"
                ),
            }
        ]
        retry_raw, retry_usage = _generate_questions(retry_messages, budget)
        usage_records.append(_usage_entry("plan_questions_retry", retry_usage))
        retry_questions = _normalize_plan(retry_raw, references, role_profile)
        if retry_questions:
            retry_questions = _improve_quality(
                retry_questions, state, role_profile, target
            )
            retry_audit = audit_questions(
                retry_questions,
                role_profile,
                jd_items=jd_items,
                resume_projects=resume_projects,
                reference_ids=reference_ids,
                reference_texts=reference_texts,
                arbiter=arbiter,
                round_index=2,
            )
            all_audit.extend(retry_audit["audit"])
            all_dropped.extend(retry_audit["dropped"])
            statuses.add(str(retry_audit["status"]))
            if retry_audit["kept"]:
                questions = list(retry_audit["kept"])
                guard_dropped.extend(_guard_dropped_records(retry_audit))

    if not questions:
        questions = [_fallback_question(job_title)]

    # 审计汇总（两轮合并后按题面去重）+ 状态可见性
    guard_summary = summarize_audit(all_audit, all_dropped)
    guard_summary.update(
        {
            "guard_audit_status": "degraded"
            if "degraded" in statuses
            else str(audit_result["status"]),
            "guard_mode": str(role_profile.get("guard_mode", "") or ""),
            "weak_focus_count": int(audit_result["summary"].get("weak_focus_count", 0) or 0),
            "guard_degrade_reason": str(
                audit_result["summary"].get("guard_degrade_reason", "") or ""
            ),
            "guard_domain_slug_missing": bool(
                audit_result["summary"].get("guard_domain_slug_missing", False)
            ),
        }
    )
    guard_audit = _truncate_audit_records(all_audit)
    raw_records = [record for record in all_audit if record.get("raw")]
    if raw_records:
        # Do not export model/source material through application log collectors.
        logger.info("guard audit raw record count: %s", len(raw_records))
    usage_records.extend(arbiter.usage_records)

    weak_skills = list(gap_report.get("weak_skills", []) or [])[:3]
    first_difficulty = "easy" if weak_skills else "medium"
    if questions and weak_skills:
        for question in questions[: max(2, len(questions) // 4)]:
            if question["difficulty"] == "hard":
                question["difficulty"] = "medium"
    return {
        "question_plan": questions,
        # M12：记录原始题量（替代题不改变该值）
        "plan_question_count": len(questions),
        # M26：简历项目的使用统计（含因超限未纳入的数量）
        "resume_project_stats": project_stats,
        "difficulty": first_difficulty,
        "role_profile": role_profile,
        "references_status": references_status,
        "guard_dropped": guard_dropped,
        "guard_audit": guard_audit,
        "guard_summary": guard_summary,
        "usage_records": usage_records,
    }
