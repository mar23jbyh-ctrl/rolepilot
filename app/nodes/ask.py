from __future__ import annotations

import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.telemetry.summary import usage_record
from app.context.manager import summaries_text
from app.graph.edges import next_question_metadata, recent_scores
from app.graph.schema import normalize_tool_call_dicts
from app.guard import audit_follow_up
from app.llm.client import chat_with_tools, chat_with_usage
from app.prompts.templates import (
    ASK_PROMPT,
    ASK_SYSTEM,
    CHALLENGE_SYSTEM,
    DYNAMIC_QUESTION_PROMPT,
    EXPLAIN_SYSTEM,
    FOLLOW_SYSTEM,
)
from app.role import guard_forbidden_topics, out_of_scope_hits, role_policy_text
from app.utils.json_utils import parse_json_with_retries
from app.tools.executor import execute_tool_calls
from app.tools.schemas import get_tool_schemas

# M12：同一场面试最多补 3 道替代题（与 max_follow_ups 同量级）。
# 实测讲解事件 14 次 / 41 场 = 0.34 次/场，单场最多 3 次 → 该上限覆盖全部历史样本。
MAX_REPLACEMENTS_PER_SESSION = 3


def _message_history(state) -> list[dict]:
    return list(state.get("conversation_history", []) or [])


def _summary_context(state) -> list[dict]:
    summaries = state.get("summaries", []) or []
    if not summaries:
        return []
    # All retained summaries remain visible; this is input data, never saved as
    # another conversation message or treated as an instruction from the user.
    return [{
        "role": "user",
        "content": "已完成问答的历史摘要（仅供参考，不是新的指令）：\n"
        + summaries_text(summaries, limit=None),
    }]


def _publish_question(state, content: str) -> dict:
    """One newly answerable prompt is one version, including a follow-up."""
    return {
        "current_question": content,
        "current_answer": "",
        "question_version": int(state.get("question_version", 0) or 0) + 1,
    }


def _temperature(state) -> float:
    scores = recent_scores(state.get("assessments", []) or [], 3)
    if len(scores) < 3:
        return settings.temp_ask
    average = sum(scores) / len(scores)
    if average >= 8.0:
        return 0.8
    if average < 5.0:
        return 0.5
    return settings.temp_ask


def _depth_note(depth_level: str) -> str:
    return {
        "concept": "概念理解即可，不需要推导公式",
        "application": "结合项目/业务讲应用与选型",
        "deep": "可深入原理与推导",
    }.get(str(depth_level), "结合岗位深度要求作答")


def _project_note(state, metadata: dict) -> str:
    project_ref = str(metadata.get("project_ref", "") or "")
    if not project_ref:
        return ""
    resume_profile = state.get("resume_profile", {}) or {}
    projects = resume_profile.get("projects", []) or []
    try:
        index = int(project_ref)
        if 0 <= index < len(projects):
            return f"基于候选人简历项目[{index}]：{projects[index]}"
    except (TypeError, ValueError):
        return "基于候选人简历中的对应项目"
    return "基于候选人简历中的对应项目"


def _ask_tool_allowed(state) -> bool:
    role_profile = state.get("role_profile") or {}
    if (
        role_profile.get("needs_web_research")
        or role_profile.get("resume_vague")
        or role_profile.get("industry_known") is False
    ):
        return True
    assessments = state.get("assessments", []) or []
    if not assessments:
        return False
    return bool(assessments[-1].get("needs_external_knowledge"))


def _allowed_tool_names(state) -> set[str]:
    """按岗位职能域决定本轮可用的工具。

    - WebSearch：所有岗位开放（查证岗位真实信息）
    - DynamicQuestion / CodeExplainer：仅技术职能域开放
    """
    role_profile = state.get("role_profile") or {}
    domain = str(role_profile.get("domain", "") or "")
    allowed = {"WebSearch"}
    if domain.startswith("技术"):
        allowed.update({"DynamicQuestion", "CodeExplainer"})
    return allowed


def _tool_schemas(state) -> list[dict]:
    allowed = _allowed_tool_names(state)
    return [
        schema
        for schema in get_tool_schemas()
        if str(schema.get("function", {}).get("name", "")) in allowed
    ]


def _asked_follow_ups(state) -> list[str]:
    """本题已经问过的追问原文，用来避免同一知识点跨轮重复。

    按 assessments 里本题的追问段精确取（is_follow_up=True 的那几条），
    不再用 assistant[-follow_up_count:]，避免插入讲解/收束语时取错消息。
    """
    asked: list[str] = []
    for item in reversed(state.get("assessments", []) or []):
        if not item.get("is_follow_up"):
            break
        text = str(item.get("question", "") or "").strip()
        if text:
            asked.append(text)
    asked.reverse()
    return asked


def _repeat_guard(state) -> str:
    """把「已问过的追问」写进提示，要求换角度而不是重复同一个点。"""
    asked = _asked_follow_ups(state)
    if not asked:
        return ""
    listed = "\n".join(f"- {item}" for item in asked)
    return (
        "\n本题已经问过的追问（严禁重复，也不要换个说法再问同一个知识点）：\n"
        f"{listed}\n"
        "如果这些点都已经问过、候选人仍未答到，就不要再揪着同一个点问第三次："
        "请换一个本题尚未涉及的相邻角度（例如选型对比、业务影响、排错思路、边界情况），"
        "或者自然收束本题、把话头交出去。"
    )


def _usage_record(node: str, usage: dict) -> dict:
    return usage_record(node, usage)


def _generate_replacement_question(
    state,
    skipped: dict,
    avoid: list[str],
) -> tuple[dict | None, list[dict]]:
    """M12：为被跳过的题目生成一道**同维度**的替代题。

    成本：**每次 1 个 LLM 调用**（复用 `DYNAMIC_QUESTION_PROMPT` 的小请求），
    仅在候选人请求讲解时触发；实测 0.34 次/场。
    失败或命中禁止话题时返回 ``None``，由调用方在报告里标注"样本不足"。
    """

    role_profile = state.get("role_profile") or {}
    skills = [str(item) for item in (skipped.get("skills") or []) if str(item).strip()]
    skill = skills[0] if skills else str(skipped.get("focus_key") or "岗位核心能力")
    depth_level = str(skipped.get("depth_level") or "application")
    category = str(skipped.get("category") or "scenario")
    difficulty = str(state.get("difficulty") or "medium")
    avoid_text = "\n".join(f"- {item}" for item in (avoid or [])[:20]) or "无"

    messages = [
        SystemMessage(content="You are an interviewer question generator. Output JSON only."),
        HumanMessage(
            content=DYNAMIC_QUESTION_PROMPT.format(
                skill=skill,
                difficulty=difficulty,
                category=category,
                avoid=avoid_text,
                role_policy=role_policy_text(role_profile, depth_level),
                depth_level=depth_level,
            )
        ),
    ]
    raw, usage = chat_with_usage(messages, temperature=settings.temp_plan)
    records = [_usage_record("ask_replacement", usage)]
    parsed = parse_json_with_retries(raw) or {}
    content = str(parsed.get("content", "") or "").strip()
    if not content:
        return None, records
    # 零成本兜底：出题提示词已带岗位政策，这里再用禁止话题做一次确定性检查
    if out_of_scope_hits(content, role_profile):
        return None, records
    return {
        "content": content,
        "skills": [str(item) for item in (parsed.get("skills") or skills or [skill]) if str(item).strip()],
        "source_id": str(parsed.get("source_id") or ("llm:" + uuid.uuid4().hex[:8])),
        "depth_level": str(parsed.get("depth_level") or depth_level),
    }, records


def _naturalize(text: str) -> str:
    return re.sub(r"(提示|追问|题目|质疑)\s*[：:]\s*", "", str(text)).strip()


def _sanitize_visible(text: str) -> str:
    """Remove leaked internal reviewer wording before rendering to the user."""
    hidden_markers = (
        "提示方向",
        "上一轮遗漏",
        "内部指令",
        "内部提示",
        "内部评审",
        "评审意见",
        "参考答案",
        "可以这样问",
        "missed_points",
        "follow_up_reason",
    )
    cleaned_lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker.lower() in lowered for marker in hidden_markers):
            continue
        cleaned_lines.append(line)
    return _naturalize("\n".join(cleaned_lines)).strip()


def _single_question(text: str) -> str:
    """Keep only the first complete question so one turn asks exactly one thing."""
    cleaned = str(text).strip()
    positions = [cleaned.find(mark) for mark in ("？", "?")]
    positions = [position for position in positions if position != -1]
    if not positions:
        return cleaned
    end = min(positions)
    if end <= 3:
        return cleaned
    return cleaned[: end + 1]


def _is_complete(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    return text[-1] in ("。", "！", "？", "!", "?", "…", '"', "”")


def _looks_truncated(content: str, usage: dict) -> bool:
    if usage.get("finish_reason") == "length":
        return True
    return not _is_complete(content)


def _safe_chat(messages, temperature: float, node: str = "ask") -> tuple[str, list[dict]]:
    """Chat with truncation detection and one automatic regeneration.

    node 用于 usage 记账：按真实节点名记录（ask / ask_follow_up / ask_explain），
    同一次重试也沿用该节点名，便于报告里的 token_by_node 按节点聚合。
    """
    records = []
    current_messages = list(messages)
    last_content = ""
    for attempt in range(2):
        content, usage = chat_with_usage(
            current_messages,
            temperature=temperature,
            max_tokens=settings.max_output_tokens * (2 if attempt else 1),
        )
        records.append(_usage_record(node, usage))
        content = _sanitize_visible(content)
        if content:
            last_content = content
        if not _looks_truncated(content, usage):
            return content, records
        current_messages = current_messages + [
            {
                "role": "user",
                "content": "上一轮输出不完整就中断了。请直接输出完整、自然的一句话问题，不要解释原因。",
            }
        ]
    return _single_question(last_content) or "请再详细说说刚才这个话题。", records


def ask(state) -> dict:
    metadata = next_question_metadata(state)
    if not metadata:
        # 题单已耗尽：不再生成新问题、不写入占位报告，
        # 由 route_after_ask 直接路由到 evaluate。
        return {"current_question": "", "current_answer": "", "explanation_pending": False}
    question_text = str(metadata.get("content", ""))
    role_profile = state.get("role_profile") or {}
    # M22：LLM 判定的岗位名优先（面试官身份提示词也用它）
    job_title = str(role_profile.get("job_title") or state.get("job_title") or "目标岗位")
    depth_level = str(metadata.get("depth_level", "application"))
    role_policy = role_policy_text(role_profile, depth_level)
    forbidden_topics = "、".join(guard_forbidden_topics(role_profile)) or "（无）"
    project_note = _project_note(state, metadata)

    payload = [
        {
            "role": "system",
            "content": ASK_SYSTEM.format(
                difficulty=state.get("difficulty", "medium"),
                role_policy=role_policy,
                job_title=job_title,
                forbidden_topics=forbidden_topics,
            ),
        }
    ]
    payload.extend(_summary_context(state))
    payload.extend(_message_history(state))
    payload.append(
        {
            "role": "user",
            "content": ASK_PROMPT.format(
                question=question_text,
                category=metadata.get("category", ""),
                skills="、".join(metadata.get("skills", []) or []),
                depth_note=_depth_note(depth_level),
                project_note=project_note or "无",
            ),
        }
    )

    temperature = _temperature(state)
    tool_records = []
    schemas = _tool_schemas(state)
    if schemas and _ask_tool_allowed(state):
        content, calls, usage = chat_with_tools(
            payload,
            schemas,
            temperature=temperature,
        )
        usage_records = [_usage_record("ask_tools", usage)]
        for _ in range(settings.max_tool_rounds):
            if not calls:
                break
            calls_dicts = normalize_tool_call_dicts(calls)
            results = execute_tool_calls(calls_dicts, allowed_tools=_allowed_tool_names(state))
            payload.append(AIMessage(content=content or "", tool_calls=calls_dicts))
            tool_ids = []
            for call in calls_dicts:
                tool_ids.append(str(call.get("id", "")))
            for tool_id, result in zip(tool_ids, results):
                payload.append(ToolMessage(content=result.content, tool_call_id=tool_id))
                tool_records.append(result.model_dump())
                if result.usage:
                    usage_records.append(_usage_record("tool:" + result.tool, result.usage))
            payload.append(
                {
                    "role": "user",
                    "content": "请基于工具结果向候选人提出本题，只输出问题本身。",
                }
            )
            content, calls, usage = chat_with_tools(
                payload,
                schemas,
                temperature=temperature,
            )
            usage_records.append(_usage_record("ask_tool_round", usage))
        if not calls:
            content = _sanitize_visible(content)
    else:
        content, usage_records = _safe_chat(payload, temperature)

    content = _single_question(content or question_text)
    history = _message_history(state) + [{"role": "assistant", "content": content}]
    return {
        **_publish_question(state, content),
        "conversation_history": history,
        "usage_records": usage_records,
        "tool_records": tool_records,
    }


def ask_follow_up(state) -> dict:
    """生成一次追问。

    追问阶段的禁止话题检查分三步（全部零 LLM 之外的额外调用）：
      ① 初次判定：`app.guard.audit_follow_up()`（只走强信号，**不调用 BatchArbiter**）；
      ② 命中则换角度重新生成一次；
      ③ 仍然命中 → 返回收束语，并把 `follow_up_count` 顶到 `max_follow_ups`，
         让下一轮路由进入 `next_question`（与模块 5 的 R5-2 一致）。
    追问的 drop 不触发整卷重生成（此时已无"整卷"概念）。
    """
    assessments = state.get("assessments", []) or []
    latest = assessments[-1] if assessments else {}
    follow_up_count = int(state.get("follow_up_count", 0)) + 1
    role_profile = state.get("role_profile") or {}
    job_title = str(
        state.get("role_profile", {}).get("job_title")
        or state.get("role_profile", {}).get("label", "")
        or state.get("job_title")
        or "目标岗位"
    )
    metadata = next_question_metadata(state)
    depth_level = str(metadata.get("depth_level", "application"))
    role_policy = role_policy_text(role_profile, depth_level)
    forbidden_topics = "、".join(guard_forbidden_topics(role_profile)) or "（无）"
    reason = str(latest.get("follow_up_reason", "") or "")
    not_direct = not bool(latest.get("is_relevant", True))
    score = float(latest.get("score", 5.0) or 5.0)
    covered_aspects = list(latest.get("covered_aspects", []) or [])
    project_note = _project_note(state, metadata)
    repeat_guard = _repeat_guard(state)
    progression = ""
    if project_note:
        covered_text = "、".join(str(item) for item in covered_aspects) if covered_aspects else "无"
        progression = (
            f"\n这是简历项目定向题：{project_note}"
            f"\n已覆盖考察角度：{covered_text}"
            "\n请从 流程→选型→踩坑→优化→取舍 中选择下一个未覆盖角度追问，不要重复。"
        )

    history = _message_history(state)
    if latest.get("resume_conflict"):
        conflict_messages = [
            SystemMessage(
                content=CHALLENGE_SYSTEM.format(
                    role_policy=role_policy,
                    conflict_explanation=str(latest.get("resume_conflict_explanation", "")),
                    claimed_projects=str(latest.get("claimed_projects", "")),
                )
            ),
            HumanMessage(
                content="候选人声称的项目与确认后的简历不一致，请当面澄清项目来源。"
            ),
        ]
        conflict_messages.extend(_summary_context(state))
        raw, usage_records = _safe_chat(
            conflict_messages, settings.temp_follow_up, node="ask_follow_up"
        )
        raw = _single_question(raw)
        return {
            **_publish_question(state, raw),
            "follow_up_count": follow_up_count,
            # P0/B2：会话级追问计数（跨题累加，供路由封顶）
            "total_follow_ups": int(state.get("total_follow_ups", 0) or 0) + 1,
            "conversation_history": history + [{"role": "assistant", "content": raw}],
            "usage_records": usage_records,
        }

    if not_direct:
        guidance = (
            "候选人没有直接回答当前题目，可能回答了上一个问题或提出反问。"
            "请面试官自然地把话题拉回本题并请其直接作答，不要扩展新问题。"
        )
    elif score < 5.0:
        guidance = (
            f"候选人得分较低（{score}）。不要加深难度：请降低难度，"
            "用一个更基础、更直白、且本题尚未问过的角度确认概念；"
            "可以适当给提示，再决定是否继续。"
        )
        guidance += repeat_guard
    else:
        guidance = (
            f"上一轮需要补充的点：{reason}\n"
            "仅针对这一个遗漏点做一次补充提问。回答若已经比较充分，本轮应转向收束或换考点，"
            "不要继续在同一知识点无限深挖。"
        )
        guidance += repeat_guard
    if follow_up_count >= settings.max_follow_ups:
        guidance += (
            "\n这是本题最后一次追问：只能选择一个新的考察角度或自然收束，"
            "绝不允许继续对同一细节做更深一层的连环追问。"
        )

    messages = [
        SystemMessage(
            content=FOLLOW_SYSTEM.format(
                follow_up_count=follow_up_count,
                max_follow_ups=state.get("max_follow_ups", settings.max_follow_ups),
                difficulty=state.get("difficulty", "medium"),
                depth_level=depth_level,
                role_policy=role_policy,
                job_title=job_title,
                forbidden_topics=forbidden_topics,
            )
        ),
    ]
    messages.extend(_summary_context(state))
    messages.extend(history)
    messages.append({"role": "user", "content": guidance + progression})
    raw, usage_records = _safe_chat(
        messages, settings.temp_follow_up, node="ask_follow_up"
    )
    raw = _single_question(raw)

    audit = audit_follow_up(raw, role_profile)
    if audit["verdict"] == "drop":
        # ① 命中 → ② 换角度重生成一次 → ③ 仍命中则收束换题
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"上一次追问涉及禁止话题（{'、'.join(audit['hits'])}），"
                    "请换一个不涉及这些话题的角度重新追问，只输出问题本身。"
                ),
            }
        ]
        retry_raw, retry_usage = _safe_chat(
            retry_messages, settings.temp_follow_up, node="ask_follow_up"
        )
        retry_raw = _single_question(retry_raw)
        usage_records = usage_records + retry_usage
        if audit_follow_up(retry_raw, role_profile)["verdict"] == "drop":
            raw = "这道题我们先聊到这里，接下来换下一个话题。"
            return {
                "current_question": raw,
                "current_answer": "",
                "follow_up_abandoned": True,
                "follow_up_count": int(
                    state.get("max_follow_ups", settings.max_follow_ups)
                ),
                "total_follow_ups": int(state.get("total_follow_ups", 0) or 0) + 1,
                "conversation_history": history + [{"role": "assistant", "content": raw}],
                "usage_records": usage_records,
            }
        raw = retry_raw

    return {
        **_publish_question(state, raw),
        "follow_up_abandoned": False,
        "follow_up_count": follow_up_count,
        # P0/B2：会话级追问计数（跨题累加，供路由封顶）
        "total_follow_ups": int(state.get("total_follow_ups", 0) or 0) + 1,
        "conversation_history": history + [{"role": "assistant", "content": raw}],
        "usage_records": usage_records,
    }


def ask_explain(state) -> dict:
    """Explain a knowledge point when the candidate asks, then let the graph
    advance to the next question so the interview stays coherent."""
    metadata = next_question_metadata(state)
    question = str(metadata.get("content", "") or state.get("current_question", ""))
    history = _message_history(state)
    request = ""
    for message in reversed(history):
        if message.get("role") == "user":
            request = str(message.get("content", ""))
            break
    if not request:
        request = "请讲解当前题目的相关知识点"
    role_policy = role_policy_text(
        state.get("role_profile") or {},
        str(metadata.get("depth_level", "application")),
    )
    payload = [
        SystemMessage(
            content=EXPLAIN_SYSTEM.format(
                request=request[:500],
                question=str(question)[:600],
                role_policy=role_policy,
            )
        )
    ]
    payload.extend(_summary_context(state))
    content, usage_records = _safe_chat(
        payload, settings.temp_follow_up, node="ask_explain"
    )
    content = _naturalize(content)

    result: dict = {
        "current_question": "",
        "current_answer": "",
        "explanation_pending": False,
        "conversation_history": history + [{"role": "assistant", "content": content}],
        "usage_records": usage_records,
    }

    # M12：讲解后补一道同维度的替代题，保证有效样本数等于题单题数。
    # 题单耗尽 / 生成失败 / 命中禁止话题时补不了，此时在报告里标注"有效样本数 X / 题单题数 Y"。
    skipped = [dict(item) for item in (state.get("skipped_questions", []) or [])]
    if skipped and skipped[-1].get("replacement_index") is None:
        plan = list(state.get("question_plan", []) or [])
        replacement_count = int(state.get("replacement_count", 0) or 0)
        if replacement_count >= MAX_REPLACEMENTS_PER_SESSION:
            skipped[-1]["replacement_skipped_reason"] = "replacement_cap_reached"
        else:
            avoid = [str(item.get("content", "")) for item in plan]
            avoid.append(str(state.get("current_question", "")))
            replacement, extra_records = _generate_replacement_question(
                state, skipped[-1], avoid
            )
            usage_records = usage_records + extra_records
            result["usage_records"] = usage_records
            if replacement is None:
                skipped[-1]["replacement_skipped_reason"] = "generation_failed_or_off_domain"
            else:
                new_index = len(plan)
                plan.append(
                    {
                        "id": new_index + 1,
                        "category": str(skipped[-1].get("category") or "scenario"),
                        "difficulty": str(state.get("difficulty") or "medium"),
                        "depth_level": replacement["depth_level"],
                        "focus_key": str(skipped[-1].get("focus_key") or ""),
                        "project_ref": "",
                        "jd_ref": "",
                        "research_ref": "",
                        "intent": "讲解后的同维度替代题",
                        "content": replacement["content"],
                        "skills": replacement["skills"],
                        "source_id": replacement["source_id"],
                        "source_type": "llm",
                        "replacement_for": int(skipped[-1].get("question_index") or 0),
                    }
                )
                skipped[-1]["replacement_index"] = new_index
                replacement_count += 1
                result["question_plan"] = plan
                result["replacement_count"] = replacement_count
        result["skipped_questions"] = skipped
    return result
