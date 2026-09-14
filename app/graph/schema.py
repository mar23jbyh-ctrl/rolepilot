"""Pydantic graph state and message object boundary helpers.

Node functions keep working with plain dict records internally, while the
LangGraph channel only ever stores LangChain BaseMessage subclasses. All
conversions happen here so no raw OpenAI message body enters the state.
"""

from __future__ import annotations

import operator
from functools import partial
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

def _cap_append(existing: list[Any], new: list[Any], limit: int) -> list[Any]:
    if not isinstance(existing, list):
        existing = []
    if not isinstance(new, list):
        new = []
    return (existing + new)[-limit:]


TurnReducer = Annotated[
    list[dict[str, Any]],
    partial(_cap_append, limit=80),
]
AssessmentReducer = Annotated[
    list[dict[str, Any]],
    partial(_cap_append, limit=100),
]
ToolReducer = Annotated[
    list[dict[str, Any]],
    partial(_cap_append, limit=60),
]
AppendReducer = Annotated[list[dict[str, Any]], operator.add]


class InterviewStateSchema(BaseModel):
    assessment_protocol_version: str = 'legacy'
    usage_schema_version: str = 'legacy'
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    llm_ledger_version: str = ""
    llm_usage_records: list[dict[str, Any]] = Field(default_factory=list)
    budget_summary: dict[str, Any] = Field(default_factory=dict)
    budget_exhausted: bool = False
    # 会话初始化时一次性注入的只读简历/岗位资料。
    # resume_text / resume_profile 只允许初始化写入，任何节点不得覆盖或注入调研结果。
    job_title: str = ""
    job_jd: str | None = None
    resume_content: str | None = None
    # job_research 只允许 research_job 节点写入，严禁混入 resume_* 字段。
    job_research: dict[str, Any] = Field(default_factory=dict)
    # 联网调研/参考可用状态（只做可见性，不参与出题逻辑）：
    # research_status 由 research_job 写（ok / failed），
    # references_status 由 plan 写（ok / empty）。
    research_status: str = ""
    references_status: str = ""
    jd_text: str = ""
    resume_text: str = ""
    resume_profile: dict[str, Any] = Field(default_factory=dict)
    jd_profile: dict[str, Any] = Field(default_factory=dict)
    gap_report: dict[str, Any] = Field(default_factory=dict)
    role_profile: dict[str, Any] = Field(default_factory=dict)

    question_plan: list[dict[str, Any]] = Field(default_factory=list)
    # M12：原始题单题数（不含替代题），用于报告里的"有效样本数 X / 题单题数 Y"
    plan_question_count: int = 0
    # M12：因请求讲解而未计分的题目记录
    skipped_questions: list[dict[str, Any]] = Field(default_factory=list)
    # M12：已生成的替代题数量（用于上限保护，避免反复讲解导致无限加题）
    replacement_count: int = 0
    # M26：简历项目使用统计（total / used / truncated）
    resume_project_stats: dict[str, Any] = Field(default_factory=dict)
    current_question: str = ""
    question_version: int = 0
    active_answer_request_id: str = ""
    last_completed_answer_request_id: str = ""
    current_answer: str = ""
    current_question_index: int = 0
    global_question_counter: int = 0
    follow_up_count: int = 0
    follow_up_abandoned: bool = False
    max_follow_ups: int = 3
    # P0/B2：会话级追问累计（跨题累加，用于封顶成本）
    total_follow_ups: int = 0
    max_total_follow_ups: int = 12
    difficulty: str = "medium"
    end_requested: bool = False
    explanation_pending: bool = False

    # LangChain BaseMessage subclasses are the only objects allowed here.
    conversation_history: list[BaseMessage] = Field(default_factory=list)
    turn_records: TurnReducer = Field(default_factory=list)
    assessments: AssessmentReducer = Field(default_factory=list)
    usage_records: AppendReducer = Field(default_factory=list)
    tool_records: ToolReducer = Field(default_factory=list)
    difficulty_events: AppendReducer = Field(default_factory=list)
    # 出题阶段因命中禁止话题被丢弃的题目（含命中的话题），用于可见性与回归排查。
    guard_dropped: list[dict[str, Any]] = Field(default_factory=list)
    # 出题审计：每题一条 AuditRecord（raw 已按 GUARD_RAW_MAX_CHARS 截断，见 plan.py）
    guard_audit: list[dict[str, Any]] = Field(default_factory=list)
    # 审计汇总（键名与 EvaluationReport 的报告字段对齐）
    guard_summary: dict[str, Any] = Field(default_factory=dict)
    summaries: list[dict[str, Any]] = Field(default_factory=list)
    summarized_ids: list[int] = Field(default_factory=list)

    evaluation_report: dict[str, Any] = Field(default_factory=dict)
    self_check_report: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


def record_to_message(record: dict[str, Any]) -> BaseMessage:
    role = str(record.get("role") or record.get("type") or "user").lower()
    content = record.get("content", "")
    if role in {"tool", "toolmessage"}:
        return ToolMessage(
            content=str(content or ""),
            tool_call_id=str(record.get("tool_call_id", "") or ""),
        )
    if role in {"system", "systemmessage"}:
        return SystemMessage(content=str(content or ""))
    if role in {"assistant", "ai", "interviewer", "aimessage"}:
        tool_calls = record.get("tool_calls") or []
        normalized = normalize_tool_call_dicts(tool_calls)
        return AIMessage(content=str(content or ""), tool_calls=normalized)
    if role in {"human", "user", "candidate", "humanmessage"}:
        return HumanMessage(content=str(content or ""))
    return HumanMessage(content=str(content or ""))


def message_to_record(message: BaseMessage) -> dict[str, Any]:
    role = message.type
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    record: dict[str, Any] = {"role": role, "content": message.content or ""}
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = message.tool_call_id or ""
    if isinstance(message, AIMessage):
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            record["tool_calls"] = normalize_tool_call_dicts(calls)
    return record


def normalize_tool_call_dicts(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Convert any tool-call representation into LangChain ToolCall dicts."""
    normalized = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            if call.get("type") == "tool_call" and "name" in call:
                if str(call.get("name", "")).strip():
                    normalized.append(dict(call))
                continue
            function = call.get("function", {}) or {}
            if isinstance(function, dict) and function.get("name"):
                name = str(function.get("name", ""))
                if name.strip():
                    normalized.append(
                        {
                            "type": "tool_call",
                            "name": name,
                            "args": _decode_arguments(function.get("arguments", "")),
                            "id": str(call.get("id", "") or ""),
                        }
                    )
                continue
            # Already LangChain ToolCall-like.
            if call.get("name"):
                if str(call.get("name", "")).strip():
                    normalized.append(dict(call))
                continue
        name = getattr(call, "name", None)
        if name is None:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "")
            args = getattr(function, "arguments", "")
            call_id = getattr(call, "id", "")
        else:
            args = getattr(call, "args", {})
            call_id = getattr(call, "id", "")
        if name and str(name).strip():
            normalized.append(
                {
                    "type": "tool_call",
                    "name": str(name),
                    "args": args if isinstance(args, dict) else _decode_arguments(args),
                    "id": str(call_id or ""),
                }
            )
    return normalized


def _decode_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    import json

    try:
        value = json.loads(raw_arguments or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def to_records(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [message_to_record(message) for message in messages]


def to_messages(records: list[Any]) -> list[BaseMessage]:
    converted = []
    for record in records:
        if isinstance(record, BaseMessage):
            converted.append(record)
        elif isinstance(record, dict):
            converted.append(record_to_message(record))
        else:
            raise TypeError(
                "Conversation history must contain BaseMessage or message dict, "
                f"got {type(record).__name__}."
            )
    return converted
