from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage


def _message_records(history: Any) -> list[dict[str, str]]:
    """Normalize graph state history for the UI.

    Live graph state stores LangChain BaseMessage objects, while sessions loaded
    from SQLite store plain dict records. Both must serialize without error
    (this was the cause of the history 500).
    """
    records: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, BaseMessage):
            role = item.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            records.append({"role": role, "content": str(item.content or "")})
        elif isinstance(item, dict):
            role = str(item.get("role") or "assistant")
            if role in {"human", "candidate", "user"}:
                role = "user"
            elif role in {"ai", "interviewer", "assistant"}:
                role = "assistant"
            records.append({"role": role, "content": str(item.get("content", "") or "")})
    return records


def _usage_total(state: dict[str, Any]) -> int | None:
    if state.get('usage_schema_version') == 'usage-v1':
        from app.telemetry.summary import summarize_usage, state_usage_records
        return summarize_usage(state_usage_records(state))['total']
    return sum(
        int(item.get("total_tokens", 0) or 0)
        for item in (state.get("usage_records", []) or [])
        if isinstance(item, dict)
    )


def session_payload(
    state: dict[str, Any],
    session_id: str,
    name: str = "",
) -> dict[str, Any]:
    """Convert a service result into a JSON-safe payload without touching logic."""
    safe_state = state if isinstance(state, dict) else {}
    report = safe_state.get("evaluation_report") or None
    done = bool(report)
    # M23：把"系统识别到的岗位"与题量暴露给前端，便于用户核对与反馈
    role_profile = safe_state.get("role_profile") or {}
    plan = safe_state.get("question_plan") or []
    effective = 0
    for item in safe_state.get("assessments") or []:
        if item.get("is_follow_up") or item.get("score_error"):
            continue
        if item.get("score") is not None:
            effective += 1
    return {
        "session_id": session_id,
        "name": name or safe_state.get("session_name", "") or "",
        "status": "failed" if safe_state.get("error") else ("completed" if done else "interviewing"),
        "done": done,
        "question": "" if done else str(safe_state.get("current_question", "") or ""),
        "question_version": int(safe_state.get("question_version", 0)),
        "messages": _message_records(safe_state.get("conversation_history") or []),
        "report": report,
        "usage_total": _usage_total(safe_state),
        "usage_summary": _usage_summary(safe_state),
        "job_title": str(role_profile.get("job_title", "") or ""),
        "domain": str(role_profile.get("domain", "") or ""),
        "industry": str(role_profile.get("industry", "") or ""),
        "plan_question_count": int(safe_state.get("plan_question_count") or len(plan)),
        "effective_sample_count": effective,
    }


def _usage_summary(state):
    from app.telemetry.summary import summarize_usage, state_usage_records
    return summarize_usage(state_usage_records(state))
