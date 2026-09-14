from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm.client import chat_with_usage
from app.models.schemas import CompressedTurn
from app.prompts.templates import SUMMARY_PROMPT
from app.prompts.templates import GLOBAL_ROLE_POLICY
from app.utils.json_utils import parse_json_with_retries
from app.utils.tokens import estimate_messages_tokens


def history_tokens(conversation_history: list[dict]) -> int:
    return estimate_messages_tokens(conversation_history or [])


def should_compress(state) -> bool:
    history = state.get("conversation_history", []) or []
    return history_tokens(history) > settings.compress_threshold or len(history) > 80


def summarize_turn(
    question: str,
    answer: str,
    assessment: dict,
    question_id: int,
    difficulty_change: str = "",
) -> tuple[CompressedTurn | None, dict]:
    """把一轮问答压缩成结构化摘要，同时返回这次调用的 usage（供 compress 节点记账）。"""
    messages = [
        SystemMessage(content="You compress an interview round into structured JSON."),
        HumanMessage(
            content=SUMMARY_PROMPT.format(
                role_policy=GLOBAL_ROLE_POLICY,
                question_id=question_id,
                question=str(question)[:1200],
                answer=str(answer),
                assessment=json.dumps(assessment, ensure_ascii=False),
            )
        ),
    ]
    try:
        raw, usage = chat_with_usage(messages, temperature=settings.temp_parse)
    except Exception as exc:
        from app.context.budget import ContextBudgetExceeded
        if isinstance(exc, ContextBudgetExceeded):
            return None, {'usage_status': 'not_called', 'unknown_calls': 0, 'budget': exc.metadata}
        return None, getattr(exc, 'usage_info', {'usage_status': 'missing', 'unknown_calls': 1})
    parsed = parse_json_with_retries(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str) or not parsed["summary"].strip():
        return None, usage
    try:
        summary = CompressedTurn(
            question_id=question_id,
            question=str(question)[:300],
            score=float(parsed.get("score", assessment.get("score", 5.0))),
            missed_points=list(
                parsed.get("missed_points", assessment.get("missed_points", [])) or []
            ),
            follow_up_reason=str(
                parsed.get("follow_up_reason", assessment.get("follow_up_reason", ""))
            ),
            difficulty_change=str(parsed.get("difficulty_change", difficulty_change)),
            summary=str(parsed.get("summary", str(answer)[:200])),
        )
        return summary, usage
    except Exception:
        return None, usage


def _difficulty_change_text(state) -> str:
    events = state.get("difficulty_events", []) or []
    if not events:
        return ""
    last = events[-1]
    return f"{last.get('previous_difficulty')}->{last.get('new_difficulty')}"


def _prune_summarized_prefix(history, turn_records, summaries, summarized_ids) -> list[dict]:
    """Remove only exact, covered Q/A pairs; unknown messages remain intact.

    History message conversion does not retain question IDs. Matching the full
    recorded question/answer is deliberately conservative, including ambiguous
    repeated pairs and messages from tool/explanation paths.
    """
    covered_ids = {
        item.get("question_id")
        for item in summaries
        if str(item.get("summary", "") or "").strip()
    }.intersection(summarized_ids)
    pairs: dict[tuple[str, str], list[int]] = {}
    for record in turn_records:
        key = (str(record.get("question", "")), str(record.get("answer", "")))
        pairs.setdefault(key, []).append(record.get("question_id"))
    keep = max(0, int(settings.keep_recent_messages))
    boundary = max(0, len(history) - keep)
    removed: dict[tuple[str, str], int] = {}
    cursor = 0
    while cursor + 2 <= boundary:
        question, answer = history[cursor : cursor + 2]
        if question.get("role") != "assistant" or answer.get("role") != "user":
            break
        key = (str(question.get("content", "")), str(answer.get("content", "")))
        ids = pairs.get(key, [])
        if not ids or not all(question_id in covered_ids for question_id in ids):
            break
        if removed.get(key, 0) >= len(ids):
            break
        removed[key] = removed.get(key, 0) + 1
        cursor += 2
    return history[cursor:]


def compress_history(state) -> dict:
    history = list(state.get("conversation_history", []) or [])
    summaries = list(state.get("summaries", []) or [])
    summarized_ids = list(state.get("summarized_ids", []) or [])
    turn_records = list(state.get("turn_records", []) or [])

    pending = [
        record
        for record in turn_records
        if int(record.get("question_id", -1)) not in summarized_ids
    ]
    created = 0
    failed = False
    usage_records: list[dict] = []
    for record in pending:
        if created >= 2 or history_tokens(history) <= settings.compress_threshold:
            break
        question_id = int(record.get("question_id", 0))
        summary, usage = summarize_turn(
            question=str(record.get("question", "")),
            answer=str(record.get("answer", "")),
            assessment=record.get("assessment", {}) or {},
            question_id=question_id,
            difficulty_change=_difficulty_change_text(state),
        )
        usage_records.append(
            {
                **usage,
                "node": "compress",
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        )
        if summary is None:
            # M29：压缩失败不得吞掉这一轮——不写 summarized_ids，
            # 保留原文等待下一轮重试（原实现会标记为"已压缩"但没写摘要，
            # 加上 conversation_history 被裁剪，该轮就永久失明了）。
            failed = True
            break
        summaries.append(summary.model_dump())
        summarized_ids.append(question_id)
        created += 1

    recent = history if failed else _prune_summarized_prefix(
        history, turn_records, summaries, summarized_ids
    )
    return {
        "conversation_history": recent,
        "summaries": summaries,
        "summarized_ids": summarized_ids,
        "usage_records": usage_records,
    }


def summaries_text(summaries: list[dict], limit: int | None = 20) -> str:
    if not summaries:
        return "暂无历史摘要"
    lines = []
    for item in (summaries if limit is None else summaries[-limit:]):
        lines.append(
            f"Q{item.get('question_id')} | score={item.get('score')} | "
            f"missed={item.get('missed_points')} | reason={item.get('follow_up_reason')} | "
            f"difficulty={item.get('difficulty_change')} | {item.get('summary')}"
        )
    return "\n".join(lines)
