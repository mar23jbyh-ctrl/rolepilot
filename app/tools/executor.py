from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import settings
from app.privacy import redact_text
from app.llm.client import chat_with_usage
from app.models.schemas import ToolResult
from app.prompts.templates import CODE_EXPLAIN_PROMPT, DYNAMIC_QUESTION_PROMPT
from app.role import role_policy_text
from app.tools.schemas import ARG_MODELS
from app.utils.json_utils import parse_json_with_retries, safe_truncate

logger = logging.getLogger(__name__)

# Bound both external-call fan-out and local worker allocation per model round.
MAX_TOOL_CALLS_PER_BATCH = 8
MAX_TOOL_WORKERS = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tavily_search(query: str, max_results: int = 5, days: int = 365) -> list[dict]:
    """M24：带时效过滤的联网检索。

    新增 `days`（时间窗）与结果侧的 `published_date` / `domain` 字段，
    供上层做"新鲜度降权 + 来源分级"。**调用次数不变**（仍是每条 query 1 次）。
    """

    import requests

    if redact_text(query) != query:
        raise ValueError("privacy_query_rejected")
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            # M24：时间窗（Tavily 对 general topic 可能忽略，因此上层还会按
            # published_date 做一次本地过滤/降权，不依赖服务端行为）
            "days": int(days),
            "search_depth": "basic",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", []) if isinstance(payload, dict) else []
    cleaned = []
    for item in results:
        url = str(item.get("url", "") or "")
        cleaned.append(
            {
                "title": str(item.get("title", ""))[:200],
                "url": url,
                "content": safe_truncate(str(item.get("content", "")), 600),
                "published_date": str(item.get("published_date", "") or ""),
                "domain": _domain_of(url),
            }
        )
    return cleaned


def _domain_of(url: str) -> str:
    """从 URL 取域名（去掉 www. 与端口）。"""

    text = str(url or "")
    match = re.match(r"^https?://([^/:]+)", text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).lower().removeprefix("www.")


def web_search_handler(arguments: dict) -> dict:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"success": False, "content": "web_search requires a non-empty query."}
    if redact_text(query) != query:
        return {"success": False, "content": "privacy_query_rejected"}
    max_results = int(arguments.get("max_results", settings.tavily_max_results))
    results = _tavily_search(query, max_results=max_results)
    if not results:
        return {"success": False, "content": f"No web results for: {query}"}
    lines = [f"Web search results for '{query}' (fetched {_now()}):"]
    for index, item in enumerate(results, 1):
        lines.append(f"[{index}] {item['title']}\n{item['content']}\nURL: {item['url']}")
    return {
        "success": True,
        "content": "\n\n".join(lines)[:6000],
        "source": " | ".join(item["url"] for item in results[:3]),
        # M24：把结构化结果一并返回，供 research_job 做分级/去重
        "raw": results,
    }


def code_explainer_handler(arguments: dict) -> dict:
    code = str(arguments.get("code", ""))
    language = str(arguments.get("language", "python"))
    focus = str(arguments.get("focus", "Explain the logic, point out problems, and show a corrected example"))
    if not code.strip():
        return {"success": False, "content": "code_explainer requires code input."}
    messages = [
        SystemMessage(content="You are a senior code reviewer."),
        HumanMessage(
            content=CODE_EXPLAIN_PROMPT.format(
                focus=focus,
                language=language,
                code=safe_truncate(code, 6000),
            )
        ),
    ]
    content, usage = chat_with_usage(messages, temperature=settings.temp_fact_check)
    return {
        "success": True,
        "content": "Code explanation (generated, no code was executed):\n" + safe_truncate(content, 5000),
        "source": "code_explainer",
        "usage": usage,
    }


def dynamic_question_handler(arguments: dict) -> dict:
    skill = str(arguments.get("skill", "")).strip()
    difficulty = str(arguments.get("difficulty", "medium"))
    category = str(arguments.get("category", "scenario"))
    avoid = arguments.get("avoid") or []
    depth_level = str(arguments.get("depth_level", "application"))
    role_policy = role_policy_text(
        arguments.get("role_profile") or {},
        depth_level,
    )
    avoid_text = "\n".join("- " + str(item) for item in avoid[:20]) or "无"
    messages = [
        SystemMessage(content="You are an interviewer question generator. Output JSON only."),
        HumanMessage(
            content=DYNAMIC_QUESTION_PROMPT.format(
                skill=skill,
                difficulty=difficulty,
                category=category,
                avoid=avoid_text,
                role_policy=role_policy,
                depth_level=depth_level,
            )
        ),
    ]
    raw, usage = chat_with_usage(messages, temperature=settings.temp_plan)
    parsed = parse_json_with_retries(raw)
    if not parsed or not parsed.get("content"):
        return {"success": False, "content": "dynamic_question failed to produce a question.", "usage": usage}
    content = str(parsed.get("content", "")).strip()
    return {
        "success": True,
        "content": f"Dynamic question ({category}/{difficulty}): {content}",
        "source": f"llm:{uuid.uuid4().hex[:8]}",
        "usage": usage,
    }


HANDLERS = {
    "WebSearch": web_search_handler,
    "CodeExplainer": code_explainer_handler,
    "DynamicQuestion": dynamic_question_handler,
}

def _arguments_to_dict(raw_arguments) -> dict | None:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        value = json.loads(raw_arguments or "{}")
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def execute_tool_call(
    name: str, raw_arguments, *, allowed_tools: set[str] | None = None
) -> ToolResult:
    handler = HANDLERS.get(name)
    model = ARG_MODELS.get(name)
    if handler is None or model is None:
        return ToolResult(
            tool=name,
            success=False,
            content=f"Unknown tool {name}.",
            error=f"Unknown tool {name}.",
        )
    if allowed_tools is not None and name not in allowed_tools:
        return ToolResult(
            tool=name, success=False, content="Tool is not allowed for this role.",
            error="tool_not_allowed",
        )
    arguments = _arguments_to_dict(raw_arguments)
    if arguments is None or set(arguments) - set(model.model_fields):
        return ToolResult(
            tool=name, success=False,
            content="Invalid tool arguments: expected a JSON object with declared fields only.",
            error="invalid_tool_arguments",
        )
    try:
        arguments = model.model_validate(arguments, strict=True).model_dump()
    except ValidationError:
        # ValidationError text contains submitted values; never reflect it back.
        return ToolResult(
            tool=name, success=False,
            content="Invalid tool arguments: schema validation failed.",
            error="invalid_tool_arguments",
        )
    required_text = {
        "WebSearch": "query",
        "CodeExplainer": "code", "DynamicQuestion": "skill",
    }[name]
    if not arguments[required_text].strip():
        return ToolResult(
            tool=name, success=False,
            content="Invalid tool arguments: required text is empty.",
            error="invalid_tool_arguments",
        )
    try:
        from app.telemetry.meter import active_meter, node_scope
        meter = active_meter()
        with node_scope(f"{meter.node}:tool:{name}" if meter else f"tool:{name}"):
            outcome = handler(arguments)
        return ToolResult(
            tool=name,
            usage=outcome.get("usage") or {},
            success=bool(outcome.get("success", True)),
            content=str(outcome.get("content", "")),
            source=str(outcome.get("source", "")),
            error=str(outcome.get("error", "")) if not outcome.get("success", True) else "",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool %s failed: %s", name, type(exc).__name__)
        label = "no external source available; answer from model knowledge and mark it accordingly"
        return ToolResult(
            tool=name,
            success=False,
            content="[degraded] " + label,
            error=f"{name} failed ({type(exc).__name__}).",
            degraded=True,
        )


def execute_tool_calls(
    tool_calls, *, allowed_tools: set[str] | None = None
) -> list[ToolResult]:
    prepared = []
    for call in tool_calls:
        if isinstance(call, dict):
            if call.get("type") == "tool_call" and call.get("name"):
                name = str(call.get("name", ""))
                raw_args = call.get("args", {})
            else:
                function = call.get("function", {}) or {}
                name = function.get("name", "") if isinstance(function, dict) else getattr(function, "name", "")
                raw_args = function.get("arguments", "") if isinstance(function, dict) else getattr(function, "arguments", "")
        else:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "")
            raw_args = getattr(function, "arguments", "")
        prepared.append((str(name or ""), raw_args))
    if len(prepared) > MAX_TOOL_CALLS_PER_BATCH:
        # Refuse the whole batch, rather than silently executing its first calls.
        return [
            ToolResult(
                tool=name, success=False,
                content="Tool batch limit exceeded; no tools were executed.",
                error="tool_batch_limit_exceeded",
            )
            for name, _ in prepared
        ]
    if len(prepared) <= 1:
        return [
            execute_tool_call(name, raw_args, allowed_tools=allowed_tools) for name, raw_args in prepared
        ]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(MAX_TOOL_WORKERS, len(prepared))) as pool:
        from app.telemetry.meter import submit_with_context
        futures = [submit_with_context(pool, execute_tool_call, name, args, allowed_tools=allowed_tools)
                   for name, args in prepared]
        return [future.result() for future in futures]
