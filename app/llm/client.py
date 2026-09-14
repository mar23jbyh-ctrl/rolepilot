import os
import json
import re
import time
import hashlib
import uuid
from urllib.parse import urlsplit
from pathlib import Path
from random import uniform

from dotenv import load_dotenv
from openai import OpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from langchain_core.messages import BaseMessage
from app.config import settings
from app.context.budget import prepare_request, ContextBudgetExceeded
from app.privacy import redact_messages
from app.telemetry.meter import active_meter
from app.telemetry.pricing import snapshot_price

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = (
    settings.api_key
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
)

def resolve_base_url():
    # Explicit generic config wins; nonempty defaults must not mask compatibility aliases.
    if 'base_url' in settings.model_fields_set and settings.base_url:
        return settings.base_url
    return os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or settings.base_url


base_url = resolve_base_url()

client = OpenAI(
    api_key=api_key or "",  # Allow health checks/uploads before credentials are configured.
    base_url=base_url,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,  # One retry policy, not SDK retries multiplied by our loop.
)


def _convert_tool_calls(tool_calls):
    """Preserve the assistant/tool protocol for SDK and persisted dict messages."""
    api_calls = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            function = call.get("function")
            if isinstance(function, dict):
                name, arguments = function.get("name", ""), function.get("arguments", "")
            else:
                name, arguments = call.get("name", ""), call.get("args", {})
            call_id = call.get("id", "")
        else:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "") if function else getattr(call, "name", "")
            arguments = getattr(function, "arguments", "") if function else getattr(call, "args", {})
            call_id = getattr(call, "id", "")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        api_calls.append({"id": str(call_id), "type": "function",
                          "function": {"name": str(name), "arguments": str(arguments or "")}})
    return api_calls


def _convert_messages(messages):
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
        "humanmessage": "user",
        "aimessage": "assistant",
        "systemmessage": "system",
        "toolmessage": "tool",
    }

    clean_messages = []
    for m in messages:
        if isinstance(m, BaseMessage):
            role = m.type
            role = role_map.get(role, role)
            converted = {
                "role": role,
                "content": m.content if isinstance(m.content, str) else str(m.content or ""),
            }
            tool_call_id = getattr(m, "tool_call_id", None)
            if role == "tool" and tool_call_id:
                converted["tool_call_id"] = tool_call_id
            tool_calls = getattr(m, "tool_calls", None)
            if role == "assistant" and tool_calls:
                converted["tool_calls"] = _convert_tool_calls(tool_calls)
            clean_messages.append(converted)
        elif isinstance(m, dict):
            role = m.get("role") or m.get("type")
            if not role:
                role = "user"
            role = role_map.get(str(role).lower(), str(role).lower())
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", part))
                    for part in content
                    if isinstance(part, dict)
                )
            payload = {"role": role, "content": content or ""}
            for key in ("name", "tool_call_id"):
                if m.get(key):
                    payload[key] = m[key]
            if role == "assistant" and m.get("tool_calls"):
                payload["tool_calls"] = _convert_tool_calls(m["tool_calls"])
            clean_messages.append(payload)
        else:
            clean_messages.append({"role": "user", "content": str(m)})
    return clean_messages


def _should_retry(exc):
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _create_with_retry(**kwargs):
    if not client.api_key:
        raise ValueError("llm_credentials_missing")
    budget = kwargs.pop('_budget_metadata', None)
    input_hash = kwargs.pop('_input_hash', '')
    meter = active_meter()
    logical_id = uuid.uuid4().hex
    last_error = None
    for attempt in range(1, 4):
        call_id = None
        if meter and budget:
            call_id = meter.ledger.reserve(logical_id=logical_id, attempt=attempt, node=meter.node,
                                           provider_host=urlsplit(str(client.base_url)).hostname or 'unknown',
                                           requested_model=kwargs['model'], input_hash=input_hash, metadata=budget,
                                           request_params={'output_limit_param':settings.llm_output_limit_param,
                                                           'output_limit':kwargs.get(settings.llm_output_limit_param),
                                                           'temperature':kwargs.get('temperature')})
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as exc:
            if call_id:
                meter.ledger.finish(call_id, {'usage_status':'missing'}, elapsed=time.perf_counter()-started, error_type=type(exc).__name__)
            last_error = exc
            if not _should_retry(exc) or attempt >= 3:
                exc.usage_info = {'usage_status': 'missing', 'unknown_calls': attempt}
                raise
            delay = min(4.0, (1.0 * (2 ** (attempt - 1)))) + uniform(0.0, 0.4)
            time.sleep(delay)
        except Exception as exc:
            if call_id:
                meter.ledger.finish(call_id, {'usage_status':'missing'}, elapsed=time.perf_counter()-started, error_type=type(exc).__name__)
            exc.usage_info = {'usage_status': 'missing', 'unknown_calls': attempt}
            raise
        else:
            # A failed audit commit must NOT resend a successful external call.
            if call_id:
                usage = _usage_dict(response)
                usage['pricing'] = snapshot_price(getattr(response, 'model', None))
                meter.ledger.finish(call_id, usage, elapsed=time.perf_counter()-started,
                                    actual_model=getattr(response, 'model', None), request_id=getattr(response, 'id', None))
            response.__dict__['_retry_unknown_calls'] = attempt - 1
            return response
    raise last_error


def _usage_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
            "usage_status": "missing", "unknown_calls": 1 + int(getattr(response, '_retry_unknown_calls', 0)),
        }
    if not all(type(getattr(usage, field, None)) is int and getattr(usage, field) >= 0 for field in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        return {'input_tokens': None, 'output_tokens': None, 'total_tokens': None, 'usage_status': 'missing',
                'unknown_calls': 1 + int(getattr(response, '_retry_unknown_calls', 0))}
    if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
        return {'input_tokens':None,'output_tokens':None,'total_tokens':None,'usage_status':'missing',
                'unknown_calls':1 + int(getattr(response,'_retry_unknown_calls',0)), 'usage_error':'inconsistent_token_total'}
    details = {}
    for group, field in [('prompt_tokens_details', 'cached_tokens'), ('completion_tokens_details', 'reasoning_tokens')]:
        value = getattr(getattr(usage, group, None), field, None)
        if type(value) is int and value >= 0:
            details[field] = value
    usage = {
        "usage_status": "known", "known_calls": 1,
        "unknown_calls": int(getattr(response, '_retry_unknown_calls', 0)),
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "usage_details": details,
    }
    try:
        usage["finish_reason"] = getattr(response.choices[0], "finish_reason", None)
    except Exception:
        usage["finish_reason"] = None
    return usage


def _request_kwargs(clean_messages, tools, temperature, max_tokens):
    input_hash = hashlib.sha256(json.dumps({'messages':clean_messages,'tools':tools,'model':settings.model},
                                          ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    try:
        reserved, budget = prepare_request(clean_messages, tools, settings.max_output_tokens if max_tokens is None else max_tokens,
                                          settings.context_budget)
    except ContextBudgetExceeded as exc:
        meter = active_meter()
        if meter:
            meter.ledger.record_blocked(exc.metadata, node=meter.node,
                                        provider_host=urlsplit(str(client.base_url)).hostname or 'unknown',
                                        requested_model=settings.model, input_hash=input_hash)
        raise
    kwargs = {
        "model": settings.model,
        "messages": clean_messages,
        settings.llm_output_limit_param: reserved,
        '_budget_metadata': budget,
        '_input_hash': input_hash,
    }
    if settings.llm_send_temperature:
        kwargs['temperature'] = temperature
    return kwargs, budget


def chat_with_usage(messages, temperature=0.7, max_tokens=None):
    clean_messages = redact_messages(_convert_messages(messages))
    kwargs, budget = _request_kwargs(clean_messages, [], temperature, max_tokens)
    response = _create_with_retry(**kwargs)
    usage = _usage_dict(response)
    usage.update(budget=budget, actual_model=getattr(response, 'model', None), pricing=snapshot_price(getattr(response, 'model', None)))
    content = response.choices[0].message.content
    if content is None:
        content = ""
    return content, usage


def chat_with_tools(messages, tools, temperature=0.2, max_tokens=None):
    clean_messages = redact_messages(_convert_messages(messages))
    kwargs, budget = _request_kwargs(clean_messages, tools, temperature, max_tokens)
    kwargs.update(tools=tools, tool_choice="auto")
    response = _create_with_retry(**kwargs)
    usage = _usage_dict(response)
    usage.update(budget=budget, actual_model=getattr(response, 'model', None), pricing=snapshot_price(getattr(response, 'model', None)))
    message = response.choices[0].message
    return (message.content or ""), (message.tool_calls or []), usage


def chat(messages, temperature=0.7):
    content, _ = chat_with_usage(messages, temperature)
    return content


def chat_json(messages, temperature=0.2):
    raw = chat(messages, temperature)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return raw
    return raw
