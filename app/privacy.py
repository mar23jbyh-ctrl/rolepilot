"""Local data minimization, not a complete PII detector or anonymity guarantee.

Cloud models still receive interview content. Only recognizable contact/identity
patterns are masked; arbitrary names, employers and personal stories can remain.
"""
from __future__ import annotations

import json
import re

MASK = "[隐私已隐藏]"
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_CN_ID = re.compile(r"(?<!\w)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\w)")
_CN_MOBILE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d(?:[- ]?\d){8}(?!\d)")
_LABELED = re.compile(
    r"(?im)^(\s*(?:姓名|候选人姓名|联系地址|家庭住址|身份证(?:号码)?|护照(?:号码)?|"
    r"电话|手机|联系方式|邮箱|full[ _]name|name|address|phone|mobile|tel(?:ephone)?|"
    r"email|passport|api[ _]key|password|密码)\s*[:：=]\s*)[^\r\n]+"
)
_PRIVATE_KEYS = frozenset({
    "姓名", "候选人姓名", "邮箱", "电话", "手机", "联系地址", "家庭住址", "身份证", "身份证号码",
    "full_name", "candidate_name", "email", "phone", "mobile", "telephone", "address", "passport",
    "api_key", "password", "token",
})


def redact_text(text: str) -> str:
    """Preserve JSON syntax when content/Function Calling arguments are JSON."""
    if not isinstance(text, str):
        return text
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            pass
        else:
            redacted = redact_value(value)
            return text if redacted == value else json.dumps(redacted, ensure_ascii=False)
    result = _LABELED.sub(lambda match: match.group(1) + MASK, text)
    for pattern in (_EMAIL, _CN_ID, _CN_MOBILE):
        result = pattern.sub(MASK, result)
    return result


def redact_value(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: (MASK if str(key).lower() in _PRIVATE_KEYS else redact_value(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def redact_messages(messages: list[dict]) -> list[dict]:
    """Mask content and function arguments, not protocol IDs or function names."""
    result = []
    for message in messages:
        item = dict(message)
        item["content"] = redact_text(str(item.get("content") or ""))
        if item.get("tool_calls"):
            item["tool_calls"] = [
                {**call, "function": {**call["function"],
                                      "arguments": redact_text(call["function"].get("arguments", ""))}}
                for call in item["tool_calls"]
            ]
        result.append(item)
    return result
