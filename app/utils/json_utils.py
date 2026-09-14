import json
import re


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_object_candidate(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _repair_unquoted_keys(text: str) -> str:
    return re.sub(
        r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:",
        lambda match: match.group(1) + '"' + match.group(2) + '":',
        text,
    )


def extract_json_object(text: str) -> dict | None:
    candidates = []
    cleaned = _strip_code_fence(text)
    if cleaned:
        candidates.append(cleaned)
    raw_object = _extract_object_candidate(cleaned)
    if raw_object:
        candidates.append(raw_object)
    for payload in candidates:
        for repaired in (payload, _remove_trailing_commas(payload)):
            try:
                value = json.loads(repaired)
                if isinstance(value, dict):
                    return value
            except (json.JSONDecodeError, TypeError):
                continue
    for payload in candidates:
        try:
            value = json.loads(_repair_unquoted_keys(payload))
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_json_with_retries(raw: str, attempts: int = 3) -> dict | None:
    for _ in range(attempts):
        parsed = extract_json_object(raw)
        if parsed is not None:
            return parsed
    return None


def safe_truncate(text: str, limit: int = 3000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"
