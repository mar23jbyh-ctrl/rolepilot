import tiktoken


_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            _ENCODING = tiktoken.get_encoding("o200k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Uploaded text is untrusted data, even when it contains tokenizer literals.
    return len(_get_encoding().encode(str(text), disallowed_special=()))


def estimate_message_tokens(message: dict) -> int:
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(str(part) for part in content)
    return 4 + estimate_tokens(str(content))


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def truncate_text(text: str, max_tokens: int) -> str:
    encoding = _get_encoding()
    tokens = encoding.encode(str(text), disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
