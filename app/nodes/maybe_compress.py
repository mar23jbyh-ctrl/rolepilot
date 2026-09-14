from app.context.manager import compress_history, should_compress


def maybe_compress(state) -> dict:
    if not should_compress(state):
        return {}
    return compress_history(state)
