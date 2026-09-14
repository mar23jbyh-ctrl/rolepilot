import random
import time


def call_with_retry(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    retry_predicate=None,
    on_retry=None,
):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transport layer decides
            last_error = exc
            if retry_predicate is not None and not retry_predicate(exc):
                raise
            if attempt >= attempts:
                break
            if on_retry is not None:
                on_retry(attempt)
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            time.sleep(delay + random.uniform(0.0, 0.4))
    raise last_error
