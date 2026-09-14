"""Optional, model-matched price snapshots. Never a provider billing statement."""
from datetime import date
from decimal import Decimal
import hashlib
import json
import re
from urllib.parse import urlsplit

from app.config import settings


def snapshot_price(actual_model):
    """Freeze the current policy at call time; legacy records are not re-priced."""
    values = (settings.cost_input_per_1m, settings.cost_output_per_1m)
    currency = settings.cost_currency.strip().upper()
    source = settings.cost_price_source.strip()
    if (any(value is None for value in values) or not settings.cost_model.strip()
            or not re.fullmatch(r"[A-Z]{3}", currency) or not source):
        return {"status": "unconfigured"}
    try:
        parsed = urlsplit(source)
    except ValueError:
        return {"status": "invalid_price_source"}
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return {"status": "invalid_price_source"}
    try:
        effective = date.fromisoformat(settings.cost_price_effective_date).isoformat()
    except (TypeError, ValueError):
        return {"status": "invalid_price_date"}
    if not actual_model or actual_model != settings.cost_model.strip():
        return {"status": "model_mismatch"}
    policy = {"status": "priced", "model": actual_model, "currency": currency,
              "input_price_per_1m": values[0], "output_price_per_1m": values[1],
              "source": source, "effective_date": effective}
    policy["version"] = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
    return policy


def estimate_price(input_tokens, output_tokens, snapshot):
    if snapshot.get("status") != "priced" or input_tokens is None or output_tokens is None:
        return None
    return float((Decimal(input_tokens) * Decimal(str(snapshot["input_price_per_1m"]))
                  + Decimal(output_tokens) * Decimal(str(snapshot["output_price_per_1m"]))) / Decimal(1000000))
