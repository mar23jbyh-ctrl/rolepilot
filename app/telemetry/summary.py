"""One accounting contract for graph records, HTTP, SQLite JSON and reports."""
from decimal import Decimal
from app.telemetry.pricing import estimate_price

TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'total_tokens')


def state_usage_records(state):
    """SDK ledger is authoritative when present; historical node records remain readable."""
    if state.get('llm_ledger_version') == 'ledger-v1':
        return state.get('llm_usage_records') or []
    return state.get('usage_records') or []

def usage_record(node, usage):
    usage = dict(usage or {})
    # Legacy numeric records remain readable; they are not retrospectively re-billed.
    status = usage.get('usage_status') or ('known' if all(isinstance(usage.get(k), int) for k in TOKEN_FIELDS) else 'missing')
    return {**usage, 'node': node, 'usage_status': status,
            **{k: usage.get(k) if status == 'known' else None for k in TOKEN_FIELDS}}

def merge_usage(total, usage):
    row = usage_record('', usage)
    for key in TOKEN_FIELDS:
        total[key] = int(total.get(key) or 0) + int(row.get(key) or 0)
    total['known_calls'] = int(total.get('known_calls') or 0) + (int(row.get('known_calls', 1)) if row['usage_status'] == 'known' else 0)
    total['unknown_calls'] = int(total.get('unknown_calls') or 0) + int(row.get('unknown_calls') or (row['usage_status'] != 'known'))
    total['usage_status'] = 'known' if total['unknown_calls'] == 0 else 'partial'
    total.setdefault('budget_events', []).extend(row.get('budget_events') or ([row['budget']] if row.get('budget') else []))
    total['finish_reason'] = row.get('finish_reason')
    total.setdefault('pricing_records', []).extend(row.get('pricing_records') or [row])
    # Keep the known subtotal separately when a retry did not return usage.
    total['known_usage'] = {k: total[k] for k in TOKEN_FIELDS}
    return total

def summarize_usage(records):
    known = {k: 0 for k in TOKEN_FIELDS}; known_calls = unknown = estimated = 0
    for raw in records or []:
        row = usage_record('', raw)
        if row['usage_status'] == 'not_called':
            continue
        if row['usage_status'] == 'known':
            known_calls += int(row.get('known_calls', 1))
            for k in TOKEN_FIELDS: known[k] += int(row.get(k) or 0)
            unknown += int(row.get('unknown_calls') or 0)
        elif row['usage_status'] == 'partial':
            known_calls += int(row.get('known_calls') or 0)
            unknown += int(row.get('unknown_calls') or 1)
            for k in TOKEN_FIELDS: known[k] += int((raw.get('known_usage') or {}).get(k) or 0)
        else:
            unknown += int(row.get('unknown_calls') or 1)
            estimated += row['usage_status'] == 'estimated'
    # Only frozen call-time snapshots can price a record. Never consult today's rates.
    priced = []; unpriced = 0; policies = {}; price_rows = []
    for raw in records or []:
        price_rows.extend(raw.get('pricing_records') or [raw])
    for raw in price_rows:
        row = usage_record('', raw)
        if row['usage_status'] == 'not_called':
            continue
        policy = raw.get('pricing') or {}
        amount = estimate_price(row.get('input_tokens'), row.get('output_tokens'), policy)
        if amount is None:
            unpriced += 1
        else:
            priced.append((Decimal(str(amount)), policy['currency']))
            policies[policy.get('version') or repr(policy)] = policy
    currencies = {currency for _, currency in priced}
    one_currency = len(currencies) == 1
    known_cost = float(sum((amount for amount, _ in priced), Decimal(0))) if one_currency else None
    complete = unknown == 0
    pricing_complete = bool(priced) and unpriced == 0 and one_currency and complete
    single_policy = next(iter(policies.values())) if len(policies) == 1 else {}
    return {'schema_version': 'usage-v1', 'usage_complete': complete,
            'known_calls': known_calls, 'unknown_calls': unknown, 'estimated_calls': estimated,
            'known_total': known['total_tokens'], 'known_input': known['input_tokens'], 'known_output': known['output_tokens'],
            'total': known['total_tokens'] if complete else None,
            'total_input': known['input_tokens'] if complete else None,
            'total_output': known['output_tokens'] if complete else None,
            'cost': known_cost if pricing_complete else None, 'known_cost_estimate': known_cost,
            'cost_status': ('configuration_estimate' if pricing_complete else
                            'mixed_currencies' if len(currencies) > 1 else
                            'partial_unknown' if unknown else 'unconfigured_or_unmatched'),
            'pricing_complete': pricing_complete,
            'currency': next(iter(currencies)) if one_currency else None, 'provider_billed_cost': None,
            'input_price_per_1m': single_policy.get('input_price_per_1m'),
            'output_price_per_1m': single_policy.get('output_price_per_1m'),
            'price_snapshots': list(policies.values())}
