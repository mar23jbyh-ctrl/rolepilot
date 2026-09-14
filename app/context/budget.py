"""Hard preflight estimate at the single SDK boundary; never truncate answers."""
import json
import logging
from app.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

class ContextBudgetExceeded(ValueError):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__('context_budget_exceeded')

def prepare_request(messages, tools, requested_output, budget):
    # JSON serialization includes roles, function arguments/IDs, schemas and results.
    # o200k is NOT the provider tokenizer. A safety margin reduces, not eliminates, drift.
    serialized = json.dumps({'messages': messages, 'tools': tools or []}, ensure_ascii=False)
    estimate = estimate_tokens(serialized)
    margin = max(256, int(estimate * .1))
    available = int(budget) - estimate - margin
    reserved = min(int(requested_output), available)
    metadata = {'tokenizer': 'o200k_base_estimate', 'estimated_input_tokens': estimate,
                'safety_margin_tokens': margin, 'context_budget': int(budget),
                'requested_output_tokens': int(requested_output),
                'reserved_output_tokens': max(0, reserved), 'trimmed_messages': 0,
                'action': 'accepted' if reserved == requested_output else 'output_cap_reduced'}
    if reserved < min(512, int(requested_output)) or requested_output <= 0:
        metadata['action'] = 'rejected'
        logger.warning('context_budget_rejected estimate=%s budget=%s', estimate, budget)
        raise ContextBudgetExceeded(metadata)
    return reserved, metadata
