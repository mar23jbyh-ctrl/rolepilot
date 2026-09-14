"""Validate model observations before deterministic score computation."""
import re

REQUIRED = ('scoreable', 'dimension_levels', 'evidence', 'missing_points',
            'hallucination_or_conflict', 'next_action', 'confidence')
ACTIONS = {'follow_up', 'next_question', 'explain', 'end'}

def _text(value):
    return re.sub(r'\s+', ' ', str(value)).strip().casefold()

def normalize_assessment(parsed, answer, rubric, intent='answer'):
    source = parsed if isinstance(parsed, dict) else {}
    errors = ['missing:' + k for k in REQUIRED if k not in source]
    if not isinstance(source.get('scoreable'), bool): errors.append('scoreable_not_bool')
    if not isinstance(source.get('hallucination_or_conflict'), bool): errors.append('conflict_not_bool')
    confidence = source.get('confidence')
    action_value = source.get('next_action')
    if not isinstance(confidence, str) or confidence not in {'low', 'medium', 'high'}:
        errors.append('invalid_confidence')
    if not isinstance(action_value, str) or action_value not in ACTIONS:
        errors.append('invalid_next_action')
    if not isinstance(source.get('missing_points'), list) or any(not isinstance(v, str) for v in source.get('missing_points', [])):
        errors.append('invalid_missing_points')
    levels = source.get('dimension_levels') if isinstance(source.get('dimension_levels'), dict) else {}
    evidence = source.get('evidence') if isinstance(source.get('evidence'), dict) else {}
    if not isinstance(source.get('dimension_levels'), dict): errors.append('invalid_levels')
    if not isinstance(source.get('evidence'), dict): errors.append('invalid_evidence')
    allowed = {str(x['key']) for x in rubric if isinstance(x, dict) and x.get('key')}
    kept, quotes = {}, {}
    for key, level in levels.items():
        if key not in allowed: errors.append('wrong_rubric_key:' + str(key)); continue
        if type(level) is not int or level not in range(1, 6):
            errors.append('invalid_level:' + key); continue
        raw = evidence.get(key, [])
        if not isinstance(raw, list) or any(not isinstance(v, str) for v in raw):
            errors.append('invalid_quotes:' + key); continue
        # Actual excerpts, not evaluator-invented paraphrases, support a high level.
        valid = [v.strip() for v in raw if len(v.strip()) >= 2 and _text(v) in _text(answer)]
        if len(valid) != len(raw): errors.append('unsupported_evidence:' + key)
        if level >= 4 and not valid: errors.append('high_level_without_evidence:' + key)
        kept[key] = level; quotes[key] = valid
    scoreable = source.get('scoreable') is True and bool(answer.strip()) and intent not in {'off_topic', 'request_stop', 'request_clarification'}
    if scoreable and (not kept or not any(quotes.values())): errors.append('no_scoring_evidence')
    conflict = source.get('hallucination_or_conflict') is True
    if conflict:
        # An unverified/conflicting story cannot earn a high practice score.
        kept = {k: min(v, 2) for k, v in kept.items()}
    if errors or not scoreable: kept = {}
    action = action_value if isinstance(action_value, str) and action_value in ACTIONS else 'next_question'
    return {'scoreable': scoreable and not errors, 'dimension_levels': kept, 'evidence': quotes,
            'missing_points': source.get('missing_points') if isinstance(source.get('missing_points'), list) else [],
            'missed_points': source.get('missing_points') if isinstance(source.get('missing_points'), list) else [],
        'covered_points': list(dict.fromkeys(q for values in quotes.values() for q in values)),
            'hallucination_or_conflict': conflict, 'next_action': action,
            'should_follow_up': action == 'follow_up',
            'confidence': confidence if isinstance(confidence, str) and confidence in {'low','medium','high'} else 'low',
            'protocol_errors': errors, 'unscoreable_reason': 'invalid_protocol' if errors else ('insufficient_or_off_topic' if not scoreable else '')}
