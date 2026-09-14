"""Offline benchmark of fixed synthetic role/question guard cases.

No model/search call. Cases requiring model arbitration remain unverified and
are excluded from deterministic-path metrics. This is not universal accuracy.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.guard import BatchArbiter, audit_questions  # noqa: E402
from app.role import build_role_profile  # noqa: E402


def evaluate_cases(args):
    if args.mode != 'offline':
        raise ValueError('This benchmark supports offline mode only')
    data = json.loads((ROOT/'tests/fixtures/guard/cases.json').read_text(encoding='utf-8'))
    cases = [case for case in data['cases'] if not args.only or case['role'] == args.only]
    if args.limit:
        cases = cases[:args.limit]
    by_role = {}
    for case in cases:
        by_role.setdefault(case['role'], []).append(case)
    results = []
    for role, role_cases in sorted(by_role.items()):
        payload = data['profiles'][role]
        profile = build_role_profile('', '', payload)
        questions = [dict(case['question']) for case in role_cases]
        audit = audit_questions(questions, profile, jd_items=payload.get('jd_items') or [],
                                resume_projects=payload.get('resume_projects') or [],
                                reference_ids=payload.get('reference_ids') or [], arbiter=BatchArbiter(max_calls=0))
        records = {int(record['question_id']): record for record in audit['audit']}
        for case in role_cases:
            record = records.get(int(case['question']['id']), {})
            results.append({'id': case['id'], 'role': role, 'category': case['category'],
                            'expected': case['expected'], 'expected_stage': case['expected_stage'],
                            'actual': 'drop' if record.get('verdict') == 'drop' else 'keep',
                            'actual_stage': record.get('stage', ''),
                            'pending_live': (record.get('raw') or {}).get('source') == 'budget_exhausted'})
    return {'mode': 'offline', 'arbiter_calls': 0, 'roles': sorted(by_role), 'results': results}


def _metrics(results):
    judgeable = [item for item in results if not item['pending_live']]
    false_drop = [item for item in judgeable if item['expected'] == 'keep' and item['actual'] == 'drop']
    missed = [item for item in judgeable if item['expected'] == 'drop' and item['actual'] == 'keep']
    mismatch = [item for item in judgeable if item['actual_stage'] != item['expected_stage']]
    return {'judgeable': len(judgeable), 'pending_live': len(results) - len(judgeable),
            'false_drop': false_drop, 'missed': missed, 'stage_mismatch': mismatch,
            'stage_hit_rate': (len(judgeable) - len(mismatch)) / len(judgeable) if judgeable else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline'], default='offline')
    parser.add_argument('--only', default='')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    result = evaluate_cases(args)
    metrics = _metrics(result['results'])
    print(json.dumps({'roles': len(result['roles']), 'cases': len(result['results']),
                      'arbiter_calls': result['arbiter_calls'], **metrics}, ensure_ascii=False))
    return int(bool(metrics['false_drop'] or metrics['missed'] or metrics['stage_mismatch']))


if __name__ == '__main__':
    raise SystemExit(main())
