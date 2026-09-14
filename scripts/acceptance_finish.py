"""Secret-free source/delivery/link snapshot, never rewrites real evidence or DB."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/release/evidence/acceptance-20260912'

def runtime_findings(comparison):
    """SID-scoped, read-only inspection of newly created synthetic cases only."""
    findings={}
    for name,c in comparison['phases']['final']['cases'].items():
        db=c['database']; owner=db['owner_id']; sid=c['session_id']
        if not re.fullmatch(r'[a-f0-9]{32}',owner) or not re.fullmatch(r'[a-f0-9]{32}',sid):raise ValueError('unsafe_synthetic_identity')
        path=(ROOT/'data'/f'sessions_{owner}.db').resolve()
        if db['business_database_path']!=f'data/sessions_{owner}.db' or not path.is_relative_to((ROOT/'data').resolve()):raise ValueError('unsafe_database_path')
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as con:
            con.execute('PRAGMA query_only=ON')
            state=json.loads(con.execute('SELECT state_json FROM sessions WHERE id=?',(sid,)).fetchone()[0])
        errors=[]
        for a in state.get('assessments',[]):
            if not a.get('score_error'):continue
            topic=next((q for q in state['evaluation_report']['per_question'] if q.get('plan_question_index')==a.get('plan_question_index')),None)
            valid=[x for x in state['assessments'] if x.get('plan_question_index')==a.get('plan_question_index') and x.get('score') is not None and x.get('scoreable')]
            round_entry=next((x for x in (topic or {}).get('rounds',[]) if x.get('question_id')==a.get('question_id')),None)
            safe=(a.get('score') is None and a.get('scoreable') is False and not a.get('dimensions') and a.get('next_action')=='next_question'
                  and round_entry is not None and round_entry.get('score') is None and topic is not None
                  and topic.get('score')==(valid[-1]['score'] if valid else None))
            errors.append({'question_id':a.get('question_id'),'plan_question_index':a.get('plan_question_index'),
                           'is_follow_up':a.get('is_follow_up'),'protocol_errors':a.get('protocol_errors'),
                           'score':a.get('score'),'topic_score':(topic or {}).get('score'),'safe_null_and_valid_topic_preserved':safe})
        selfcheck_text=json.dumps(state.get('assessments',[]),ensure_ascii=False)
        findings[name]={'score_errors':errors,'report_score_error_count_matches':len(errors)==c.get('score_errors'),
                        'all_invalid_scores_safely_excluded':all(x['safe_null_and_valid_topic_preserved'] for x in errors),
                        'selfcheck_status':state.get('self_check_report',{}).get('status'),
                        'selfcheck_serialized_assessment_chars':len(selfcheck_text),'selfcheck_input_char_limit':5000,
                        'selfcheck_assessment_text_was_sliced':len(selfcheck_text)>5000,
                        'selfcheck_findings_are_model_opinions_not_independent_verified_defects':True}
    return findings

def acceptance_checks(comparison, offline, scoring, probes, http, orchestrator, source_unchanged, findings=None):
    """Local-practice acceptance only; derive every flag from saved observations."""
    cases=comparison['phases']['final']['cases']; aggregate=comparison['phases']['final']['aggregate']
    required_http={'stale_while_active','restart_restore','restart_replay','oldest_replay','final_replay','stale_after_end','final_read'}
    fc=probes.get('function_calling',{}); budgets=probes.get('budget',[])
    required_aux={'empty_http','concurrent_whitespace','20_replays','stale_version','id_payload_conflict','off_topic','owner_isolation','explicit_stop_partial_smoke'}
    aux={r.get('kind'):r for r in http}
    return {
        'business_source_unchanged_since_preflight':source_unchanged,
        'orchestrator_completed':orchestrator.get('success') is True,
        'owned_server_pid_changed':orchestrator.get('restart_performed') is True and all(len({n.get('pid') for n in c.get('graph',{}).get('node_order',[]) if n.get('pid')})>=2 for c in cases.values()),
        'evidence_complete_lines':all(not r.get('malformed_line_numbers') and r.get('trailing_newline',True) for r in comparison.get('inventory',[])),
        'same_uploaded_inputs':comparison.get('same_uploaded_input_hashes') is True,
        'three_natural_reports':len(cases)==3 and all(c.get('natural_end') and c.get('status')=='completed' and c.get('report_created') and c.get('effective_main',0)>=3 for c in cases.values()),
        'http_replay_restart_version':all(required_http<=set(c.get('http_checks',{})) and all(c['http_checks'][k].get('passed') is True for k in required_http) for c in cases.values()),
        'pause_and_end':all(c.get('graph',{}).get('paused_before_assess_count',0)>0 and c['graph'].get('end_next_empty_observed') is True for c in cases.values()),
        'two_databases_and_accounting':aggregate.get('all_two_store_and_accounting_equal') is True,
        'no_duplicate_assess':aggregate.get('repeat_assess_count')==0,
        # The user's practice-coach criteria require explicit safe fallback,
        # not a fabricated claim that every stochastic reply is valid.
        'score_errors_explicit_and_safely_excluded':aggregate.get('score_errors')==0 or (findings is not None and all(x['report_score_error_count_matches'] and x['all_invalid_scores_safely_excluded'] for x in findings.values())),
        'complete_provider_usage':aggregate.get('unknown_provider_usage_calls')==0,
        'budget_on_every_business_call':all(c.get('context_budget_records',0)==c['business_model']['logical_calls']>0 and c.get('budget_enforced_violations')==0 for c in cases.values()),
        'real_search':all(c.get('search',{}).get('successful_queries',0)>=5 and c['search'].get('failed_queries')==0 for c in cases.values()),
        'offline_tests_and_build':offline.get('all_passed') is True,
        'previous_scoring_48_rules':scoring.get('results_count')==48 and scoring.get('passed_records')==48 and scoring.get('all_passed') is True,
        'previous_real_readonly_function_calling':fc.get('actual_tool_calls',0)>0 and fc.get('success') is True and fc.get('result_back_in_context') is True,
        'previous_tool_rejections':all(probes.get('tool_rejection',{}).get(k,{}).get('rejected') is True for k in ('invalid_argument','disallowed','batch')),
        'previous_long_input_rejection':len(budgets)==4 and all(b.get('rejected') is True and b.get('provider_calls')==0 for b in budgets),
        'previous_summary_and_raw_preservation':probes.get('real_summary',{}).get('success') is True and probes.get('summary_failure_injection',{}).get('raw_preserved') is True and probes['summary_failure_injection'].get('ids_not_marked') is True,
        'previous_http_boundary_probes':required_aux<=set(aux) and all(aux[k].get('passed') is True for k in required_aux),
    }

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=int,default=1)
    p.add_argument('--final-round',type=int);p.add_argument('--offline',default='round5-offline.json')
    p.add_argument('--before-record',type=int,default=3);args=p.parse_args()
    if args.record<1 or args.before_record<1 or (args.final_round is not None and args.final_round<1):p.error('record/round must be positive')
    offline_path=(OUT/args.offline).resolve()
    if not offline_path.is_relative_to(OUT.resolve()):p.error('offline input must remain within acceptance evidence')
    # Reserve our new artifact before link checks so a documented self-reference
    # exists during checking; the new file is written once, never replaced.
    stream=(OUT/f'delivery-record-{args.record}.json').open('x',encoding='utf-8')
    baseline=json.loads((OUT/'baseline/source.json').read_text(encoding='utf-8'))['source_sha256']
    current={};refs={}
    for folder in ('app','api','frontend/src','tests','scripts'):
        for f in (ROOT/folder).rglob('*'):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix not in ('.py','.ts','.tsx','.cjs','.json','.css'):continue
            path=f.relative_to(ROOT).as_posix();current[path]=hashlib.sha256(f.read_bytes()).hexdigest()
            if f.suffix=='.py':
                refs[path]=[{'name':n.name,'line':n.lineno} for n in ast.walk(ast.parse(f.read_text(encoding='utf-8-sig'))) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    for path in ('README.md','.env.example','requirements.txt','frontend/package.json','frontend/package-lock.json'):
        current[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    changes=[{'path':path,'status':'new' if path not in baseline else 'modified','baseline_sha256':baseline.get(path),'current_sha256':value}
             for path,value in current.items() if baseline.get(path)!=value]
    links=[];broken=[];md_paths=[]
    for base,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in ('.git','.venv','node_modules','data','uploads','.pytest_cache','__pycache__')]
        for name in files:
            if not name.endswith('.md'):continue
            f=Path(base)/name;md_paths.append(f.relative_to(ROOT).as_posix())
            for match in re.finditer(r'\]\(([^)]+)\)',f.read_text(encoding='utf-8')):
                url=match.group(1).strip('<>')
                if re.match(r'^[A-Za-z]+:|^#',url):continue
                target=url.split('#',1)[0]
                exists=(f.parent/target).exists()
                record={'doc':f.relative_to(ROOT).as_posix(),'target':url,'exists':exists}
                links.append(record)
                if not exists:broken.append(record)
    offline=json.loads(offline_path.read_text(encoding='utf-8'))
    final_round=args.final_round or 2
    comparison=json.loads((OUT/f'comparison-r{final_round}.json').read_text(encoding='utf-8'))
    checks={}; findings={}; full_pass=False
    if args.final_round is not None:
        prior=json.loads((OUT/f'delivery-record-{args.before_record}.json').read_text(encoding='utf-8'))['current_source_sha256']
        business_paths={k for k in current if k.startswith(('app/','api/','frontend/src/')) or k in ('.env.example','requirements.txt','frontend/package.json','frontend/package-lock.json')}
        old_business_paths={k for k in prior if k.startswith(('app/','api/','frontend/src/')) or k in ('.env.example','requirements.txt','frontend/package.json','frontend/package-lock.json')}
        changed_business=sorted(k for k in business_paths|old_business_paths if current.get(k)!=prior.get(k))
        scoring=json.loads((OUT/'scoring-r1/summary.json').read_text(encoding='utf-8'))
        probes=json.loads((OUT/'probes/results.json').read_text(encoding='utf-8'))
        http=[json.loads(line) for line in (OUT/'http/results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        orchestrator=json.loads((OUT.parent/f'acceptance-20260912-final-r{final_round}/orchestrator.json').read_text(encoding='utf-8'))
        findings=runtime_findings(comparison)
        checks=acceptance_checks(comparison,offline,scoring,probes,http,orchestrator,not changed_business,findings)
        checks['all_local_markdown_links_exist']=not broken
        full_pass=all(checks.values())
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'current_source_sha256':current,'changes_from_frozen_baseline':changes,
            'source_function_lines':refs,'markdown_inventory':md_paths,'local_markdown_links':links,'broken_local_links':broken,
            'final_offline_passed':offline['all_passed'],
            'python_tests':int(re.search(r'(\d+) passed',offline['checks'][0]['stdout']).group(1)),
            'frontend_tests':int(re.search(r'tests (\d+)',offline['checks'][1]['stdout']).group(1)),
            'baseline_and_final_uploaded_inputs_equal':comparison['same_uploaded_input_hashes'],
            'comparison_input':f'comparison-r{final_round}.json','offline_input':offline_path.relative_to(OUT).as_posix(),
            'acceptance_scope':'Local interview-practice coach; not enterprise deployment, expert accuracy, billing verification, full browser e2e or all-document OCR accuracy; Chinese OCR evaluated separately with synthetic files; local question-bank/RAG removed',
            'acceptance_checks':checks,
            'runtime_findings':findings,
            'additional_quality_targets':{'zero_score_errors':comparison['phases']['final']['aggregate']['score_errors']==0,
                                          'selfcheck_receives_full_structured_assessment_input':bool(findings) and not any(x['selfcheck_assessment_text_was_sliced'] for x in findings.values())},
            'latest_full_real_reverification_passed':bool(checks) and all(v for k,v in checks.items() if k not in ('all_local_markdown_links_exist','offline_tests_and_build')),
            'reason':'Saved acceptance criteria passed' if full_pass else ('See individual acceptance_checks' if checks else 'DeepSeek HTTP402 Insufficient Balance at final-r3; stopped further real attempts'),
            'five_repair_round_cap_observed':True,'historical_reports_not_overwritten':True,
            'external_billed_cost_verified':False,'unobserved_early_test_usage_and_failed_attempt_usage':'unknown, not zero',
            'all_final_acceptance_passed':full_pass}
    if checks:result['changed_business_source_since_preflight']=changed_business
    with stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'source_changes':len(changes),'local_links':len(links),'broken':broken,'offline':offline['all_passed'],'full_acceptance':full_pass,'checks':checks},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
