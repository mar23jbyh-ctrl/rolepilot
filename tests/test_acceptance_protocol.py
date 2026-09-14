"""New protocol tests use synthetic data and the suite's external-network guard."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from app.context.budget import prepare_request, ContextBudgetExceeded
from app.telemetry.summary import summarize_usage, merge_usage, usage_record
from app.scoring.protocol import normalize_assessment
from app.llm import client
from app.nodes import assess, evaluate

RUBRIC = [{'key':'d1','weight':1}]
ANSWER = '先隔离环境测试，再验证回滚和数据完整性。'

def payload(**changes):
    return {'scoreable':True,'dimension_levels':{'d1':4}, 'evidence':{'d1':['验证回滚和数据完整性']},
            'missing_points':[], 'hallucination_or_conflict':False, 'next_action':'next_question',
            'confidence':'medium', **changes}

@pytest.mark.parametrize('name', ['long_resume','long_jd','multi_follow','long_tool_result'])
def test_budget_rejects_entire_oversize_payload(name):
    messages = [{'role':'system','content':'policy'}, {'role':'user','content':name + '数据库测试 ' * 10000}]
    original = deepcopy(messages)
    with pytest.raises(ContextBudgetExceeded) as error:
        prepare_request(messages, [], 2048, 16000)
    assert messages == original and error.value.metadata['action'] == 'rejected'

def test_tools_and_arguments_are_counted_and_output_reserved():
    _, plain = prepare_request([{'role':'user','content':'x'}], [], 2048, 16000)
    output, rich = prepare_request([{'role':'user','content':'x'}, {'role':'tool','tool_call_id':'abc','content':'result ' * 100}], [{'schema':'description ' * 500}], 20000, 16000)
    assert rich['estimated_input_tokens'] > plain['estimated_input_tokens']
    assert output + rich['estimated_input_tokens'] + rich['safety_margin_tokens'] <= 16000
    assert rich['action'] == 'output_cap_reduced'

@pytest.mark.parametrize('tools', [False, True])
def test_sdk_not_called_on_budget_rejection(monkeypatch, tools):
    monkeypatch.setattr(client, '_create_with_retry', lambda **k: pytest.fail('SDK called'))
    fn = client.chat_with_tools if tools else client.chat_with_usage
    args = ([{'role':'user','content':'very long ' * 30000}],) + (([],) if tools else ())
    with pytest.raises(ContextBudgetExceeded): fn(*args)

def test_known_zero_and_missing_are_distinct():
    zero = client._usage_dict(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0,completion_tokens=0,total_tokens=0), choices=[]))
    missing = client._usage_dict(SimpleNamespace(usage=None))
    assert summarize_usage([zero])['total'] == 0
    result = summarize_usage([missing]); assert result['total'] is None and result['cost'] is None
    assert result['unknown_calls'] == 1 and missing['total_tokens'] is None

def test_partial_retry_keeps_known_subtotal():
    total = {}; merge_usage(total, {'input_tokens':10,'output_tokens':5,'total_tokens':15})
    merge_usage(total, {'usage_status':'missing'})
    result = summarize_usage([usage_record('assess',total)])
    assert result['total'] is None and result['known_total'] == 15 and result['unknown_calls'] == 1

def test_estimated_not_added_to_provider_total():
    result = summarize_usage([{'usage_status':'estimated','total_tokens':900}])
    assert result['total'] is None and result['known_total'] == 0 and result['estimated_calls'] == 1

@pytest.mark.parametrize('level', [0,6,7,3.5,True,'4',None])
def test_illegal_levels_cannot_become_high_scores(level):
    out = normalize_assessment(payload(dimension_levels={'d1':level}), ANSWER, RUBRIC)
    assert not out['scoreable'] and out['dimension_levels'] == {} and out['protocol_errors']

@pytest.mark.parametrize('bad', [{}, {'d1':['not in answer']}, {'d1':[]}])
def test_high_level_without_real_quote_is_rejected(bad):
    out = normalize_assessment(payload(evidence=bad), ANSWER, RUBRIC)
    assert not out['scoreable'] and out['protocol_errors']

def test_conflict_caps_score_and_offtopic_is_unscoreable():
    out = normalize_assessment(payload(hallucination_or_conflict=True), ANSWER, RUBRIC)
    assert out['dimension_levels'] == {'d1':2}
    assert not normalize_assessment(payload(), ANSWER, RUBRIC, 'off_topic')['scoreable']

def test_full_current_answer_reaches_assessment(monkeypatch):
    answer = ANSWER + '很长但不能静默丢弃 ' * 800 + 'TAIL_CURRENT_ANSWER'
    captured=[]
    def fake(messages, **kw):
        captured.append(messages[1].content)
        return json.dumps(payload(),ensure_ascii=False), {'input_tokens':10,'output_tokens':5,'total_tokens':15}
    monkeypatch.setattr(assess,'chat_with_usage',fake)
    monkeypatch.setattr(assess,'validate_resume_claims',lambda **kw: {})
    out=assess.assess({'current_question':'如何验证','current_answer':answer, 'role_profile':{'rubric':RUBRIC}, 'assessment_protocol_version':'practice-v1'})
    assert 'TAIL_CURRENT_ANSWER' in captured[0] and out['turn_records'][0]['answer'] == answer
    assert out['assessments'][0]['score'] == 8

def test_empty_no_llm_and_no_numeric_score(monkeypatch):
    monkeypatch.setattr(assess,'chat_with_usage',lambda *a,**k:pytest.fail('empty calls LLM'))
    out=assess.assess({'current_answer':' ', 'current_question':'question'})['assessments'][0]
    assert out['score'] is None and not out['scoreable']

def test_invalid_protocol_fallback_is_null_not_extra_llm(monkeypatch):
    calls=[]
    monkeypatch.setattr(assess,'chat_with_usage',lambda *a,**k:(calls.append(1) or json.dumps({'dimension_levels':{'d1':5}}), {'input_tokens':1,'output_tokens':1,'total_tokens':2}))
    monkeypatch.setattr(assess,'validate_resume_claims',lambda **k: {})
    out=assess.assess({'current_answer':ANSWER,'role_profile':{'rubric':RUBRIC},'assessment_protocol_version':'practice-v1'})['assessments'][0]
    assert out['score'] is None and out['score_error'] and len(calls)==1

def test_empty_follow_does_not_erase_previous_valid_topic():
    merged=evaluate._topic_records([{'question_id':1,'score':8,'dimensions':{'d1':8},'scoreable':True,'plan_question_index':0}, {'question_id':2,'is_follow_up':True,'score':None,'scoreable':False,'plan_question_index':0}])[0]
    assert merged['score']==8 and merged['question_id']==1 and len(merged['rounds'])==2

def test_report_uses_same_summary_and_does_not_recompute_legacy():
    records=[{'usage_status':'missing'}]
    assert evaluate._token_summary({'usage_records':records,'usage_schema_version':'usage-v1'}) == summarize_usage(records)

def test_decimal_estimate_exact_known_fixture():
    # Frozen synthetic policy, not today's deployment rates or an actual bill.
    summary=summarize_usage([{'input_tokens':252621,'output_tokens':31711,'total_tokens':284332,
                             'pricing':{'status':'priced','model':'synthetic','currency':'CNY',
                                        'input_price_per_1m':1,'output_price_per_1m':2,'version':'fixture'}}])
    assert summary['cost']==0.316043

def test_noncall_not_unknown_usage():
    summary=summarize_usage([{'usage_status':'not_called','unknown_calls':0,'budget':{'action':'rejected'}}])
    assert summary['unknown_calls']==0 and summary['total']==0

def test_sidebar_api_and_cli_keep_missing_unknown():
    from api.routers.sessions import _list_tokens
    from app.telemetry.usage import UsageTracker
    assert _list_tokens({'state_json':json.dumps({'usage_schema_version':'usage-v1','usage_records':[{'usage_status':'missing'}]}),'total_tokens':0}) is None
    tracker=UsageTracker();tracker.add_usage('missing',{})
    assert tracker.totals()['total'] is None and tracker.cost() is None

def test_legacy_report_projection_preserves_historical_cost():
    from app.service import InterviewService
    report={'overall_score':6,'cost':9.123456,'token_totals':{'total':999}}
    state={'evaluation_report':report,'usage_records':[{'total_tokens':1}]}
    graph=SimpleNamespace(get_state=lambda config:SimpleNamespace(values=state,next=(),config={}),update_state=lambda *a,**k:pytest.fail('historic report rewritten'))
    instance=InterviewService.__new__(InterviewService);instance.graph=graph
    projected,_=instance._projection('synthetic')
    assert projected['evaluation_report']==report

def test_provider_missing_after_retries_is_not_one_zero_call():
    row=client._usage_dict(SimpleNamespace(usage=None,_retry_unknown_calls=2))
    assert row['unknown_calls']==3 and row['total_tokens'] is None

@pytest.mark.parametrize('transport',['httpx','requests'])
def test_external_url_blocked_even_if_local_proxy_exists(transport):
    if transport=='httpx':
        import httpx
        with pytest.raises(RuntimeError,match='External request URL'):
            httpx.get('https://example.com')
    else:
        import requests
        with pytest.raises(RuntimeError,match='External request URL'):
            requests.get('https://example.com')

def test_default_external_tracing_disabled():
    import os
    from app.config import settings
    assert settings.enable_external_tracing is False
    assert os.environ['LANGSMITH_TRACING']=='false'
