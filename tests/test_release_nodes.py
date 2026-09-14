"""Release regressions for D3/D5/D8/D9/D10; synthetic and offline only."""

import json
import operator
from copy import deepcopy

import httpx
import pytest
import requests
from langgraph.graph import END, START, StateGraph

from app.context import manager
from app.graph.runtime import wrap_node
from app.graph.schema import InterviewStateSchema, to_messages, to_records
from app.models.schemas import CompressedTurn
from app.nodes import ask as ask_mod
from app.nodes import evaluate as evaluate_mod
from app.nodes.advance import advance
from app.tools import executor

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "finish_reason": "stop"}
QUESTION = "你会怎样验证这个合成业务项目的结果？"


def test_tool_internal_llm_usage_is_not_discarded(monkeypatch):
    monkeypatch.setattr(executor, "chat_with_usage", lambda *a, **k: ("synthetic explanation", dict(USAGE)))
    result = executor.execute_tool_call("CodeExplainer", {"code": "print('synthetic')"})
    assert result.success
    assert result.usage["total_tokens"] == 15


def test_failed_dynamic_output_keeps_usage(monkeypatch):
    monkeypatch.setattr(executor, "chat_with_usage", lambda *a, **k: ("not json", dict(USAGE)))
    result = executor.execute_tool_call("DynamicQuestion", {"skill": "synthetic"})
    assert not result.success
    assert result.usage["total_tokens"] == 15


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Release node tests must not invoke network, real LLM or retrieval")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(manager, "history_tokens", lambda messages: sum(len(str(m.get("content", ""))) for m in messages))
    for module in (manager, ask_mod, evaluate_mod, executor):
        monkeypatch.setattr(module, "chat_with_usage", forbidden)
    monkeypatch.setattr(ask_mod, "chat_with_tools", forbidden)
    monkeypatch.setattr(executor, "_tavily_search", forbidden)


def _state(**changes):
    state = {
        "question_plan": [{"id": 1, "content": QUESTION, "skills": ["业务分析"], "category": "scenario", "depth_level": "application"}],
        "current_question": QUESTION,
        "current_answer": "上一题的合成回答",
        "question_version": 4,
        "current_question_index": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "total_follow_ups": 0,
        "difficulty": "medium",
        "role_profile": {"domain": "运营-业务分析", "job_title": "业务分析员"},
        "assessments": [{"question_id": 1, "score": 6, "is_follow_up": False, "is_relevant": True, "follow_up_reason": "补充验证"}],
        "conversation_history": [{"role": "assistant", "content": QUESTION}],
    }
    state.update(changes)
    return state


def _stub_ask(monkeypatch, text=QUESTION):
    captured = []

    def fake(messages, **kwargs):
        captured.append(messages)
        return text, dict(USAGE)

    monkeypatch.setattr(ask_mod, "chat_with_usage", fake)
    monkeypatch.setattr(ask_mod, "audit_follow_up", lambda *args: {"verdict": "allow", "hits": []})
    return captured


def _compression_state(rounds=4):
    history, turns = [], []
    for index in range(1, rounds + 1):
        question, answer = f"合成问题{index}？", f"合成事实{index}"
        history.extend([{"role": "assistant", "content": question}, {"role": "user", "content": answer}])
        turns.append({"question_id": index, "question": question, "answer": answer, "assessment": {"score": 6}})
    return {"conversation_history": history, "turn_records": turns, "summaries": [], "summarized_ids": []}


def _summary(**kwargs):
    return CompressedTurn(question_id=kwargs["question_id"], question=kwargs["question"], score=6, summary=kwargs["answer"]), dict(USAGE)


def test_d5_failure_preserves_all_twenty_messages_and_usage(monkeypatch):
    monkeypatch.setattr(manager.settings, "compress_threshold", 0)
    monkeypatch.setattr(manager.settings, "keep_recent_messages", 12)
    monkeypatch.setattr(manager, "summarize_turn", lambda **kwargs: (None, dict(USAGE)))
    state = _compression_state(10)
    original = deepcopy(state)
    result = manager.compress_history(state)
    assert result["conversation_history"] == original["conversation_history"]
    assert len(result["conversation_history"]) == 20
    assert result["summaries"] == result["summarized_ids"] == []
    assert result['usage_records'] == [{'node':'compress', **USAGE}]
    assert state == original


def test_d5_failure_after_one_success_does_not_cut_history(monkeypatch):
    monkeypatch.setattr(manager.settings, "compress_threshold", 0)
    monkeypatch.setattr(manager.settings, "keep_recent_messages", 2)
    outcomes = iter([_summary(question_id=1, question="q", answer="a"), (None, dict(USAGE))])
    monkeypatch.setattr(manager, "summarize_turn", lambda **kwargs: next(outcomes))
    state = _compression_state()
    result = manager.compress_history(state)
    assert result["conversation_history"] == state["conversation_history"]
    assert result["summarized_ids"] == [1]
    assert len(result["usage_records"]) == 2


@pytest.mark.parametrize("raw", ["not JSON", "{}", '{"summary": ""}', "[]", '{"summary": 123}', '{"summary": ["not text"]}', '{"summary": "ok", "score": "not a score"}'])
def test_d5_invalid_model_summary_is_not_marked_success(monkeypatch, raw):
    monkeypatch.setattr(manager, "chat_with_usage", lambda *a, **k: (raw, dict(USAGE)))
    summary, usage = manager.summarize_turn("q", "a", {"score": 6}, 1)
    assert summary is None
    assert usage["total_tokens"] == 15


def test_d5_success_preserves_pending_rounds_and_reinjects_summaries(monkeypatch):
    monkeypatch.setattr(manager.settings, "compress_threshold", 0)
    monkeypatch.setattr(manager.settings, "keep_recent_messages", 2)
    monkeypatch.setattr(manager, "summarize_turn", _summary)
    state = _compression_state()
    result = manager.compress_history(state)
    assert result["summarized_ids"] == [1, 2]
    assert result["conversation_history"] == state["conversation_history"][4:]
    assert "合成事实3" in str(result["conversation_history"])
    captured = _stub_ask(monkeypatch)
    question_state = _state(**result)
    asked = ask_mod.ask(question_state)
    text = str(captured[0])
    assert "合成事实1" in text and "合成事实2" in text and "合成事实3" in text
    assert len(asked["conversation_history"]) == len(result["conversation_history"]) + 1
    assert "历史摘要" not in str(asked["conversation_history"])


def test_d5_real_graph_keeps_summary_visible_after_successful_pruning(monkeypatch):
    monkeypatch.setattr(manager.settings, "compress_threshold", 0)
    monkeypatch.setattr(manager.settings, "keep_recent_messages", 2)
    answers = iter(["合成事实1", "合成事实2"])
    monkeypatch.setattr(manager, "chat_with_usage", lambda *a, **k: (json.dumps({"summary": next(answers), "score": 6}), dict(USAGE)))
    captured = _stub_ask(monkeypatch)
    graph = StateGraph(InterviewStateSchema)
    graph.add_node("compress", wrap_node(manager.compress_history))
    graph.add_node("ask", wrap_node(ask_mod.ask))
    graph.add_edge(START, "compress")
    graph.add_edge("compress", "ask")
    graph.add_edge("ask", END)
    state = _state(**_compression_state())
    state["conversation_history"] = to_messages(state["conversation_history"])
    result = graph.compile().invoke(state)
    assert result["summarized_ids"] == [1, 2]
    assert len(result["conversation_history"]) == 5
    assert "合成事实1" in str(captured[0]) and "合成事实3" in str(captured[0])
    assert result["question_version"] == 5 and result["current_answer"] == ""
    assert len(result["usage_records"]) == 3


@pytest.mark.parametrize("node", [ask_mod.ask, ask_mod.ask_follow_up, ask_mod.ask_explain])
def test_d5_oldest_of_more_than_twenty_summaries_is_visible(monkeypatch, node):
    captured = _stub_ask(monkeypatch)
    summaries = [{"question_id": i, "summary": f"不可丢失的合成摘要{i}"} for i in range(25)]
    node(_state(summaries=summaries))
    assert "不可丢失的合成摘要0" in str(captured[0])
    assert "不可丢失的合成摘要24" in str(captured[0])


@pytest.mark.parametrize("case", ["unknown_prefix", "ambiguous_pair", "missing_summary", "no_turns"])
def test_d5_does_not_prune_uncovered_or_ambiguous_messages(monkeypatch, case):
    monkeypatch.setattr(manager.settings, "keep_recent_messages", 2)
    state = _compression_state()
    state["summarized_ids"] = [1]
    state["summaries"] = [{"question_id": 1, "summary": "合成事实1"}]
    if case == "unknown_prefix":
        state["conversation_history"].insert(0, {"role": "assistant", "content": "未记录的讲解"})
    elif case == "ambiguous_pair":
        state["turn_records"].append({**state["turn_records"][0], "question_id": 99})
    elif case == "missing_summary":
        state["summaries"] = []
    else:
        state["turn_records"] = []
    result = manager._prune_summarized_prefix(state["conversation_history"], state["turn_records"], state["summaries"], state["summarized_ids"])
    assert result == state["conversation_history"]


@pytest.mark.parametrize("arguments", [
    {"query": "q", "max_results": 0}, {"query": "q", "max_results": -1},
    {"query": "q", "max_results": 10000}, {"query": "q", "max_results": "5"},
    {"query": "q", "max_results": True}, {"query": "q", "max_results": {}},
    {"query": "q", "unexpected": "never echo this synthetic value"},
    {"query": "  "}, {}, {"query": ["q"]}, "broken JSON", 'prefix {"query":"q"}', "[]",
])
def test_d8_invalid_arguments_do_not_reach_handler_or_fallback(monkeypatch, arguments):
    calls = []
    monkeypatch.setitem(executor.HANDLERS, "WebSearch", lambda args: calls.append(args) or {"success": True})
    result = executor.execute_tool_call("WebSearch", arguments)
    assert not result.success and result.error == "invalid_tool_arguments"
    assert not result.degraded and calls == []
    assert "never echo this synthetic value" not in result.model_dump_json()


@pytest.mark.parametrize("name,args", [("CodeExplainer", {"code": "x", "language": 7}), ("CodeExplainer", {"code": " "}), ("DynamicQuestion", {"skill": "q", "avoid": "not a list"})])
def test_d8_other_tool_schemas_are_validated(name, args):
    result = executor.execute_tool_call(name, args)
    assert not result.success and result.error == "invalid_tool_arguments"


def test_d8_valid_json_gets_schema_defaults(monkeypatch):
    received = []
    monkeypatch.setitem(executor.HANDLERS, "WebSearch", lambda args: received.append(args) or {"success": True, "content": "ok"})
    result = executor.execute_tool_call("WebSearch", '{"query": "合成查询"}')
    assert result.success and result.content == "ok"
    assert received == [{"query": "合成查询", "max_results": 5}]


@pytest.mark.parametrize("name,allowed,error", [("NotRegistered", None, "Unknown tool"), ("CodeExplainer", {"WebSearch"}, "tool_not_allowed")])
def test_d8_unknown_and_role_disallowed_tools_are_refused(name, allowed, error):
    result = executor.execute_tool_call(name, {"code": "print(1)"}, allowed_tools=allowed)
    assert not result.success and error in result.error


def test_d8_search_failure_is_explicit_without_local_fallback(monkeypatch):
    received = []

    def fail(args):
        received.append(args)
        raise RuntimeError("SYNTHETIC_PRIVATE_SEARCH_FAILURE")

    monkeypatch.setitem(executor.HANDLERS, "WebSearch", fail)
    result = executor.execute_tool_call(
        "WebSearch", {"query": "q", "max_results": 5}, allowed_tools={"WebSearch"}
    )
    assert not result.success and result.degraded and result.tool == "WebSearch"
    assert received == [{"query": "q", "max_results": 5}]
    assert result.source == ""
    assert "no external source available" in result.content
    assert "SYNTHETIC_PRIVATE_SEARCH_FAILURE" not in result.model_dump_json()


def test_d8_runtime_exception_values_are_not_exposed(monkeypatch, caplog):
    def fail(args):
        raise RuntimeError("SYNTHETIC_PRIVATE_EXCEPTION_VALUE")

    monkeypatch.setitem(executor.HANDLERS, "CodeExplainer", fail)
    result = executor.execute_tool_call("CodeExplainer", {"code": "print(1)"})
    assert not result.success and result.degraded
    assert "SYNTHETIC_PRIVATE_EXCEPTION_VALUE" not in result.model_dump_json()
    assert "SYNTHETIC_PRIVATE_EXCEPTION_VALUE" not in caplog.text


def _calls(count):
    return [{"type": "tool_call", "name": "WebSearch", "id": f"call-{i}", "args": {"query": str(i)}} for i in range(count)]


def test_d8_oversized_batch_executes_zero_tools(monkeypatch):
    received = []
    monkeypatch.setitem(executor.HANDLERS, "WebSearch", lambda args: received.append(args) or {"success": True})
    results = executor.execute_tool_calls(_calls(9))
    assert len(results) == 9 and received == []
    assert all(result.error == "tool_batch_limit_exceeded" for result in results)


def test_d8_valid_batch_has_bounded_workers_and_ordered_results(monkeypatch):
    import concurrent.futures

    real_pool, workers = concurrent.futures.ThreadPoolExecutor, []

    def pool(*args, **kwargs):
        workers.append(kwargs["max_workers"])
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", pool)
    monkeypatch.setitem(executor.HANDLERS, "WebSearch", lambda args: {"success": True, "content": args["query"]})
    results = executor.execute_tool_calls(_calls(8))
    assert workers == [4]
    assert [result.content for result in results] == [str(i) for i in range(8)]
    assert executor.execute_tool_calls([]) == []


def test_d8_ask_passes_actual_role_allowlist_to_executor(monkeypatch):
    unauthorized = {"type": "tool_call", "name": "CodeExplainer", "id": "unauthorized", "args": {"code": "print(1)"}}
    responses = iter([("", [unauthorized], dict(USAGE)), (QUESTION, [], dict(USAGE))])
    monkeypatch.setattr(ask_mod, "chat_with_tools", lambda *a, **k: next(responses))
    result = ask_mod.ask(_state(role_profile={"domain": "运营-业务分析", "needs_web_research": True}))
    assert result["tool_records"][0]["error"] == "tool_not_allowed"
    assert result["current_question"] == QUESTION
    assert result["current_answer"] == "" and result["question_version"] == 5


def test_d9_append_reducer_only_receives_new_event():
    old = {"question_index": 0, "previous_difficulty": "easy", "new_difficulty": "medium", "reason": "synthetic old event"}
    state = _state(difficulty_events=[old], assessments=[{"score": 8}] * 3)
    original = deepcopy(state)
    first = advance(state)
    events = operator.add(state["difficulty_events"], first.get("difficulty_events", []))
    assert len(events) == 2 and events[0] == old
    assert first["difficulty_events"][0]["new_difficulty"] == "hard"
    next_state = {**state, **first, "difficulty_events": events}
    second = advance(next_state)
    assert second.get("difficulty_events", []) == []
    assert len(operator.add(events, second.get("difficulty_events", []))) == 2
    assert state == original


def test_d9_real_graph_append_contract_does_not_duplicate():
    graph = StateGraph(InterviewStateSchema)
    graph.add_node("advance_once", wrap_node(advance))
    graph.add_node("advance_twice", wrap_node(advance))
    graph.add_edge(START, "advance_once")
    graph.add_edge("advance_once", "advance_twice")
    graph.add_edge("advance_twice", END)
    result = graph.compile().invoke({"difficulty": "medium", "assessments": [{"score": 8}] * 3, "difficulty_events": [{"reason": "old"}]})
    assert len(result["difficulty_events"]) == 2
    assert result["current_question_index"] == 2


def _assessment_pair():
    return [
        {"question_id": 1, "plan_question_index": 0, "is_follow_up": False, "question": "主问题短题干", "original_question": QUESTION, "question_summary": "主问题摘要", "score": 6, "dimensions": {"d1": 6}},
        {"question_id": 2, "plan_question_index": 0, "is_follow_up": True, "question": "追问题干", "original_question": "请解释你如何设计合成验证样本？", "question_summary": "追問摘要", "score": 7, "dimensions": {"d1": 8, "d2": 6}},
    ]


def test_d10_main_identity_and_last_scoring_are_both_preserved():
    assessments = _assessment_pair()
    before = deepcopy(assessments)
    record = evaluate_mod._topic_records(assessments)[0]
    assert record["question_id"] == 1 and record["original_question"] == QUESTION
    assert record["question_summary"] == "主问题摘要" and not record["is_follow_up"]
    assert record["score"] == 7 and record["dimensions"] == {"d1": 8, "d2": 6}
    assert record["round_scores"] == [6, 7] and record["follow_up_rounds"] == 1
    assert record["last_assessment_question_id"] == 2
    assert [item["question_id"] for item in record["rounds"]] == [1, 2]
    assert assessments == before


def test_d10_report_keeps_original_main_question_without_changing_score(monkeypatch):
    monkeypatch.setattr(evaluate_mod, "chat_with_usage", lambda *a, **k: (json.dumps({"text_analysis": "合成评语"}), dict(USAGE)))
    state = _state(assessments=_assessment_pair(), role_profile={"rubric": [{"key": "d1", "weight": 1}, {"key": "d2", "weight": 1}]}, summaries=[], usage_records=[], turn_records=[], difficulty_events=[], plan_question_count=1)
    report = evaluate_mod.evaluate(state)["evaluation_report"]
    assert report["overall_score"] == 7 and report["grade"] == "B"
    record = report["per_question"][0]
    assert record["question_id"] == 1 and record["question"] == QUESTION
    assert record["round_scores"] == [6, 7] and record["follow_up_rounds"] == 1
    assert record["rounds"][1]["question"] == "请解释你如何设计合成验证样本？"


def test_d10_legacy_records_without_index_keep_main_identity():
    records = _assessment_pair()
    for record in records:
        record.pop("plan_question_index")
    merged = evaluate_mod._topic_records(records)[0]
    assert merged["question_id"] == 1 and merged["score"] == 7
    assert merged["plan_question_index"] == 0


@pytest.mark.parametrize("version", [0, 4, 99])
def test_d3_main_publication_clears_answer_and_advances_version(monkeypatch, version):
    _stub_ask(monkeypatch)
    state = _state(question_version=version)
    original = deepcopy(state)
    result = ask_mod.ask(state)
    assert result["current_answer"] == "" and result["question_version"] == version + 1
    assert state == original


def test_d3_initial_dict_without_version_publishes_one(monkeypatch):
    _stub_ask(monkeypatch)
    state = _state()
    state.pop("question_version")
    assert ask_mod.ask(state)["question_version"] == 1


@pytest.mark.parametrize("changes", [{}, {"assessments": [{"score": 6, "resume_conflict": True}]}, {"assessments": [{"score": 4, "is_relevant": False}]}])
def test_d3_follow_up_publication_clears_answer_and_advances_version(monkeypatch, changes):
    _stub_ask(monkeypatch)
    result = ask_mod.ask_follow_up(_state(**changes))
    assert result["current_answer"] == "" and result["question_version"] == 5
    assert result["follow_up_count"] == 1 and result["total_follow_ups"] == 1


def test_d3_guard_retry_only_publishes_one_version(monkeypatch):
    _stub_ask(monkeypatch)
    verdicts = iter([{"verdict": "drop", "hits": ["合成禁止主题"]}, {"verdict": "allow", "hits": []}])
    monkeypatch.setattr(ask_mod, "audit_follow_up", lambda *a: next(verdicts))
    result = ask_mod.ask_follow_up(_state())
    assert result["question_version"] == 5 and result["current_answer"] == ""
    assert len(result["usage_records"]) == 2


def test_d3_guard_closure_is_not_a_new_answerable_question(monkeypatch):
    _stub_ask(monkeypatch)
    monkeypatch.setattr(ask_mod, "audit_follow_up", lambda *a: {"verdict": "drop", "hits": ["合成禁止主题"]})
    result = ask_mod.ask_follow_up(_state())
    assert "换下一个话题" in result["current_question"]
    assert "question_version" not in result and result["current_answer"] == ""


def test_d3_exhausted_plan_does_not_publish_version():
    result = ask_mod.ask(_state(current_question_index=1))
    assert result["current_question"] == result["current_answer"] == ""
    assert "question_version" not in result


def test_d3_explanation_schedules_replacement_without_publishing_until_ask(monkeypatch):
    _stub_ask(monkeypatch, "这里用合成数据演示验证的步骤。")
    skipped = {"question_index": 0, "skills": ["业务分析"], "focus_key": "d1"}
    replacement = {"content": "你如何设计另一组业务验证样本？", "skills": ["业务分析"], "source_id": "llm:synthetic", "depth_level": "application"}
    monkeypatch.setattr(ask_mod, "_generate_replacement_question", lambda *a: (replacement, []))
    state = _state(skipped_questions=[skipped], conversation_history=[{"role": "user", "content": "请讲解"}])
    original = deepcopy(state)
    result = ask_mod.ask_explain(state)
    assert result["current_question"] == result["current_answer"] == ""
    assert "question_version" not in result and len(result["question_plan"]) == 2
    assert state == original
    next_state = {**state, **result}
    next_state.update(advance(next_state))
    _stub_ask(monkeypatch, replacement["content"])
    published = ask_mod.ask(next_state)
    assert published["current_question"] == replacement["content"]
    assert published["question_version"] == 5 and published["current_answer"] == ""


def test_d3_ask_adapter_returns_message_objects_and_version(monkeypatch):
    _stub_ask(monkeypatch)
    state = _state(active_answer_request_id="synthetic-answer-request")
    state["conversation_history"] = to_messages(state["conversation_history"])
    result = wrap_node(ask_mod.ask)(InterviewStateSchema(**state))
    assert result["question_version"] == 5 and result["current_answer"] == ""
    assert to_records(result["conversation_history"])[-1]["content"] == QUESTION
    assert isinstance(result["conversation_history"][0], type(result["conversation_history"][-1]))
    assert result["last_completed_answer_request_id"] == "synthetic-answer-request"


def test_d3_real_graph_preserves_version_and_request_completion_marker(monkeypatch):
    _stub_ask(monkeypatch)
    graph = StateGraph(InterviewStateSchema)
    graph.add_node("ask", wrap_node(ask_mod.ask))
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    state = _state(active_answer_request_id="synthetic-answer-request")
    state["conversation_history"] = to_messages(state["conversation_history"])
    result = graph.compile().invoke(state)
    assert result["question_version"] == 5 and result["current_answer"] == ""
    assert result["last_completed_answer_request_id"] == "synthetic-answer-request"
