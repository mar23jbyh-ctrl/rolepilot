"""Runtime boundary validators between dict-style node code and graph state."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.graph.schema import InterviewStateSchema, to_messages, to_records
from app.errors import NodeExecutionError
from app.telemetry.meter import node_scope, active_meter
from app.telemetry.ledger import SessionBudgetExceeded


def _budget_stop(current, name):
    """Finish safely; retain a submitted answer even if its assessment was blocked."""
    output = {"budget_exhausted": True, "end_requested": True}
    if name in {"ask", "ask_follow_up"}:
        output["current_question"] = ""
    if name == "assess" and str(current.get("current_answer", "")).strip():
        assessment = {"question_id":current.get("global_question_counter",0),
                      "plan_question_index":current.get("current_question_index",0),
                      "question":current.get("current_question",""),"score":None,"scoreable":False,
                      "score_error":True,"budget_skipped":True,"next_action":"evaluate",
                      "is_follow_up":bool(current.get("follow_up_count",0)),"dimensions":{}}
        history = list(current.get("conversation_history") or [])
        history.append({"role":"user","content":current["current_answer"]})
        output.update(assessments=[assessment],conversation_history=history,
                      turn_records=[{"question_id":assessment["question_id"],"question":assessment["question"],
                                     "answer":current["current_answer"],"assessment":assessment}])
    return output


def node_input(state: BaseModel) -> dict[str, Any]:
    """Copy every field from the Pydantic state into a dict for node functions.

    NOTE: this is a full shallow copy; node functions should never mutate the
    returned nested structures (they are copies of lists but not deep-copied).
    Keep this adapter cheap because it runs before and after every node.
    """
    data: dict[str, Any] = {}
    for field in InterviewStateSchema.model_fields:
        if not hasattr(state, field):
            continue
        value = getattr(state, field)
        if field == "conversation_history":
            data[field] = to_records(value)
        else:
            data[field] = value
    return data


def validate_node_output(
    result: dict[str, Any],
    current_state: dict[str, Any],
    node_name: str,
) -> dict[str, Any]:
    output = dict(result or {})

    # Resume/JD are initialized once per session and are read-only afterwards.
    if "resume_text" in output or "jd_text" in output:
        raise PermissionError(
            f"Node {node_name} tried to overwrite read-only resume/jd fields."
        )
    current_profile = current_state.get("resume_profile") or {}
    output_profile = output.get("resume_profile")
    if output_profile is not None and (current_profile or node_name != "analyze"):
        raise PermissionError(
            f"Node {node_name} tried to overwrite read-only resume_profile."
        )
    if "job_research" in output and node_name != "research_job":
        raise PermissionError(
            f"Node {node_name} is not allowed to write job_research."
        )

    history = output.get("conversation_history")
    if history is not None:
        messages = to_messages(history)
        for message in messages:
            if not isinstance(message, BaseMessage):
                raise TypeError(
                    f"conversation_history item must be BaseMessage, got {type(message).__name__}"
                )
        output["conversation_history"] = messages
    return output


def wrap_node(func, node_name: str | None = None):
    name = node_name or getattr(func, "__name__", func.__class__.__name__)

    def wrapped(state):
        current = node_input(state)
        try:
            if current.get("error"):
                raise RuntimeError("previous_node_failed")
            with node_scope(name):
                meter = active_meter()
                final = name in {"evaluate", "self_check"}
                exhausted = meter and meter.ledger.budget_summary()["exhausted"]
                try:
                    if exhausted and not final:
                        result = _budget_stop(current, name)
                    elif exhausted and name == "self_check":
                        result = {"self_check_report":{"status":"skipped","findings":[],"reason":"session_budget"}}
                    else:
                        result = func(current)
                except SessionBudgetExceeded:
                    if final:
                        raise
                    result = _budget_stop(current, name)
                if meter:
                    result = dict(result or {})
                    result.update(llm_ledger_version="ledger-v1",llm_usage_records=meter.ledger.records(),
                                  budget_summary=meter.ledger.budget_summary())
                    if result["budget_summary"]["exhausted"] and not final:
                        result.update(budget_exhausted=True,end_requested=True)
                        if name in {"ask", "ask_follow_up"}:
                            result["current_question"] = ""
            if result is None:
                raise TypeError("node_returned_none")
            output = validate_node_output(result, current, name)
            if current.get('usage_schema_version') == 'usage-v1':
                from app.telemetry.summary import summarize_usage, state_usage_records
                merged = {**current, **output}
                if merged.get('llm_ledger_version') != 'ledger-v1':
                    merged['usage_records'] = list(current.get('usage_records') or []) + list(output.get('usage_records') or [])
                output['usage_summary'] = summarize_usage(state_usage_records(merged))
        except Exception as exc:  # noqa: BLE001
            # Do not turn a failed prerequisite into a successful graph update.
            # Raw exception text may contain provider credentials or user material.
            failure = NodeExecutionError(name, type(exc).__name__)
            failure.usage_info = getattr(exc, 'usage_info', None)
            failure.budget = getattr(exc, 'metadata', None)
            raise failure from exc
        request_id = current.get("active_answer_request_id", "")
        published = bool(output.get("current_question")) and (
            int(output.get("question_version", 0)) > int(current.get("question_version", 0))
        )
        if request_id and (published or name == "self_check"):
            output["last_completed_answer_request_id"] = request_id
        return output

    return wrapped


def wrap_edge(func):
    def wrapped(state):
        return func(node_input(state))

    return wrapped


def dump_state_for_store(state: dict[str, Any]) -> dict[str, Any]:
    """Convert graph state to JSON-safe records for the session store."""
    dumped = dict(state)
    history = dumped.get("conversation_history")
    if history is not None:
        dumped["conversation_history"] = to_records(history)
    return dumped
