"""Explicit session/node context, including propagation to worker threads."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, replace

from app.telemetry.ledger import CallLedger


@dataclass(frozen=True)
class Meter:
    ledger: CallLedger
    node: str = "service"


_active = ContextVar("interview_call_meter", default=None)


def active_meter():
    return _active.get()


@contextmanager
def metering_scope(store, session_id, node="service"):
    ledger = CallLedger(store, session_id)
    ledger.ensure_budget()
    token = _active.set(Meter(ledger, node))
    try:
        yield _active.get()
    finally:
        _active.reset(token)


@contextmanager
def node_scope(node):
    meter = active_meter()
    token = _active.set(replace(meter, node=node) if meter else None)
    try:
        yield active_meter()
    finally:
        _active.reset(token)


def submit_with_context(executor, func, *args, **kwargs):
    # A separate copied context per future; one Context cannot run concurrently.
    return executor.submit(copy_context().run, func, *args, **kwargs)


def ledger_projection(store, session_id):
    ledger = CallLedger(store, session_id)
    with store._connect() as conn:
        exists = conn.execute("SELECT 1 FROM llm_session_budgets WHERE session_id=?", (session_id,)).fetchone()
    if not exists:
        return {}
    return {"llm_ledger_version":"ledger-v1","llm_usage_records":ledger.records(),"budget_summary":ledger.budget_summary()}
