"""Per-attempt SQLite audit and conservative, restart-safe session reservations.

No prompts, resumes, answers, credentials or full provider URLs are stored here.
An uncertain external attempt retains its reservation; it is never free on restart.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import uuid

from app.config import settings
from app.context.budget import ContextBudgetExceeded
from app.telemetry.summary import summarize_usage, state_usage_records


class SessionBudgetExceeded(ContextBudgetExceeded):
    def __init__(self, metadata):
        super().__init__(metadata)
        self.usage_info = {"usage_status": "not_called", "unknown_calls": 0, "budget": metadata}


def initialize_ledger(connection):
    connection.execute("""CREATE TABLE IF NOT EXISTS llm_session_budgets (
        session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        token_limit INTEGER NOT NULL, report_reserve INTEGER NOT NULL,
        baseline_tokens INTEGER NOT NULL, baseline_input INTEGER NOT NULL,
        baseline_output INTEGER NOT NULL, baseline_known_calls INTEGER NOT NULL,
        baseline_unknown_calls INTEGER NOT NULL, baseline_unknown_reserve INTEGER NOT NULL,
        exhausted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS llm_calls (
        call_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        logical_call_id TEXT NOT NULL, attempt INTEGER NOT NULL, node TEXT NOT NULL,
        provider_host TEXT NOT NULL, requested_model TEXT NOT NULL, actual_model TEXT,
        input_hash TEXT NOT NULL, status TEXT NOT NULL,
        usage_status TEXT NOT NULL, reserved_tokens INTEGER NOT NULL,
        input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
        budget_json TEXT NOT NULL, pricing_json TEXT NOT NULL DEFAULT '{}',
        usage_details_json TEXT NOT NULL DEFAULT '{}', provider_request_id TEXT,
        finish_reason TEXT, error_type TEXT, elapsed_seconds REAL,
        started_at TEXT NOT NULL, finished_at TEXT,
        UNIQUE(session_id, logical_call_id, attempt)
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id)")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(llm_calls)")}
    if "request_params_json" not in columns:
        connection.execute("ALTER TABLE llm_calls ADD COLUMN request_params_json TEXT NOT NULL DEFAULT '{}'")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class CallLedger:
    def __init__(self, store, session_id):
        self.store, self.session_id = store, session_id

    def ensure_budget(self):
        with self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM llm_session_budgets WHERE session_id=?", (self.session_id,)).fetchone():
                return
            row = conn.execute("SELECT state_json,status FROM sessions WHERE id=?", (self.session_id,)).fetchone()
            if not row or row["status"] == "deleting":
                raise RuntimeError("metering_session_unavailable")
            state = json.loads(row["state_json"] or "{}")
            baseline = summarize_usage(state_usage_records(state))
            unknown = int(baseline["unknown_calls"])
            conn.execute("""INSERT INTO llm_session_budgets VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                         (self.session_id, settings.session_token_budget, settings.final_report_token_reserve,
                          baseline["known_total"], baseline["known_input"], baseline["known_output"], baseline["known_calls"], unknown,
                          unknown * settings.context_budget, 0, _now()))

    def _amounts(self, conn):
        budget = conn.execute("SELECT * FROM llm_session_budgets WHERE session_id=?", (self.session_id,)).fetchone()
        if not budget:
            raise RuntimeError("metering_budget_missing")
        amounts = conn.execute("""SELECT
            COALESCE(SUM(CASE WHEN usage_status='known' THEN total_tokens ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN usage_status='missing' THEN reserved_tokens ELSE 0 END),0)
            FROM llm_calls WHERE session_id=?""", (self.session_id,)).fetchone()
        return budget, budget["baseline_tokens"] + amounts[0], budget["baseline_unknown_reserve"] + amounts[1]

    def reserve(self, *, logical_id, attempt, node, provider_host, requested_model, input_hash, metadata, request_params=None):
        """Reserve before EACH external attempt under BEGIN IMMEDIATE, across workers."""
        call_id = uuid.uuid4().hex
        amount = (int(metadata["estimated_input_tokens"]) + int(metadata["safety_margin_tokens"])
                  + int(metadata["reserved_output_tokens"]))
        final = node.split(":", 1)[0] in {"evaluate", "self_check"}
        with self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            budget, used, held = self._amounts(conn)
            row = conn.execute("SELECT status FROM sessions WHERE id=?", (self.session_id,)).fetchone()
            if not row or row[0] == "deleting":
                raise RuntimeError("metering_session_unavailable")
            limit = budget["token_limit"]
            cap = limit if final else limit - budget["report_reserve"]
            rejected = bool(limit and (used + held + amount > cap or (budget["exhausted"] and not final)))
            event = dict(metadata)
            if rejected:
                event.update(action="session_rejected", session_limit=limit, reserved_for_report=budget["report_reserve"],
                             known_used_tokens=used, held_tokens=held, requested_reservation=amount)
                if not final:
                    conn.execute("UPDATE llm_session_budgets SET exhausted=1 WHERE session_id=?", (self.session_id,))
            conn.execute("""INSERT INTO llm_calls
                (call_id,session_id,logical_call_id,attempt,node,provider_host,requested_model,input_hash,
                 status,usage_status,reserved_tokens,budget_json,started_at,finished_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                         (call_id,self.session_id,logical_id,attempt,node,provider_host,requested_model,input_hash,
                          "blocked" if rejected else "started", "not_called" if rejected else "missing",
                          0 if rejected else amount,json.dumps(event),_now(),_now() if rejected else None))
            conn.execute("UPDATE llm_calls SET request_params_json=? WHERE call_id=?",
                         (json.dumps(request_params or {}),call_id))
        if rejected:
            raise SessionBudgetExceeded(event)
        return call_id

    def record_blocked(self, metadata, *, node, provider_host, requested_model, input_hash):
        call_id = uuid.uuid4().hex
        with self.store._connect() as conn:
            conn.execute("""INSERT INTO llm_calls
                (call_id,session_id,logical_call_id,attempt,node,provider_host,requested_model,input_hash,
                 status,usage_status,reserved_tokens,budget_json,started_at,finished_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                         (call_id,self.session_id,call_id,0,node,provider_host,requested_model,input_hash,
                          "blocked","not_called",0,json.dumps(metadata),_now(),_now()))

    def finish(self, call_id, usage, *, elapsed, actual_model=None, request_id=None, error_type=None):
        known = usage.get("usage_status") == "known"
        with self.store._connect() as conn:
            conn.execute("""UPDATE llm_calls SET status=?,usage_status=?,actual_model=?,
                input_tokens=?,output_tokens=?,total_tokens=?,pricing_json=?,usage_details_json=?,
                provider_request_id=?,finish_reason=?,error_type=?,elapsed_seconds=?,finished_at=?
                WHERE call_id=? AND session_id=? AND status='started'""",
                         ("failed" if error_type else "succeeded","known" if known else "missing",actual_model,
                          usage.get("input_tokens") if known else None,usage.get("output_tokens") if known else None,
                          usage.get("total_tokens") if known else None,json.dumps(usage.get("pricing") or {}),
                          json.dumps(usage.get("usage_details") or {}),request_id,usage.get("finish_reason"),
                          error_type,elapsed,_now(),call_id,self.session_id))

    def records(self):
        with self.store._connect() as conn:
            baseline = conn.execute("SELECT * FROM llm_session_budgets WHERE session_id=?", (self.session_id,)).fetchone()
            rows = conn.execute("SELECT * FROM llm_calls WHERE session_id=? ORDER BY started_at,call_id", (self.session_id,)).fetchall()
        records = []
        if baseline and (baseline["baseline_known_calls"] or baseline["baseline_tokens"] or baseline["baseline_unknown_calls"]):
            records.append({"node":"legacy_baseline","usage_status":"partial" if baseline["baseline_unknown_calls"] else "known",
                            "known_calls":baseline["baseline_known_calls"],"unknown_calls":baseline["baseline_unknown_calls"],
                            "input_tokens":baseline["baseline_input"],"output_tokens":baseline["baseline_output"],"total_tokens":baseline["baseline_tokens"],
                            "known_usage":{"input_tokens":baseline["baseline_input"],"output_tokens":baseline["baseline_output"],"total_tokens":baseline["baseline_tokens"]}})
        for row in rows:
            records.append({"call_id":row["call_id"],"logical_call_id":row["logical_call_id"],"attempt":row["attempt"],
                            "node":row["node"],"provider_host":row["provider_host"],"requested_model":row["requested_model"],
                            "actual_model":row["actual_model"],"status":row["status"],"usage_status":row["usage_status"],
                            "input_hash":row["input_hash"],"request_params":json.loads(row["request_params_json"]),
                            "known_calls":int(row["usage_status"] == "known"),"unknown_calls":int(row["usage_status"] == "missing"),
                            "input_tokens":row["input_tokens"],"output_tokens":row["output_tokens"],"total_tokens":row["total_tokens"],
                            "budget":json.loads(row["budget_json"]),"pricing":json.loads(row["pricing_json"]),
                            "elapsed_seconds":row["elapsed_seconds"],"finish_reason":row["finish_reason"]})
        return records

    def budget_summary(self):
        with self.store._connect() as conn:
            budget, used, held = self._amounts(conn)
            rows = conn.execute("SELECT budget_json,status,usage_status FROM llm_calls WHERE session_id=?", (self.session_id,)).fetchall()
        actions = Counter(json.loads(row[0]).get("action", "unknown") for row in rows)
        limit = budget["token_limit"]
        return {"schema_version":"budget-v1","token_limit":limit,"report_reserve":budget["report_reserve"],
                "known_used_tokens":used,"held_tokens":held,"remaining_tokens":max(0,limit-used-held) if limit else None,
                "normal_remaining_tokens":max(0,limit-budget["report_reserve"]-used-held) if limit else None,
                "exhausted":bool(budget["exhausted"]),"overrun":bool(limit and used+held > limit),
                "attempts":sum(row[1] != "blocked" for row in rows),"blocked_calls":sum(row[1] == "blocked" for row in rows),
                "unknown_attempts":sum(row[2] == "missing" for row in rows),"actions":dict(actions),
                "control_basis":"provider_usage_plus_conservative_reservations_not_billing_cap"}
