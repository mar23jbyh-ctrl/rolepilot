from __future__ import annotations

import re
import uuid
import threading
import json
from contextlib import contextmanager
from pathlib import Path

from app.config import settings
from app.graph.builder import build_graph
from app.graph.checkpointer import get_checkpointer, delete_checkpoints
from app.graph.runtime import dump_state_for_store
from app.session.store import SessionStore
from app.session.store import safe_owner
from app.errors import NodeExecutionError, ServiceError
from app.telemetry.meter import metering_scope, node_scope, ledger_projection
from app.telemetry.summary import state_usage_records
from app.telemetry.ledger import SessionBudgetExceeded

# M2：只认**显式**结束指令——整句（去掉空白与标点后）必须完全等于下列命令之一。
#
# 原实现是子串匹配（含"结束""退出""终止"这类高频词），实测在 208 条真实回答里
# 命中 6 次、且 6 次全部是误判（例如"一旦信息公开，保密义务自动终止"）。
# 隐式的结束意图改由 assess 的 candidate_intent 判断（见 app/nodes/assess.py）。
STOP_COMMANDS = (
    "结束面试",
    "停止面试",
    "终止面试",
    "退出面试",
    "结束吧",
    "不面了",
    "不想面了",
    "就到这里",
    "先这样",
)

# 显式命令最多只有几个字；超过该长度一律不认为是"命令整句"
STOP_COMMAND_MAX_CHARS = 12

_STOP_PUNCTUATION = re.compile(r"[\s，。、；：,.;:!！?？~～\"'“”‘’()（）\[\]【】]+")

# Multiple service objects can share one owner database in a process (for
# example, parallel requests in a test runner or a development server).  The
# database fence protects separate processes; this lock also serializes the
# cross-database checkpoint cleanup phase, whose SQLite connection is owned by
# a different service object and therefore cannot share ``_session_lock``.
_DELETE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_DELETE_LOCKS_GUARD = threading.Lock()


def _delete_lock(store: SessionStore, session_id: str) -> threading.Lock:
    key = (str(store.db_path.resolve()), str(session_id))
    with _DELETE_LOCKS_GUARD:
        return _DELETE_LOCKS.setdefault(key, threading.Lock())


def _normalize_command(text: str) -> str:
    return _STOP_PUNCTUATION.sub("", str(text or "")).lower()


def _is_explicit_stop_command(answer: str) -> bool:
    """M2：整句等于显式结束命令才返回 True（不再做子串匹配）。"""

    text = _normalize_command(answer)
    if not text or len(text) > STOP_COMMAND_MAX_CHARS:
        return False
    return any(_normalize_command(command) == text for command in STOP_COMMANDS)


class InterviewService:
    def __init__(self, owner: str = "local", store: SessionStore | None = None):
        self.owner = safe_owner(owner)
        if store is None:
            store = SessionStore(owner=self.owner)
        self.store = store
        checkpoint_file = (
            Path(settings.checkpoint_db_path).parent
            / f"checkpoints_{self.owner}.db"
        )
        self._checkpointer = get_checkpointer(checkpoint_file)
        self.graph = build_graph(checkpointer=self._checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._busy: set[str] = set()
        for session_id in self.store.cleanup_stale():
            self.delete_session(session_id)

    def _session_lock(self, session_id: str) -> threading.Lock:
        return self._locks.setdefault(session_id, threading.Lock())

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    @contextmanager
    def _mutation(self, session_id: str, kind: str):
        operation = kind + ":" + uuid.uuid4().hex
        with self._session_lock(session_id):
            self.store.claim_operation(session_id, operation)
            try:
                yield
            finally:
                self.store.release_operation(session_id, operation)

    def _projection(self, session_id: str) -> tuple[dict, str]:
        snapshot = self.graph.get_state(self._config(session_id))
        state = dict(snapshot.values or {})
        if getattr(self, "store", None) is not None:
            state.update(ledger_projection(self.store, session_id))
        if state.get("evaluation_report") and state.get('usage_schema_version') == 'usage-v1':
            from app.nodes.evaluate import _token_by_node, _token_summary
            report = dict(state["evaluation_report"])
            totals = _token_summary(state)
            report.update(token_totals=totals, cost=totals["cost"],
                          token_by_node=_token_by_node(state_usage_records(state)),
                          budget_summary=state.get("budget_summary") or {})
            if report != state["evaluation_report"] and not snapshot.next:
                self.graph.update_state(self._config(session_id), {"evaluation_report": report}, as_node="self_check")
                snapshot = self.graph.get_state(self._config(session_id))
            state["evaluation_report"] = report
        if state.get('usage_schema_version') == 'usage-v1':
            from app.telemetry.summary import summarize_usage
            state['usage_summary'] = summarize_usage(state_usage_records(state))
        state["session_phase"] = "finished" if state.get("evaluation_report") else "awaiting_answer"
        checkpoint_id = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id", "")
        return dump_state_for_store(state), checkpoint_id

    def _outcome(self, session_id: str, state: dict, replayed: bool = False) -> dict:
        return {"session_id": session_id, "state": state,
                "question": state.get("current_question", ""),
                "report": state.get("evaluation_report") or None,
                "done": bool(state.get("evaluation_report")), "replayed": replayed}

    def _repair_answer(self, session_id: str, request_id: str) -> bool:
        snapshot = self.graph.get_state(self._config(session_id))
        values = snapshot.values or {}
        if (values.get("last_completed_answer_request_id") == request_id
                and (tuple(snapshot.next) == ("assess",) or not snapshot.next)):
            state, checkpoint_id = self._projection(session_id)
            self.store.finish_answer(session_id, request_id, state, checkpoint_id)
            return True
        return False

    def _save_failure(self, session_id: str, exc: Exception):
        state = dump_state_for_store(dict(self.graph.get_state(self._config(session_id)).values or {}))
        state.update(error=str(exc) if isinstance(exc, NodeExecutionError) else "execution_failed",
                     session_phase="failed")
        if getattr(exc, 'usage_info', None):
            from app.telemetry.summary import usage_record, summarize_usage
            state['usage_records'] = list(state.get('usage_records') or []) + [usage_record(getattr(exc, 'node', 'failed'), exc.usage_info)]
            state['usage_summary'] = summarize_usage(state['usage_records'])
        if getattr(exc, 'budget', None):
            state['failed_budget'] = exc.budget
        state.update(ledger_projection(self.store, session_id))
        if state.get('llm_ledger_version') == 'ledger-v1':
            from app.telemetry.summary import summarize_usage
            state['usage_summary'] = summarize_usage(state_usage_records(state))
        self.store.update(session_id, state)

    def start_session(
        self,
        jd_text: str,
        resume_text: str,
        session_name: str = "",
    ) -> dict:
        if not str(jd_text or "").strip() or not str(resume_text or "").strip():
            raise ValueError("请上传有效的简历与岗位要求材料")
        session_id = uuid.uuid4().hex
        with self._session_lock(session_id):
            if session_id in self._busy:
                raise RuntimeError("该会话正在生成中，请稍候。")
            self._busy.add(session_id)
            try:
                self.store.create(
                    session_id,
                    jd_text=jd_text,
                    resume_text=resume_text,
                    state={"session_phase": "generating"},
                    name=session_name,
                )
                try:
                    with metering_scope(self.store, session_id):
                        state = self.graph.invoke(
                            {"jd_text": jd_text, "resume_text": resume_text,
                             "assessment_protocol_version": "practice-v1", "usage_schema_version": "usage-v1"},
                            config=self._config(session_id),
                        )
                except NodeExecutionError as exc:
                    self._save_failure(session_id, exc)
                    raise ServiceError(str(exc), 502) from exc
                state["session_phase"] = "finished" if state.get("evaluation_report") else "awaiting_answer"
                self.store.update(session_id, dump_state_for_store(state))
                return {
                    "session_id": session_id,
                    "state": state,
                    "question": state.get("current_question", ""),
                }
            finally:
                self.store.release_operation(session_id, "start:" + session_id)
                self._busy.discard(session_id)

    def submit_answer(self, session_id: str, answer: str,
                      answer_request_id: str | None = None,
                      expected_question_version: int | None = None) -> dict:
        # CLI callers may create fresh operations; HTTP callers MUST supply both.
        request_id = str(uuid.UUID(answer_request_id)) if answer_request_id else str(uuid.uuid4())
        if expected_question_version is None:
            expected_question_version = int(self.restore(session_id)["state"].get("question_version", 0))
        claim = self.store.claim_answer(session_id, request_id, expected_question_version, answer)
        if claim["status"] != "claimed":
            if claim["status"] != "succeeded":
                self._repair_answer(session_id, request_id)
                claim = self.store.get_answer_request(session_id, request_id) or claim
            if claim["status"] == "succeeded":
                return self._outcome(session_id, json.loads(claim["response_json"]), replayed=True)
            if claim["status"] == "recovery_required":
                raise ServiceError("recovery_required")
            return {"processing": True, "session_id": session_id, "answer_request_id": request_id}
        with self._session_lock(session_id):
            try:
                config = self._config(session_id)
                values = {"current_answer": answer, "active_answer_request_id": request_id}
                if _is_explicit_stop_command(answer):
                    values["end_requested"] = True
                    # 显式命令本身不含作答内容，一律不参与评分
                    values["current_answer"] = ""
                self.graph.update_state(config, values)
                with metering_scope(self.store, session_id):
                    self.graph.invoke(None, config=config)
                state, checkpoint_id = self._projection(session_id)
                self.store.finish_answer(session_id, request_id, state, checkpoint_id)
                return self._outcome(session_id, state)
            except Exception as exc:
                # External LLM completion and graph checkpoint commit cannot be
                # atomically committed together. Never blindly rerun uncertainty.
                self.store.mark_recovery_required(session_id, request_id)
                if isinstance(exc, NodeExecutionError):
                    self._save_failure(session_id, exc)
                    raise ServiceError(str(exc), 502) from exc
                if isinstance(exc, ServiceError):
                    raise
                raise ServiceError("recovery_required", 503) from exc

    def answer_request_status(self, session_id: str, request_id: str) -> dict:
        record = self.store.get(session_id)
        if not record:
            raise ServiceError("session_not_found", 404)
        if record.get("status") == "deleting":
            raise ServiceError("session_deleting", 410)
        request_id = str(uuid.UUID(request_id))
        record = self.store.get_answer_request(session_id, request_id)
        if not record:
            raise ServiceError("answer_request_not_found", 404)
        if record["status"] != "succeeded":
            self._repair_answer(session_id, request_id)
            record = self.store.get_answer_request(session_id, request_id) or record
        if record["status"] == "succeeded":
            return self._outcome(session_id, json.loads(record["response_json"]), replayed=True)
        return {"processing": record["status"] == "processing", "status": record["status"],
                "session_id": session_id, "answer_request_id": request_id}

    def stop_session(self, session_id: str) -> dict:
        record = self.store.get(session_id)
        if not record:
            raise ValueError("Session not found")
        with self._mutation(session_id, "stop"):
            record = self.store.get(session_id) or {}
            if record.get("status") == "failed":
                raise ServiceError("session_failed")
            if session_id in self._busy:
                raise RuntimeError("该会话正在处理中，请稍候再结束。")
            if record.get("status") == "completed":
                restored = self.restore(session_id)
                return restored
            snapshot = self.graph.get_state(self._config(session_id))
            if snapshot.values.get("evaluation_report") and not snapshot.next:
                state, _ = self._projection(session_id)
                self.store.update(session_id, state)
                return self._outcome(session_id, state)
            self._busy.add(session_id)
            try:
                config = self._config(session_id)
                self.graph.update_state(
                    config,
                    {"current_answer": "", "end_requested": True},
                )
                with metering_scope(self.store, session_id):
                    self.graph.invoke(None, config=config)
                state, _ = self._projection(session_id)
                state["session_phase"] = "finished"
                self.store.update(session_id, state)
                return {
                    "session_id": session_id,
                    "state": state,
                    "report": state.get("evaluation_report"),
                    "done": True,
                }
            except NodeExecutionError as exc:
                self._save_failure(session_id, exc)
                raise ServiceError(str(exc), 502) from exc
            finally:
                self._busy.discard(session_id)

    def restore(self, session_id: str) -> dict:
        record = self.store.get(session_id)
        if not record:
            raise ServiceError("session_not_found", 404)
        if record.get("status") == "deleting":
            raise ServiceError("session_deleting", 410)
        operation = record.get("active_operation") or ""
        if operation.startswith("answer:"):
            self._repair_answer(session_id, operation.split(":", 1)[1])
        elif not operation and record.get("status") != "failed":
            # A finished graph is authoritative for recovering a failed report
            # projection. Do not project a graph while another mutator is active.
            snapshot = self.graph.get_state(self._config(session_id))
            safe_boundary = tuple(snapshot.next) == ("assess",) or (
                snapshot.values.get("evaluation_report") and not snapshot.next
            )
            if snapshot.values and safe_boundary:
                state, _ = self._projection(session_id)
                if state != self.store.load_state(session_id):
                    # A GET racing with another worker must not overwrite its
                    # active operation or a newer committed business snapshot.
                    self.store.project_if_idle(session_id, state, record["state_json"])
        state = self.store.load_state(session_id)
        return {
            "session_id": session_id,
            "state": state,
            "question": state.get("current_question", ""),
            "report": state.get("evaluation_report"),
            "done": bool(state.get("evaluation_report")),
        }

    def delete_session(self, session_id: str) -> bool:
        """Remove private business data, replay ledger and graph history.

        Uploaded originals are already removed by the extraction endpoints;
        this application creates no retained session attachment/report files.
        Keep a durable deletion fence on failure instead of exposing partial data.
        """
        # The process-wide lock is intentionally acquired outside the
        # per-service lock: distinct InterviewService instances may point at
        # the same owner database and checkpoint file.
        with _delete_lock(self.store, session_id):
            with self._session_lock(session_id):
                if session_id in self._busy:
                    raise ServiceError("session_busy")
                if not self.store.begin_delete(session_id):
                    return False
                try:
                    delete_checkpoints(self._checkpointer, session_id)
                    self.store.finish_delete(session_id)
                except Exception as exc:
                    raise ServiceError("session_cleanup_failed", 503) from exc
                return True

    def rename_session(self, session_id: str, name: str) -> dict:
        """重命名本会话；唯一性校验由 SessionStore.rename 完成。"""
        record = self.store.get(session_id)
        if not record:
            raise ValueError("Session not found")
        with self._mutation(session_id, "rename"):
            if session_id in self._busy:
                raise RuntimeError("该会话正在处理中，请稍候再重命名。")
            self.store.rename(session_id, name)
            updated = self.store.get(session_id) or {}
            return {"id": session_id, "name": updated.get("name", "")}

    def update_job_title(
        self,
        session_id: str,
        job_title: str,
        regenerate: bool = True,
    ) -> dict:
        """P1/A1：修正系统识别到的岗位名。

        - 任何阶段都允许修正岗位名（用于后续提问的面试官身份与岗位政策）
        - **仅当会话还没产生任何评分记录时**，才允许顺带**重新出题**
          （重新出题会换掉整份题单，已在答题的会话不允许，避免答案与题目错位）
        """
        record = self.store.get(session_id)
        if not record:
            raise ValueError("Session not found")
        cleaned = str(job_title or "").strip()
        if not cleaned:
            raise ValueError("岗位名不能为空")
        if len(cleaned) > 60:
            raise ValueError("岗位名过长（最多 60 字）")
        with self._mutation(session_id, "job_title"):
            if (self.store.get(session_id) or {}).get("status") != "interviewing":
                raise ServiceError("session_not_answerable")
            if session_id in self._busy:
                raise RuntimeError("该会话正在处理中，请稍候再修改。")
            config = self._config(session_id)
            state = self.store.load_state(session_id) or {}
            assessments = state.get("assessments") or []
            if regenerate and assessments:
                raise RuntimeError("该会话已经开始答题，只能修正岗位名，不能重新出题。")
            role_profile = dict(state.get("role_profile") or {})
            role_profile["job_title"] = cleaned
            updates: dict = {"job_title": cleaned, "role_profile": role_profile,
                             "question_version": int(state.get("question_version", 0)) + 1,
                             "current_answer": ""}
            regenerated = False
            self.graph.update_state(config, updates)
            if regenerate:
                # 重新出题：直接调用 plan/ask 的纯函数，复用当前 state
                from app.nodes import ask as ask_node
                from app.nodes import plan as plan_node

                fresh = dict(state)
                fresh["job_title"] = cleaned
                fresh["role_profile"] = role_profile
                fresh["current_question_index"] = 0
                try:
                    with metering_scope(self.store, session_id):
                        with node_scope("plan"):
                            planned = plan_node.plan(fresh)
                        fresh.update(planned)
                        with node_scope("ask"):
                            asked = ask_node.ask(fresh)
                except SessionBudgetExceeded as exc:
                    self.graph.update_state(config,{"budget_exhausted":True,"end_requested":True})
                    projected,_ = self._projection(session_id)
                    self.store.update(session_id,projected)
                    raise ServiceError("session_budget_exhausted") from exc
                fresh.update(asked)
                self.graph.update_state(
                    config,
                    {
                        "job_title": cleaned,
                        "role_profile": role_profile,
                        "question_plan": planned.get("question_plan", []),
                        "plan_question_count": planned.get("plan_question_count", 0),
                        "difficulty": planned.get("difficulty", "medium"),
                        "current_question_index": 0,
                        "current_question": asked.get("current_question", ""),
                        "current_answer": "",
                        "question_version": asked.get("question_version", int(state.get("question_version", 0)) + 1),
                        "conversation_history": asked.get("conversation_history", []),
                        "usage_records": (planned.get("usage_records", []) + asked.get("usage_records", [])),
                    },
                )
                regenerated = True
            else:
                # Changing the role changes the answer contract even if wording
                # is unchanged. Invalidate previously displayed question versions.
                self.graph.update_state(config, {"question_version": int(state.get("question_version", 0)) + 1,
                                                 "current_answer": ""})
            state = self.graph.get_state(config).values
            state = {**state, **ledger_projection(self.store, session_id)}
            self.store.update(session_id, dump_state_for_store(state))
            return {
                "session_id": session_id,
                "state": state,
                "question": state.get("current_question", ""),
                "regenerated": regenerated,
            }


def get_service() -> InterviewService:
    return InterviewService()
