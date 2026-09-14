import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "../api/client";
import type {
  AnswerRequest,
  AnswerResult,
  ChatMessage,
  Report,
  SessionPayload,
  SessionSummary,
} from "../types";

const ACTIVE_SESSION_KEY = "interview_active_session";
const PENDING_KEY = "interview_pending_answer_v1";

interface PendingAnswer {
  sessionId: string;
  body: AnswerRequest;
  recoveryRequired?: boolean;
}

function readPending(): PendingAnswer | null {
  try {
    const saved = sessionStorage.getItem(PENDING_KEY);
    if (!saved) return null;
    const pending = JSON.parse(saved) as PendingAnswer;
    return pending.sessionId && typeof pending.body?.answer === "string" &&
      /^[0-9a-f-]{36}$/i.test(pending.body.answer_request_id) &&
      Number.isInteger(pending.body.expected_question_version) && pending.body.expected_question_version >= 1
      ? pending : null;
  } catch {
    return null;
  }
}

/** 名称留空时生成默认标识：面试会话 09-10 15:30（重名自动加序号） */
function defaultSessionName(existing: SessionSummary[]): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  const base = `面试会话 ${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(
    now.getHours()
  )}:${pad(now.getMinutes())}`;
  const taken = new Set(existing.map((item) => (item.name || "").trim().toLowerCase()));
  if (!taken.has(base.toLowerCase())) return base;
  let index = 2;
  while (taken.has(`${base} (${index})`.toLowerCase())) index += 1;
  return `${base} (${index})`;
}

export interface ViewingSession {
  sessionId: string;
  name: string;
  messages: ChatMessage[];
  question: string;
  questionVersion: number;
  report: Report | null;
  status: string;
  // M23：系统识别到的岗位与题量
  jobTitle: string;
  domain: string;
  planQuestionCount: number;
  effectiveSampleCount: number;
}

export interface InterviewState {
  // 当前面试（active）：所有写入都只影响它
  resumeText: string;
  resumeName: string;
  resumeMethod: string;
  jdText: string;
  jdOcr: string;
  sessionName: string;
  activeSessionId: string;
  status: string;
  messages: ChatMessage[];
  question: string;
  questionVersion: number;
  pendingAnswer: boolean;
  recoveryRequired: boolean;
  report: Report | null;
  // M23：系统识别到的岗位与题量（TopBar 展示，供用户核对）
  jobTitle: string;
  domain: string;
  planQuestionCount: number;
  effectiveSampleCount: number;
  // 历史会话只读查看（viewing）：打开时不覆盖 active
  viewing: ViewingSession | null;
  sessions: SessionSummary[];
  nameError: string;
  loading: boolean;
  error: string;
}

const initialState: InterviewState = {
  resumeText: "",
  resumeName: "",
  resumeMethod: "",
  jdText: "",
  jdOcr: "",
  sessionName: "",
  activeSessionId: "",
  status: "idle",
  messages: [],
  question: "",
  questionVersion: 0,
  pendingAnswer: false,
  recoveryRequired: false,
  report: null,
  jobTitle: "",
  domain: "",
  planQuestionCount: 0,
  effectiveSampleCount: 0,
  viewing: null,
  sessions: [],
  nameError: "",
  loading: false,
  error: "",
};

function toViewing(payload: SessionPayload): ViewingSession {
  return {
    sessionId: payload.session_id,
    name: payload.name,
    messages: payload.messages,
    question: payload.question,
    questionVersion: payload.question_version,
    report: payload.report,
    status: payload.status,
    jobTitle: payload.job_title ?? "",
    domain: payload.domain ?? "",
    planQuestionCount: payload.plan_question_count ?? 0,
    effectiveSampleCount: payload.effective_sample_count ?? 0,
  };
}

export function useInterview() {
  const [state, setState] = useState<InterviewState>(initialState);
  const pendingRef = useRef<PendingAnswer | null>(readPending());
  const mutationInFlight = useRef(false);
  const activeSnapshot = useRef({ sessionId: "", questionVersion: 0, done: false });

  const patch = useCallback((next: Partial<InterviewState>) => {
    setState((prev) => ({ ...prev, ...next }));
  }, []);

  const applyPayload = useCallback(
    (payload: SessionPayload, fallbackName = "") => {
      if (payload.processing) return;
      const current = activeSnapshot.current;
      if (current.sessionId === payload.session_id && (
        payload.question_version < current.questionVersion || (current.done && !payload.done)
      )) {
        patch({ loading: false });
        return;
      }
      activeSnapshot.current = { sessionId: payload.session_id, questionVersion: payload.question_version, done: payload.done };
      if (payload.session_id) {
        localStorage.setItem(ACTIVE_SESSION_KEY, payload.session_id);
      }
      setState((prev) => ({
        ...prev,
        activeSessionId: payload.session_id,
        status: payload.status,
        sessionName: payload.name || fallbackName || prev.sessionName,
        messages: payload.messages,
        question: payload.question,
        questionVersion: payload.question_version,
        pendingAnswer: Boolean(pendingRef.current),
        recoveryRequired: Boolean(pendingRef.current?.recoveryRequired),
        report: payload.report,
        jobTitle: payload.job_title ?? "",
        domain: payload.domain ?? "",
        planQuestionCount: payload.plan_question_count ?? 0,
        effectiveSampleCount: payload.effective_sample_count ?? 0,
        viewing: null,
        loading: false,
        error: "",
      }));
    },
    [patch]
  );

  const clearPending = useCallback(() => {
    sessionStorage.removeItem(PENDING_KEY);
    pendingRef.current = null;
    patch({ pendingAnswer: false, recoveryRequired: false });
  }, [patch]);

  const refreshSessions = useCallback(async () => {
    try {
      const sessions = await api.listSessions();
      patch({ sessions });
    } catch (error) {
      patch({ error: (error as Error).message });
    }
  }, [patch]);

  const settleAnswer = useCallback(async (pending: PendingAnswer, first: AnswerResult) => {
    let result = first;
    if (result.status === "recovery_required") {
      throw new ApiError(409, "recovery_required");
    }
    // Only GET polling, never a fresh answer ID or a second POST during processing.
    for (let attempt = 0; result.processing && attempt < 20; attempt += 1) {
      if (result.status === "recovery_required") {
        throw new ApiError(409, "recovery_required");
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
      result = await api.answerStatus(pending.sessionId, pending.body.answer_request_id);
      if (result.status === "recovery_required") {
        throw new ApiError(409, "recovery_required");
      }
    }
    if (result.processing) throw new Error("回答仍在处理中，原请求已保留；稍后重试会查询同一请求");
    if (!("question_version" in result) || !Number.isInteger(result.question_version)) {
      throw new Error("回答状态响应不完整，已保留原请求，请重试恢复");
    }
    // A replay can describe an older question. Always display the current projection.
    const complete = result as SessionPayload;
    const latest = complete.replayed ? await api.getSession(pending.sessionId) : complete;
    clearPending();
    applyPayload(latest);
    void refreshSessions();
  }, [applyPayload, clearPending, refreshSessions]);

  useEffect(() => {
    void (async () => {
      await refreshSessions();
      const saved = localStorage.getItem(ACTIVE_SESSION_KEY);
      if (!saved) return;
      try {
        const payload = await api.getSession(saved);
        applyPayload(payload);
        const pending = pendingRef.current;
        if (pending) {
          patch({ pendingAnswer: true, recoveryRequired: Boolean(pending.recoveryRequired), error: "有未确认的回答，请恢复原请求后再继续面试" });
        }
      } catch (error) {
        // Transient failures and a bad credential must not destroy recovery data.
        patch({ error: (error as Error).message });
      }
    })();
  }, [refreshSessions, applyPayload, patch]);

  const run = useCallback(
    async <T,>(task: () => Promise<T>, onSuccess: (value: T) => void) => {
      if (mutationInFlight.current) return;
      mutationInFlight.current = true;
      patch({ loading: true, error: "" });
      try {
        const value = await task();
        onSuccess(value);
      } catch (error) {
        patch({ loading: false, error: (error as Error).message });
      } finally {
        mutationInFlight.current = false;
      }
    },
    [patch]
  );

  const uploadResume = useCallback(
    (file: File) =>
      run(
        () => api.uploadResume(file),
        (result) =>
          patch({
            resumeText: result.text,
            resumeName: result.file_name,
            resumeMethod: result.method,
            loading: false,
            error: "",
          })
      ),
    [run, patch]
  );

  const uploadJdImage = useCallback(
    (file: File) =>
      run(
        () => api.uploadJdImage(file),
        // One editable JD is authoritative; don't silently merge stale OCR later.
        (result) => patch({ jdOcr: result.text, jdText: result.text, loading: false, error: "" })
      ),
    [run, patch]
  );

  const start = useCallback(
    () => {
      if (pendingRef.current) {
        patch({ error: "请先恢复未确认的回答，不能创建新面试" });
        return;
      }
      const name = state.sessionName.trim() || defaultSessionName(state.sessions);
      return run(
        () =>
          api.startSession({
            jd_text: state.jdText.trim(),
            resume_text: state.resumeText,
            session_name: name,
          }),
        (payload) => {
          applyPayload(payload, name);
          void refreshSessions();
        }
      );
    },
    [
      run,
      state.jdText,
      state.jdOcr,
      state.resumeText,
      state.sessionName,
      state.sessions,
      applyPayload,
      refreshSessions,
      patch,
    ]
  );

  const submitPending = useCallback(async (pending: PendingAnswer, retry: boolean) => {
    if (mutationInFlight.current) return;
    mutationInFlight.current = true;
    patch({ loading: true, pendingAnswer: true, error: "" });
    try {
      let result: AnswerResult;
      if (retry) {
        try {
          result = await api.answerStatus(pending.sessionId, pending.body.answer_request_id);
        } catch (error) {
          if (!(error instanceof ApiError) || error.status !== 404) throw error;
          // No ledger entry: resend the exact frozen request, never a new UUID.
          result = await api.answer(pending.sessionId, pending.body);
        }
      } else {
        result = await api.answer(pending.sessionId, pending.body);
      }
      await settleAnswer(pending, result);
    } catch (error) {
      if (error instanceof ApiError && error.code === "recovery_required") {
        pending.recoveryRequired = true;
        pendingRef.current = pending;
        try {
          sessionStorage.setItem(PENDING_KEY, JSON.stringify(pending));
        } catch {
          // In-memory state still protects this tab; the backend gate remains closed.
        }
        patch({ recoveryRequired: true });
      }
      const staleCodes = ["stale_question", "stale_question_version", "question_version_conflict", "session_completed", "session_not_answerable"];
      if (error instanceof ApiError && error.status === 409 && staleCodes.includes(error.code)) {
        try {
          const latest = await api.getSession(pending.sessionId);
          clearPending();
          applyPayload(latest);
        } catch {
          // Keep pending if reconciliation failed; do not invent a new request.
        }
      }
      patch({ loading: false, error: (error as Error).message });
    } finally {
      mutationInFlight.current = false;
    }
  }, [applyPayload, clearPending, patch, settleAnswer]);

  const retryPendingAnswer = useCallback(() => {
    const pending = pendingRef.current;
    if (pending) return submitPending(pending, true);
  }, [submitPending]);

  const preserveFailedSessionAndReset = useCallback(() => {
    if (mutationInFlight.current || !pendingRef.current?.recoveryRequired) return;
    // Explicit escape only after confirmed recovery_required. No server DELETE,
    // gate release, new answer ID, or graph invocation for the old session.
    localStorage.removeItem(ACTIVE_SESSION_KEY);
    clearPending();
    activeSnapshot.current = { sessionId: "", questionVersion: 0, done: false };
    setState((prev) => ({
      ...initialState,
      resumeText: prev.resumeText, resumeName: prev.resumeName, resumeMethod: prev.resumeMethod,
      jdText: prev.jdText, jdOcr: prev.jdOcr, sessions: prev.sessions,
      error: "已保留故障会话和服务端请求记录，不会重执行旧回答；可开始一场新的面试。",
    }));
  }, [clearPending]);

  const answer = useCallback((text: string) => {
    if (mutationInFlight.current) return;
    if (pendingRef.current) return submitPending(pendingRef.current, true);
    const active = activeSnapshot.current;
    if (!active.sessionId || active.done || active.questionVersion < 1 || !text.trim()) return;
    const pending: PendingAnswer = {
      sessionId: active.sessionId,
      body: { answer: text, answer_request_id: crypto.randomUUID(), expected_question_version: active.questionVersion },
    };
    try {
      // Tab-scoped persistence survives reload without sharing resume text across tabs.
      sessionStorage.setItem(PENDING_KEY, JSON.stringify(pending));
    } catch {
      patch({ error: "无法保存待确认回答；未发送请求，请检查浏览器存储权限" });
      return;
    }
    pendingRef.current = pending;
    setState((prev) => ({ ...prev, messages: [...prev.messages, { role: "user", content: text }] }));
    return submitPending(pending, false);
  }, [patch, submitPending]);

  const stop = useCallback(
    () => {
      if (pendingRef.current) {
        patch({ error: "回答尚未确认，请先恢复原请求，不能结束面试" });
        return;
      }
      const sessionId = activeSnapshot.current.sessionId;
      if (!sessionId) return;
      return run(
        () => api.stop(sessionId),
        (payload) => {
          applyPayload(payload, state.sessionName);
          void refreshSessions();
        }
      );
    },
    [run, state.activeSessionId, state.sessionName, applyPayload, refreshSessions, patch]
  );

  /** 只读打开历史会话：不触碰 active 面试的任何字段。 */
  /** P1/A1：修正岗位名（未答题时可重新出题）。 */
  const updateJobTitle = useCallback(
    (jobTitle: string, regenerate?: boolean) => {
      if (pendingRef.current) {
        patch({ error: "回答尚未确认，请先恢复原请求，不能修改岗位或重新出题" });
        return;
      }
      const sessionId = activeSnapshot.current.sessionId;
      if (!sessionId) return;
      return run(
        () => api.updateJobTitle(sessionId, jobTitle,
          regenerate ?? !state.messages.some((message) => message.role === "user")),
        (payload) => {
          applyPayload(payload, state.sessionName);
          void refreshSessions();
        }
      );
    },
    [run, state.activeSessionId, state.sessionName, state.messages, applyPayload, refreshSessions, patch]
  );

  const openHistory = useCallback(
    (sessionId: string) =>
      run(
        () => api.getSession(sessionId),
        (payload) => patch({ viewing: toViewing(payload), loading: false, error: "" })
      ),
    [run, patch]
  );

  /** 关闭历史视图，回到当前面试（active 从未被修改）。 */
  const closeHistory = useCallback(() => patch({ viewing: null }), [patch]);

  const deleteSession = useCallback(
    async (sessionId: string) => {
      if (mutationInFlight.current || pendingRef.current?.sessionId === sessionId) {
        patch({ error: "会话正在处理或有未确认回答，不能删除" });
        return;
      }
      mutationInFlight.current = true;
      patch({ loading: true, error: "" });
      try {
        await api.deleteSession(sessionId);
        localStorage.removeItem(`interview_draft_${sessionId}`);
        if (localStorage.getItem(ACTIVE_SESSION_KEY) === sessionId) {
          localStorage.removeItem(ACTIVE_SESSION_KEY);
          activeSnapshot.current = { sessionId: "", questionVersion: 0, done: false };
          setState((prev) => ({
            ...prev,
            resumeText: "",
            resumeName: "",
            resumeMethod: "",
            jdText: "",
            jdOcr: "",
            activeSessionId: "",
            status: "idle",
            sessionName: "",
            messages: [],
            question: "",
            questionVersion: 0,
            report: null,
            jobTitle: "",
            domain: "",
            planQuestionCount: 0,
            effectiveSampleCount: 0,
            viewing: null,
            loading: false,
            error: "",
          }));
        } else {
          patch({ loading: false });
        }
        setState((prev) =>
          prev.viewing?.sessionId === sessionId ? { ...prev, viewing: null } : prev
        );
        await refreshSessions();
      } catch (error) {
        patch({ loading: false, error: (error as Error).message });
      } finally {
        mutationInFlight.current = false;
      }
    },
    [patch, refreshSessions]
  );

  /** 重命名会话（新建与重命名共用同一套重名校验规则）。 */
  const renameSession = useCallback(
    async (sessionId: string, name: string): Promise<boolean> => {
      const cleaned = name.trim();
      if (!cleaned) {
        patch({ nameError: "会话名称不能为空" });
        return false;
      }
      const duplicate = state.sessions.some(
        (item) =>
          item.id !== sessionId &&
          (item.name || "").trim().toLowerCase() === cleaned.toLowerCase()
      );
      if (duplicate) {
        patch({ nameError: "该会话名称已存在，请更换一个名称" });
        return false;
      }
      try {
        const updated = await api.renameSession(sessionId, cleaned);
        setState((prev) => ({
          ...prev,
          nameError: "",
          sessionName:
            prev.activeSessionId === sessionId ? updated.name : prev.sessionName,
          viewing:
            prev.viewing && prev.viewing.sessionId === sessionId
              ? { ...prev.viewing, name: updated.name }
              : prev.viewing,
        }));
        await refreshSessions();
        return true;
      } catch (error) {
        patch({ nameError: (error as Error).message });
        return false;
      }
    },
    [patch, refreshSessions, state.sessions]
  );

  return {
    state,
    patch,
    start,
    answer,
    retryPendingAnswer,
    preserveFailedSessionAndReset,
    stop,
    updateJobTitle,
    uploadResume,
    uploadJdImage,
    openHistory,
    closeHistory,
    deleteSession,
    renameSession,
    refreshSessions,
  };
}
