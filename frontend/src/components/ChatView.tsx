import { ArrowLeft, ArrowUp, MessagesSquare, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { ChatMessage, Report } from "../types";
import { MessageBubble } from "./MessageBubble";

interface Props {
  sessionId: string;
  messages: ChatMessage[];
  report: Report | null;
  loading: boolean;
  error: string;
  onAnswer: (text: string) => void;
  onStop: () => void;
  pendingAnswer?: boolean;
  recoveryRequired?: boolean;
  onRetry?: () => void;
  onPreserveFailed?: () => void;
  readOnly?: boolean;
  onBack?: () => void;
}

export function ChatView({
  sessionId,
  messages,
  report,
  loading,
  error,
  onAnswer,
  onStop,
  pendingAnswer = false,
  recoveryRequired = false,
  onRetry,
  onPreserveFailed,
  readOnly = false,
  onBack,
}: Props) {
  const draftKey = sessionId ? `interview_draft_${sessionId}` : "";
  const [draft, setDraft] = useState(() =>
    draftKey ? localStorage.getItem(draftKey) || "" : ""
  );
  const streamRef = useRef<HTMLDivElement>(null);
  const lastLengthRef = useRef(messages.length);

  useEffect(() => {
    setDraft(draftKey ? localStorage.getItem(draftKey) || "" : "");
  }, [draftKey]);

  // 新消息到达 / 进入思考态时自动锚定到最新内容
  useEffect(() => {
    const el = streamRef.current;
    if (!el) return;
    const grew = messages.length > lastLengthRef.current;
    lastLengthRef.current = messages.length;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 180;
    if (grew || nearBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages.length, loading]);

  useEffect(() => {
    const el = streamRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    lastLengthRef.current = messages.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const updateDraft = (value: string) => {
    setDraft(value);
    if (draftKey) localStorage.setItem(draftKey, value);
  };

  const send = () => {
    const text = draft.trim();
    if (!sessionId || !text || loading) return;
    setDraft("");
    if (draftKey) localStorage.removeItem(draftKey);
    onAnswer(text);
  };

  const hasReport = Boolean(report);

  return (
    <div className={hasReport ? "chat card chat--compact" : "chat card"}>
      {readOnly && (
        <div className="history-banner">
          <span className="history-banner__text">正在查看历史会话 · 只读</span>
          <button type="button" className="btn btn-secondary history-back" onClick={onBack}>
            <ArrowLeft size={14} strokeWidth={1.6} aria-hidden="true" />
            返回当前面试
          </button>
        </div>
      )}
      <div className="chat-body">
        <div className="chat-stream" ref={streamRef}>
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-state__icon" aria-hidden="true">
                <MessagesSquare size={20} strokeWidth={1.6} />
              </div>
              <div className="empty-state__title">准备开始</div>
              <div className="empty-state__text">
                左侧上传简历与岗位要求，点击“开始面试”
              </div>
            </div>
          )}
          {messages.map((message, index) => (
            <MessageBubble
              key={`${index}-${message.role}`}
              message={message}
              index={index}
            />
          ))}
          {loading && (
            <div className="thinking">
              <span className="thinking__dot" />
              <span className="thinking__dot" />
              <span className="thinking__dot" />
              <span className="thinking__label">面试官正在思考…</span>
            </div>
          )}
        </div>

      </div>

      {error && <div className="alert">{error}</div>}
      {pendingAnswer && !readOnly && !loading && (
        <div className="alert" role="status">
          {recoveryRequired && (
            <p>执行结果尚无法安全确认，旧会话已锁定，不会自动重复调用模型。</p>
          )}
          <button type="button" className="btn btn-secondary" disabled={loading} onClick={onRetry}>
            恢复原回答请求
          </button>
          {recoveryRequired && <button type="button" className="btn btn-secondary" disabled={loading} onClick={onPreserveFailed}>
            保留故障会话，开始新面试
          </button>}
        </div>
      )}

      {!report && !readOnly && !pendingAnswer && (
        <>
          <div className="chat-composer">
            <textarea
              className="chat-composer__input"
              rows={1}
              placeholder="输入你的回答..."
              value={draft}
              onChange={(event) => updateDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  send();
                }
              }}
            />
            <button
              type="button"
              className="chat-composer__send"
              onClick={send}
              disabled={!sessionId || loading || !draft.trim()}
              aria-label="发送回答"
              title="发送回答"
            >
              <ArrowUp size={18} strokeWidth={1.6} />
            </button>
          </div>
          <div className="chat-footer">
            <button
              type="button"
              className="btn btn-secondary stop"
              onClick={onStop}
              disabled={loading || messages.length === 0}
              aria-label="结束并评估"
            >
              <Square size={12} strokeWidth={1.6} aria-hidden="true" />
              结束并评估
            </button>
            <span className="text-aux">结束后生成评估报告，可在左侧历史会话中回看</span>
          </div>
        </>
      )}
    </div>
  );
}
