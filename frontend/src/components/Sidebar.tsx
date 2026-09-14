import { Briefcase, Check, FileText, History, Pencil, Trash2, Upload, X } from "lucide-react";
import { useState } from "react";
import type { ChangeEvent } from "react";

import type { SessionSummary } from "../types";
import type { InterviewState } from "../hooks/useInterview";
import { isDuplicateName, SessionNameInput } from "./SessionNameInput";
import { PRODUCT_NAME, PRODUCT_DESCRIPTION, sessionStatusLabel } from "../branding";

interface Props {
  state: InterviewState;
  onPatch: (patch: Partial<InterviewState>) => void;
  onResume: (file: File) => void;
  onJdImage: (file: File) => void;
  onStart: () => void;
  onView: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onRename: (sessionId: string, name: string) => Promise<boolean>;
}

export function Sidebar({
  state,
  onPatch,
  onResume,
  onJdImage,
  onStart,
  onView,
  onDelete,
  onRename,
}: Props) {
  const ready = Boolean(state.resumeText.trim() && state.jdText.trim());
  const trimmedName = state.sessionName.trim();
  const nameTaken =
    Boolean(trimmedName) &&
    isDuplicateName(trimmedName, state.sessions, state.activeSessionId);
  const [editingId, setEditingId] = useState("");
  const [draftName, setDraftName] = useState("");
  const [localError, setLocalError] = useState("");
  const pick = (handler: (file: File) => void) => (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) handler(file);
    event.target.value = "";
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-dot" />
        <div>
          <div className="brand-title">{PRODUCT_NAME}</div>
          <div className="brand-sub">{PRODUCT_DESCRIPTION}</div>
        </div>
      </div>

      <section className="sidebar-section">
        <header className="sidebar-title">
          <FileText size={16} strokeWidth={1.6} aria-hidden="true" />
          <span>简历</span>
        </header>
        <label className="dropzone">
          <Upload size={18} strokeWidth={1.6} aria-hidden="true" />
          <span>{state.resumeName || "上传 PDF / Word / TXT / 图片"}</span>
          <input
            type="file"
            aria-label="上传简历文件"
            accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.bmp,.webp"
            onChange={pick(onResume)}
          />
        </label>
        <textarea
          className="form-input"
          rows={5}
          placeholder="识别结果将显示在这里，可编辑"
          value={state.resumeText}
          onChange={(event) => onPatch({ resumeText: event.target.value })}
        />
        {state.resumeText.trim() && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => onPatch({ resumeText: "", resumeName: "", resumeMethod: "" })}
          >
            清除简历
          </button>
        )}
      </section>

      <section className="sidebar-section">
        <header className="sidebar-title">
          <Briefcase size={16} strokeWidth={1.6} aria-hidden="true" />
          <span>岗位要求</span>
        </header>
        <label className="dropzone dropzone--sm">
          <Upload size={15} strokeWidth={1.6} aria-hidden="true" />
          <span>{state.jdOcr ? "已识别 JD 图片" : "上传 JD 图片（可选）"}</span>
          <input
            type="file"
            aria-label="上传 JD 图片"
            accept=".png,.jpg,.jpeg,.bmp,.webp"
            onChange={pick(onJdImage)}
          />
        </label>
        <textarea
          className="form-input"
          rows={5}
          placeholder="粘贴岗位 JD 文本"
          value={state.jdText}
          onChange={(event) => onPatch({ jdText: event.target.value })}
        />
      </section>

      <section className="sidebar-section">
        <SessionNameInput
          value={state.sessionName}
          onChange={(value) => onPatch({ sessionName: value })}
          currentSessionId={state.activeSessionId}
          sessions={state.sessions}
        />
        <button
          type="button"
          className="btn btn-primary"
          disabled={!ready || state.loading || nameTaken}
          onClick={onStart}
        >
          开始面试
        </button>
        {!ready && <p className="text-aux">请先提供简历与岗位要求材料</p>}
      </section>

      <section className="sidebar-section">
        <header className="sidebar-title">
          <History size={16} strokeWidth={1.6} aria-hidden="true" />
          <span>历史会话</span>
        </header>
        {state.sessions.length === 0 && <p className="text-aux">暂无历史会话</p>}
        <ul className="session-list">
          {state.sessions.map((item: SessionSummary) => {
            const isViewing = state.viewing?.sessionId === item.id;
            const isHighlighted =
              isViewing || (!state.viewing && item.id === state.activeSessionId);
            const startEdit = () => {
              setEditingId(item.id);
              setDraftName(item.name || "");
              setLocalError("");
            };
            const commitEdit = async () => {
              const cleaned = draftName.trim();
              if (!cleaned) {
                setLocalError("会话名称不能为空");
                return;
              }
              if (isDuplicateName(cleaned, state.sessions, item.id)) {
                setLocalError("该会话名称已存在，请换一个名称");
                return;
              }
              const ok = await onRename(item.id, cleaned);
              if (ok) {
                setEditingId("");
                setDraftName("");
                setLocalError("");
              } else {
                setLocalError(state.nameError || "重命名失败，请重试");
              }
            };
            if (editingId === item.id) {
              return (
                <li key={item.id} className="session-edit">
                  <SessionNameInput
                    value={draftName}
                    onChange={setDraftName}
                    currentSessionId={item.id}
                    sessions={state.sessions}
                    autoFocus
                    placeholder="会话名称"
                    onCommit={() => void commitEdit()}
                    onCancel={() => {
                      setEditingId("");
                      setLocalError("");
                    }}
                  />
                  <div className="session-edit__actions">
                    <button
                      type="button"
                      className="session-action"
                      aria-label="确认重命名"
                      title="确认"
                      onClick={() => void commitEdit()}
                    >
                      <Check size={15} strokeWidth={1.6} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="session-action"
                      aria-label="取消重命名"
                      title="取消"
                      onClick={() => {
                        setEditingId("");
                        setLocalError("");
                      }}
                    >
                      <X size={15} strokeWidth={1.6} aria-hidden="true" />
                    </button>
                  </div>
                  {(localError || state.nameError) && (
                    <p className="inline-error" role="status">
                      {localError || state.nameError}
                    </p>
                  )}
                </li>
              );
            }
            return (
              <li key={item.id} className="session-row">
                <button
                  type="button"
                  className={isHighlighted ? "session-item active" : "session-item"}
                  onClick={() => onView(item.id)}
                  aria-current={isHighlighted}
                >
                  <span className="session-item__name">
                    {item.name || item.id.slice(0, 8)}
                  </span>
                  <span className="session-item__meta">
                    {isViewing ? "查看中 · " : ""}
                    {sessionStatusLabel(item.status)}
                  </span>
                </button>
                <button
                  type="button"
                  className="session-action"
                  aria-label={`重命名会话 ${item.name || item.id.slice(0, 8)}`}
                  title="重命名会话"
                  onClick={startEdit}
                >
                  <Pencil size={15} strokeWidth={1.6} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="session-delete"
                  aria-label={`删除会话 ${item.name || item.id.slice(0, 8)}`}
                  title="删除会话"
                  onClick={() => {
                    const label = item.name || item.id.slice(0, 8);
                    if (window.confirm(`确定删除会话「${label}」吗？`)) {
                      onDelete(item.id);
                    }
                  }}
                >
                  <Trash2 size={15} strokeWidth={1.6} aria-hidden="true" />
                </button>
              </li>
            );
          })}
        </ul>
      </section>
    </aside>
  );
}
