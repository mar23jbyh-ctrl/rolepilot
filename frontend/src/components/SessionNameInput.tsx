import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { SessionSummary } from "../types";

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** 当前正在命名 / 重命名的会话 id，用于排除自身重名误判 */
  currentSessionId?: string;
  /** 已有会话列表；不传时组件自行拉取（复用统一 api client，带上 owner 头） */
  sessions?: SessionSummary[];
  /** 把校验结果同步给父组件，便于禁用提交按钮 */
  onValidityChange?: (valid: boolean, message: string) => void;
  autoFocus?: boolean;
  placeholder?: string;
  onCommit?: () => void;
  onCancel?: () => void;
}

export function isDuplicateName(
  inputVal: string,
  sessionList: SessionSummary[],
  currentSessionId = ""
): boolean {
  const trimmed = inputVal.trim().toLowerCase();
  if (!trimmed) return false;
  return sessionList.some(
    (item) =>
      (item.name || "").trim().toLowerCase() === trimmed &&
      item.id !== currentSessionId
  );
}

export function SessionNameInput({
  value,
  onChange,
  currentSessionId = "",
  sessions,
  onValidityChange,
  autoFocus = false,
  placeholder = "会话名称（留空自动生成）",
  onCommit,
  onCancel,
}: Props) {
  const [sessionList, setSessionList] = useState<SessionSummary[]>(sessions ?? []);
  const [nameError, setNameError] = useState("");

  useEffect(() => {
    if (sessions) {
      setSessionList(sessions);
      return;
    }
    let alive = true;
    api
      .listSessions()
      .then((list) => {
        if (alive) setSessionList(list);
      })
      .catch(() => {
        /* 拉取失败时退化为后端校验，不阻断输入 */
      });
    return () => {
      alive = false;
    };
  }, [sessions]);

  const validateName = (inputVal: string) => {
    const trimmed = inputVal.trim();
    if (!trimmed) {
      setNameError("");
      onValidityChange?.(true, "");
      return true;
    }
    if (isDuplicateName(trimmed, sessionList, currentSessionId)) {
      const message = "该会话名称已存在，请换一个名称";
      setNameError(message);
      onValidityChange?.(false, message);
      return false;
    }
    setNameError("");
    onValidityChange?.(true, "");
    return true;
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = event.target.value;
    onChange(next);
    validateName(next);
  };

  return (
    <div className="form-group">
      <input
        className="form-input"
        placeholder={placeholder}
        value={value}
        autoFocus={autoFocus}
        aria-label="会话名称"
        aria-invalid={Boolean(nameError)}
        onChange={handleChange}
        onBlur={() => validateName(value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && onCommit) {
            event.preventDefault();
            onCommit();
          }
          if (event.key === "Escape" && onCancel) {
            event.preventDefault();
            onCancel();
          }
        }}
      />
      {nameError && (
        <p className="inline-error" role="status">
          {nameError}
        </p>
      )}
    </div>
  );
}

export default SessionNameInput;
