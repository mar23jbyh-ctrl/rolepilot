import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { useState } from "react";

import { useInterview } from "./hooks/useInterview";
import { ChatView } from "./components/ChatView";
import { ReportView } from "./components/ReportView";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";

export default function App() {
  const [navOpen, setNavOpen] = useState(false);
  const [roleEdit, setRoleEdit] = useState<{ sessionId: string; text: string } | null>(null);
  const {
    state,
    patch,
    start,
    answer,
    retryPendingAnswer,
    preserveFailedSessionAndReset,
    stop,
    uploadResume,
    uploadJdImage,
    openHistory,
    closeHistory,
    deleteSession,
    renameSession,
    updateJobTitle,
  } = useInterview();

  const viewing = state.viewing;
  const isViewing = Boolean(viewing);
  const status = isViewing
    ? viewing!.status
    : state.status;
  const hasReport = isViewing ? Boolean(viewing!.report) : Boolean(state.report);
  const canEditRole = Boolean(state.activeSessionId) && !isViewing && !hasReport &&
    !state.loading && !state.pendingAnswer && status !== "failed";

  return (
    <div className={navOpen ? "app-frame nav-open" : "app-frame"}>
      <TopBar
        name={isViewing ? viewing!.name : state.sessionName}
        status={status}
        jobTitle={isViewing ? viewing!.jobTitle : state.jobTitle}
        domain={isViewing ? viewing!.domain : state.domain}
        planQuestionCount={
          isViewing ? viewing!.planQuestionCount : state.planQuestionCount
        }
        effectiveSampleCount={
          isViewing ? viewing!.effectiveSampleCount : state.effectiveSampleCount
        }
        onEditJobTitle={
          canEditRole
            ? () => setRoleEdit({ sessionId: state.activeSessionId, text: state.jobTitle || "" })
            : undefined
        }
      />

      {canEditRole && roleEdit?.sessionId === state.activeSessionId && (
        <form
          className="job-title-editor"
          aria-label="修正岗位名称"
          onSubmit={(event) => {
            event.preventDefault();
            const next = roleEdit.text.trim();
            if (!next || !canEditRole) return;
            setRoleEdit(null);
            if (next !== state.jobTitle) void updateJobTitle(next);
          }}
          onKeyDown={(event) => {
            if (event.key === "Escape") { event.preventDefault(); setRoleEdit(null); }
          }}
        >
          <label htmlFor="job-title-edit">修正岗位名称</label>
          <input id="job-title-edit" value={roleEdit.text} maxLength={200} autoFocus
            onChange={(event) => setRoleEdit({ ...roleEdit, text: event.target.value })} />
          <button type="submit" disabled={!roleEdit.text.trim()}>保存岗位名称</button>
          <button type="button" onClick={() => setRoleEdit(null)}>取消岗位修改</button>
        </form>
      )}

      <div className="main-layout">
        <Sidebar
          state={state}
          onPatch={patch}
          onResume={uploadResume}
          onJdImage={uploadJdImage}
          onStart={start}
          onView={openHistory}
          onDelete={deleteSession}
          onRename={renameSession}
        />

        <main className={hasReport ? "main-content main-content--scroll" : "main-content"}>
          {isViewing ? (
            <>
              <ChatView
                sessionId={viewing!.sessionId}
                messages={viewing!.messages}
                report={viewing!.report}
                loading={false}
                error=""
                onAnswer={() => undefined}
                onStop={() => undefined}
                readOnly
                onBack={closeHistory}
              />
              {viewing!.report && <ReportView report={viewing!.report} />}
            </>
          ) : (
            <>
              <ChatView
                sessionId={state.activeSessionId}
                messages={state.messages}
                report={state.report}
                loading={state.loading}
                error={state.error}
                onAnswer={answer}
                onStop={stop}
                pendingAnswer={state.pendingAnswer}
                recoveryRequired={state.recoveryRequired}
                onRetry={retryPendingAnswer}
                onPreserveFailed={preserveFailedSessionAndReset}
              />
              {state.report && <ReportView report={state.report} />}
            </>
          )}
        </main>
      </div>

      <button
        type="button"
        className="sidebar-toggle"
        aria-expanded={navOpen}
        aria-label={navOpen ? "收起配置面板" : "展开配置面板"}
        onClick={() => setNavOpen((value) => !value)}
      >
        {navOpen ? (
          <PanelLeftClose size={15} strokeWidth={1.6} aria-hidden="true" />
        ) : (
          <PanelLeftOpen size={15} strokeWidth={1.6} aria-hidden="true" />
        )}
        {navOpen ? "收起配置" : "展开配置"}
      </button>
    </div>
  );
}
