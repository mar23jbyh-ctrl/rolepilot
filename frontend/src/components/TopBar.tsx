import { PRODUCT_NAME, sessionStatusLabel } from "../branding";

interface Props {
  name: string;
  status: string;
  /** M23：系统识别到的岗位与职能域，供用户核对 */
  jobTitle?: string;
  domain?: string;
  planQuestionCount?: number;
  effectiveSampleCount?: number;
  /** P1/A1：点击岗位名可修正 */
  onEditJobTitle?: () => void;
}

export function TopBar({
  name,
  status,
  jobTitle,
  domain,
  planQuestionCount,
  effectiveSampleCount,
  onEditJobTitle,
}: Props) {
  const finished = status === "completed";
  const roleText = [jobTitle, domain].filter(Boolean).join(" · ");
  const sampleText =
    planQuestionCount && planQuestionCount > 0
      ? `有效样本 ${effectiveSampleCount ?? 0} / ${planQuestionCount}`
      : "";
  return (
    <header className="top-nav">
      <div className="top-nav-brand">
        <span className="brand-dot" />
        <span className="top-nav-title">{name ? `${PRODUCT_NAME} · ${name}` : PRODUCT_NAME}</span>
        {roleText ? (
          <button
            type="button"
            className="top-nav-sub"
            title="系统识别到的岗位 —— 点击可修正"
            onClick={onEditJobTitle}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              cursor: onEditJobTitle ? "pointer" : "default",
              textDecoration: onEditJobTitle ? "underline dotted" : "none",
            }}
          >
            {roleText}
            {onEditJobTitle ? " ✎" : ""}
          </button>
        ) : null}
      </div>
      <div className="top-nav-right">
        {sampleText ? <span className="top-nav-sub">{sampleText}</span> : null}
        <span className={finished ? "badge badge-done" : "badge badge-doing"}>
          {sessionStatusLabel(status)}
        </span>
      </div>
    </header>
  );
}
