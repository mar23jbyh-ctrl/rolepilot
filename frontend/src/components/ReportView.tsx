import { ChevronDown, Sparkles } from "lucide-react";
import { useState } from "react";

import type { QuestionDetail, Report } from "../types";

/**
 * 综合评语只谈整体：先拆到句子粒度，再把点名到具体题目的句子剔除，
 * 剩下的整体性句子按语义成段，避免整段压迫感，也避免与「各题详情」重复。
 */
const QUESTION_REF =
  /(?:Q\s*\d+|第\s*[0-9一二三四五六七八九十]+\s*题|本\s*题|该\s*题|这\s*道\s*题)/;

function toSentences(text: string): string[] {
  const raw = text
    .split(/\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
  const sentences: string[] = [];
  raw.forEach((para) => {
    const parts = para.match(/[^。！？；!?;]+[。！？；!?;]?/g) || [para];
    parts.forEach((part) => {
      const piece = part.trim();
      if (piece) sentences.push(piece);
    });
  });
  return sentences;
}

/** 把整体性句子按长度聚合成自然段 */
function groupSentences(sentences: string[], maxLen = 160): string[] {
  const blocks: string[] = [];
  let buffer = "";
  sentences.forEach((piece) => {
    if (buffer && (buffer + piece).length > maxLen) {
      blocks.push(buffer);
      buffer = piece;
    } else {
      buffer += piece;
    }
  });
  if (buffer) blocks.push(buffer);
  return blocks;
}

function isOverallOnly(text: string): boolean {
  return !QUESTION_REF.test(text);
}

/**
 * 支撑依据：全部来自本轮汇总口径（平均分、证据条数、前后半程趋势），
 * 不涉及任何单题得分 / 扣分 / 遗漏点，因此不会与「各题详情」重复。
 */
function buildEvidence(report: Report): string {
  const per = report.per_question || [];
  const scored = per.filter(
    (item): item is QuestionDetail & { score: number } =>
      typeof item.score === "number" && Number.isFinite(item.score)
  );
  const sentences: string[] = [];

  if (scored.length) {
    const avg = scored.reduce((sum, item) => sum + item.score, 0) / scored.length;
    const covered = per.reduce(
      (sum, item) => sum + new Set((item.covered_points || []).filter(Boolean)).size,
      0
    );
    const missed = per.reduce(
      (sum, item) => sum + new Set((item.missed_points || []).filter(Boolean)).size,
      0
    );
    let head = `从汇总数据看，本轮共评估 ${scored.length} 道题，平均得分 ${avg.toFixed(
      1
    )} 分（满分 10 分）`;
    if (covered + missed > 0) {
      // Evidence is quoted from answers, not an exhaustive gold-answer checklist.
      head += `，全部作答合计记录回答证据 ${covered} 条、待补充要点 ${missed} 条`;
    }
    sentences.push(`${head}。`);
  }

  // 能力维度的强项与弱项（仅当报告带维度分时才统计，否则不写）
  const dimensionTotals = new Map<string, { sum: number; count: number }>();
  per.forEach((item) => {
    Object.entries(item.dimensions || {}).forEach(([name, value]) => {
      if (typeof value !== "number" || !Number.isFinite(value)) return;
      const entry = dimensionTotals.get(name) || { sum: 0, count: 0 };
      entry.sum += value;
      entry.count += 1;
      dimensionTotals.set(name, entry);
    });
  });
  if (dimensionTotals.size >= 2) {
    const ranked = [...dimensionTotals.entries()]
      .map(([name, entry]) => ({ name, avg: entry.sum / entry.count }))
      .sort((a, b) => b.avg - a.avg);
    const best = ranked[0];
    const worst = ranked[ranked.length - 1];
    if (best.avg - worst.avg >= 0.5) {
      sentences.push(
        `在可量化的能力维度上，「${best.name}」的平均得分最高（${best.avg.toFixed(
          1
        )} 分），「${worst.name}」相对偏弱（${worst.avg.toFixed(1)} 分）。`
      );
    }
  }

  if (scored.length >= 4) {
    const half = Math.floor(scored.length / 2);
    const front = scored.slice(0, half);
    const back = scored.slice(half);
    const frontAvg = front.reduce((sum, item) => sum + item.score, 0) / front.length;
    const backAvg = back.reduce((sum, item) => sum + item.score, 0) / back.length;
    const delta = backAvg - frontAvg;
    const trend =
      delta >= 0.8
        ? "回答质量呈上行趋势，越靠后越进入状态"
        : delta <= -0.8
          ? "回答质量呈回落趋势，后半程的把握度不如前半程"
          : "回答质量整体平稳，前后半程没有明显起伏";
    sentences.push(
      `从题目推进看，前半程平均 ${frontAvg.toFixed(1)} 分、后半程平均 ${backAvg.toFixed(
        1
      )} 分，${trend}。`
    );
  }

  return sentences.join("");
}

const SUMMARY_FALLBACK =
  "本轮未产出整体定性评语，单题表现请参考下方各题详情。";

function ScoreHero({ score, grade }: { score?: number; grade?: string }) {
  return (
    <div className="hero-score">
      <span className="hero-score__label">综合得分</span>
      <div className="hero-score__value">
        <b>{score ?? "—"}</b>
        <span className="hero-score__unit">/ 10</span>
      </div>
      <span className="hero-score__grade">
        <span className="hero-score__grade-cap">等级</span>
        {grade ?? "—"}
      </span>
    </div>
  );
}

function ScoreRing({ score }: { score?: number }) {
  const value = typeof score === "number" ? Math.max(0, Math.min(100, score * 10)) : 0;
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const dash = (value / 100) * circumference;
  return (
    <div className="score-ring" role="img" aria-label={`综合得分 ${score ?? "—"} 分`}>
      <svg viewBox="0 0 132 132" aria-hidden="true">
        <defs>
          <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#15803d" />
          </linearGradient>
        </defs>
        <circle cx="66" cy="66" r={radius} fill="none" stroke="#e6e8f0" strokeWidth="7" />
        <circle
          cx="66"
          cy="66"
          r={radius}
          fill="none"
          stroke="url(#ringGradient)"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference - dash}`}
        />
      </svg>
      <div className="score-ring__center">
        <span className="score-ring-value">{score ?? "—"}</span>
        <span className="score-ring-label">满分 10</span>
      </div>
    </div>
  );
}

function QuestionItem({ item, index }: { item: QuestionDetail; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const qid = item.question_id ?? index + 1;
  const max = item.max_score ?? 10;
  const score = item.score;
  const fullQuestion = (item.question || item.question_summary || "无").trim();
  const covered = [...new Set((item.covered_points || []).filter(Boolean))];
  const missed = [...new Set((item.missed_points || []).filter(Boolean))];
  const tone =
    score === undefined || score === null
      ? "mid"
      : score >= 8
        ? "ok"
        : score >= 6
          ? "mid"
          : score >= 5
            ? "warn"
            : "low";
  return (
    <div className={open ? "question-card open" : "question-card"}>
      <button
        type="button"
        className="question-card-header"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="question-card-title">
          <span className="question-card-qid">Q{qid}</span>
          <span className="question-card-title__text">{fullQuestion}</span>
        </span>
        <span className="question-card-meta">
          <span className={`question-score-tag question-score-tag--${tone}`}>
            {score ?? "—"} / {max}
          </span>
          <ChevronDown
            size={16}
            strokeWidth={1.6}
            className="question-card-chevron"
            aria-hidden="true"
          />
        </span>
      </button>
      <div className="question-card-body">
        <div className="question-card-body-inner">
          <div className="question-card-field">
            <span className="question-card-field__label">原题文本</span>
            <p className="question-card-field__text">{fullQuestion}</p>
          </div>
          <div className="question-card-grid">
            <div className="question-card-field">
              <span className="question-card-field__label">本题满分</span>
              <p className="question-card-field__text">{max}</p>
            </div>
            <div className="question-card-field">
              <span className="question-card-field__label">得分</span>
              <p className="question-card-field__text">{item.score ?? "—"}</p>
            </div>
          </div>
          <div className="question-card-field">
                <span className="question-card-field__label">回答证据</span>
            {covered.length ? (
              <ul className="text-list">
                {covered.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            ) : (
              <p className="question-card-field__text">本题未记录覆盖点</p>
            )}
          </div>
          <div className="question-card-field">
            <span className="question-card-field__label">遗漏点</span>
            {missed.length ? (
              <ul className="text-list">
                {missed.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            ) : (
              <p className="question-card-field__text">本题未记录遗漏点</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function ReportView({ report }: { report: Report }) {
  const overallSentences = toSentences(report.text_analysis || "").filter(isOverallOnly);
  const qualitative = groupSentences(overallSentences);
  const paragraphs = qualitative.length ? qualitative : [SUMMARY_FALLBACK];
  const evidence = buildEvidence(report);
  const questionCount = (report.per_question || []).length;
  return (
    <div className="report stack-lg">
      <h1 className="sr-only">面试评估报告</h1>

      <section className="card hero-card">
        <ScoreHero score={report.overall_score} grade={report.grade} />
        <div className="score-ring-card">
          <ScoreRing score={report.overall_score} />
        </div>
      </section>

      <div className="meta-row">
        <span className="meta-item">
          作答题目 <b>{questionCount}</b>
        </span>
      </div>

      <section className="card summary-card">
        <div className="section-head">
          <Sparkles size={17} strokeWidth={1.7} aria-hidden="true" />
          <h2 className="text-h2">综合评语</h2>
        </div>
        <div className="summary-paragraphs">
          {paragraphs.map((para, index) => (
            <p className="summary-paragraph" key={`${index}-${para.slice(0, 8)}`}>
              {para}
            </p>
          ))}
          {evidence && <p className="summary-paragraph">{evidence}</p>}
        </div>
      </section>

      <div className="two-col">
        <section className="card col-card col-card--ok">
          <div className="section-head">
            <h3 className="text-h2">优势</h3>
          </div>
          {(report.strengths || []).length ? (
            <ul className="text-list">
              {(report.strengths || []).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="text-list">暂无</p>
          )}
        </section>
        <section className="card col-card col-card--warn">
          <div className="section-head">
            <h3 className="text-h2">待改进</h3>
          </div>
          {(report.weaknesses || []).length ? (
            <ul className="text-list">
              {(report.weaknesses || []).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="text-list">暂无</p>
          )}
        </section>
      </div>

      <section className="stack-md report-block">
        <div className="section-head section-head--divider">
          <h2 className="text-h1">各题详情</h2>
          <span className="section-head__meta">{questionCount} 题</span>
        </div>
        {(report.per_question || []).length ? (
          (report.per_question || []).map((item, index) => (
            <QuestionItem key={`${item.question_id}-${index}`} item={item} index={index} />
          ))
        ) : (
          <p className="text-aux">暂无逐题评分数据</p>
        )}
      </section>

      <section className="card stack-md report-block">
        <div className="section-head">
          <h2 className="text-h2">学习路径</h2>
        </div>
        {(report.learning_path || []).length ? (
          <div className="timeline">
            {(report.learning_path || []).map((step, index) => (
              <div className="timeline-item" key={`${index}-${step}`}>
                <span className="timeline-item__order" aria-hidden="true">
                  {index + 1}
                </span>
                <div className="timeline-item-content">{step}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-aux">暂无</p>
        )}
      </section>
    </div>
  );
}
