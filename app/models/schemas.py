from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# M34：报告结构版本号（与 docs/scoring_contract.md 的 contract_version 配套）
SCHEMA_VERSION = "3.4.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResumeSection(BaseModel):
    heading: str = Field(description="Section heading found in the resume")
    content: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    summary: str = Field(default="", description="One-paragraph candidate summary")
    skills: list[str] = Field(default_factory=list, description="Normalized skills")
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    sections: list[ResumeSection] = Field(
        default_factory=list,
        description="Anything else found in the resume, kept dynamically",
    )
    concerns: list[str] = Field(default_factory=list)


class JDRequirement(BaseModel):
    category: Literal["tech", "experience", "education", "soft", "plus"]
    text: str
    skills: list[str] = Field(default_factory=list)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class SkillMatch(BaseModel):
    requirement: str
    status: Literal["mastered", "possible", "missing"]
    evidence: str = ""
    severity: Literal["severe", "moderate", "mild"] = "mild"
    likely_question: str = ""


class GapReport(BaseModel):
    requirements: list[JDRequirement] = Field(default_factory=list)
    matches: list[SkillMatch] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    weak_skills: list[str] = Field(default_factory=list)
    strong_skills: list[str] = Field(default_factory=list)
    summary: str = ""


class InterviewQuestion(BaseModel):
    id: int
    category: Literal["foundation", "project", "scenario", "algorithm"]
    difficulty: Literal["easy", "medium", "hard"]
    content: str
    skills: list[str] = Field(default_factory=list)
    source_id: str = Field(default="", description="Traceable source, e.g. research:job:1 / llm:uuid")
    source_type: Literal["web", "llm"] = "llm"


class QuestionPlan(BaseModel):
    questions: list[InterviewQuestion]


class Assessment(BaseModel):
    question_id: int = 0
    is_follow_up: bool = False
    score: float | None = Field(default=None, ge=0, le=10)
    # M16：不再预置英文六维默认值。维度**完全**由 rubric 定义；
    # rubric 缺失时报告降级为"仅总分 + 文字评语"（见 evaluate._dimension_scores）。
    dimensions: dict[str, float] = Field(default_factory=dict)
    is_relevant: bool = True
    should_follow_up: bool = False
    follow_up_reason: str = ""
    hint: str = Field(default="", description="A hint shown before the follow-up")
    missed_points: list[str] = Field(default_factory=list)
    covered_points: list[str] = Field(default_factory=list)
    needs_external_knowledge: bool = False


class DifficultyEvent(BaseModel):
    question_index: int
    previous_difficulty: str
    new_difficulty: str
    reason: str


class CompressedTurn(BaseModel):
    question_id: int
    question: str
    score: float
    missed_points: list[str] = Field(default_factory=list)
    follow_up_reason: str = ""
    difficulty_change: str = ""
    summary: str


class UsageRecord(BaseModel):
    node: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    timestamp: str = Field(default_factory=utc_now)


class EvaluationReport(BaseModel):
    overall_score: float
    # M9：等级只允许 A/B/C/D/—，且完全由代码按配置阈值推导
    grade: Literal["A", "B", "C", "D", "—"] = "—"
    text_analysis: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    per_question: list[dict] = Field(default_factory=list)
    difficulty_events: list[dict] = Field(default_factory=list)
    token_totals: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    budget_summary: dict = Field(default_factory=dict)
    report_generation_status: str = "generated"
    practice_disclaimer: str = '面试练习参考分，不是招聘决策、专家评定或能力认证。'
    unscoreable_count: int = 0
    learning_path: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    # 联网调研 / 调研参考状态（ok|failed / ok|empty），前端不消费，仅作诊断。
    research_status: str = ""
    references_status: str = ""
    # M5：评分失败（模型未给出可用分数）而被排除出总分的记录数
    score_errors: int = 0
    # M8：模型自己给的总分意见（仅审计留档，不参与最终取值）
    llm_overall_opinion: float | None = None
    # M9：模型自己给的等级意见（仅审计留档，不参与最终取值）
    llm_grade_opinion: str | None = None
    # M10：最终总分是否因越界被 clamp
    out_of_range: bool = False
    # ---- M12：样本完整性（替代题保证有效样本数等于题单题数）----
    # 原始题单题数（替代题不计入，保证跨会话可比）
    plan_question_count: int = 0
    # 真正产生可用评分的主问题数（不含追问、不含评分失败）
    effective_sample_count: int = 0
    # 因请求讲解等原因未计分的题目（含是否补到替代题）
    skipped_questions: list[dict] = Field(default_factory=list)
    # 仅当有效样本 < 题单题数时给出"有效样本数 X / 题单题数 Y"
    sample_note: str = ""
    # ---- M15 / M16：维度口径 ----
    # 未被任何题覆盖的维度（不按 0 分计入加权）
    not_covered_dimensions: list[str] = Field(default_factory=list)
    # 聚合口径：rubric_weighted（默认）/ topic_mean（rubric 缺失时降级为仅总分+评语）
    aggregation_mode: str = ""
    # M26：简历项目的使用情况（total / used / truncated）
    resume_project_stats: dict = Field(default_factory=dict)
    # ---- M34：可追溯元信息 ----
    # model / temperature / prompt_version / rubric_version / schema_version / generated_at
    scoring_meta: dict = Field(default_factory=dict)
    # M35：代码算出的总分（与 overall_score 同值，语义上区分"代码值"便于跨版本审计）
    overall_code: float | None = None
    # ---- 3.2.4：跨职能域审计与按节点 token 明细（诊断可见性，前端不消费）----
    guard_audit_status: str = ""            # ok / degraded / disabled
    guard_uncertain_count: int = 0          # 保留但未能验证（kept_unverified）的题数
    guard_dropped_count: int = 0            # 按归一化题面去重后的丢弃题数
    guard_dropped_samples: list[str] = Field(default_factory=list)  # 去重后前 5 条
    weak_focus_count: int = 0               # 画像中不合格 focus 数
    guard_evidence_mismatch_count: int = 0  # 引用来源与题面内容对不上的题数（只标记不丢弃）
    guard_degrade_reason: str = ""           # 守卫降级成因（多成因用 + 拼接，空=正常）
    guard_domain_slug_missing: bool = False   # 画像缺少规范化职能域标识（诊断字段）
    # 按 usage_records[].node 聚合的 total_tokens；未产生 usage 的节点不出现（不补 0）
    token_by_node: dict[str, int | None] = Field(default_factory=dict)


class ToolResult(BaseModel):
    usage: dict[str, Any] = Field(default_factory=dict)
    tool: str
    success: bool = True
    content: str = ""
    source: str = ""
    timestamp: str = Field(default_factory=utc_now)
    degraded: bool = False
    error: str = ""


# Function-calling argument schemas (standard OpenAI tool format is derived
# from these Pydantic models in app/tools/schemas.py).


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=10)


class CodeExplainerArgs(BaseModel):
    code: str = Field(description="The code snippet to explain")
    language: str = Field(default="python")
    focus: str = Field(
        default="Explain the logic, point out problems, and show a corrected example"
    )


class DynamicQuestionArgs(BaseModel):
    skill: str = Field(description="Skill or topic to ask about")
    difficulty: str = Field(default="medium", description="easy|medium|hard")
    category: str = Field(default="scenario", description="question category")
    avoid: list[str] = Field(
        default_factory=list,
        description="Question texts already asked, to avoid duplicates",
    )
