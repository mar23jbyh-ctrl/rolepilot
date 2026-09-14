from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.telemetry.summary import usage_record, merge_usage
from app.llm.client import chat_with_usage
from app.models.schemas import Assessment
from app.prompts.templates import ASSESS_PROMPT, ASSESS_SYSTEM
from app.role import role_policy_text
from app.utils.json_utils import parse_json_with_retries
from app.utils.numeric import parse_number
from app.validation import validate_resume_claims


DIMENSIONS = (
    "completeness",
    "depth",
    "expression",
    "practice",
    "followup_questions",
    "thinking",
)

# M1：候选人意图标签的白名单。评分调用顺带输出，不增加 LLM 调用次数。
CANDIDATE_INTENTS = (
    "answer",
    "request_explanation",
    "request_clarification",
    "request_stop",
    "off_topic",
)

# M6：评分调用最多 2 次（首次 + 1 次重试），与 ask._safe_chat 的重试口径一致。
ASSESS_MAX_ATTEMPTS = 2

# M18：档位 → 分数映射（1-5 档 → 0-10 分）。
# 用 ×2 线性映射，保持与既有等级阈值（A≥8 / B≥6 / C≥4）同一量纲——
# 4 档（"结合经历讲清流程/选型/排错"）恰好等于 8 分，即 A 线。
LEVEL_TO_SCORE: dict[int, float] = {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 10.0}

# M19：动机/经历题使用独立的两维量规，不再用专业维度评"你为什么应聘"。
# 这与出题期守卫对动机题的豁免（role.motivation_exempt）保持一致。
MOTIVATION_RUBRIC: list[dict] = [
    {
        "key": "m1",
        "label": "求职动机清晰度",
        "weight": 0.5,
        "definition": "表达的职业选择理由是否具体、可信、与自身经历一致",
        "threshold": "能说清选择这个岗位的具体原因，并给出与经历相关的依据",
    },
    {
        "key": "m2",
        "label": "岗位认知与规划",
        "weight": 0.5,
        "definition": "对岗位真实工作内容的理解，以及入职后的能力补齐计划",
        "threshold": "能说出岗位的真实工作要点，并给出可执行的学习路径",
    },
]


def _levels_to_dimensions(levels, rubric: list) -> tuple[dict[str, float], list[str]]:
    """M18：把模型给的**档位**换算成维度分；返回 (维度分, 缺失的维度 key)。

    - 档位必须是 1-5；越界、分数档位和非数字视为缺失，不升级为高分
    - rubric 里有、模型没给的维度记入 missing（= 未覆盖），**不补中性分**
    """

    dimensions: dict[str, float] = {}
    missing: list[str] = []
    source = levels if isinstance(levels, dict) else {}
    for item in rubric or []:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        key = str(item["key"])
        parsed = parse_number(source.get(key), low=-1e9, high=1e9)
        if parsed.value is None or parsed.value not in LEVEL_TO_SCORE:
            missing.append(key)
            continue
        dimensions[key] = LEVEL_TO_SCORE[int(parsed.value)]
    return dimensions, missing


def _weighted_question_score(dimensions: dict[str, float], rubric: list) -> float | None:
    """M18：单题分**完全由维度分与 rubric 权重推导**，不再由模型直接给。"""

    weights: dict[str, float] = {}
    for item in rubric or []:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        parsed_weight = parse_number(item.get("weight", 0.0), low=-1e9, high=1e9)
        if parsed_weight.value is None or parsed_weight.value <= 0:
            continue
        weights[str(item["key"])] = parsed_weight.value
    available = [
        (score, weights[key]) for key, score in (dimensions or {}).items() if key in weights
    ]
    if not available:
        return None
    weight_sum = sum(weight for _, weight in available)
    if weight_sum <= 0:
        return None
    return round(sum(score * weight for score, weight in available) / weight_sum, 2)


def _default_assessment(question_id: int) -> dict:
    model = Assessment(question_id=question_id)
    return model.model_dump()


def _next_question_id(state: dict) -> int:
    counter = int(state.get("global_question_counter", 0) or 0)
    assessed = len(state.get("assessments", []) or [])
    return max(counter + 1, assessed + 1)


def _normalize_intent(raw) -> str:
    """M1：把模型给的意图标签收敛到白名单，拿不准一律按正常作答处理。"""

    text = str(raw or "").strip().lower()
    return text if text in CANDIDATE_INTENTS else "answer"


def _needs_retry(parsed: dict | None, usage: dict) -> bool:
    """M6：判断是否需要重试。

    只在两种确定性信号下重试，避免把"模型在 JSON 后多写一句话"也当成失败
    （多余重试会直接推高成本）：

    1. 解析不出 JSON；
    2. 模型因长度上限被截断（``finish_reason == "length"``）。
    """

    if parsed is None:
        return True
    return usage.get("finish_reason") == "length"


def _add_usage(total: dict, usage: dict) -> dict:
    """累加多次调用的 token 用量（M6：重试的消耗也要计入成本）。"""

    merge_usage(total, usage)
    total["attempts"] = int(total.get("attempts", 0) or 0) + 1
    return total


def _chat_for_assessment(messages: list) -> tuple[str, dict | None, dict]:
    """M6：评分调用带 max_tokens 与截断检测，最多重试 1 次。

    成本：正常路径不增加调用；触发重试时增加一次评分调用。
    返回 (最后一个原始回复, 解析结果或 None, 累计 usage)。
    """

    usage_total: dict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "finish_reason": None,
        "attempts": 0,
    }
    last_content = ""
    current = list(messages)
    for attempt in range(ASSESS_MAX_ATTEMPTS):
        content, usage = chat_with_usage(
            current,
            temperature=settings.temp_assess,
            max_tokens=settings.max_output_tokens * (2 if attempt else 1),
        )
        _add_usage(usage_total, usage)
        last_content = content or last_content
        parsed = parse_json_with_retries(content)
        if isinstance(parsed, dict) and not _needs_retry(parsed, usage):
            return content, parsed, usage_total
        current = current + [
            {
                "role": "user",
                "content": (
                    "上一次输出不是完整可解析的 JSON。请只输出一个 JSON 对象，"
                    "不要解释、不要 Markdown 代码块、不要省略字段。"
                ),
            }
        ]
    return last_content, None, usage_total


def _usage_record(node: str, usage: dict) -> dict:
    return usage_record(node, usage)


# P0/B3：评分失败的**第二次补救**。首次失败（模型没给维度档位）时，
# 再用一个极小的请求要一个"总体档位"，避免整题丢分。
# 成本：仅在维度评分不可用时增加一次小调用。
FALLBACK_LEVEL_PROMPT = """请只输出一个 JSON：{{"overall_level": 1-5}}
档位含义：1=完全没体现 / 2=只谈概念 / 3=做法正确但笼统 / 4=结合经历讲清流程·选型·排错 / 5=有量化结果或方法论。
请仅依据本题量规判断总体档位，不得改用其他专业或通用维度：
{rubric_text}
题目：{question}
候选人回答：{answer}"""


def _fallback_overall_level(
    state, question: str, answer: str, messages: list, rubric: list | None = None
) -> tuple[dict[str, float], dict]:
    """Use the selected rubric for a labeled, overall-level fallback.

    Derived uniform scores are not independent per-dimension assessments.
    Keep the legacy four-argument call compatible; explicit [] stays empty.
    """

    if rubric is None:
        rubric = (state.get("role_profile") or {}).get("rubric") or []
    payload = [
        SystemMessage(content="You are a strict interviewer. Output JSON only."),
        HumanMessage(
            content=FALLBACK_LEVEL_PROMPT.format(
                question=question[:800], answer=answer,
                rubric_text=json.dumps(rubric, ensure_ascii=False),
            )
        ),
    ]
    try:
        raw, usage = chat_with_usage(payload, temperature=settings.temp_assess)
    except Exception:  # noqa: BLE001 - 兜底路径不允许抛异常
        return {}, _usage_record("assess_fallback", {})
    parsed = parse_json_with_retries(raw) or {}
    level = parse_number(parsed.get("overall_level"), low=1, high=5).value
    record = _usage_record("assess_fallback", usage)
    if level is None:
        return {}, record
    score = LEVEL_TO_SCORE[int(round(level))]
    return {str(item["key"]): score for item in rubric if isinstance(item, dict) and item.get("key")}, record


def assess(state) -> dict:
    if state.get("end_requested") and not str(state.get("current_answer", "") or "").strip():
        return {}
    question_index = int(state.get("current_question_index", 0))
    question = str(state.get("current_question", "") or "")
    answer = str(state.get("current_answer", "") or "")
    question_plan = state.get("question_plan", []) or []
    active_meta = (
        dict(question_plan[question_index])
        if 0 <= question_index < len(question_plan)
        else {}
    )
    question_id = _next_question_id(state)
    is_follow_up = int(state.get("follow_up_count", 0) or 0) > 0
    # M13：题单下标。追问轮不会推进 current_question_index（只有 advance 会），
    # 因此主问题与它的每一轮追问共用同一个下标——这正是"按题合并"需要的地基。
    plan_question_index = question_index if active_meta else -1
    depth_level = str(active_meta.get("depth_level", "application"))
    role_policy = role_policy_text(state.get("role_profile") or {}, depth_level)
    project_note = ""
    project_ref = str(active_meta.get("project_ref", "") or "")
    projects = (state.get("resume_profile", {}) or {}).get("projects", []) or []
    try:
        project_index = int(project_ref)
        if 0 <= project_index < len(projects):
            project_note = f"本题对应简历项目[{project_index}]：{projects[project_index]}"
    except (TypeError, ValueError):
        project_note = ""
    if not answer.strip():
        result = _default_assessment(question_id)
        result["is_follow_up"] = is_follow_up
        result["plan_question_index"] = plan_question_index
        result["question"] = question[:500]
        result["covered_aspects"] = []
        for meta_key, value in (
            ("knowledge_points", active_meta.get("skills", []) or []),
            ("project_ref", active_meta.get("project_ref", "")),
            ("jd_ref", active_meta.get("jd_ref", "")),
            ("source_id", active_meta.get("source_id", "")),
        ):
            result[meta_key] = value
        result.update(score=None, scoreable=False, dimension_levels={}, evidence={},
                      missing_points=['没有提供回答'], hallucination_or_conflict=False,
                      confidence='low', next_action='follow_up', unscoreable_reason='empty_answer')
        result["should_follow_up"] = True
        result["follow_up_reason"] = "候选人没有给出回答，需要鼓励并再次追问。"
        history = list(state.get("conversation_history", []) or [])
        return {
            "assessments": [result],
            "conversation_history": history + [
                {"role": "user", "content": answer, "question": question, "question_id": question_id}
            ],
            "turn_records": [
                {
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "assessment": result,
                }
            ],
        }

    # Select once before prompting: parsing and fallback must use these same keys.
    # Legacy behavioral metadata uses the same no-focus/no-source rule as before.
    is_motivation = str(active_meta.get("question_kind") or "") == "motivation" or (
        not str(active_meta.get("focus_key") or "").strip()
        and str(active_meta.get("question_type") or "") == "behavioral"
        and not any(
            str(active_meta.get(field) or "").strip()
            for field in ("jd_ref", "project_ref", "research_ref")
        )
    )
    rubric = (
        MOTIVATION_RUBRIC
        if is_motivation
        else ((state.get("role_profile") or {}).get("rubric") or [])
    )
    messages = [
        SystemMessage(content=ASSESS_SYSTEM.format(role_policy=role_policy)),
        HumanMessage(
            content=ASSESS_PROMPT.format(
                question=question,
                answer=answer,
                question_id=question_id,
                depth_level=depth_level,
                project_note=project_note or "无",
                rubric_text=json.dumps(rubric, ensure_ascii=False),
            )
        ),
    ]
    raw, parsed, usage = _chat_for_assessment(messages)

    # M3：讲解/结束意图改由模型判断（评分调用顺带输出 candidate_intent），
    # 彻底移除关键词子串匹配。注意这里已经产生一次 LLM 调用，
    # 因此该轮必须记账（见下方 usage_records）。
    intent = _normalize_intent((parsed or {}).get("candidate_intent"))
    if intent == "request_explanation":
        history = list(state.get("conversation_history", []) or [])
        # M12：登记"本题被跳过"，便于报告统计有效样本数并补一道同维度替代题
        skipped = list(state.get("skipped_questions", []) or [])
        skipped.append(
            {
                "question_index": question_index,
                "focus_key": str(active_meta.get("focus_key", "") or ""),
                "skills": list(active_meta.get("skills", []) or []),
                "category": str(active_meta.get("category", "") or ""),
                "depth_level": depth_level,
                "reason": "request_explanation",
                "replacement_index": None,
            }
        )
        return {
            "explanation_pending": True,
            "skipped_questions": skipped,
            "conversation_history": history + [
                {
                    "role": "user",
                    "content": answer,
                    "question": question,
                    "question_id": question_id,
                }
            ],
            "usage_records": [_usage_record("assess_intent", usage)],
        }

    result = _default_assessment(question_id)
    if isinstance(parsed, dict):
        result.update(parsed)
    result["question_id"] = question_id
    result["is_follow_up"] = is_follow_up
    result["plan_question_index"] = plan_question_index
    result["question"] = question[:500]
    result.setdefault("covered_aspects", [])
    result["candidate_intent"] = intent
    if intent == "request_stop":
        # M2/M3：模型判定候选人想结束面试 → 交给路由收束（本题分数照常记录）
        result["stop_suggested"] = True
    for meta_key, value in (
        ("knowledge_points", active_meta.get("skills", []) or []),
        ("project_ref", active_meta.get("project_ref", "")),
        ("jd_ref", active_meta.get("jd_ref", "")),
        ("source_id", active_meta.get("source_id", "")),
    ):
        result[meta_key] = value
    # M18：单题分 = "模型标维档位 → 代码换算分数 → 按 rubric 权重合成"。
    # 分数不再由模型直接给出，因此总分与维度分天然同源、可复算。
    # M19/D12: rubric was already selected before constructing the prompt.
    if is_motivation:
        result["question_kind"] = "motivation"
    else:
        result["question_kind"] = "professional"
    # M21：把题目难度带进评分记录，供聚合阶段做难度加权
    result["difficulty"] = str(active_meta.get("difficulty") or state.get("difficulty") or "medium")
    strict = state.get('assessment_protocol_version') == 'practice-v1'
    if strict:
        from app.scoring.protocol import normalize_assessment
        validated = normalize_assessment(parsed, answer, rubric, intent)
        result.update(validated)
        parsed = {**(parsed or {}), 'dimension_levels': validated['dimension_levels']}
    dimensions, missing_dimensions = _levels_to_dimensions((parsed or {}).get("dimension_levels"), rubric)
    result["dimensions"] = dimensions
    if missing_dimensions:
        result["dimensions_not_covered"] = missing_dimensions
    score_value = _weighted_question_score(dimensions, rubric)
    result["score"] = score_value
    if strict and not result.get('scoreable'):
        result['score'] = None
        result['dimensions'] = {}
        score_value = None
    if score_value is None and not strict:
        # P0/B3：第二次补救——再要一个"总体档位"，均匀应用到所有维度。
        fallback_dims, fallback_record = _fallback_overall_level(
            state, question, answer, messages, rubric=rubric
        )
        extra_usage = [fallback_record]
        if fallback_dims:
            dimensions = fallback_dims
            result["dimensions"] = dimensions
            result["score_fallback"] = "single_overall_level"
            score_value = _weighted_question_score(dimensions, rubric)
            result["score"] = score_value
        if score_value is None:
            # 两次都拿不到 → M5 语义：不写假值，标记并排除出总分
            result["score_error"] = True
            result["score_error_status"] = "no_dimension_levels"
    if strict and result.get('protocol_errors'):
        result.update(score_error=True, score_error_status='invalid_assessment_protocol', score=None,
                      dimensions={}, should_follow_up=False, next_action='next_question')
    result["original_question"] = question[:1000]
    result["question_summary"] = question[:80]
    result["max_score"] = 10.0
    if not result.get("covered_points") and not result.get("missed_points"):
        result["missed_points"] = ["回答未能给出具体要点、案例或清晰的思路"]
    if result["score"] is not None and result["score"] >= 7.0:
        result["should_follow_up"] = False
    next_action = str(result.get("next_action", "") or "")
    if strict and next_action == 'explain':
        result['next_action'] = 'next_question'
        result['explanation_recommended'] = True
        # Only an explicit request enters the existing explanation/replacement branch.
        # Do not erase a scored answer because the model recommends teaching.
    if strict and next_action == 'end':
        result['next_action'] = 'evaluate'
    next_action = str(result.get('next_action') or '')
    if next_action not in {"follow_up", "next_question", "evaluate"}:
        result["next_action"] = (
            "follow_up"
            if result.get("should_follow_up") and int(state.get("follow_up_count", 0)) < int(state.get("max_follow_ups", settings.max_follow_ups))
            else "next_question"
        )
    if result.get("score_error"):
        # M5：分数都拿不到时不对同一题继续追问（追问也评不出分），直接换题
        result["next_action"] = "next_question"
    # M15/M16：维度只来自 rubric，且"没评到"保持缺失（不补 5.0 中性分）

    usage_records = [_usage_record("assess", usage)] + locals().get("extra_usage", [])

    # Only the first answer of each question is checked against the resume so
    # follow-up clarifications are not repeatedly flagged as fabrication.
    if int(state.get("follow_up_count", 0) or 0) == 0:
        validation = validate_resume_claims(
            answer=answer,
            resume_profile=state.get("resume_profile", {}) or {},
            question_category=str(active_meta.get("category", "")),
        )
        if validation and validation.get("conflict"):
            claims = validation.get("claims", []) or []
            claimed_projects = "、".join(
                str(claim.get("name", "未具名项目"))
                for claim in claims
                if isinstance(claim, dict)
            )
            result["resume_conflict"] = True
            if strict:
                result['hallucination_or_conflict'] = True
                result['dimension_levels'] = {k:min(v,2) for k,v in result.get('dimension_levels',{}).items()}
                result['dimensions'] = {k:min(v,4.0) for k,v in result.get('dimensions',{}).items()}
                result['score'] = _weighted_question_score(result['dimensions'],rubric)
            result["resume_conflict_explanation"] = str(validation.get("explanation", ""))
            result["claimed_projects"] = claimed_projects
            result["should_follow_up"] = True
            result["follow_up_reason"] = "口述项目与简历不符，需先澄清项目来源"
            result["hint"] = (
                f"你提到的项目（{claimed_projects}）不在确认后的简历中，"
                "请说明该项目属于哪段经历或何时完成。"
            )
            if claimed_projects not in (result.get("missed_points", []) or []):
                result.setdefault("missed_points", []).append("口述项目与简历不一致")
        if validation and validation.get("usage"):
            validation_usage = validation["usage"]
            usage_records.append(
                {
                    **validation_usage,
                    "node": "resume_check",
                    "input_tokens": int(validation_usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(validation_usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(validation_usage.get("total_tokens", 0) or 0),
                }
            )

    history = list(state.get("conversation_history", []) or [])
    return {
        "assessments": [result],
        **({'explanation_pending': True} if strict and result.get('explanation_recommended') else {}),
        "conversation_history": history + [
            {"role": "user", "content": answer, "question": question, "question_id": question_id}
        ],
        "turn_records": [
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "assessment": result,
            }
        ],
        "usage_records": usage_records,
        "global_question_counter": question_id,
    }
