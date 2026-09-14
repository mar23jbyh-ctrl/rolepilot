from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.telemetry.summary import summarize_usage, state_usage_records
from app.telemetry.ledger import SessionBudgetExceeded
from app.context.manager import summaries_text
from app.llm.client import chat_with_usage
from app.models.schemas import SCHEMA_VERSION, EvaluationReport
from app.prompts.templates import EVALUATE_SYSTEM, PROMPT_VERSION
from app.role import role_policy_text
from app.utils.json_utils import parse_json_with_retries
from app.utils.numeric import (
    SCORE_MAX,
    SCORE_MIN,
    STATUS_CLAMPED_HIGH,
    STATUS_CLAMPED_LOW,
    parse_number,
    safe_score,
)

ALLOWED_GRADES = ("A", "B", "C", "D", "—")

# M28：两块注入终评的数据各自的字符预算（按"条"裁剪，不再按字符截断 JSON）
FINAL_INPUT_BUDGET = 8000


def _trim_to_budget(records: list, budget_chars: int = FINAL_INPUT_BUDGET) -> tuple[list, bool]:
    """M28：按"条"裁剪，保证 json.dumps 出来永远是**合法 JSON**。

    原实现是 `json.dumps(...)[:8000]`——按字符截断会切断 JSON 结构，
    长会话里模型拿到的是半个对象。这里改为保留**首尾各一半**的条目。
    """

    original = list(records or [])
    trimmed = list(original)
    while len(trimmed) > 2 and (
        len(json.dumps(trimmed, ensure_ascii=False)) > budget_chars
    ):
        keep = (len(trimmed) - 1) // 2
        trimmed = trimmed[:keep] + trimmed[-keep:]
    return trimmed, len(trimmed) < len(original)

# M21：难度权重（可用 config.difficulty_weighting 整体关闭）
DIFFICULTY_WEIGHT = {"easy": 0.8, "medium": 1.0, "hard": 1.2}


def _topic_weight(record: dict) -> float:
    """M21：按题目难度给权重；关闭开关后一律 1.0（等价于不加权）。"""

    if not getattr(settings, "difficulty_weighting", True):
        return 1.0
    return DIFFICULTY_WEIGHT.get(str(record.get("difficulty") or "medium").lower(), 1.0)


def _normalize_text(raw, default: str = "") -> str:
    """M7：文本字段规范化——``null`` 不再变成字面量 ``"None"``。"""

    if raw is None:
        return default
    if isinstance(raw, (list, tuple, set)):
        parts = [str(item).strip() for item in raw if item is not None]
        joined = "\n".join(part for part in parts if part)
        return joined or default
    text = str(raw).strip()
    return text if text else default


def _normalize_text_list(raw) -> list[str]:
    """M7：列表字段规范化。

    关键修复：字符串必须变成**单元素列表**，而不是被 ``list()`` 拆成一个个字符
    （原先 `list("沟通能力强")` → `["沟","通","能","力","强"]`）。
    """

    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, dict):
        # 结构不符合契约，宁可为空也不要塞一段 str(dict) 进报告
        return []
    if isinstance(raw, (list, tuple, set)):
        items: list[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(raw).strip()
    return [text] if text else []


def _grade_from_score(score: float, cutoffs: tuple[float, float, float] | None = None) -> str:
    """M9：等级**完全**由代码按配置阈值推导，模型不再参与。

    阈值来自 ``settings.grade_cutoffs``（由 ``GRADE_THRESHOLDS`` 解析），
    默认 A ≥ 8 / B ≥ 6 / C ≥ 4 / 其余 D。
    """

    a, b, c = cutoffs if cutoffs is not None else settings.grade_cutoffs
    if score >= a:
        return "A"
    if score >= b:
        return "B"
    if score >= c:
        return "C"
    return "D"


def _safe_grade(value: str) -> str:
    """防御：任何越界取值都不允许让报告生成失败（宁可降级为 ``—``）。"""

    text = str(value or "").strip()
    return text if text in ALLOWED_GRADES else "—"


def _sample_fields(state, assessments: list[dict]) -> dict:
    """M12：有效样本数 / 题单题数 / 被跳过的题目。

    - ``plan_question_count``：**原始题单题数**（替代题不计入，保证跨会话可比）
    - ``effective_sample_count``：真正产生可用评分的**主问题**数（不含追问、不含评分失败）
    - ``sample_note``：仅当有效样本少于题单题数时给出"有效样本数 X / 题单题数 Y"
    """

    plan = state.get("question_plan") or []
    plan_count = int(state.get("plan_question_count") or 0) or len(plan)
    effective = sum(
        1
        for item in assessments or []
        if not item.get("is_follow_up")
        and not item.get("score_error")
        and safe_score(item.get("score")) is not None
    )
    note = ""
    if plan_count and effective < plan_count:
        note = f"有效样本数 {effective} / 题单题数 {plan_count}"
    return {
        "plan_question_count": plan_count,
        "effective_sample_count": effective,
        "skipped_questions": list(state.get("skipped_questions") or []),
        "sample_note": note,
    }

DIMENSION_LABELS = {
    "completeness": "完整性",
    "depth": "深度",
    "expression": "表达",
    "practice": "实战",
    "followup_questions": "反问质量",
    "thinking": "思考过程",
}


def _topic_records(assessments: list[dict]) -> list[dict]:
    """M14：把同一题的**多轮追问合并成一条**记录。

    - 合并键：`plan_question_index`（M13 引入）
    - 老数据没有该字段时，按 `is_follow_up` 顺序切段还原（与
      `app/graph/edges.py::_question_segment` 同口径）
    - 结果取该题的**最后一轮**（体现追问后的最终表现），并记录 `follow_up_rounds`
    - 维度取各轮中出现过的最后一个有效值
    """

    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    fallback = -1
    for item in assessments or []:
        if not item.get("is_follow_up"):
            fallback += 1
        index = item.get("plan_question_index")
        key = index if isinstance(index, int) and index >= 0 else fallback
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    merged: list[dict] = []
    for key in order:
        items = groups[key]
        usable = [x for x in items if x.get('scoreable') is not False and not x.get('score_error') and safe_score(x.get('score')) is not None]
        record = dict(usable[-1] if usable and any('scoreable' in x for x in items) else items[-1])
        # Keep the final scoring result, but identify the original main question.
        main = next((item for item in items if not item.get("is_follow_up")), items[0])
        for field in ("question_id", "question", "original_question", "question_summary"):
            record[field] = main.get(field, "")
        record["is_follow_up"] = bool(main.get("is_follow_up"))
        record["last_assessment_question_id"] = items[-1].get("question_id")
        record["rounds"] = [
            {
                "question_id": item.get("question_id"),
                "question": item.get("original_question") or item.get("question", ""),
                "is_follow_up": bool(item.get("is_follow_up")),
                "score": safe_score(item.get("score")),
            }
            for item in items
        ]
        record["plan_question_index"] = key
        record["follow_up_rounds"] = sum(1 for x in items if x.get("is_follow_up"))
        record["round_scores"] = [
            value for value in (safe_score(x.get("score")) for x in items) if value is not None
        ]
        dimensions: dict[str, float] = {}
        for item in items:
            if item.get('scoreable') is False or item.get('score_error'):
                continue
            for dim_key, dim_value in (item.get("dimensions") or {}).items():
                parsed = parse_number(dim_value)
                if parsed.value is not None:
                    dimensions[str(dim_key)] = parsed.value
        record["dimensions"] = dimensions
        merged.append(record)
    return merged


def _dimension_scores(
    topic_records: list[dict], role_profile: dict
) -> tuple[dict[str, float], list[str]]:
    """M15 + M16：维度分**只认 rubric**，分母是"覆盖该维度的题数"。

    与旧实现的三处区别：
    1. 不再把模型自报的非 rubric 维度（英文六维）混进报告；
    2. 分母从"全部记录数"改成"**覆盖该维度的**题数"，消除稀释假低分；
    3. 未被任何题覆盖的维度记入 `not_covered`，而不是当成 0 分参与平均。

    返回 ``(已覆盖维度的 {label: 分数}, 未覆盖维度的 label 列表)``。
    """

    rubric = role_profile.get("rubric") or []
    if not isinstance(rubric, list) or not rubric:
        return {}, []
    scores: dict[str, float] = {}
    not_covered: list[str] = []
    for item in rubric:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        key = str(item["key"])
        label = str(item.get("label") or key)
        # M21：难度参与维度均值（按难度权重加权平均）
        weighted: list[tuple[float, float]] = []
        for record in topic_records:
            parsed = parse_number((record.get("dimensions") or {}).get(key))
            if parsed.value is not None:
                weighted.append((parsed.value, _topic_weight(record)))
        if weighted:
            weight_sum = sum(weight for _, weight in weighted)
            scores[label] = round(
                sum(value * weight for value, weight in weighted) / (weight_sum or 1), 2
            )
        else:
            not_covered.append(label)
    return scores, not_covered


def _has_out_of_range_values(records: list[dict]) -> bool:
    """M10 辅助：本次评分里是否出现过越界数值（在**原始值**层面判定）。

    `safe_score` 会把越界值先夹到边界，到最终总分时已经看不出越界了，
    因此必须在源头扫描。
    """

    for record in records or []:
        status = parse_number(record.get("score")).status
        if status in {STATUS_CLAMPED_LOW, STATUS_CLAMPED_HIGH}:
            return True
        for value in (record.get("dimensions") or {}).values():
            dim_status = parse_number(value).status
            if dim_status in {STATUS_CLAMPED_LOW, STATUS_CLAMPED_HIGH}:
                return True
    return False


def _rubric_dimension_keys(role_profile: dict) -> set[str]:
    rubric = role_profile.get("rubric") or []
    if not isinstance(rubric, list):
        rubric = []
    keys = {
        str(item["key"])
        for item in rubric
        if isinstance(item, dict) and item.get("key")
    }
    # M19：动机题有独立量规，其维度也要出现在报告里
    keys.update({"m1", "m2"})
    return keys


def _filter_dimensions(dimensions, allowed_keys: set[str]) -> dict[str, float]:
    """M15/M16：报告里只保留 rubric 定义的维度。

    模型仍可能按提示词示例回显英文六维；若让它们进入报告，就会出现
    "两套维度并存"的老问题。原始值仍完整保留在 state.assessments 里，
    报告作为对外产物只呈现契约内的维度。
    """

    if not allowed_keys:
        return {}
    kept: dict[str, float] = {}
    for key, value in (dimensions or {}).items():
        name = str(key)
        if name not in allowed_keys:
            continue
        parsed = parse_number(value)
        if parsed.value is not None:
            kept[name] = parsed.value
    return kept


def _token_summary(state) -> dict:
    return summarize_usage(state_usage_records(state))


def _token_by_node(records: list[dict]) -> dict[str, int | None]:
    """按 usage_records[].node 聚合 total_tokens。

    - 同一节点多条记录 → 求和；
    - plan_questions 与 plan_questions_retry 分开计（不合并）；
    - 未产生 usage 的节点不出现（不补 0）。
    """
    totals: dict[str, int | None] = {}
    for item in records or []:
        if not isinstance(item, dict):
            continue
        node = str(item.get("node", "") or "").strip()
        if not node or item.get("usage_status") == "not_called":
            continue
        if (item.get("usage_status") in {"missing", "partial", "estimated"}
                or item.get("unknown_calls",0) or item.get("total_tokens") is None):
            totals[node] = None
        elif node not in totals or totals[node] is not None:
            totals[node] = int(totals.get(node, 0) or 0) + int(item["total_tokens"])
    return totals


def _guard_report_fields(state) -> dict:
    """把 plan 写入状态的 guard_summary 映射成报告字段（3.2.4）。"""
    summary = state.get("guard_summary") or {}
    samples = [str(item) for item in (summary.get("guard_dropped_samples") or [])]
    return {
        "guard_audit_status": str(summary.get("guard_audit_status", "") or ""),
        "guard_uncertain_count": int(summary.get("guard_uncertain_count", 0) or 0),
        "guard_dropped_count": int(summary.get("guard_dropped_count", 0) or 0),
        "guard_dropped_samples": samples[:5],
        "weak_focus_count": int(summary.get("weak_focus_count", 0) or 0),
        "guard_evidence_mismatch_count": int(
            summary.get("evidence_mismatch_count", 0) or 0
        ),
        "guard_degrade_reason": str(summary.get("guard_degrade_reason", "") or ""),
        "guard_domain_slug_missing": bool(
            summary.get("guard_domain_slug_missing", False)
        ),
    }


def _rubric_weighted_score(topic_records: list[dict], role_profile: dict) -> float | None:
    """M15：rubric 加权总分，分母同样是"覆盖该维度的题数"。

    未被覆盖的维度**不参与**加权（也不按 0 分计入），避免"没考到"被当成"考砸了"。
    """

    rubric = role_profile.get("rubric", []) or []
    if not isinstance(rubric, list) or not rubric:
        return None
    weights: dict[str, float] = {}
    for item in rubric:
        if isinstance(item, dict) and item.get("key"):
            parsed_weight = parse_number(item.get("weight", 0.0), low=-1e9, high=1e9)
            if parsed_weight.value is None:
                continue
            weights[str(item["key"])] = parsed_weight.value
    if not weights:
        return None

    available: list[tuple[float, float]] = []
    for key, weight in weights.items():
        values: list[tuple[float, float]] = []
        for record in topic_records:
            parsed = parse_number((record.get("dimensions") or {}).get(key))
            if parsed.value is not None:
                values.append((parsed.value, _topic_weight(record)))
        if values:
            local_weight = sum(item_weight for _, item_weight in values)
            mean = sum(value * item_weight for value, item_weight in values) / (local_weight or 1)
            available.append((mean, weight))
    if not available:
        return None
    weight_sum = sum(weight for _, weight in available)
    if weight_sum <= 0:
        return None
    return round(
        sum(mean * weight for mean, weight in available) / weight_sum,
        2,
    )


def evaluate(state) -> dict:
    assessments = state.get("assessments", []) or []
    history = list(state.get("conversation_history", []) or [])
    closing = "本次模拟面试到这里，接下来我会给出评估报告。"
    if not (
        history
        and history[-1].get("role") == "assistant"
        and closing in str(history[-1].get("content", ""))
    ):
        history.append({"role": "assistant", "content": closing})
    if not assessments:
        report = EvaluationReport(
            overall_score=0.0,
            grade="—",
            text_analysis="本轮没有可评估的完整回答，未生成评分。下次面试请至少完整回答一道题后再结束。",
            learning_path=[],
            resources=[],
            research_status=str(state.get("research_status", "") or ""),
            references_status=str(state.get("references_status", "") or ""),
            # M12：零评估分支同样标注有效样本数
            **_sample_fields(state, []),
            **_guard_report_fields(state),
            token_by_node=_token_by_node(state.get("usage_records") or []),
        )
        return {"evaluation_report": report.model_dump(), "conversation_history": history}
    summaries = summaries_text(state.get("summaries", []) or [])
    # M5：评分失败的记录（score_error）不进任何聚合，只计数上报
    score_errors = sum(bool(item.get('score_error')) for item in assessments)
    role_profile = state.get("role_profile", {}) or {}
    rubric_keys = _rubric_dimension_keys(role_profile)
    # M14：先按题合并（多轮追问 → 一条），再做聚合
    # 报告用含评分失败的题（用户要看得见），聚合用排除失败后的记录
    topic_records = _topic_records(assessments)
    scoring_records = [item for item in topic_records if not item.get("score_error") and item.get('scoreable') is not False]
    # M15 + M16：维度只认 rubric；未被覆盖的维度单独标出
    dimension_scores, not_covered_dimensions = _dimension_scores(scoring_records, role_profile)
    weighted_overall = _rubric_weighted_score(scoring_records, role_profile)
    if weighted_overall is not None:
        overall = weighted_overall
        aggregation_mode = "rubric_weighted"
    else:
        # M16：rubric 缺失 → 降级为"仅总分 + 文字评语"
        # M21：题均分同样按难度加权
        weighted_topics = [
            (value, _topic_weight(item))
            for value, item in (
                (safe_score(record.get("score")), record) for record in scoring_records
            )
            if value is not None
        ]
        total_weight = sum(weight for _, weight in weighted_topics)
        overall = (
            round(
                sum(value * weight for value, weight in weighted_topics) / (total_weight or 1), 2
            )
            if weighted_topics
            else 0.0
        )
        aggregation_mode = "topic_mean"

    # M27：逐题评分补上"是不是追问 / 难度 / 题型"，让终评能分辨追问与专业/动机题
    assessment_records = [
        {
            "q": item.get("question_id"),
            "idx": item.get("plan_question_index"),
            "follow_ups": item.get("follow_up_rounds", 0),
            "is_follow_up": bool(item.get("is_follow_up")),
            "difficulty": str(item.get("difficulty") or "medium"),
            "question_kind": str(item.get("question_kind") or "professional"),
            "score": item.get("score"),
            "dims": _filter_dimensions(item.get("dimensions"), rubric_keys),
            "covered": item.get("covered_points", []),
            "missed": item.get("missed_points", []),
            "reason": item.get("follow_up_reason", ""),
            # M5：只有评分失败时才多带一个标记，正常路径不增加 token
            **({"err": "score_unavailable"} if item.get("score_error") else {}),
        }
        for item in topic_records
    ]
    qa_records = [
        {
            "question": str(record.get("question", ""))[:3000],
            "answer": str(record.get("answer", ""))[:3000],
        }
        for record in (state.get("turn_records", []) or [])
    ]
    # M28：按"条"裁剪，保证两块都是合法 JSON（不再按字符截断）
    assessment_records, assessment_trimmed = _trim_to_budget(assessment_records)
    qa_records, qa_trimmed = _trim_to_budget(qa_records)
    assessment_text = json.dumps(assessment_records, ensure_ascii=False)
    qa_text = json.dumps(qa_records, ensure_ascii=False)

    # M27：把聚合摘要、题数、追问数、被丢弃题数、调研状态一并交给终评，
    # 让评语有依据（原先模型看不到任何聚合与诊断信息）。
    guard_summary = state.get("guard_summary") or {}
    follow_up_total = sum(int(item.get("follow_up_rounds", 0) or 0) for item in topic_records)
    context_block = json.dumps(
        {
            "aggregation_mode": aggregation_mode,
            "plan_question_count": int(state.get("plan_question_count") or len(state.get("question_plan") or [])),
            "matched_question_count": len(topic_records),
            "follow_up_rounds_total": follow_up_total,
            "dimension_scores": dimension_scores,
            "not_covered_dimensions": not_covered_dimensions,
            "guard_dropped_count": int(guard_summary.get("guard_dropped_count", 0) or 0),
            "guard_uncertain_count": int(guard_summary.get("guard_uncertain_count", 0) or 0),
            "research_status": str(state.get("research_status", "") or ""),
            "references_status": str(state.get("references_status", "") or ""),
            "score_errors": score_errors,
            "difficulty_weighting": bool(getattr(settings, "difficulty_weighting", True)),
            "assessment_trimmed": assessment_trimmed,
            "qa_trimmed": qa_trimmed,
        },
        ensure_ascii=False,
    )
    user_content = (
        f"压缩摘要：\n{summaries}\n\n"
        f"本轮统计（供你判断整体口径，不要直接复述数字）：\n{context_block}\n\n"
        f"逐题评分：\n{assessment_text}\n\n"
        f"完整问答原文：\n{qa_text or '（无）'}\n\n"
        f"平均分 {overall}。"
    )
    messages = [
        SystemMessage(
            content=EVALUATE_SYSTEM.format(
                role_policy=role_policy_text(state.get("role_profile") or {}),
                rubric_text=json.dumps(
                    (state.get("role_profile") or {}).get("rubric", []),
                    ensure_ascii=False,
                ),
            )
        ),
        HumanMessage(content=user_content),
    ]
    report_generation_status = "generated"
    try:
        raw, usage = chat_with_usage(messages, temperature=settings.temp_evaluate)
        parsed = parse_json_with_retries(raw) or {}
    except SessionBudgetExceeded as exc:
        # Deterministic scoring survives exhaustion; never fabricate model commentary.
        parsed = {"text_analysis":"本次反馈根据已完成回答汇总；文字分析生成未完成。"}
        usage = exc.usage_info
        report_generation_status = "budget_skipped"

    token_summary = _token_summary(state)
    # M8：总分归属反转——最终分**一律**取代码算出的 overall；
    #    模型给出的 overall_score 只作为审计留档，不参与取值。
    overall_result = parse_number(overall, low=SCORE_MIN, high=SCORE_MAX)
    final_overall = round(
        overall_result.value if overall_result.value is not None else 0.0, 2
    )
    # M10：越界留痕（原先 85 会原样进报告，界面出现"85/10"）
    out_of_range = overall_result.status in {STATUS_CLAMPED_LOW, STATUS_CLAMPED_HIGH} or (
        _has_out_of_range_values(scoring_records)
    )
    llm_overall_opinion = None
    if isinstance(parsed, dict) and "overall_score" in parsed:
        opinion = parse_number(parsed.get("overall_score"), low=-1e9, high=1e9)
        llm_overall_opinion = opinion.value
    # M9：等级同样收归代码——由最终总分按配置阈值推导；
    #    模型给出的 grade 只作为审计留档（与 llm_overall_opinion 同构）。
    llm_grade_opinion = _normalize_text(parsed.get("grade")) or None
    final_grade = _safe_grade(_grade_from_score(final_overall)) if scoring_records else '—'

    report = EvaluationReport(
        overall_score=final_overall,
        # M35：代码值单独留档（当前与 overall_score 同值，便于跨版本审计）
        overall_code=final_overall,
        # M34：可追溯元信息
        scoring_meta={
            "model": str(settings.model or ""),
            "temperature": float(settings.temp_evaluate),
            "prompt_version": PROMPT_VERSION,
            "rubric_version": str(
                (role_profile.get("rubric_version") or PROMPT_VERSION)
            ),
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        llm_overall_opinion=llm_overall_opinion,
        out_of_range=out_of_range,
        grade=final_grade,
        llm_grade_opinion=llm_grade_opinion,
        text_analysis=_normalize_text(parsed.get("text_analysis")),
        strengths=_normalize_text_list(parsed.get("strengths")),
        weaknesses=_normalize_text_list(parsed.get("weaknesses")),
        dimension_scores=dimension_scores,
        # M15：未被任何题覆盖的维度（不按 0 分计入）
        not_covered_dimensions=not_covered_dimensions,
        # M16/M35：本次用的是哪种聚合口径
        aggregation_mode=aggregation_mode,
        per_question=[
            {
                "question_id": item.get("question_id"),
                # M13：题单下标（M14 按题合并、报告可追溯的依据）
                "plan_question_index": item.get("plan_question_index", -1),
                "question": item.get("original_question") or item.get("question", ""),
                "question_summary": item.get("question_summary", ""),
                "score": item.get("score"),
                "max_score": item.get("max_score", 10),
                # M14：合并后每题一行；追问轮数单独记录，原始各轮分数保留在 round_scores
                "is_follow_up": False,
                "follow_up_rounds": item.get("follow_up_rounds", 0),
                "round_scores": item.get("round_scores", []),
                "rounds": item.get("rounds", []),
                "last_assessment_question_id": item.get("last_assessment_question_id"),
                "dimensions": _filter_dimensions(item.get("dimensions"), rubric_keys),
                "missed_points": item.get("missed_points", []),
                "covered_points": item.get("covered_points", []),
                "follow_up_reason": item.get("follow_up_reason", ""),
                # M5：让报告显式暴露"这条没评出分"，而不是悄悄当成 0
                "score_error": bool(item.get("score_error")),
                "scoreable": item.get('scoreable', item.get('score') is not None),
                "evidence": item.get('evidence', {}),
                "confidence": item.get('confidence', 'low'),
                "question_kind": item.get('question_kind', 'professional'),
                "hallucination_or_conflict": item.get('hallucination_or_conflict', False),
            }
            for item in topic_records
        ],
        difficulty_events=list(state.get("difficulty_events", []) or []),
        token_totals=token_summary,
        budget_summary=state.get("budget_summary") or {},
        report_generation_status=report_generation_status,
        cost=token_summary["cost"],
        learning_path=_normalize_text_list(parsed.get("learning_path")),
        resources=_normalize_text_list(parsed.get("resources")),
        research_status=str(state.get("research_status", "") or ""),
        references_status=str(state.get("references_status", "") or ""),
        # M5：评分失败的记录数（已排除出总分）
        score_errors=score_errors,
        unscoreable_count=sum(item.get('scoreable') is False for item in assessments),
        # M12：样本完整性
        **_sample_fields(state, assessments),
        # M26：简历项目使用统计
        resume_project_stats=dict(state.get("resume_project_stats") or {}),
        **_guard_report_fields(state),
        token_by_node=_token_by_node(state.get("usage_records") or []),
    )
    return {
        "evaluation_report": report.model_dump(),
        "conversation_history": history,
        "usage_records": [
            {
                **usage,
                "node": "evaluate",
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        ],
    }
