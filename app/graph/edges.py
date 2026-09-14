from __future__ import annotations

import math
import re

from app.config import settings
from app.utils.numeric import safe_score

# M11：早停门槛 = max(min_questions, ceil(target × 该比例))
EARLY_STOP_TARGET_RATIO = 0.6


def _normalize_point(text: str) -> str:
    """把遗漏点/追问理由归一化，用于判断是否是同一个知识点。"""
    return re.sub(
        r"[\s，。、；：,.;:!！?？\-—_()（）\[\]【】「」{}<>\"'“”‘’/\\|]+",
        "",
        str(text or ""),
    ).lower()


# 追问理由里几乎每句都会出现的外壳词，比对知识点前先剥掉，
# 否则「候选人完全未回答 X」和「候选人没有提到 Y」会因为是同一句式被误判成同一个点。
_REASON_FILLERS = (
    "该候选人",
    "候选人",
    "应试者",
    "答非所问",
    "偏题",
    "本题",
    "这道题",
    "上一轮",
    "本轮",
    "重新",
    "再次",
    "需要",
    "建议",
    "应该",
    "可以",
    "展示",
    "体现",
    "表明",
    "表示",
    "说明",
    "属于",
    "严重",
    "明显",
    "完全",
    "仍然",
    "依然",
    "无法",
    "未能",
    "没有",
    "不能",
    "回答",
    "提问",
    "追问",
    "引导",
    "确认",
    "一点",
    "任何",
    "具体",
    "详细",
    "比较",
    "非常",
    "相当",
    "基础",
    "这个问题",
)


def _topic(text: str) -> str:
    """从追问理由里抽出真正的知识点：归一化后剥掉外壳词。"""
    value = _normalize_point(text)
    if not value:
        return ""
    stripped = value
    for filler in _REASON_FILLERS:
        stripped = stripped.replace(filler, "")
    stripped = stripped.strip("的了吗呢")
    return stripped or value


def _question_segment(assessments: list[dict]) -> list[dict]:
    """取本题的记录：从最近一条「主问题」评分一直到末尾（含各轮追问）。"""
    segment: list[dict] = []
    for item in reversed(assessments or []):
        segment.append(item)
        if not item.get("is_follow_up"):
            break
    segment.reverse()
    return segment


def _asked_points(segment: list[dict]) -> set[str]:
    """
    本题此前每一轮追问各自盯的那个知识点。

    只取 follow_up_reason（即「这次追问是为了补哪个点」），
    不取主问题的整张 missed_points 列表——那张表太宽，会把还没问过的点也算成已问过。
    """
    points: list[str] = []
    for item in segment:
        if str(item.get("next_action", "") or "") != "follow_up":
            continue
        reason = _topic(item.get("follow_up_reason", ""))
        if reason:
            points.append(reason)
    return points


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _same_point(left: str, right: str, threshold: float = 0.4) -> bool:
    """
    中文短语没有空格，用两个近似信号判断是不是同一个知识点：
    1) 共享任意一个三字片段（如「原始文」「始文档」）；
    2) 短句的字符二元组大部分能在长句里找到（包含度 ≥ threshold）。
    命中任意一条即视为同一个知识点。
    """
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    for index in range(len(shorter) - 2):
        if shorter[index : index + 3] in longer:
            return True
    first = _char_bigrams(shorter)
    second = _char_bigrams(longer)
    if not first or not second:
        return False
    return len(first & second) / len(first) >= threshold


def _current_points(latest: dict) -> list[str]:
    """
    本轮追问实际盯的那个知识点。

    以 follow_up_reason 为准：它才是「这一轮打算深挖什么」的定义。
    不能把整张 missed_points 也算进来——那是本题的知识点清单，
    每轮重新生成、措辞会变，拿它比对会把同一件事永远判成「新知识点」。
    只有模型没给 reason 时才退回 missed_points。
    """
    reason = _topic(latest.get("follow_up_reason", ""))
    if reason:
        return [reason]
    points = [_topic(point) for point in (latest.get("missed_points") or [])]
    return [point for point in points if point]


def _has_fresh_point(latest: dict, asked: list[str]) -> bool:
    """latest 这一轮里，是否还有跟此前追问过的知识点「不是同一个」的遗漏点。"""
    points = _current_points(latest)
    if not points:
        return False
    return any(
        all(not _same_point(point, item) for item in asked) for point in points
    )


def would_repeat_knowledge_point(assessments: list[dict]) -> bool:
    """
    同一知识点是否已经被追问过：是则本轮不再追问。

    只有在候选人这一轮暴露出「全新的遗漏点」时才允许再追问一次；
    如果仍然是同一个没答上来的知识点，就不许跨轮重复提问。
    """
    segment = _question_segment(assessments)
    if len(segment) < 2:
        return False
    asked = _asked_points(segment[:-1])
    if not asked:
        return False
    return not _has_fresh_point(segment[-1], asked)


def recent_scores(assessments: list[dict], count: int = 3) -> list[float]:
    """取最近 ``count`` 个**可用**分数（M5）。

    M5 之后 ``score`` 可能是 ``None``（评分失败），必须跳过而不是当成 0 分——
    否则一次解析失败就会把难度判断和早停判断一起拉偏。
    """

    scores: list[float] = []
    for item in assessments or []:
        value = safe_score((item or {}).get("score"))
        if value is not None:
            scores.append(value)
    return scores[-count:]


def route_after_assessment(state) -> str:
    if state.get("budget_exhausted"):
        return "evaluate"
    if state.get("explanation_pending"):
        return "ask_explain"
    assessments = state.get("assessments", []) or []
    if not assessments:
        return "evaluate"

    latest = assessments[-1]
    if state.get("end_requested") or latest.get("stop_suggested"):
        return "evaluate"
    action = str(latest.get("next_action", "") or "")
    action_map = {
        "follow_up": "ask_follow_up",
        "next_question": "advance",
        "evaluate": "evaluate",
    }
    if action in action_map:
        # P0/B2：会话级追问预算（与下面 should_follow_up 分支同口径）
        if action == "follow_up" and int(state.get("total_follow_ups", 0) or 0) >= int(
            state.get("max_total_follow_ups", settings.max_total_follow_ups)
        ):
            action = "next_question"
        if action == "follow_up" and would_repeat_knowledge_point(assessments):
            # 同一知识点已经追问过一轮仍未答上来：不再重复追问，直接换下一题
            action = "next_question"
        if action == "follow_up" and int(state.get("follow_up_count", 0)) >= int(
            state.get("max_follow_ups", settings.max_follow_ups)
        ):
            action = "next_question"
        if action == "next_question":
            index = int(state.get("current_question_index", 0))
            question_plan = state.get("question_plan", []) or []
            if index + 1 >= len(question_plan):
                return "evaluate"
            # M11：早停修复——原先只在 next_action 失效的兜底分支才判定，
            # 而 assess 总会写入合法三值，导致早停永不触发。现在放进正常分支。
            if _should_stop_early(assessments):
                return "evaluate"
            return "advance"
        return action_map.get(action, "advance")
    follow_up_count = int(state.get("follow_up_count", 0))
    max_follow_ups = int(state.get("max_follow_ups", settings.max_follow_ups))

    if latest.get("should_follow_up") and follow_up_count < max_follow_ups:
        # P0/B2：会话级追问预算封顶——单场所有追问之和达到上限后不再追问，
        # 直接换题，把最坏情况成本压回预算内。
        if int(state.get("total_follow_ups", 0) or 0) >= int(
            state.get("max_total_follow_ups", settings.max_total_follow_ups)
        ):
            index = int(state.get("current_question_index", 0))
            question_plan = state.get("question_plan", []) or []
            return "advance" if index + 1 < len(question_plan) else "evaluate"
        if would_repeat_knowledge_point(assessments):
            index = int(state.get("current_question_index", 0))
            question_plan = state.get("question_plan", []) or []
            return "advance" if index + 1 < len(question_plan) else "evaluate"
        return "ask_follow_up"

    index = int(state.get("current_question_index", 0))
    question_plan = state.get("question_plan", []) or []
    if index + 1 < len(question_plan):
        if _should_stop_early(assessments):
            return "evaluate"
        return "advance"

    # 题单已耗尽：一律进入评估，不再补题、不再重复最后一题
    return "evaluate"


def route_after_ask(state) -> str:
    """ask 节点没有生成问题（题单耗尽）时直接进入评估。"""
    if state.get("budget_exhausted"):
        return "evaluate"
    if str(state.get("current_question", "") or "").strip():
        return "assess"
    return "evaluate"


def after_explain(state) -> str:
    index = int(state.get("current_question_index", 0))
    question_plan = state.get("question_plan", []) or []
    if state.get("end_requested"):
        return "evaluate"
    if index + 1 < len(question_plan):
        return "advance"
    return "evaluate"


def route_after_follow_up(state) -> str:
    """A guard-rejected closing statement is not an answerable question."""
    if state.get("budget_exhausted"):
        return "evaluate"
    if state.get("follow_up_abandoned"):
        return after_explain(state)
    return "assess"


def _main_question_count(assessments: list[dict]) -> int:
    """M11：早停只数**主问题**，追问不算题量。"""

    return sum(1 for item in assessments or [] if not (item or {}).get("is_follow_up"))


def early_stop_threshold() -> int:
    """M11：早停门槛 = ``max(min_questions, ceil(target × 0.6))``。"""

    return max(
        int(settings.min_questions),
        int(math.ceil(int(settings.target_questions) * EARLY_STOP_TARGET_RATIO)),
    )


def _should_stop_early(assessments: list[dict]) -> bool:
    if _main_question_count(assessments) < early_stop_threshold():
        return False
    recent = recent_scores(assessments, 4)
    # 评分失败的记录已被 recent_scores 过滤，因此要求至少 4 个有效分数
    return len(recent) >= 4 and min(recent) >= 8.0


def next_question_metadata(state) -> dict:
    index = int(state.get("current_question_index", 0))
    plan = state.get("question_plan", []) or []
    if index < len(plan):
        return dict(plan[index])
    return {}
