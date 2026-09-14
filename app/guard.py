"""跨职能域审计的编排层（3.2.0 的"白名单 + 证据约束 + 默认保留"）。

职责分工（与 3.2.1 的约定一致）：
- 纯判定在 app/role.py：match_focus / out_of_scope_hits / evidence_status /
  is_motivation_question / motivation_exempt / guard_forbidden_topics / render_focus_whitelist
- 本模块只做编排：分层命中 → 批量 LLM 仲裁 → D1/D2/D3 三票合成 → AuditRecord

核心规则：**删除需要 ≥2 票反对**，任何不确定/服务不可用/预算耗尽一律保留。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Callable, TypedDict

from app.config import settings
from app.llm.client import chat_with_usage
from app.role import (
    GUARD_MODE_LEGACY,
    GUARD_MODE_WEAK_WHITELIST,
    evidence_status,
    guard_forbidden_topics,
    guard_is_active,
    is_motivation_question,
    match_focus,
    motivation_exempt,
    out_of_scope_hits,
    profile_quality,
    render_focus_whitelist,
)
from app.utils.json_utils import parse_json_with_retries

logger = logging.getLogger(__name__)

# ---- 审计状态 ----
GUARD_STATUS_OK = "ok"
GUARD_STATUS_DEGRADED = "degraded"
GUARD_STATUS_DISABLED = "disabled"

# ---- 成本与判定参数 ----
ARBITER_MAX_CALLS_PER_SESSION = 1
CACHE_LIMIT = 512
DROP_VOTES_REQUIRED = 2

# ---- 判定结果 ----
VERDICT_KEEP = "keep"
VERDICT_DROP = "drop"
VERDICT_KEPT_UNVERIFIED = "kept_unverified"

ARBITER_KEEP = "keep"
ARBITER_OUT_OF_SCOPE = "out_of_scope"
ARBITER_UNCERTAIN = "uncertain"
_ARBITER_VERDICTS = {ARBITER_KEEP, ARBITER_OUT_OF_SCOPE, ARBITER_UNCERTAIN}

# ---- 审计阶段（stage 是超集，便于离线评测定位问题）----
STAGE_BINDING = "binding"            # 生成期声明的 focus_key 命中白名单
STAGE_WHITELIST_HIT = "whitelist_hit"  # 题面文本命中白名单
STAGE_EXEMPT = "exempt"              # 通用动机/经历题豁免
STAGE_UNVERIFIED = "unverified"      # 未命中白名单且无任何选票
STAGE_STRONG_SIGNAL = "strong_signal"  # 命中跨域强信号（含 D1+D2 短路）
STAGE_ARBITER = "arbiter"            # 由仲裁参与裁决
STAGE_LEGACY = "legacy"              # legacy 降级路径


class AuditRecord(TypedDict, total=False):
    """单题的审计记录。字段名与 3.2.4 的 EvaluationReport 汇总字段对齐。"""

    question_id: int
    question: str
    verdict: str            # keep / drop / kept_unverified
    stage: str
    match_stage: str        # 白名单命中的细分层：binding/ascii/substring/cooccur/similarity
    focus_key: str
    matched_subtopic: str
    evidence: str
    reason: str
    votes: dict[str, bool]          # D1 无归属无来源 / D2 强信号命中 / D3 仲裁确认跨域
    vote_evidence: dict[str, str]   # 每一票的可复现证据
    binding_status: str
    evidence_status: str
    evidence_invalid: list[str]
    guard_mode: str
    round: int
    raw: dict


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()


def _question_id(question: dict, fallback: int = 0) -> int:
    try:
        return int((question or {}).get("id", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _question_blob(question: dict) -> str:
    """参与判定与命中的文本：题面 + 考察意图 + 技能。"""
    return " ".join(
        [
            str(question.get("content", "") or ""),
            str(question.get("intent", "") or ""),
            " ".join(str(item) for item in (question.get("skills") or [])),
        ]
    )


def _focus_signature(profile: dict) -> str:
    parts: list[str] = []
    for item in (profile or {}).get("assessment_focus") or []:
        if isinstance(item, dict):
            parts.append(f"{item.get('key', '')}:{item.get('name', '')}")
        else:
            parts.append(str(item))
    return "|".join(parts)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 判定缓存（进程内 LRU，不落盘）
# --------------------------------------------------------------------------


class JudgementCache:
    """仲裁结果缓存：key = sha1(domain + focus 集合 + 归一化题面)。

    画像或白名单发生变化 → key 变化 → 不会误用旧结论。
    """

    def __init__(self, limit: int = CACHE_LIMIT) -> None:
        self.limit = max(1, int(limit))
        self._data: "OrderedDict[str, dict]" = OrderedDict()

    @staticmethod
    def key(profile: dict, text: str) -> str:
        """sha1(domain + focus 集合 + out_of_scope 哈希 + 归一化题面)。

        out_of_scope 也参与键：画像的任何变更（含禁止话题调整）都不会误用旧判定。
        """
        data = profile or {}
        domain = str(data.get("domain", "") or "")
        out_of_scope = "、".join(guard_forbidden_topics(data))
        out_of_scope_hash = hashlib.sha1(out_of_scope.encode("utf-8")).hexdigest()[:12]
        payload = (
            f"{domain}||{_focus_signature(data)}||{out_of_scope_hash}||{_normalize_text(text)}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return dict(value)

    def set(self, key: str, value: dict) -> None:
        self._data[key] = dict(value)
        self._data.move_to_end(key)
        while len(self._data) > self.limit:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------
# 批量 LLM 仲裁
# --------------------------------------------------------------------------

ARBITER_SYSTEM = """你是面试题与岗位的匹配审核员。只输出 JSON，不要任何解释、不要输出多余文字。
规则：
1. 白名单是该岗位的考察范围（依据 JD 与简历生成）；题目属于白名单或它的直接延伸时判 keep。
2. 判 out_of_scope 必须有正面证据：题目考查的知识或技能明显属于另一个职能域，
   且不属于白名单中任何一条 focus 的直接或间接延伸。
3. 拿不准一律 keep：宁可保留一道无关题，也绝不许误删一道合法题。
4. evidence 必须是题目原文里的连续片段，不得改写、不得编造；
   引用不出来的判定会被系统降级为 uncertain。
5. verdict 只能是 keep / out_of_scope / uncertain。"""


def build_arbiter_messages(profile: dict, pending: list[dict]) -> list[dict]:
    """构造仲裁消息：白名单只渲染一次（render_focus_whitelist），不重复贴 subtopics。"""
    whitelist = render_focus_whitelist(profile)
    out_of_scope = "、".join(guard_forbidden_topics(profile)) or "（无）"
    items = [
        {
            "qid": _question_id(question, index),
            "content": str(question.get("content", ""))[:300],
            "skills": [str(item) for item in (question.get("skills") or [])],
            "intent": str(question.get("intent", ""))[:80],
        }
        for index, question in enumerate(pending, 1)
    ]
    user_content = (
        f"岗位：{(profile or {}).get('job_title', '未知岗位')}\n"
        f"职能域：{(profile or {}).get('domain', 'general')}\n"
        f"白名单（考察范围）：\n{whitelist}\n\n"
        f"已知的跨域话题：{out_of_scope}\n\n"
        f"待判定题目：\n{json.dumps(items, ensure_ascii=False)}\n\n"
        '只输出 JSON：{"judgements":[{"qid":1,"verdict":"keep|out_of_scope|uncertain",'
        '"focus_key":"f1 或空","reason":"一句话","evidence":"题目原文片段","confidence":0.0}]}'
    )
    return [
        {"role": "system", "content": ARBITER_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _evidence_in_text(evidence: str, content: str) -> bool:
    """校验 evidence 是否真的是题面原文片段（压缩空白与标点后做包含判断）。"""
    needle = _normalize_text(evidence).replace(" ", "")
    haystack = _normalize_text(content).replace(" ", "")
    if not needle or not haystack:
        return False
    return needle in haystack


def parse_arbiter_response(raw: str, pending: list[dict]) -> dict[int, dict]:
    """解析并校验仲裁返回：非法 verdict 归 uncertain；证据不可校验 → 降级 uncertain。"""
    parsed = parse_json_with_retries(str(raw or "")) or {}
    entries = parsed.get("judgements") if isinstance(parsed, dict) else None
    contents = {
        _question_id(question, index): str(question.get("content", "") or "")
        for index, question in enumerate(pending, 1)
    }
    results: dict[int, dict] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        qid = _question_id({"id": entry.get("qid")}, 0)
        if not qid:
            continue
        verdict = str(entry.get("verdict", "") or "").strip().lower()
        if verdict not in _ARBITER_VERDICTS:
            verdict = ARBITER_UNCERTAIN
        evidence = str(entry.get("evidence", "") or "").strip()
        valid = True
        if verdict == ARBITER_OUT_OF_SCOPE:
            valid = _evidence_in_text(evidence, contents.get(qid, ""))
            if not valid:
                verdict = ARBITER_UNCERTAIN
        results[qid] = {
            "verdict": verdict,
            "focus_key": str(entry.get("focus_key", "") or "").strip(),
            "reason": str(entry.get("reason", "") or "").strip(),
            "evidence": evidence,
            "confidence": _coerce_float(entry.get("confidence")),
            "valid": valid,
        }
    return results


class BatchArbiter:
    """批量仲裁器：一次调用判所有未命中题；单会话调用硬上限（默认 1）。"""

    def __init__(
        self,
        max_calls: int = ARBITER_MAX_CALLS_PER_SESSION,
        cache: JudgementCache | None = None,
        judge_fn: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self.max_calls = max(0, int(max_calls))
        self.cache = cache if cache is not None else JudgementCache()
        self.judge_fn = judge_fn
        self._calls = 0
        # 仲裁调用的 token 记账（node="guard_arbiter"）。
        # 注入 judge_fn、预算耗尽、调用抛异常时都不会产生记录（不写假的 0 usage）。
        self.usage_records: list[dict] = []

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def exhausted(self) -> bool:
        return self._calls >= self.max_calls

    def _invoke(self, messages: list[dict]) -> str:
        if self.judge_fn is not None:
            return self.judge_fn(messages)
        content, usage = chat_with_usage(messages, temperature=settings.temp_fact_check)
        self.usage_records.append(
            {
                **usage,
                "node": "guard_arbiter",
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        )
        return content

    def judge(self, pending: list[dict], profile: dict) -> dict[int, dict]:
        """返回 {question_id: judgement}；judgement 含 verdict/reason/evidence/valid/source。"""
        results: dict[int, dict] = {}
        unresolved: list[tuple[str, dict]] = []
        for index, question in enumerate(pending, 1):
            key = self.cache.key(profile, str(question.get("content", "")))
            cached = self.cache.get(key)
            if cached is not None:
                results[_question_id(question, index)] = {**cached, "source": "cache"}
                continue
            unresolved.append((key, question))
        if not unresolved:
            return results
        if self.exhausted:
            for index, (_, question) in enumerate(unresolved, 1):
                results[_question_id(question, index)] = {
                    "verdict": ARBITER_UNCERTAIN,
                    "reason": "仲裁调用预算已用尽（单会话上限 1 次），按默认保留处理",
                    "focus_key": "",
                    "evidence": "",
                    "confidence": 0.0,
                    "valid": False,
                    "source": "budget_exhausted",
                }
            return results

        self._calls += 1
        pending_questions = [question for _, question in unresolved]
        parsed: dict[int, dict] = {}
        degraded = False
        try:
            parsed = parse_arbiter_response(
                self._invoke(build_arbiter_messages(profile, pending_questions)),
                pending_questions,
            )
        except Exception as exc:  # noqa: BLE001 - 仲裁失败必须降级而不是中断面试
            logger.warning("Guard arbiter call failed: %s", type(exc).__name__)
            degraded = True

        for index, (key, question) in enumerate(unresolved, 1):
            qid = _question_id(question, index)
            judgement = parsed.get(qid)
            if judgement is None:
                judgement = {
                    "verdict": ARBITER_UNCERTAIN,
                    "focus_key": "",
                    "reason": "仲裁服务不可用，按默认保留处理"
                    if degraded
                    else "仲裁未返回该题结论，按默认保留处理",
                    "evidence": "",
                    "confidence": 0.0,
                    "valid": False,
                }
            judgement = {**judgement, "source": "degraded" if degraded else "llm"}
            self.cache.set(key, {k: v for k, v in judgement.items() if k != "source"})
            results[qid] = judgement
        return results


# --------------------------------------------------------------------------
# 逐题审计
# --------------------------------------------------------------------------


def _base_record(question: dict, profile: dict, index: int, round_index: int) -> AuditRecord:
    return {
        "question_id": _question_id(question, index),
        "question": str(question.get("content", ""))[:200],
        "verdict": VERDICT_KEEP,
        "stage": "",
        "match_stage": "",
        "focus_key": str(question.get("focus_key", "") or ""),
        "matched_subtopic": "",
        "evidence": "",
        "reason": "",
        "votes": {"D1": False, "D2": False, "D3": False},
        "vote_evidence": {"D1": "", "D2": "", "D3": ""},
        "binding_status": "n/a",
        "evidence_status": "n/a",
        "evidence_invalid": [],
        "guard_mode": str((profile or {}).get("guard_mode", "") or ""),
        "round": int(round_index),
        "raw": {},
    }


def _d1_evidence(question: dict, match: dict) -> str:
    declared = str(question.get("focus_key", "") or "")
    missing = [
        field
        for field in ("jd_ref", "project_ref", "research_ref")
        if not str(question.get(field, "") or "").strip()
    ]
    return (
        f"未绑定白名单（声明 focus_key={declared or '空'}，"
        f"匹配状态={match.get('binding_status', 'n/a')}）；"
        f"来源为空：{'、'.join(missing) if missing else '无'}"
    )


def _legacy_verdict(question: dict, profile: dict) -> dict:
    """legacy 降级路径：复刻模块 5 的行为（4 组词表 + profile_source 检查）。

    ⚠️ 删除语义差异（新人最容易误解之处，务必留意）：
    - **legacy 分支：单票删除** —— 命中 4 组词表即丢弃（与模块 5 完全一致）；
    - **whitelist 分支：两票删除** —— D1/D2/D3 中至少 2 票反对才丢弃。
    两套语义针对两类画像（旧画像 / 新画像），审计记录里用 stage="legacy" 区分。
    """
    if not guard_is_active(profile):
        return {
            "verdict": VERDICT_KEEP,
            "stage": STAGE_LEGACY,
            "reason": "legacy/兜底画像：只做提示词约束，不做硬丢弃",
        }
    hits = out_of_scope_hits(_question_blob(question), profile)
    if hits:
        return {
            "verdict": VERDICT_DROP,
            "stage": STAGE_LEGACY,
            "reason": "legacy 4 组守卫命中：" + "、".join(hits),
            "evidence": "、".join(hits),
            "votes": {"D1": False, "D2": True, "D3": False},
            "vote_evidence": {"D1": "", "D2": "、".join(hits), "D3": ""},
        }
    return {
        "verdict": VERDICT_KEEP,
        "stage": STAGE_LEGACY,
        "reason": "legacy 4 组守卫未命中",
    }


def audit_questions(
    questions: list[dict],
    profile: dict,
    *,
    jd_items: list[str] | None = None,
    resume_projects: list[str] | None = None,
    reference_ids: list[str] | None = None,
    reference_texts: dict[str, str] | None = None,
    arbiter: BatchArbiter | None = None,
    round_index: int = 1,
) -> dict:
    """审计一份题单：返回 kept / dropped / audit / status / arbiter_calls / drop_rate / summary。

    决策规则：D1（无归属无来源）、D2（命中跨域强信号）、D3（仲裁确认跨域）三票中
    **≥2 票反对才丢弃**；其余情况一律保留（未定题记 kept_unverified）。
    """
    data = profile or {}
    mode = str(data.get("guard_mode", "") or GUARD_MODE_LEGACY)
    audited: list[tuple[dict, AuditRecord]] = []
    pending: list[tuple[dict, AuditRecord]] = []
    status = GUARD_STATUS_OK

    if mode == GUARD_MODE_LEGACY:
        status = GUARD_STATUS_DISABLED
    elif mode == GUARD_MODE_WEAK_WHITELIST:
        # 弱白名单：只标记不丢弃
        status = GUARD_STATUS_DEGRADED

    for index, question in enumerate(questions, 1):
        record = _base_record(question, data, index, round_index)

        if mode == GUARD_MODE_LEGACY:
            record.update(_legacy_verdict(question, data))
            audited.append((question, record))
            continue

        blob = _question_blob(question)
        evidence = evidence_status(
            question, jd_items, resume_projects, reference_ids, reference_texts
        )
        record["evidence_status"] = evidence.get("evidence_status", "n/a")
        record["evidence_invalid"] = list(evidence.get("evidence_invalid") or [])

        matched = match_focus(
            blob, data, declared_focus_key=str(question.get("focus_key", "") or "")
        )
        record["binding_status"] = str(matched.get("binding_status", "n/a"))
        record["match_stage"] = str(matched.get("stage", "none"))
        if matched.get("focus_key"):
            record["focus_key"] = str(matched["focus_key"])
        record["matched_subtopic"] = str(matched.get("matched_subtopic", "") or "")
        record["evidence"] = str(matched.get("evidence", "") or "")

        if motivation_exempt(question, data):
            record.update(
                {
                    "verdict": VERDICT_KEEP,
                    "stage": STAGE_EXEMPT,
                    "reason": "通用动机/经历题豁免（无专业术语、无来源引用）",
                    "binding_status": "exempt",
                }
            )
            audited.append((question, record))
            continue

        hits = out_of_scope_hits(blob, data)
        d1 = (
            matched.get("verdict") != "keep"
            and not evidence.get("source_refs")
            and not is_motivation_question(question)
        )
        d2 = bool(hits)
        record["votes"] = {"D1": d1, "D2": d2, "D3": False}
        record["vote_evidence"] = {
            "D1": _d1_evidence(question, matched) if d1 else "",
            "D2": "、".join(hits),
            "D3": "",
        }

        if mode == GUARD_MODE_WEAK_WHITELIST:
            # 弱白名单：只标记不丢弃（与 R-3.2.1-5 一致），也不消耗仲裁预算
            if matched.get("verdict") == "keep":
                record.update(
                    {
                        "verdict": VERDICT_KEEP,
                        "stage": STAGE_BINDING
                        if matched.get("stage") == "binding"
                        else STAGE_WHITELIST_HIT,
                        "reason": f"命中白名单（{matched.get('stage')}）；弱白名单画像只标记不丢弃",
                    }
                )
            else:
                record.update(
                    {
                        "verdict": VERDICT_KEPT_UNVERIFIED,
                        "stage": STAGE_STRONG_SIGNAL if d2 else STAGE_UNVERIFIED,
                        "reason": "弱白名单画像：只标记不丢弃"
                        + ("；命中跨域强信号：" + "、".join(hits) if d2 else ""),
                    }
                )
            audited.append((question, record))
            continue

        if d1 and d2:
            # 两票已足，短路省下仲裁预算
            record.update(
                {
                    "verdict": VERDICT_DROP,
                    "stage": STAGE_STRONG_SIGNAL,
                    "reason": "无归属且无来源，同时命中跨域强信号（D1+D2）",
                    "evidence": "、".join(hits),
                }
            )
            audited.append((question, record))
            continue

        if not d1 and not d2:
            # 无法凑够两票：保留（可能有仲裁价值，但改变不了结局，省预算）
            if matched.get("verdict") == "keep":
                record.update(
                    {
                        "verdict": VERDICT_KEEP,
                        "stage": STAGE_BINDING
                        if matched.get("stage") == "binding"
                        else STAGE_WHITELIST_HIT,
                        "reason": f"命中白名单（{matched.get('stage')}）",
                    }
                )
            else:
                record.update(
                    {
                        "verdict": VERDICT_KEPT_UNVERIFIED,
                        "stage": STAGE_UNVERIFIED,
                        "reason": "未命中白名单，但也无跨域证据，按默认保留处理",
                    }
                )
            audited.append((question, record))
            continue

        # 需要第三票：白名单命中但命中强信号（冲突），或已有 1 票
        pending.append((question, record))
        audited.append((question, record))

    active_arbiter: BatchArbiter | None = None
    if pending:
        active_arbiter = arbiter if arbiter is not None else BatchArbiter()
        judgements = active_arbiter.judge([question for question, _ in pending], data)
        degraded_sources = {"budget_exhausted", "degraded", "missing"}
        sources: set[str] = set()
        for question, record in pending:
            judgement = judgements.get(_question_id(question, record["question_id"]), {})
            sources.add(str(judgement.get("source", "")))
            record["raw"] = dict(judgement)
            record["votes"]["D3"] = judgement.get("verdict") == ARBITER_OUT_OF_SCOPE
            record["vote_evidence"]["D3"] = str(judgement.get("evidence", "") or "")
            if sum(1 for value in record["votes"].values() if value) >= DROP_VOTES_REQUIRED:
                record.update(
                    {
                        "verdict": VERDICT_DROP,
                        "stage": STAGE_ARBITER,
                        "reason": "仲裁确认跨域："
                        + (str(judgement.get("reason", "")) or "证据可校验"),
                        "evidence": str(judgement.get("evidence", "") or record.get("evidence", "")),
                    }
                )
            else:
                bound = record.get("binding_status") in {"matched", "normalized", "repaired"}
                record.update(
                    {
                        "verdict": VERDICT_KEEP if bound else VERDICT_KEPT_UNVERIFIED,
                        "stage": STAGE_ARBITER,
                        "reason": str(judgement.get("reason", ""))
                        or "仲裁未确认跨域，按默认保留处理",
                    }
                )
        if sources & degraded_sources:
            status = GUARD_STATUS_DEGRADED

    kept: list[dict] = []
    dropped: list[dict] = []
    for question, record in audited:
        if record["verdict"] == VERDICT_DROP:
            dropped.append(question)
        else:
            kept.append(question)
    for index, question in enumerate(kept, 1):
        question["id"] = index

    summary = summarize_audit([record for _, record in audited], dropped)
    quality = profile_quality(data)
    summary.update(
        {
            "guard_audit_status": status,
            "guard_mode": mode,
            "weak_focus_count": int(quality.get("weak_focus_count", 0)),
            # 降级成因与"规范化岗位 slug 是否缺失"（后者只是画像诊断，不是守卫降级）
            "guard_degrade_reason": str(quality.get("degrade_reason", "") or ""),
            "guard_domain_slug_missing": bool(quality.get("domain_slug_missing", False)),
        }
    )
    total = len(kept) + len(dropped)
    return {
        "kept": kept,
        "dropped": dropped,
        "audit": [record for _, record in audited],
        "status": status,
        "arbiter_calls": active_arbiter.calls if active_arbiter is not None else 0,
        "drop_rate": (len(dropped) / total) if total else 0.0,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# 追问专用（不调用仲裁）
# --------------------------------------------------------------------------


def audit_follow_up(text: str, profile: dict | None) -> dict:
    """追问审计：只走 D2 强信号，**不调用仲裁**（零 LLM 调用）。

    3.2.3 在 ask.py 里的用法：

        result = audit_follow_up(follow_up_text, role_profile)
        if result["verdict"] == "drop":
            # 与模块 5 的 R5-2 一致：不重生成整卷（追问阶段已无"整卷"概念），
            # 直接把 follow_up_count 顶到 max_follow_ups 并返回收束语，
            # 让路由在下一轮进入 next_question。
        else:
            直接采用该追问。

    弱白名单与兜底画像不做硬丢弃（与 R-3.2.1-5 一致）。
    """
    data = profile or {}
    mode = str(data.get("guard_mode", "") or GUARD_MODE_LEGACY)
    if mode == GUARD_MODE_WEAK_WHITELIST:
        return {"verdict": VERDICT_KEEP, "hits": [], "reason": "弱白名单画像：只标记不丢弃"}
    if mode == GUARD_MODE_LEGACY and not guard_is_active(data):
        return {"verdict": VERDICT_KEEP, "hits": [], "reason": "兜底画像：不做硬丢弃"}
    hits = out_of_scope_hits(str(text or ""), data)
    if hits:
        return {
            "verdict": VERDICT_DROP,
            "hits": hits,
            "reason": "命中跨域强信号：" + "、".join(hits),
        }
    return {"verdict": VERDICT_KEEP, "hits": [], "reason": "未命中跨域强信号"}


# --------------------------------------------------------------------------
# 汇总（字段名与 3.2.4 的报告字段对齐）
# --------------------------------------------------------------------------


def summarize_audit(audit: list[AuditRecord], dropped: list[dict]) -> dict:
    uncertain = sum(1 for record in audit if record.get("verdict") == VERDICT_KEPT_UNVERIFIED)
    unique_samples: list[str] = []
    seen: set[str] = set()
    for question in dropped:
        text = str(question.get("content", "") or "")
        key = _normalize_text(text)
        if key and key not in seen:
            seen.add(key)
            unique_samples.append(text)
    return {
        "guard_uncertain_count": uncertain,
        "guard_dropped_count": len(unique_samples),
        "guard_dropped_samples": unique_samples[:5],
        "evidence_mismatch_count": sum(
            1 for record in audit if record.get("evidence_status") == "mismatch"
        ),
    }
