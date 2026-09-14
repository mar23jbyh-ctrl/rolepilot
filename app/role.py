"""岗位画像的校验、兜底与守卫。

设计原则：岗位职能域由 LLM 依据 JD 与简历判断，代码不预设
"AI 岗 / 技术岗 / 非技术岗"这类分类，也不做关键词分类，只负责三件事：

- 兜底：LLM 失败或置信度过低时降级为通用岗
- 守卫：把跨职能域的话题补进禁止清单，防止串岗
- 校验：修正 LLM 输出的字段类型与取值范围

历史字段（role_family / focus_areas / forbidden_drills / theory_depth）
只做兼容读取，不再作为业务概念使用。
"""

from __future__ import annotations

import re
from typing import Any

# LLM 画像低于该置信度时，整条链路降级为通用岗。
MIN_CONFIDENCE = 0.7

DEFAULT_FOCUS = ("岗位真实业务场景", "简历经历深挖", "求职动机")

# ---- 守卫模式（3.2.0 起）----
# whitelist      ：LLM 画像 + 每条 focus 都有 3-8 条可判定 subtopics → 白名单为主，不追加写死词表
# weak_whitelist ：LLM 画像但至少一条 focus 的 subtopics 不合格 → 只做提示词约束，不硬丢弃
# legacy         ：旧格式画像或 LLM 兜底 → 沿用 4 组兜底词表（模块 5 的行为）
GUARD_MODE_WHITELIST = "whitelist"
GUARD_MODE_WEAK_WHITELIST = "weak_whitelist"
GUARD_MODE_LEGACY = "legacy"

# ---- 白名单质量参数 ----
FOCUS_MIN_SUBTOPICS = 3
FOCUS_MAX_SUBTOPICS = 8
# 子项过短（不足 3 字）视为不可判定：例如"法律""合同"无法判断一道题是否属于它。
SUBTOPIC_MIN_CHARS = 3
# out_of_scope_topics 必须用复合词：压缩空白与标点后至少 3 个字符（"编程""模型"不合格）。
COMPOSITE_TOPIC_MIN_CHARS = 3

# domain_slug：英文小写 + 连字符，作为规范化岗位标识。三条约束：
#   ① 正则 ^[a-z0-9]+(-[a-z0-9]+)*$（只允许小写字母、数字与连字符）
#   ② 长度 3-30 字符（避免 LLM 生成超长 slug）
#   ③ 与 domain 一一对应（同一份画像内不可能出现两个 domain 共用一个 slug；
#      合并多份画像时由调用方保证不冲突）
# 首尾空白与大小写会被归一化，其余不符合约束的一律视为 invalid。
DOMAIN_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DOMAIN_SLUG_MIN_CHARS = 3
DOMAIN_SLUG_MAX_CHARS = 30

# ---- 文本匹配参数（3.2.5 会用标注集校准）----
# 相似度层：subtopic 的字符 bigram 至少有该比例出现在题面里才算命中。
# 取值依据：同义改写（如"合同主体资格审查"vs"合同主体的资格审查流程"）的包含度接近 1.0，
# 而只有个别字重合的无关题接近 0；0.5 是"宁可保留"取向下的保守值。
SIMILARITY_THRESHOLD = 0.5
# 短词共现：中文 2 字片段单独出现不算命中，必须与同一条 subtopic 的另外至少
# COOCCUR_MIN_SEGMENTS-1 个不同片段同时出现，避免"模型/检索/对话/开发"这类宽泛词误伤。
COOCCUR_MIN_SEGMENTS = 2

# ---- 来源引用与动机题 ----
# 自动补来源：题面与来源清单条目的 bigram 包含度达到该值才允许自动绑定。
AUTO_REPAIR_THRESHOLD = 0.3
# 引用内容对不上的判定线：低于该相似度记为 mismatch（只标记，不丢弃）。
EVIDENCE_MISMATCH_THRESHOLD = 0.15
# 通用动机/经历题额度（零风险题：不含专业术语，结构上不可能跨域）。
MOTIVATION_QUOTA = 2
# True 时才允许丢弃超额或不合格的动机题；默认 False（只标记不删除）。
MOTIVATION_STRICT = False

# 守卫规则：(规则名, 考察重心关键词, 重心里没命中时要补进禁止清单的话题)
# 3.2.0 起仅用于 guard_mode == "legacy" 的降级场景。
_LEGACY_TOPIC_GUARDS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "ai",
        (
            "RAG", "大模型", "LLM", "LangChain", "Agent", "提示词",
            "向量库", "向量检索", "embedding", "微调", "Transformer",
        ),
        (
            "RAG", "大模型", "LLM", "Agent", "提示词", "向量库",
            # R6-1 加宽：覆盖模型/训练/推理/检索/对话/问答这类自然表述
            "模型", "训练", "推理", "检索", "对话", "问答",
        ),
    ),
    (
        "code",
        ("编程", "代码", "开发", "算法", "系统设计", "架构"),
        (
            "编程", "算法", "系统架构",
            # R6-1 加宽：覆盖代码/开发/语言/接口/数据库这类自然表述
            "代码", "开发", "Python", "Java", "接口", "函数", "SQL", "数据库",
        ),
    ),
    (
        "medical",
        ("护理", "临床", "诊断", "病案", "医患"),
        (
            "护理实务", "临床诊断",
            # R6-1 加宽：覆盖护理/临床/医院/医生/患者这类自然表述
            "护理", "临床", "诊断", "病案", "医患", "医院", "医生", "患者",
        ),
    ),
    (
        "legal",
        ("法律实务", "诉讼流程", "合同审查", "合规"),
        ("法律实务", "诉讼流程"),
    ),
)

# 旧 role_profile 字段 → 新画像字段的兼容映射（仅用于历史会话与既有测试）
_LEGACY_FAMILY_DOMAIN = {
    "research": "技术-算法研究",
    "application": "技术-工程应用",
}

_LEGACY_INDUSTRY_LABEL = {
    "tech": "技术",
    "legal": "法律",
    "finance": "财务",
    "operation": "运营",
    "education": "教育",
    "sales": "销售",
    "hr": "人力资源",
}

_UNKNOWN_INDUSTRIES = {"", "unknown", "未知", "general", "n/a"}

_ASCII_TERM = re.compile(r"[A-Za-z0-9 .+#/_-]+")


def _contains(term: str, text: str) -> bool:
    """ASCII 话题词按词边界匹配（避免 email 命中 ai），中文话题词按包含匹配。"""
    if not term or not text:
        return False
    if _ASCII_TERM.fullmatch(term):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        return re.search(pattern, text, re.IGNORECASE) is not None
    return term.lower() in text.lower()


def _any_topic(terms: tuple[str, ...], items: list[str]) -> bool:
    return any(_contains(term, item) for term in terms for item in items)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


_PUNCTUATION = re.compile(r"[\s，。、；：,.;:!！?？\-—_()（）\[\]【】「」{}<>\"'“”‘’/\\|]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RUN = re.compile(r"[A-Za-z][A-Za-z0-9+#./_-]*")


def _compact(text: Any) -> str:
    """去掉空白与常见标点，便于做子串与 bigram 计算。"""
    return _PUNCTUATION.sub("", str(text or ""))


def _char_bigrams(text: Any) -> set[str]:
    value = _compact(text)
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _bigram_containment(needle: Any, haystack: Any) -> float:
    """needle 的字符 bigram 有多大比例出现在 haystack 里（0-1）。"""
    left = _char_bigrams(needle)
    if not left:
        return 0.0
    right = _char_bigrams(haystack)
    if not right:
        return 0.0
    return len(left & right) / len(left)


def _subtopic_segments(subtopic: str) -> tuple[list[str], list[str], list[str]]:
    """把 subtopic 拆成 (ASCII 词, 中文长片段>=3字, 中文短片段==2字)。"""
    ascii_terms = [item for item in _ASCII_RUN.findall(str(subtopic or "")) if len(item) >= 2]
    long_runs: list[str] = []
    short_runs: list[str] = []
    for run in _CJK_RUN.findall(str(subtopic or "")):
        if len(run) >= 3:
            long_runs.append(run)
        elif len(run) == 2:
            short_runs.append(run)
    return ascii_terms, long_runs, short_runs


def _is_judgeable_subtopic(subtopic: str, focus_name: str = "") -> bool:
    """subtopic 是否"能判定一道题是否属于它"（结构判断，不使用任何词表）。"""
    core = _compact(subtopic)
    if len(core) < SUBTOPIC_MIN_CHARS:
        return False
    name_core = _compact(focus_name)
    if not name_core:
        return True
    if core == name_core:
        return False
    if len(core) <= len(name_core) + 1 and (core in name_core or name_core in core):
        return False
    return True


def _is_composite_topic(term: str) -> bool:
    """out_of_scope_topics 合规判定：必须是复合词（≥3 字符），拒绝"编程""模型"这类通用短词。"""
    return len(_compact(term)) >= COMPOSITE_TOPIC_MIN_CHARS


def _normalize_domain_slug(value: Any) -> tuple[str, str]:
    """校验 domain_slug，返回 (slug, source)；source ∈ {llm, missing, invalid}。

    约束见 DOMAIN_SLUG_PATTERN 上方的三条说明：正则、长度 3-30、与 domain 一一对应。
    """
    text = str(value or "").strip().lower()
    if not text:
        return "", "missing"
    if not DOMAIN_SLUG_MIN_CHARS <= len(text) <= DOMAIN_SLUG_MAX_CHARS:
        return "", "invalid"
    if not DOMAIN_SLUG_PATTERN.fullmatch(text):
        return "", "invalid"
    return text, "llm"


def _as_focus_items(value: Any) -> list[dict]:
    """把 assessment_focus 归一化成对象数组（兼容旧的字符串数组）。"""
    raw_items = value if isinstance(value, (list, tuple)) else ([value] if value else [])
    items: list[dict] = []
    for index, raw in enumerate(raw_items, 1):
        if isinstance(raw, dict):
            name = str(raw.get("name", "") or raw.get("focus", "") or "").strip()
            if not name:
                continue
            items.append(
                {
                    "key": str(raw.get("key", "") or f"f{index}").strip() or f"f{index}",
                    "name": name,
                    "subtopics": _as_str_list(raw.get("subtopics")),
                    "why_relevant": str(raw.get("why_relevant", "") or "").strip(),
                    "source_jd_ref": str(raw.get("source_jd_ref", "") or "").strip(),
                    "focus_source": "llm",
                }
            )
            continue
        name = str(raw).strip()
        if not name:
            continue
        items.append(
            {
                "key": f"f{index}",
                "name": name,
                "subtopics": [],
                "why_relevant": "",
                "source_jd_ref": "",
                "focus_source": "legacy",
            }
        )
    if items:
        return items
    return [
        {
            "key": f"f{index}",
            "name": name,
            "subtopics": [],
            "why_relevant": "",
            "source_jd_ref": "",
            "focus_source": "legacy",
        }
        for index, name in enumerate(DEFAULT_FOCUS, 1)
    ]


def _focus_terms(items: list[dict]) -> list[str]:
    """把 focus 对象数组摊平成字符串列表（供 legacy 4 组守卫判断方向）。"""
    terms: list[str] = []
    for item in items or []:
        name = str(item.get("name", "") or "").strip()
        if name:
            terms.append(name)
        terms.extend(str(sub) for sub in (item.get("subtopics") or []) if str(sub).strip())
    return terms


def _focus_items(data: dict) -> list[dict]:
    """取画像的 focus 内部视图（始终是对象数组，兼容外部两种形态）。"""
    raw = data.get("assessment_focus") or data.get("focus_areas")
    return _as_focus_items(raw)


def _focus_items_qualified(items: list[dict]) -> bool:
    """白名单是否合格：全部 focus 都有 3-8 条可判定 subtopics，且都来自 LLM。"""
    if not items:
        return False
    for item in items:
        if item.get("focus_source") == "legacy":
            return False
        subtopics = item.get("subtopics") or []
        if not FOCUS_MIN_SUBTOPICS <= len(subtopics) <= FOCUS_MAX_SUBTOPICS:
            return False
        if any(not _is_judgeable_subtopic(sub, item.get("name", "")) for sub in subtopics):
            return False
    return True


def _resolve_guard_mode(data: dict) -> str:
    """判定守卫模式（三态）。"""
    if str(data.get("profile_source", "") or "") != "llm":
        return GUARD_MODE_LEGACY
    items = _focus_items(data)
    if not items or any(item.get("focus_source") == "legacy" for item in items):
        return GUARD_MODE_LEGACY
    # 白名单模式要求：focus 全部合格 **且** out_of_scope 非空（跨域拦截依赖它）。
    # out_of_scope 缺失（必填但 LLM 没给）→ 降级 weak_whitelist：只提示、不硬丢弃，也不退回 legacy。
    if _focus_items_qualified(items) and _as_str_list(data.get("out_of_scope_topics")):
        return GUARD_MODE_WHITELIST
    return GUARD_MODE_WEAK_WHITELIST


def _fallback_profile() -> dict:
    """LLM 失败时的通用兜底画像：不预设任何考察方向。"""
    return {
        "job_title": "未知岗位",
        "domain": "general",
        "industry": "unknown",
        "core_responsibilities": [],
        "key_skills": [],
        # 兜底画像保持旧的字符串形态（与模块 2 完全一致）→ guard_mode 为 legacy。
        "assessment_focus": list(DEFAULT_FOCUS),
        "out_of_scope_topics": [],
        "forbidden_topics": [],
        "confidence": 0.0,
        "profile_source": "fallback",
    }


def _legacy_domain(data: dict) -> str:
    """把历史 role_profile 的 role_family / industry 折算成职能域。"""
    family = str(data.get("role_family", "") or "")
    if family in _LEGACY_FAMILY_DOMAIN:
        return _LEGACY_FAMILY_DOMAIN[family]
    return _LEGACY_INDUSTRY_LABEL.get(str(data.get("industry", "") or ""), "general")


def _validate_profile(profile: dict | None) -> dict:
    """校验画像字段，缺失的补默认值（同时兼容旧字段）。"""
    data: dict = dict(profile or {})
    if not str(data.get("job_title", "") or "").strip():
        data["job_title"] = "未知岗位"

    domain = str(data.get("domain", "") or "").strip()
    data["domain"] = domain or _legacy_domain(data)

    industry = str(data.get("industry", "") or "").strip()
    data["industry"] = industry or "unknown"

    # domain_slug：作为规范化岗位标识；缺失或非法时置空，并保持首次判定出的 source。
    slug, slug_source = _normalize_domain_slug(data.get("domain_slug"))
    if not slug:
        recorded = str(data.get("domain_slug_source", "") or "")
        if recorded in {"missing", "invalid"}:
            slug_source = recorded
    data["domain_slug"] = slug
    data["domain_slug_source"] = slug_source

    # focus 有两种形态：
    # - 新格式（LLM 返回对象数组）→ 校验后写回对象数组
    # - 旧格式（字符串数组 / focus_areas）→ 保持字符串形态，历史会话与既有读取方零影响
    focus_items = _focus_items(data)
    legacy_shape = all(item.get("focus_source") == "legacy" for item in focus_items)
    data["assessment_focus"] = (
        [str(item.get("name", "")) for item in focus_items]
        if legacy_shape
        else focus_items
    )
    data["forbidden_topics"] = _as_str_list(data.get("forbidden_topics"))
    # out_of_scope 的来源要区分开：
    #   llm    = LLM 直接给出的复合词（新格式画像）
    #   mirror = 由 forbidden_topics 兼容镜像而来（旧格式/降级画像）
    #   none   = 两边都为空
    out_of_scope = _as_str_list(data.get("out_of_scope_topics"))
    if out_of_scope:
        data["out_of_scope_source"] = str(data.get("out_of_scope_source") or "llm")
    else:
        mirrored = list(data["forbidden_topics"])
        data["out_of_scope_topics"] = mirrored
        # out_of_scope 为必填：缺失时记 "missing"，并由 _resolve_guard_mode 降级为弱白名单。
        data["out_of_scope_source"] = "mirror" if mirrored else "missing"
    data["out_of_scope_topics"] = _as_str_list(data.get("out_of_scope_topics"))
    data["core_responsibilities"] = _as_str_list(data.get("core_responsibilities"))
    data["key_skills"] = _as_str_list(data.get("key_skills"))

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    data["confidence"] = min(1.0, max(0.0, confidence))

    data["industry_known"] = data["industry"].strip().lower() not in _UNKNOWN_INDUSTRIES
    data["needs_web_research"] = bool(
        data.get("needs_web_research", False)
    ) or not data["industry_known"]
    if not str(data.get("profile_source", "") or "").strip():
        data["profile_source"] = "llm" if profile else "fallback"
    data["guard_mode"] = _resolve_guard_mode(data)
    return data


def _legacy_forbidden(profile: dict) -> list[str]:
    """降级场景的 4 组兜底守卫（guard_mode == "legacy" 时使用）。

    3.2.0 起这是"降级路径"：只有旧格式画像或 LLM 兜底画像才走这里。
    """
    focus = _focus_terms(_focus_items(profile)) or list(DEFAULT_FOCUS)
    merged = _as_str_list(profile.get("forbidden_topics"))
    # R1：技术职能域本身就要考编程/算法/工程，不再把这类话题列为禁止项；
    # 其他守则（AI / 医疗 / 法律）照常生效，技术岗若不做 AI 仍会禁止 AI 话题。
    technical = str(profile.get("domain", "") or "").startswith("技术")
    for name, focus_terms, forbidden_terms in _LEGACY_TOPIC_GUARDS:
        if name == "code" and technical:
            continue
        if _any_topic(focus_terms, focus):
            # 该方向确实是这个岗位的考察重心，不做禁止。
            continue
        for term in forbidden_terms:
            # 去重判断的是"检出能力"而不是字面包含：
            # 只有当已存在的条目更宽或等价（即已有条目本身就出现在新词里）时才算重复。
            # 反过来必须补进来——例如已有「护理实务」时，仍要补更宽的「护理」，
            # 否则题干只写「护理」的自然表述会漏检。
            if any(_contains(item, term) for item in merged):
                continue
            merged.append(term)
    return merged


def _collect_out_of_scope(profile: dict) -> list[str]:
    """产出用于"正面证据"（D2 强信号）的禁止话题清单。

    - whitelist / weak_whitelist：使用 LLM 生成的 out_of_scope_topics（复合词），
      不追加任何写死词表；
    - legacy：沿用 4 组兜底词表。
    """
    mode = str(profile.get("guard_mode", "") or _resolve_guard_mode(profile))
    if mode == GUARD_MODE_LEGACY:
        return _legacy_forbidden(profile)
    merged = _as_str_list(profile.get("out_of_scope_topics")) or _as_str_list(
        profile.get("forbidden_topics")
    )
    return merged


def _merge_forbidden(profile: dict) -> list[str]:
    """（保留的旧名字）等价于 _collect_out_of_scope，供历史调用点使用。"""
    return _collect_out_of_scope(profile)


def guard_forbidden_topics(profile: dict | None) -> list[str]:
    """对外提供“校验 + 守卫后”的禁止话题清单（提问等节点复用同一份）。"""
    return _collect_out_of_scope(_validate_profile(profile))


def focus_keys(profile: dict | None) -> list[str]:
    """白名单 key 列表（供 3.2.3 校验题目声明的 focus_key）。"""
    return [
        str(item.get("key", "")).strip()
        for item in _focus_items(_validate_profile(profile))
        if str(item.get("key", "")).strip()
    ]


def render_focus_whitelist(profile: dict | None) -> str:
    """渲染 [f1] 名称 —— 子项：a / b / c 形式的白名单文本（供 PLAN_PROMPT 使用）。"""
    lines: list[str] = []
    for item in _focus_items(_validate_profile(profile)):
        subtopics = item.get("subtopics") or []
        suffix = (
            " —— 子项：" + " / ".join(subtopics)
            if subtopics
            else "（无子项：该维度不可判定，只作提示）"
        )
        lines.append(f"[{item.get('key')}] {item.get('name')}{suffix}")
    return "\n".join(lines) or "（无白名单）"


def _match_focus_by_text(text: str, items: list[dict]) -> dict | None:
    """白名单文本匹配的第 2-5 层（不含生成期绑定）。"""
    content = str(text or "")
    if not content:
        return None
    compact = _compact(content)
    for item in items:
        focus_key = str(item.get("key", "") or "")
        for subtopic in item.get("subtopics") or []:
            ascii_terms, long_runs, short_runs = _subtopic_segments(subtopic)
            for term in ascii_terms:
                if _contains(term, content):
                    return {
                        "verdict": "keep",
                        "stage": "ascii",
                        "focus_key": focus_key,
                        "matched_subtopic": subtopic,
                        "evidence": term,
                    }
            for run in long_runs:
                if run in compact:
                    return {
                        "verdict": "keep",
                        "stage": "substring",
                        "focus_key": focus_key,
                        "matched_subtopic": subtopic,
                        "evidence": run,
                    }
            if short_runs:
                matched = [run for run in short_runs if run in compact]
                if len(matched) >= COOCCUR_MIN_SEGMENTS:
                    return {
                        "verdict": "keep",
                        "stage": "cooccur",
                        "focus_key": focus_key,
                        "matched_subtopic": subtopic,
                        "evidence": "、".join(matched),
                    }
            # 相似度层只在 subtopic 足够长（>=3 个 bigram）时启用，避免短词误命中。
            if len(_char_bigrams(subtopic)) >= 3 and _bigram_containment(
                subtopic, content
            ) >= SIMILARITY_THRESHOLD:
                return {
                    "verdict": "keep",
                    "stage": "similarity",
                    "focus_key": focus_key,
                    "matched_subtopic": subtopic,
                    "evidence": subtopic,
                }
    return None


def match_focus(
    text: str,
    profile: dict | None,
    declared_focus_key: str = "",
) -> dict:
    """判断一段文本（通常是题面）是否落在岗位白名单内。

    分层顺序（返回第一个命中，均携带证据）：
      1. binding：declared_focus_key 命中白名单 key。
         3.2.3 会把出题阶段 LLM 声明的 focus_key 传进来：
         完全一致 → binding_status="matched"；仅大小写/空格差异 → "normalized"；
         未知 key 但题面能反查到 focus → "repaired"（并用反查结果改写 focus_key）；
         未知 key 且反查不到 → "unverified"（调用方按"默认保留"处理）。
      2. ascii：subtopic 中的英文词按词边界命中题面
      3. substring：subtopic 的中文长片段（≥3 字）作为子串出现在题面里
      4. cooccur：中文 2 字片段必须与同一条 subtopic 的另外至少
         COOCCUR_MIN_SEGMENTS-1 个不同片段同时出现
      5. similarity：subtopic 的字符 bigram 包含度 ≥ SIMILARITY_THRESHOLD
    都不命中 → verdict="unverified"（不构成丢弃依据）。
    """
    data = _validate_profile(profile)
    items = _focus_items(data)
    raw_declared = str(declared_focus_key or "").strip()
    base = {
        "verdict": "unverified",
        "stage": "none",
        "focus_key": "",
        "matched_subtopic": "",
        "evidence": "",
        "binding_status": "n/a",
        "declared_focus_key": raw_declared,
        "guard_mode": data.get("guard_mode", GUARD_MODE_LEGACY),
    }
    if not items:
        return base

    by_key = {str(item.get("key", "")).lower(): item for item in items}
    if raw_declared:
        item = by_key.get(raw_declared.lower())
        if item is not None:
            subtopics = item.get("subtopics") or []
            status = "matched" if raw_declared == str(item.get("key")) else "normalized"
            return {
                **base,
                "verdict": "keep",
                "stage": "binding",
                "focus_key": str(item.get("key")),
                "matched_subtopic": subtopics[0] if subtopics else "",
                "evidence": raw_declared,
                "binding_status": status,
            }

    hit = _match_focus_by_text(text, items)
    if raw_declared:
        if hit:
            return {**hit, "binding_status": "repaired", "declared_focus_key": raw_declared}
        return {**base, "binding_status": "unverified"}
    if hit:
        return {**base, **hit, "binding_status": "n/a"}
    return base


def out_of_scope_hits(text: str, profile: dict | None) -> list[str]:
    """D2 强信号：文本命中的"禁止话题"。

    whitelist / weak_whitelist 模式只匹配 LLM 生成的复合词（不再追加写死词表）；
    legacy 模式沿用 4 组词表。
    """
    data = _validate_profile(profile)
    return text_hits_topics(text, _collect_out_of_scope(data))


def evidence_status(
    question: dict,
    jd_items: list[str] | None = None,
    resume_projects: list[str] | None = None,
    reference_ids: list[str] | None = None,
    reference_texts: dict[str, str] | None = None,
) -> dict:
    """校验题目的来源引用（纯函数：只修复与标记，绝不丢弃题目）。

    - jd_ref / project_ref：必须是清单下标（0..n-1）；非数字或越界 → 置空并记入 evidence_invalid
    - research_ref：必须命中 reference_ids；否则置空并记入 evidence_invalid
    - 三来源全空时：用 bigram 反查自动补一个来源（阈值 AUTO_REPAIR_THRESHOLD）
    - 已填来源：用 bigram 包含度判断内容是否对得上，低于 EVIDENCE_MISMATCH_THRESHOLD 记 mismatch

    返回：
      {jd_ref, project_ref, research_ref, evidence_status, evidence_invalid,
       evidence_similarity, source_refs}
    evidence_status 取值：ok / repaired / missing / mismatch（evidence_invalid 独立上报）
    """
    jd_list = [str(item) for item in (jd_items or [])]
    project_list = [str(item) for item in (resume_projects or [])]
    ref_set = {str(item) for item in (reference_ids or [])}
    content = str((question or {}).get("content", "") or "")

    invalid: list[str] = []
    similarity: dict[str, float] = {}

    def _check_index(raw: Any, items: list[str], field: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        if not text.isdigit():
            invalid.append(field)
            return ""
        index = int(text)
        if index < 0 or index >= len(items):
            invalid.append(field)
            return ""
        similarity[field] = round(_bigram_containment(items[index], content), 3)
        return str(index)

    jd_ref = _check_index((question or {}).get("jd_ref"), jd_list, "jd_ref")
    project_ref = _check_index((question or {}).get("project_ref"), project_list, "project_ref")
    research_ref = str((question or {}).get("research_ref", "") or "").strip()
    if research_ref and research_ref not in ref_set:
        invalid.append("research_ref")
        research_ref = ""
    elif research_ref and reference_texts:
        # M24：调研片段此前只校验 ID 是否存在；现在补上**内容相关性**校验，
        # 题面与片段原文对不上时记为 mismatch（只标记，不丢弃题目）。
        text = str(reference_texts.get(research_ref, "") or "")
        if text:
            similarity["research_ref"] = round(_bigram_containment(text, content), 3)

    status = "ok"
    if not (jd_ref or project_ref or research_ref):
        best: tuple[str, str, float] | None = None
        for field, items in (("jd_ref", jd_list), ("project_ref", project_list)):
            for index, item in enumerate(items):
                score = _bigram_containment(item, content)
                if score >= AUTO_REPAIR_THRESHOLD and (best is None or score > best[2]):
                    best = (field, str(index), score)
        if best is not None:
            field, index, score = best
            if field == "jd_ref":
                jd_ref = index
            else:
                project_ref = index
            similarity[field] = round(score, 3)
            status = "repaired"
        else:
            status = "missing"
    elif similarity and min(similarity.values()) < EVIDENCE_MISMATCH_THRESHOLD:
        status = "mismatch"

    source_refs = [ref for ref in (jd_ref, project_ref, research_ref) if ref]
    return {
        "jd_ref": jd_ref,
        "project_ref": project_ref,
        "research_ref": research_ref,
        "evidence_status": status,
        "evidence_invalid": invalid,
        "evidence_similarity": similarity,
        "source_refs": source_refs,
    }


def is_motivation_question(question: dict | None) -> bool:
    """通用动机/经历题：无 focus 归属、声明为 behavioral、且没有任何来源引用。"""
    data = question or {}
    if str(data.get("focus_key", "") or "").strip():
        return False
    if str(data.get("question_type", "") or "") != "behavioral":
        return False
    return not any(
        str(data.get(field, "") or "").strip()
        for field in ("jd_ref", "project_ref", "research_ref")
    )


def motivation_exempt(question: dict | None, profile: dict | None = None) -> bool:
    """动机题豁免判定：通用动机题且不含本岗位的禁止话题（供 3.2.2 审计调用）。"""
    if not is_motivation_question(question):
        return False
    if profile is None:
        return True
    text = " ".join(
        [
            str((question or {}).get("content", "") or ""),
            " ".join(str(item) for item in ((question or {}).get("skills") or [])),
        ]
    )
    return not out_of_scope_hits(text, profile)


def profile_quality(profile: dict | None) -> dict:
    """画像质量指标（供 3.2.3/3.2.5 诊断：白名单是否可用、哪些 subtopic 不合格）。"""
    data = _validate_profile(profile)
    items = _focus_items(data)
    invalid: dict[str, list[str]] = {}
    weak_keys: set[str] = set()
    for item in items:
        key = str(item.get("key", "") or "")
        subtopics = [str(sub) for sub in (item.get("subtopics") or [])]
        bad = [sub for sub in subtopics if not _is_judgeable_subtopic(sub, item.get("name", ""))]
        if bad:
            invalid[key] = bad
            weak_keys.add(key)
        if item.get("focus_source") == "legacy" or not (
            FOCUS_MIN_SUBTOPICS <= len(subtopics) <= FOCUS_MAX_SUBTOPICS
        ):
            weak_keys.add(key)
    mode = str(data.get("guard_mode", GUARD_MODE_LEGACY))
    out_of_scope = _as_str_list(data.get("out_of_scope_topics"))
    slug = str(data.get("domain_slug", "") or "")
    # 降级成因（机器可读，多成因用 + 拼接）：
    #   legacy_profile / fallback_profile / invalid_subtopics / out_of_scope_missing
    reasons: list[str] = []
    if mode == GUARD_MODE_LEGACY:
        reasons.append(
            "fallback_profile"
            if str(data.get("profile_source", "")) == "fallback"
            else "legacy_profile"
        )
    else:
        if weak_keys:
            reasons.append("invalid_subtopics")
        if str(data.get("out_of_scope_source", "")) != "llm":
            reasons.append("out_of_scope_missing")
    return {
        "guard_mode": mode,
        "focus_count": len(items),
        "weak_focus_count": len(weak_keys),
        "invalid_subtopics": invalid,
        "out_of_scope_ok": str(data.get("out_of_scope_source", "")) == "llm",
        "out_of_scope_source": str(data.get("out_of_scope_source", "") or ""),
        "out_of_scope_count": len(out_of_scope),
        "invalid_out_of_scope": [
            term for term in out_of_scope if not _is_composite_topic(term)
        ],
        "domain_slug": slug,
        "domain_slug_missing": not slug,
        "domain_slug_source": str(data.get("domain_slug_source", "") or ""),
        "degrade_reason": "+".join(reasons),
        "focus_source_mix": sorted({str(item.get("focus_source", "")) for item in items}),
        "focus_key_available": mode != GUARD_MODE_LEGACY
        and all(str(item.get("key", "")).strip() for item in items),
    }


def text_hits_topics(text: str, topics: list[str]) -> list[str]:
    """返回文本命中的话题清单（ASCII 词按词边界，中文按包含匹配）。"""
    content = str(text or "")
    return [
        str(topic)
        for topic in (topics or [])
        if str(topic).strip() and _contains(str(topic), content)
    ]


def guard_is_active(profile: dict | None) -> bool:
    """R2：只有 LLM 画像（profile_source == "llm"）才启用硬丢弃守卫。

    兜底画像（LLM 失败/低置信度）的禁止清单是"全部跨域话题"，
    对它做硬丢弃会把题单删空，因此只保留提示词级约束。
    """
    return str((profile or {}).get("profile_source", "") or "") == "llm"


def build_role_profile(
    jd_text: str,
    resume_text: str,
    llm_profile: dict | None = None,
) -> dict:
    """把 LLM 画像做校验、兜底、补守卫后交给状态。"""
    if not isinstance(llm_profile, dict) or not llm_profile:
        profile = _fallback_profile()
    else:
        profile = _validate_profile(llm_profile)
        if float(profile.get("confidence", 0.0)) < MIN_CONFIDENCE:
            # 置信度不足时按通用岗处理，避免顺着错误方向深考。
            profile = _fallback_profile()
        else:
            profile["profile_source"] = "llm"

    # 守卫模式与禁止话题：whitelist/weak_whitelist 用 LLM 生成的话题，
    # legacy 才走 4 组兜底词表；forbidden_topics 继续作为兼容镜像输出。
    profile["guard_mode"] = _resolve_guard_mode(profile)
    profile["out_of_scope_topics"] = _as_str_list(profile.get("out_of_scope_topics"))
    if not profile["out_of_scope_topics"] and profile["guard_mode"] != GUARD_MODE_LEGACY:
        profile["out_of_scope_topics"] = _as_str_list(profile.get("forbidden_topics"))
    profile["forbidden_topics"] = _collect_out_of_scope(profile)
    profile["needs_web_research"] = bool(
        profile.get("needs_web_research", False)
    ) or not bool(profile.get("industry_known"))
    return profile


def role_policy_text(profile: dict | None, depth_level: str = "") -> str:
    """渲染岗位政策文本，注入到出题 / 提问 / 评分 / 终评各节点。"""
    data = _validate_profile(profile)
    forbidden = _as_str_list(data.get("forbidden_topics")) or _collect_out_of_scope(data)
    items = _focus_items(data)

    if str(data.get("guard_mode", "")) == GUARD_MODE_LEGACY:
        # legacy 画像：渲染结果与模块 5 逐字一致（回归保护）。
        focus_line = "考察重心：" + "、".join(
            str(item.get("name", "")) for item in items
        ) + "。"
    else:
        rendered: list[str] = []
        for item in items:
            subtopics = [str(sub) for sub in (item.get("subtopics") or [])]
            rendered.append(
                f"{item.get('name')}（{'、'.join(subtopics)}）"
                if subtopics
                else str(item.get("name", ""))
            )
        focus_line = "考察重心：" + "；".join(rendered) + "。"

    lines = [
        f"岗位：{data.get('job_title', '未知岗位')}。",
        f"职能域：{data.get('domain', 'general')}。",
        f"行业：{data.get('industry', 'unknown')}。",
        focus_line,
    ]
    if forbidden:
        lines.append("禁止话题：" + "、".join(forbidden) + "。")
    lines.append("禁止在题目、追问、评分、评估报告中出现以上禁止话题。")
    lines.append("若出现，视为出题失败，必须重新生成。")
    lines.append(
        "当不确定该岗位的业务术语或真实面试官提问风格时，"
        "必须调用 web_search 查证后再提问。"
    )

    if not data.get("industry_known", True):
        lines.append(
            "该岗位的行业归属不确定：不得凭印象编造行业要求或假面试题，必须先查证再提问。"
        )
    if data.get("resume_vague") or data.get("needs_web_research"):
        lines.append(
            "候选人简历技能/业务描述较模糊：优先用 web_search 补充该岗位高频真实面试题作为提问依据，"
            "避免脱离真实岗位提问。"
        )

    match_level = str(data.get("resume_jd_match_level", "") or "")
    if match_level == "HIGH":
        lines.append("匹配等级 HIGH：优先深挖简历中与岗位重合的项目/技能，验证真实经验。")
    elif match_level == "MEDIUM":
        lines.append(
            "匹配等级 MEDIUM：一半考察简历与岗位重合部分，一半考察岗位通用专业知识。"
        )
    elif match_level == "LOW":
        lines.append(
            "匹配等级 LOW：简历缺少本岗位经验；不要反复追问简历无关内容、不要编造简历项目；"
            "重点考察岗位基础专业知识、行业认知、求职动机与学习规划，"
            "可提出‘为何转行/如何补足能力’等动机问题。"
        )

    experience = str(data.get("experience_level", "") or "")
    if experience == "intern":
        lines.append(
            "候选人经验等级：在校/实习。以基础实务流程、工作习惯、基础专业知识为主，复杂难题选考。"
        )
    elif experience == "senior":
        lines.append(
            "候选人经验等级：资深。可启用高难度实务题与需要多任务权衡的复杂场景题。"
        )
    elif experience == "junior":
        lines.append("候选人经验等级：初级。以基础概念 + 常见业务场景为主，少量深挖。")

    lines.append(
        "考察覆盖要求：专业知识、业务场景、简历/项目经历、求职动机与软素质四类都要覆盖；"
        "同一知识点不得在两道以上题目中重复考察。"
    )
    if depth_level:
        allowed = {
            "concept": "本题只要求概念级理解",
            "application": "本题要求结合业务/项目的应用级理解",
            "deep": "本题可深入原理与推导",
        }
        lines.append("本题深度：" + allowed.get(depth_level, allowed["application"]))
    return " ".join(lines)
