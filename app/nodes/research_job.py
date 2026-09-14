from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from app.tools.executor import web_search_handler

# ---- M24：来源分级与时效 ----
# 高可信：政府/教育/官方文档/大型云厂商；中：主流技术社区；低：其余（含 UGC 聚合站）
HIGH_TRUST_SUFFIXES = (
    ".gov.cn", ".edu.cn", ".gov", ".edu",
    "docs.python.org", "learn.microsoft.com", "developer.mozilla.org",
    "cloud.google.com", "aws.amazon.com", "kubernetes.io", "w3.org",
)
MEDIUM_TRUST_SUFFIXES = (
    "github.com", "stackoverflow.com", "zhihu.com", "juejin.cn",
    "infoq.cn", "csdn.net", "segmentfault.com", "cnblogs.com",
)
TRUST_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4, "unknown": 0.5}

# 超过该天数的资料只保留、不参与排序（freshness 权重降到 0.2）
STALE_DAYS = 730
MAX_FRAGMENTS = 5


def _domain_tier(domain: str) -> str:
    text = str(domain or "").lower()
    if not text:
        return "unknown"
    def matches(suffix):
        hostname = suffix.lstrip(".")
        return text == hostname or text.endswith("." + hostname)

    if any(matches(suffix) for suffix in HIGH_TRUST_SUFFIXES):
        return "high"
    if any(matches(suffix) for suffix in MEDIUM_TRUST_SUFFIXES):
        return "medium"
    return "low"


def _freshness_weight(published_date: str) -> float:
    """M24：按发布日期给新鲜度权重；解析不出日期按中性 0.7 处理。"""

    text = str(published_date or "").strip()
    if not text:
        return 0.7
    published = None
    for fmt, length in (("%Y-%m-%dT%H:%M:%S", 19), ("%Y-%m-%d", 10), ("%Y/%m/%d", 10)):
        try:
            published = datetime.strptime(text[:length], fmt)
            break
        except ValueError:
            continue
    if published is None:
        return 0.7
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - published).days
    if age_days <= 365:
        return 1.0
    if age_days <= STALE_DAYS:
        return 0.6
    return 0.2


def _content_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", "", str(text or ""))[:160]
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _rank_and_dedup(results: list[dict]) -> list[dict]:
    """M24：去重（URL + 内容指纹）+ 按"信任度 × 新鲜度"排序。"""

    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    kept: list[dict] = []
    for item in results or []:
        url = str(item.get("url", "") or "")
        fingerprint = _content_fingerprint(item.get("content", ""))
        if (url and url in seen_urls) or fingerprint in seen_fingerprints:
            continue
        if url:
            seen_urls.add(url)
        seen_fingerprints.add(fingerprint)
        tier = _domain_tier(str(item.get("domain", "") or ""))
        freshness = _freshness_weight(str(item.get("published_date", "") or ""))
        enriched = dict(item)
        enriched["tier"] = tier
        enriched["freshness"] = freshness
        enriched["rank_score"] = round(TRUST_WEIGHT.get(tier, 0.5) * freshness, 3)
        kept.append(enriched)
    kept.sort(key=lambda entry: entry.get("rank_score", 0.0), reverse=True)
    return kept


def _infer_job_title(jd_text: str, resume_text: str) -> str:
    lines = [line.strip() for line in str(jd_text or "").splitlines() if line.strip()]
    for line in lines[:3]:
        if re.search(r"(岗位|职位|招聘|方向|JD)[：:，,\s]", line) and len(line) < 80:
            return re.sub(r"^[\s\d.、\-—]+", "", line)[:50]
    return (lines[0] if lines else str(jd_text or ""))[:50]


def _run_one(query: str, max_results: int = 5) -> dict:
    try:
        return web_search_handler({"query": query, "max_results": max_results})
    except Exception:
        return {"success": False, "content": "", "source": ""}


def research_job_node(state) -> dict:
    """One-time industry/job research before the interview loop starts."""
    jd_text = str(state.get("jd_text", "") or "")
    resume_text = str(state.get("resume_text", "") or "")
    # M22：优先用 analyze 阶段 LLM 读全文判定的岗位名；
    # `_infer_job_title`（只看 JD 前三行）降级为兜底。
    role_profile = state.get("role_profile") or {}
    job_title = str(role_profile.get("job_title") or "").strip()
    if not job_title or job_title == "未知岗位":
        job_title = _infer_job_title(jd_text, resume_text)
    if not job_title:
        job_title = "目标岗位"

    queries = [
        f"{job_title} 岗位职责 硬性技能要求",
        f"{job_title} 面试题 高频真题 考察点",
        f"{job_title} 日常工作 业务痛点",
        f"{job_title} 胜任力模型 核心能力 考察维度",
        f"{job_title} 典型工作场景 案例分析",
    ]
    gathered = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_run_one, query) for query in queries]
        for future in futures:
            outcome = future.result()
            if outcome.get("success") and outcome.get("content"):
                gathered.append(
                    {
                        "query": queries[futures.index(future)],
                        "content": str(outcome.get("content", ""))[:2400],
                        "sources": str(outcome.get("source", "")),
                        "raw": outcome.get("raw") or [],
                    }
                )
    # M24：跨 query 汇总后统一做去重与分级排序，只保留排序靠前的片段
    merged_raw: list[dict] = []
    for item in gathered:
        merged_raw.extend(item.get("raw") or [])
    ranked = _rank_and_dedup(merged_raw)[:MAX_FRAGMENTS]
    job_research = {
        "job_title": job_title,
        "hard_skills": [],
        "soft_skills": [],
        "business_scenarios": [],
        "question_directions": [],
        "sources": [item["sources"] for item in gathered],
        "raw_notes": "\n\n".join(item["content"] for item in gathered),
        # M24：分级/去重后的检索结果（含域名、发布日期、信任分级与排序分）
        "ranked_sources": ranked,
        # 逐条保留调研片段（content 不截断，注入出题提示时再按总量截断），
        # 让 research_ref 有真实取值空间：research:job:1..5。
        "fragments": [
            {
                "id": f"research:job:{index}",
                "query": item["query"],
                "content": item["content"],
            }
            for index, item in enumerate(gathered, 1)
        ],
    }
    assert "resume_text" not in job_research and "resume_profile" not in job_research
    # M22：调研结果不得反过来覆盖 LLM 判定的岗位名
    state_job_title = str(role_profile.get("job_title") or job_title)
    return {
        "job_title": state_job_title,
        "job_jd": jd_text or None,
        "job_research": job_research,
        # 5 条查询全部失败时明确标记，便于在报告里暴露"本次没有联网调研"。
        "research_status": "ok" if gathered else "failed",
        "usage_records": [],
    }
