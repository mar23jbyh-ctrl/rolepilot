"""批次 6 验证：M24 调研时效 / 来源分级 / 去重 / research_ref 内容相关性。"""

from __future__ import annotations

import json

from app.nodes.research_job import (
    MAX_FRAGMENTS,
    _domain_tier,
    _freshness_weight,
    _rank_and_dedup,
)
from app.role import evidence_status


def _five_results() -> list[dict]:
    """构造 5 条：1 条高可信新资料、1 条过期低质、2 条重复、1 条中等。"""

    return [
        {
            "url": "https://docs.python.org/3/library/asyncio.html",
            "domain": "docs.python.org",
            "published_date": "2026-03-01",
            "content": "asyncio 事件循环与协程调度官方说明",
        },
        {
            "url": "https://random-blog.example.com/old-post",
            "domain": "random-blog.example.com",
            "published_date": "2018-01-01",
            "content": "十年前的 asyncio 入门（已过时）",
        },
        {
            "url": "https://github.com/org/repo/issues/1",
            "domain": "github.com",
            "published_date": "2025-06-01",
            "content": "关于事件循环调度的讨论与结论",
        },
        {
            # 与第 3 条 URL 不同但内容完全相同 → 内容指纹去重
            "url": "https://mirror.example.net/copy",
            "domain": "mirror.example.net",
            "published_date": "2025-06-02",
            "content": "关于事件循环调度的讨论与结论",
        },
        {
            # 与第 1 条 URL 完全相同 → URL 去重
            "url": "https://docs.python.org/3/library/asyncio.html",
            "domain": "docs.python.org",
            "published_date": "2026-03-01",
            "content": "asyncio 事件循环与协程调度官方说明",
        },
    ]


def test_m24_domain_tiers():
    assert _domain_tier("docs.python.org") == "high"
    assert _domain_tier("github.com") == "medium"
    assert _domain_tier("random-blog.example.com") == "low"
    assert _domain_tier("") == "unknown"


def test_m24_freshness_weight_decays():
    assert _freshness_weight("2026-09-01") == 1.0
    assert _freshness_weight("2025-01-01") == 0.6
    assert _freshness_weight("2018-01-01") == 0.2
    assert _freshness_weight("") == 0.7


def test_m24_dedup_and_ranking():
    ranked = _rank_and_dedup(_five_results())
    # 5 条 → 去掉 1 条 URL 重复 + 1 条内容重复 → 3 条
    assert len(ranked) == 3, [item["url"] for item in ranked]
    # 高可信 + 新鲜 排第一
    assert ranked[0]["domain"] == "docs.python.org"
    assert ranked[0]["tier"] == "high"
    # 过期低质排最后，且被降权
    assert ranked[-1]["domain"] == "random-blog.example.com"
    assert ranked[-1]["freshness"] == 0.2
    scores = [item["rank_score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)
    # 重复内容只保留一条
    contents = [item["content"] for item in ranked]
    assert contents.count("关于事件循环调度的讨论与结论") == 1


def test_m24_fragment_cap():
    many = [
        {
            "url": f"https://example.com/{index}",
            "domain": "example.com",
            "published_date": "2026-01-01",
            "content": f"内容 {index}",
        }
        for index in range(20)
    ]
    assert len(_rank_and_dedup(many)[:MAX_FRAGMENTS]) == MAX_FRAGMENTS


def test_m24_search_does_not_add_calls(monkeypatch):
    """确认新增参数不改变调用次数：每条 query 仍只发 1 次请求。"""

    import app.tools.executor as ex

    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"url": "https://docs.python.org/x", "title": "t", "content": "c"}]}

    def _post(url, json=None, timeout=None):
        calls["n"] += 1
        assert json["days"] == 365 and json["search_depth"] == "basic"
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "post", _post)
    monkeypatch.setattr(ex.settings, "tavily_api_key", "tvly-test")
    out = ex._tavily_search("q", max_results=5)
    assert calls["n"] == 1
    assert out[0]["domain"] == "docs.python.org"
    assert "published_date" in out[0]


def test_m24_research_ref_content_mismatch_is_flagged():
    """题目引用了调研片段，但内容完全对不上 → 记 mismatch。"""

    question = {
        "content": "请说明合同主体资格审查的要点",
        "jd_ref": "",
        "project_ref": "",
        "research_ref": "research:job:1",
    }
    matched = evidence_status(
        question,
        reference_ids=["research:job:1"],
        reference_texts={"research:job:1": "合同主体资格审查要点包括营业执照与授权文件"},
    )
    assert matched["evidence_status"] == "ok"

    mismatched = evidence_status(
        question,
        reference_ids=["research:job:1"],
        reference_texts={"research:job:1": "静脉输液操作规范与无菌技术要点说明"},
    )
    assert mismatched["evidence_status"] == "mismatch"
    assert mismatched["evidence_similarity"]["research_ref"] < 0.15
