import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.schema import normalize_tool_call_dicts, record_to_message

USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "finish_reason": "stop"}


def _json(messages, temperature=0.7, max_tokens=None):
    def _text(message):
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))

    content = " ".join(_text(m) for m in messages)
    if "rubric" in content.lower() and "动态生成" in content:
        if "法学" in content or "律师" in content or "法务" in content:
            dims = [
                {"key": "law1", "label": "法律实务", "weight": 0.4, "definition": "合同/合规实务", "threshold": "能结合真实案件讲清审查流程与风险判断"},
                {"key": "law2", "label": "法律文书与检索", "weight": 0.3, "definition": "文书能力", "threshold": "能说明检索路径与文书要点"},
                {"key": "law3", "label": "诉讼流程", "weight": 0.15, "definition": "诉讼程序", "threshold": "能说清关键节点"},
                {"key": "law4", "label": "沟通与执业素养", "weight": 0.15, "definition": "执业素养", "threshold": "能说明与当事人沟通的具体做法"},
            ]
        elif "AI" in content or "大模型" in content or "RAG" in content:
            dims = [
                {"key": "ai1", "label": "RAG工程能力", "weight": 0.4, "definition": "检索与生成", "threshold": "能结合项目讲清检索链路与调优"},
                {"key": "ai2", "label": "工程落地", "weight": 0.2, "definition": "工程实现", "threshold": "能说明部署与排错"},
                {"key": "ai3", "label": "评估与迭代", "weight": 0.2, "definition": "评估体系", "threshold": "能给出量化评估方式"},
                {"key": "ai4", "label": "协作与表达", "weight": 0.2, "definition": "协作沟通", "threshold": "表达清晰有条理"},
            ]
        else:
            dims = [
                {"key": "job1", "label": "岗位基础", "weight": 0.4, "definition": "岗位能力", "threshold": "能结合场景说清核心做法"},
                {"key": "job2", "label": "实务处理", "weight": 0.3, "definition": "实务能力", "threshold": "能讲清处理流程与取舍"},
                {"key": "job3", "label": "沟通协作", "weight": 0.15, "definition": "协作", "threshold": "能说明协作方式"},
                {"key": "job4", "label": "学习成长", "weight": 0.15, "definition": "学习能力", "threshold": "能给出具体学习路径"},
            ]
        return json.dumps({"dimensions": dims}), USAGE
    if "岗位 JD 摘要" in content:
        if "匹配等级 LOW" in content or "匹配等级：LOW" in content:
            questions = [
                {
                    "id": 1,
                    "category": "scenario",
                    "difficulty": "easy",
                    "depth_level": "application",
                    "project_ref": "",
                    "jd_ref": "",
                    "content": "你如何理解这个岗位的日常工作，并计划如何补齐所需能力？",
                    "skills": ["岗位认知"],
                    "source_id": "llm:low-1",
                    "source_type": "llm",
                }
            ]
        elif "法学" in content or "律师" in content or "法务" in content:
            questions = [
                {
                    "id": 1,
                    "category": "scenario",
                    "difficulty": "medium",
                    "depth_level": "application",
                    "project_ref": "",
                    "jd_ref": "0",
                    "content": "请说明你审核合同时的重点步骤。",
                    "skills": ["合同审查"],
                    "source_id": "llm:law-1",
                    "source_type": "llm",
                }
            ]
        elif "AI" in content or "大模型" in content or "RAG" in content:
            questions = [
                {
                    "id": 1,
                    "category": "project",
                    "difficulty": "medium",
                    "depth_level": "application",
                    "project_ref": "0",
                    "jd_ref": "0",
                    "content": "介绍一下你的RAG项目",
                    "skills": ["RAG"],
                    "source_id": "llm:1",
                    "source_type": "llm",
                }
            ]
        else:
            questions = [
                {
                    "id": 1,
                    "category": "scenario",
                    "difficulty": "medium",
                    "depth_level": "application",
                    "project_ref": "0",
                    "jd_ref": "",
                    "content": "请介绍你负责过的一个典型业务项目",
                    "skills": ["业务理解"],
                    "source_id": "llm:g-1",
                    "source_type": "llm",
                }
            ]
        return json.dumps({"questions": questions}), USAGE
    if "候选人回答" in content:
        return (
            json.dumps(
                {
                    # M18：评分契约改为"模型标档位，代码算分"
                    "dimension_levels": {k: 4 for k in ("law1", "law2", "law3", "law4", "ai1", "ai2", "ai3", "ai4", "job1", "job2", "job3", "job4")},
                    "is_relevant": True,
                    "should_follow_up": False,
                    "missed_points": [],
                    "covered_points": [],
                    "covered_aspects": ["流程"],
                }
            ),
            USAGE,
        )
    if "压缩摘要" in content or "面试流程质检员" in content:
        return json.dumps({"status": "ok", "findings": [], "suggestions": []}), USAGE
    if "简历文本" in content:
        if "法学" in content or "律师" in content or "法务" in content:
            return json.dumps(
                {"skills": ["法务", "合同管理"], "projects": ["合同管理台账"], "concerns": []}
            ), USAGE
        return json.dumps({"skills": ["rag"], "projects": ["问答系统"], "concerns": []}), USAGE
    if "JD 文本" in content:
        return json.dumps({"requirements": []}), USAGE
    if "简历原文" in content:
        return json.dumps({"matches": [], "weak_skills": [], "strong_skills": [], "summary": "s"}), USAGE
    if "法学" in content or "律师" in content or "法务" in content:
        return "你最近一次合同审查里最关注的风险点是什么？", USAGE
    return "请结合你简历中提到的真实经历回答这个问题。", USAGE


@pytest.fixture(autouse=True)
def mock_chat(monkeypatch):
    import app.context.manager as cm
    import app.nodes.analyze as na
    import app.nodes.ask as nq
    import app.nodes.assess as ns
    import app.nodes.evaluate as ne
    import app.nodes.plan as np
    import app.nodes.self_check as nsc
    import app.validation as v

    for module in (na, np, ns, ne, cm, v, nsc, nq):
        monkeypatch.setattr(module, "chat_with_usage", _json)
    monkeypatch.setattr(nq, "chat_with_tools", lambda *a, **k: ("x", [], USAGE))


def _base_state(**overrides):
    state = {
        "jd_text": "AI大模型应用工程师",
        "resume_text": "RAG问答系统",
        "resume_profile": {"skills": ["RAG"], "projects": ["问答系统"]},
        "jd_profile": {"requirements": []},
        "gap_report": {"weak_skills": [], "strong_skills": ["RAG"], "matches": [], "missing_skills": [], "summary": "s"},
        "role_profile": {
            "role_family": "application",
            "industry": "tech",
            "industry_known": True,
            "needs_web_research": False,
            "resume_vague": False,
            "theory_depth": "concept",
            "forbidden_drills": [],
            "focus_areas": ["RAG"],
        },
        "question_plan": [],
        "conversation_history": [],
        "summaries": [],
        "summarized_ids": [],
        "turn_records": [],
        "assessments": [],
        "usage_records": [],
        "tool_records": [],
        "difficulty_events": [],
        "current_question_index": 0,
        "follow_up_count": 0,
        "max_follow_ups": 3,
        "difficulty": "medium",
    }
    state.update(overrides)
    return state


def test_record_to_message_tool_calls():
    record = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "tool_call",
                "name": "web_search",
                "args": {"query": "x"},
                "id": "call_1",
            }
        ],
    }
    message = record_to_message(record)
    assert isinstance(message, AIMessage)
    assert message.tool_calls[0]["name"] == "web_search"


def test_normalize_openai_style_tool_calls():
    calls = normalize_tool_call_dicts(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "WebSearch", "arguments": '{"query":"x"}'},
            }
        ]
    )
    assert calls[0]["type"] == "tool_call"
    assert calls[0]["args"] == {"query": "x"}


def test_analyze_node():
    from app.nodes.analyze import analyze

    result = analyze(_base_state())
    assert "role_profile" in result
    assert result["resume_profile"]["skills"]
    assert result["usage_records"]


def test_plan_node():
    from app.nodes.plan import plan

    result = plan(_base_state())
    assert result["question_plan"][0]["source_id"]
    assert result["question_plan"][0]["project_ref"] == "0"


def test_plan_low_match_uses_motivation_questions(monkeypatch):
    import app.nodes.plan as np

    state = _base_state(
        jd_text="法学岗位",
        job_title="企业律师（法学方向）",
        role_profile={
            "role_family": "general",
            "industry": "legal",
            "industry_known": True,
            "needs_web_research": True,
            "resume_vague": True,
            "resume_jd_match_level": "LOW",
            "theory_depth": "application",
            "forbidden_drills": [],
            "focus_areas": ["岗位业务"],
            "label": "通用行业岗",
        },
        gap_report={
            "weak_skills": [],
            "strong_skills": [],
            "matches": [],
            "missing_skills": ["法学"],
            "summary": "转行",
            "resume_jd_match_level": "LOW",
        },
        resume_profile={"skills": ["销售"], "projects": []},
        resume_text="五年销售，无法学经验",
    )
    result = np.plan(state)
    first = result["question_plan"][0]
    assert "岗位" in first["content"] or "能力" in first["content"]
    assert first["project_ref"] == ""


def test_ask_node_returns_message_records():
    from app.nodes.ask import ask

    state = _base_state(
        question_plan=[
            {
                "id": 1,
                "category": "project",
                "difficulty": "medium",
                "depth_level": "application",
                "project_ref": "0",
                "jd_ref": "0",
                "content": "介绍RAG项目",
                "skills": ["RAG"],
                "source_id": "llm:1",
                "source_type": "llm",
            }
        ]
    )
    result = ask(state)
    assert result["current_question"]
    assert result["conversation_history"][-1]["role"] == "assistant"


def test_assess_explanation_request_uses_model_intent(monkeypatch):
    """M3：讲解请求改由模型意图判断（原先靠关键词子串，误判率 4.8%）。"""

    import app.nodes.assess as ns

    monkeypatch.setattr(
        ns,
        "chat_with_usage",
        lambda *a, **k: (
            json.dumps({"score": 3, "candidate_intent": "request_explanation"}, ensure_ascii=False),
            USAGE,
        ),
    )
    state = _base_state(
        current_answer="请讲解一下直接操作数据库和调用API的区别",
        current_question="介绍一下你的RAG项目",
        question_plan=[{"id": 1, "category": "project", "skills": ["RAG"], "depth_level": "application", "project_ref": "0"}],
    )
    result = ns.assess(state)
    assert result.get("explanation_pending") is True
    assert "assessments" not in result
    # M3：该轮现在会产生一次 LLM 调用，必须记账，否则成本不可见
    assert result["usage_records"][0]["node"] == "assess_intent"


def test_assess_question_id_and_follow_up_flag():
    from app.nodes.assess import assess

    state = _base_state(
        current_answer="我负责合同合规审查",
        current_question="请说明你审核合同时的重点步骤",
        current_question_index=0,
        follow_up_count=1,
        question_plan=[
            {
                "id": 12,
                "category": "scenario",
                "skills": ["合同审查"],
                "depth_level": "application",
                "project_ref": "",
                "jd_ref": "0",
            }
        ],
    )
    result = assess(state)
    assessment = result["assessments"][0]
    assert assessment["question_id"] == 1
    assert assessment["is_follow_up"] is True


def test_assess_primary_question_is_not_follow_up():
    from app.nodes.assess import assess

    state = _base_state(
        current_answer="这是对主问题的回答",
        follow_up_count=0,
        question_plan=[{"id": 3, "category": "scenario", "skills": ["合同"], "depth_level": "application", "project_ref": "", "jd_ref": ""}],
    )
    assessment = assess(state)["assessments"][0]
    assert assessment["question_id"] == 1
    assert assessment["is_follow_up"] is False


def test_global_question_ids_increment():
    from app.nodes.assess import assess

    base = _base_state(
        current_answer="我负责合同审查",
        question_plan=[{"id": 99, "category": "scenario", "skills": ["合同"], "depth_level": "application", "project_ref": "", "jd_ref": ""}],
    )
    first = assess(base)
    second = assess(
        _base_state(
            current_answer="追问的回答",
            assessments=first["assessments"],
            global_question_counter=first["global_question_counter"],
            question_plan=[{"id": 99, "category": "scenario", "skills": ["合同"], "depth_level": "application", "project_ref": "", "jd_ref": ""}],
        )
    )
    assert first["assessments"][0]["question_id"] == 1
    assert second["assessments"][0]["question_id"] == 2


def test_route_after_assessment_no_extra_question_at_plan_end():
    from app.graph.edges import route_after_assessment

    state = {
        "assessments": [
            {"question_id": 1, "score": 9.0},
            {"question_id": 2, "score": 3.0},
            {"question_id": 3, "score": 9.0},
            {"question_id": 4, "score": 3.0},
        ],
        "question_plan": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
        "current_question_index": 3,
        "follow_up_count": 0,
        "max_follow_ups": 3,
    }
    assert route_after_assessment(state) == "evaluate"


def test_route_after_ask_without_question_goes_to_evaluate():
    from app.graph.edges import route_after_ask

    assert route_after_ask({"current_question": ""}) == "evaluate"
    assert route_after_ask({"current_question": "请介绍一下你负责的项目？"}) == "assess"


def test_session_name_unique_and_delete(tmp_path):
    from app.session.store import SessionStore

    store = SessionStore(db_path=tmp_path / "sessions.db")
    store.create("a", jd_text="jd", resume_text="resume", state={}, name="法治岗")
    with pytest.raises(ValueError):
        store.create("b", jd_text="jd", resume_text="resume", state={}, name="法治岗")
    assert store.name_exists("法治岗") is True
    assert store.delete("a") is True
    assert store.name_exists("法治岗") is False
    assert store.delete("a") is False


def test_cleanup_stale_disabled_by_default(tmp_path, monkeypatch):
    from app.config import settings
    from app.session.store import SessionStore

    monkeypatch.setattr(settings, "session_stale_days", 0)
    store = SessionStore(db_path=tmp_path / "sessions2.db")
    store.create("old", jd_text="jd", resume_text="resume", state={}, name="")
    store.cleanup_stale()
    assert store.get("old") is not None


def test_session_rename_rules(tmp_path):
    from app.session.store import SessionStore

    store = SessionStore(db_path=tmp_path / "rename.db")
    store.create("a", jd_text="jd", resume_text="resume", state={}, name="会话A")
    store.create("b", jd_text="jd", resume_text="resume", state={}, name="会话B")

    with pytest.raises(ValueError):
        store.rename("b", "会话A")
    with pytest.raises(ValueError):
        store.rename("b", "   ")

    store.rename("b", "会话C")
    assert store.get("b")["name"] == "会话C"
    # 改成自己当前的名字是允许的
    store.rename("b", "会话C")
    assert store.get("b")["name"] == "会话C"


def test_advance_node():
    from app.nodes.advance import advance

    state = _base_state(
        assessments=[{"question_id": 0, "score": 8}],
        question_plan=[{"id": 1}, {"id": 2}],
    )
    result = advance(state)
    assert result["current_question_index"] == 1
    assert result["follow_up_count"] == 0


def test_evaluate_and_self_check_message_objects():
    from app.nodes.evaluate import evaluate
    from app.nodes.self_check import self_check

    state = _base_state(
        assessments=[{"question_id": 0, "score": 7, "question": "q"}],
        usage_records=[{"node": "assess", "total_tokens": 4}],
        conversation_history=[
            {"role": "assistant", "content": "question?"},
            {"role": "user", "content": "answer"},
        ],
    )
    evaluated = evaluate(state)
    assert evaluated["evaluation_report"]["overall_score"] == 7
    checked = self_check({**state, "evaluation_report": evaluated["evaluation_report"]})
    assert checked["self_check_report"]["status"] in {"ok", "warning", "error"}


def test_schema_roundtrip():
    from app.graph.schema import to_messages, to_records

    messages = [HumanMessage(content="hi"), SystemMessage(content="sys")]
    records = to_records(messages)
    assert records[0]["role"] == "user"
    assert records[1]["role"] == "system"
    restored = to_messages(records)
    assert isinstance(restored[0], HumanMessage)


def test_research_job_node_only_research_stage():
    import app.nodes.research_job as rj

    def fake_search(args):
        return {
            "success": True,
            "content": "律师 高频面试：合同审查流程；法务 业务痛点：合规风险",
            "source": "https://example.com/lawyer",
        }

    rj.web_search_handler = fake_search
    result = rj.research_job_node(
        {
            "jd_text": "企业律师（法务方向）\n负责合同与合规",
            "resume_text": "五年公司法务",
        }
    )
    assert "律师" in result["job_title"] or "法务" in result["job_title"]
    assert "法务" in result["job_research"]["raw_notes"]
    assert "assessments" not in result


def test_rubric_does_not_cross_roles():
    from app.nodes.analyze import analyze

    law = analyze(_base_state(jd_text="律师助理（法学方向）", resume_text="法学本科"))
    ai = analyze(_base_state(jd_text="AI大模型开发工程师", resume_text="RAG项目"))
    law_labels = " ".join(d.get("label", "") for d in law["role_profile"].get("rubric", []))
    ai_labels = " ".join(d.get("label", "") for d in ai["role_profile"].get("rubric", []))
    assert "RAG" not in law_labels and "大模型" not in law_labels
    assert "RAG" in ai_labels


def test_full_graph_sessions_isolated(monkeypatch, tmp_path):
    import app.nodes.research_job as rj
    from app.config import settings
    from app.service import InterviewService

    # 会话库与断点库都改写到 pytest 临时目录：data_root / checkpoint_db_path
    # 是运行期读取的属性，测试跑完由 pytest 自动清理，
    # 不再往真实 data/ 里写文件（也避开 Windows 下 SQLite 连接未释放导致的删除失败）。
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))

    rj.web_search_handler = lambda args: {
        "success": True,
        "content": "法学 高频面试：合同审查；财务 高频面试：审计流程",
        "source": "https://example.test",
    }
    owner_a = "unit_a_" + uuid.uuid4().hex[:6]
    owner_b = "unit_b_" + uuid.uuid4().hex[:6]
    service_a = InterviewService(owner=owner_a)
    service_b = InterviewService(owner=owner_b)
    started_a = service_a.start_session("法学岗位 JD", "简历：法务三年")
    assert started_a["question"]
    result_a = service_a.submit_answer(started_a["session_id"], "我负责合同与合规审查。")
    follow_up_rounds = 0
    while not result_a.get("report") and follow_up_rounds < 6:
        result_a = service_a.submit_answer(
            started_a["session_id"], "补充说明：这是我对该问题的进一步回答。"
        )
        follow_up_rounds += 1
    assert result_a["report"]
    plan_text = json.dumps(started_a["state"].get("question_plan", []), ensure_ascii=False)
    history_messages = result_a["state"].get("conversation_history", []) or []
    history_text = " ".join(
        str(getattr(m, "content", "")) for m in history_messages
    )
    assert "RAG" not in plan_text and "大模型" not in plan_text and "AI" not in plan_text
    assert "RAG" not in history_text and "AI" not in history_text
    report_text = json.dumps(result_a["report"], ensure_ascii=False)
    assert "RAG" not in report_text and "大模型" not in report_text
    per_question = result_a["report"].get("per_question", [])
    assert all(int(item.get("question_id", 0)) > 0 for item in per_question)
    sessions_a = [s["id"] for s in service_a.store.list_sessions()]
    sessions_b = [s["id"] for s in service_b.store.list_sessions()]
    assert sessions_a and not sessions_b


# ---------------------------------------------------------------------------
# 模块 2：岗位画像（LLM 判断职能域 + 代码只做兜底/守卫/校验）
# 以下均为新增用例，不改动上面的既有测试。
# ---------------------------------------------------------------------------


def _llm_profile(**overrides):
    """一份合法的 LLM 岗位画像样例（默认为诉讼律师）。"""
    profile = {
        "job_title": "诉讼律师",
        "domain": "法律-诉讼",
        "industry": "法律服务",
        "core_responsibilities": ["案件事实梳理与证据组织", "法律文书起草"],
        "key_skills": ["民商法", "证据规则"],
        "assessment_focus": ["法律实务", "案例分析", "合规意识", "沟通谈判"],
        "forbidden_topics": [],
        "confidence": 0.9,
    }
    profile.update(overrides)
    return profile


def test_role_profile_for_lawyer():
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "五年诉讼经验", _llm_profile())
    assert "法律" in profile["domain"]
    joined = "、".join(profile["forbidden_topics"])
    assert "编程" in joined
    assert "AI" in joined or "RAG" in joined
    assert profile["profile_source"] == "llm"


def test_role_profile_for_nurse():
    from app.role import build_role_profile

    profile = build_role_profile(
        "临床护士",
        "三年三甲医院护理经验",
        _llm_profile(
            job_title="临床护士",
            domain="医疗-临床护理",
            industry="医疗健康",
            assessment_focus=["护理流程", "临床判断", "医患沟通", "应急处理"],
        ),
    )
    assert "护理" in profile["domain"] or "医疗" in profile["domain"]
    joined = "、".join(profile["forbidden_topics"])
    assert "编程" in joined


def test_role_profile_for_ai_engineer():
    from app.role import build_role_profile

    profile = build_role_profile(
        "AI 应用工程师",
        "RAG 问答系统",
        _llm_profile(
            job_title="AI 应用工程师",
            domain="技术-AI应用",
            industry="互联网",
            assessment_focus=["RAG 全流程", "Agent/工具调用", "提示词工程", "工程落地"],
        ),
    )
    assert "AI" in profile["domain"]
    assert any("RAG" in item for item in profile["assessment_focus"])
    # 岗位本身就是 AI 方向，守卫不能把 RAG 误加进禁止清单
    assert not any(item.upper() == "RAG" for item in profile["forbidden_topics"])
    joined = "、".join(profile["forbidden_topics"])
    # R1：技术职能域不该把编程/算法列为禁止话题（AI 岗也要写代码）
    assert "编程" not in joined
    assert "算法" not in joined
    # 但跨职能域话题仍必须保留
    assert "护理实务" in joined
    assert "法律实务" in joined


def test_role_profile_for_backend():
    from app.role import build_role_profile

    profile = build_role_profile(
        "Java 后端开发工程师",
        "五年 Java 服务开发",
        _llm_profile(
            job_title="Java 后端开发工程师",
            domain="技术-后端开发",
            industry="互联网",
            assessment_focus=["Java 并发与 JVM", "数据库与 SQL 优化", "分布式与缓存", "系统设计"],
        ),
    )
    assert "后端" in profile["domain"]
    joined = "、".join(profile["forbidden_topics"])
    assert "护理" in joined
    assert "法律" in joined
    # 技术岗不该把编程/算法列为禁止话题
    assert "编程" not in joined
    assert "算法" not in joined


def test_role_profile_fallback():
    from app.role import DEFAULT_FOCUS, build_role_profile

    invalid = build_role_profile("某岗位", "某简历", {"confidence": "not-a-number"})
    assert invalid["domain"] == "general"
    assert invalid["profile_source"] == "fallback"
    assert invalid["confidence"] == 0.0
    assert invalid["assessment_focus"] == list(DEFAULT_FOCUS)

    low_confidence = build_role_profile(
        "某岗位", "某简历", _llm_profile(confidence=0.3)
    )
    assert low_confidence["domain"] == "general"
    assert low_confidence["profile_source"] == "fallback"


def test_merge_forbidden_guard():
    from app.role import build_role_profile

    profile = build_role_profile(
        "企业律师",
        "法律从业者",
        _llm_profile(assessment_focus=["法律实务", "案例分析"], forbidden_topics=[]),
    )
    joined = "、".join(profile["forbidden_topics"])
    assert "RAG" in joined
    assert "大模型" in joined
    assert "编程" in joined


def test_analyze_writes_new_role_profile_schema():
    from app.nodes.analyze import analyze

    result = analyze(_base_state())
    profile = result["role_profile"]
    for key in (
        "job_title",
        "domain",
        "industry",
        "assessment_focus",
        "forbidden_topics",
        "confidence",
    ):
        assert key in profile
    assert isinstance(profile["assessment_focus"], list) and profile["assessment_focus"]
    assert isinstance(profile["forbidden_topics"], list)
    # 新画像不再产出旧的岗位族 / 理论深度概念
    assert "role_family" not in profile
    assert "theory_depth" not in profile


# ---------------------------------------------------------------------------
# 3.2.1：画像 schema 升级（白名单 focus + out_of_scope + 3.2.0 接口预留）
# ---------------------------------------------------------------------------


def _focus_payload(**overrides):
    """新格式画像 payload：assessment_focus 为对象数组 + out_of_scope_topics。"""
    payload = {
        "job_title": "诉讼律师",
        "domain": "法律-诉讼",
        "industry": "法律服务",
        "core_responsibilities": ["案件事实梳理与证据组织"],
        "key_skills": ["民商法", "证据规则"],
        "assessment_focus": [
            {
                "key": "f1",
                "name": "合同审查实务",
                "subtopics": ["合同主体资格审查", "违约责任与违约金", "争议解决条款"],
                "why_relevant": "JD 要求独立完成合同审核",
                "source_jd_ref": "0",
            },
            {
                "key": "f2",
                "name": "诉讼流程与证据",
                "subtopics": ["举证期限与证据交换", "庭审质证要点", "管辖与送达"],
                "why_relevant": "岗位职责含民商事诉讼",
                "source_jd_ref": "1",
            },
        ],
        "out_of_scope_topics": ["编程开发", "模型训练", "护理实务"],
        "forbidden_topics": ["编程开发", "模型训练", "护理实务"],
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def test_role_profile_accepts_new_focus_schema():
    from app.role import build_role_profile, focus_keys

    profile = build_role_profile("诉讼律师", "三年诉讼经验", _focus_payload())
    assert profile["guard_mode"] == "whitelist"
    items = profile["assessment_focus"]
    assert isinstance(items, list) and isinstance(items[0], dict)
    assert items[0]["key"] == "f1" and items[0]["name"] == "合同审查实务"
    assert len(items[0]["subtopics"]) == 3 and items[0]["focus_source"] == "llm"
    assert profile["out_of_scope_topics"] == ["编程开发", "模型训练", "护理实务"]
    # 兼容镜像：forbidden_topics 与 out_of_scope_topics 保持一致
    assert profile["forbidden_topics"] == profile["out_of_scope_topics"]
    assert focus_keys(profile) == ["f1", "f2"]


def test_role_profile_accepts_legacy_string_focus():
    from app.role import build_role_profile, profile_quality

    profile = build_role_profile(
        "诉讼律师",
        "三年诉讼经验",
        {
            "job_title": "诉讼律师",
            "domain": "法律-诉讼",
            "industry": "法律服务",
            "assessment_focus": ["法律实务", "案例分析"],
            "forbidden_topics": [],
            "confidence": 0.92,
        },
    )
    assert profile["guard_mode"] == "legacy"
    # 旧格式保持字符串形态：历史会话与既有读取方零影响
    assert profile["assessment_focus"] == ["法律实务", "案例分析"]
    joined = "、".join(profile["forbidden_topics"])
    assert "编程" in joined and "大模型" in joined  # 4 组兜底仍生效
    assert profile_quality(profile)["guard_mode"] == "legacy"


def test_role_profile_weak_subtopics_marked():
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(
        assessment_focus=[
            {
                "key": "f1",
                "name": "法律实务",
                "subtopics": ["法律", "合同", "合规"],
                "why_relevant": "",
                "source_jd_ref": "",
            }
        ]
    )
    profile = build_role_profile("诉讼律师", "简历", payload)
    assert profile["guard_mode"] == "weak_whitelist"
    quality = profile_quality(profile)
    assert quality["guard_mode"] == "weak_whitelist"
    assert quality["weak_focus_count"] == 1
    assert quality["invalid_subtopics"]["f1"] == ["法律", "合同", "合规"]


def test_match_focus_layers():
    from app.role import build_role_profile, match_focus

    lawyer = build_role_profile("诉讼律师", "简历", _focus_payload())
    ascii_profile = build_role_profile(
        "Java 后端开发工程师",
        "五年 Java",
        _focus_payload(
            job_title="Java 后端开发工程师",
            domain="技术-后端开发",
            industry="互联网",
            assessment_focus=[
                {
                    "key": "f1",
                    "name": "并发与性能",
                    "subtopics": ["JVM 内存模型与调优", "Java 线程池参数设计", "GC 日志分析"],
                    "why_relevant": "",
                    "source_jd_ref": "",
                }
            ],
            out_of_scope_topics=["护理实务"],
            forbidden_topics=["护理实务"],
        ),
    )
    cooccur_profile = build_role_profile(
        "某岗位",
        "简历",
        _focus_payload(
            assessment_focus=[
                {
                    "key": "f1",
                    "name": "跨职能沟通",
                    "subtopics": ["对话 反馈 闭环"],
                    "why_relevant": "",
                    "source_jd_ref": "",
                }
            ],
            out_of_scope_topics=["护理实务"],
            forbidden_topics=["护理实务"],
        ),
    )

    ascii_hit = match_focus("请讲讲你调 JVM 内存模型的经验", ascii_profile)
    assert ascii_hit["verdict"] == "keep" and ascii_hit["stage"] == "ascii"

    substring_hit = match_focus("举证期限与证据交换怎么把握", lawyer)
    assert substring_hit["stage"] == "substring" and substring_hit["focus_key"] == "f2"

    # 中文 2 字片段单独出现不算命中（避免"对话/模型"这类宽泛词误伤）
    assert match_focus("这里只提到对话两个字", cooccur_profile)["verdict"] == "unverified"
    cooccur_hit = match_focus("对话与反馈怎么形成闭环", cooccur_profile)
    assert cooccur_hit["stage"] == "cooccur"

    # 断开字面的同义改写由 bigram 相似度层兜住
    similarity_hit = match_focus("合同主体的资格审查流程要注意什么", lawyer)
    assert similarity_hit["stage"] == "similarity" and similarity_hit["focus_key"] == "f1"


def test_role_policy_text_renders_focus_and_out_of_scope():
    from app.role import build_role_profile, role_policy_text

    new_profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    text = role_policy_text(new_profile)
    assert "合同审查实务（合同主体资格审查、违约责任与违约金、争议解决条款）" in text
    assert "禁止话题：编程开发、模型训练、护理实务。" in text

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "domain": "法律-诉讼",
            "industry": "法律服务",
            "assessment_focus": ["法律实务", "案例分析"],
            "forbidden_topics": [],
            "confidence": 0.92,
        },
    )
    legacy_text = role_policy_text(legacy)
    # legacy 渲染与模块 5 逐字一致（回归保护）
    assert "考察重心：法律实务、案例分析。" in legacy_text
    assert "禁止话题：" in legacy_text and "编程" in legacy_text


def test_role_profile_prompt_formats():
    from app.prompts.templates import ROLE_PROFILE_PROMPT

    text = ROLE_PROFILE_PROMPT.format(jd="JD 内容", resume="简历内容")
    assert "JD 内容" in text and "简历内容" in text
    for token in ("subtopics", "out_of_scope_topics", "forbidden_topics", "示例 4"):
        assert token in text, f"新 prompt 缺少 {token}"


def test_focus_keys_and_render_focus_whitelist():
    from app.role import build_role_profile, focus_keys, render_focus_whitelist

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    assert focus_keys(profile) == ["f1", "f2"]
    rendered = render_focus_whitelist(profile)
    assert (
        "[f1] 合同审查实务 —— 子项：合同主体资格审查 / 违约责任与违约金 / 争议解决条款"
        in rendered
    )
    assert "[f2] 诉讼流程与证据" in rendered


def test_guard_mode_three_states():
    from app.role import build_role_profile

    whitelist = build_role_profile("诉讼律师", "简历", _focus_payload())
    assert whitelist["guard_mode"] == "whitelist"

    weak = build_role_profile(
        "诉讼律师",
        "简历",
        _focus_payload(
            assessment_focus=[
                {
                    "key": "f1",
                    "name": "法律实务",
                    "subtopics": ["法律", "合同", "合规"],
                    "why_relevant": "",
                    "source_jd_ref": "",
                }
            ]
        ),
    )
    assert weak["guard_mode"] == "weak_whitelist"
    # 弱白名单不追加写死词表（避免"质量不合格"变成"误删"）
    assert weak["forbidden_topics"] == weak["out_of_scope_topics"]

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "assessment_focus": ["法律实务"],
            "forbidden_topics": [],
            "confidence": 0.9,
        },
    )
    assert legacy["guard_mode"] == "legacy"

    fallback = build_role_profile("某岗位", "简历", {"confidence": 0.2})
    assert fallback["guard_mode"] == "legacy"


def test_match_focus_binding_layer():
    from app.role import build_role_profile, match_focus

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())

    matched = match_focus("随便聊聊你的职业规划", profile, declared_focus_key="f1")
    assert matched["verdict"] == "keep" and matched["stage"] == "binding"
    assert matched["binding_status"] == "matched" and matched["focus_key"] == "f1"

    normalized = match_focus("随便聊聊", profile, declared_focus_key=" F1 ")
    assert normalized["binding_status"] == "normalized" and normalized["focus_key"] == "f1"

    repaired = match_focus("合同主体资格审查要注意什么", profile, declared_focus_key="f9")
    assert repaired["binding_status"] == "repaired" and repaired["focus_key"] == "f1"

    unverified = match_focus("完全无关的一句话", profile, declared_focus_key="f9")
    assert unverified["verdict"] == "unverified"
    assert unverified["binding_status"] == "unverified"


def test_out_of_scope_hits_composite_only():
    from app.role import build_role_profile, out_of_scope_hits

    whitelist = build_role_profile("诉讼律师", "简历", _focus_payload())
    assert out_of_scope_hits("请写一段代码实现模型训练流水线", whitelist) == ["模型训练"]
    # 复合词守卫不再误伤"增长模型"这类合法表述（收尾 3 的误伤样本）
    assert out_of_scope_hits("你如何搭建用户增长模型？", whitelist) == []

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "assessment_focus": ["法律实务"],
            "forbidden_topics": [],
            "confidence": 0.9,
        },
    )
    assert out_of_scope_hits("你如何搭建用户增长模型？", legacy)  # legacy 仍走 4 组


def test_evidence_status_branches():
    from app.role import evidence_status

    jd_items = ["独立完成合同审核", "负责民商事诉讼"]
    projects = ["合同管理台账系统"]
    refs = ["research:job", "qb:9"]

    ok = evidence_status({"content": "你如何做合同审核？", "jd_ref": "0"}, jd_items, projects, refs)
    assert ok["jd_ref"] == "0" and ok["evidence_status"] == "ok"
    assert ok["evidence_invalid"] == []

    non_digit = evidence_status({"content": "随便聊聊", "jd_ref": "第一题"}, jd_items, projects, refs)
    assert non_digit["jd_ref"] == "" and "jd_ref" in non_digit["evidence_invalid"]
    assert non_digit["evidence_status"] == "missing"

    out_of_range = evidence_status({"content": "随便聊聊", "jd_ref": "9"}, jd_items, projects, refs)
    assert out_of_range["jd_ref"] == "" and "jd_ref" in out_of_range["evidence_invalid"]

    bad_research = evidence_status(
        {"content": "随便聊聊", "research_ref": "web:不存在"}, jd_items, projects, refs
    )
    assert bad_research["research_ref"] == ""
    assert "research_ref" in bad_research["evidence_invalid"]

    mismatch = evidence_status(
        {"content": "护理中如何评估患者的疼痛程度", "jd_ref": "0"}, jd_items, projects, refs
    )
    assert mismatch["evidence_status"] == "mismatch" and mismatch["jd_ref"] == "0"

    repaired = evidence_status({"content": "负责民商事诉讼的流程", "jd_ref": ""}, jd_items, projects, refs)
    assert repaired["evidence_status"] == "repaired" and repaired["jd_ref"] == "1"


def test_motivation_quota_and_exemption():
    from app.role import (
        MOTIVATION_QUOTA,
        MOTIVATION_STRICT,
        build_role_profile,
        is_motivation_question,
        motivation_exempt,
    )

    assert MOTIVATION_QUOTA == 2
    assert MOTIVATION_STRICT is False

    motivation = {
        "focus_key": "",
        "question_type": "behavioral",
        "content": "你为什么选择这个岗位？",
        "jd_ref": "",
        "project_ref": "",
        "research_ref": "",
    }
    assert is_motivation_question(motivation) is True
    assert is_motivation_question({**motivation, "focus_key": "f1"}) is False
    assert is_motivation_question({**motivation, "jd_ref": "0"}) is False
    assert is_motivation_question({**motivation, "question_type": "scenario"}) is False

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    assert motivation_exempt(motivation, profile) is True
    contaminated = {**motivation, "content": "你为什么想转行做模型训练？"}
    assert motivation_exempt(contaminated, profile) is False


def test_profile_quality_focus_key_available():
    from app.role import build_role_profile, profile_quality

    whitelist = build_role_profile("诉讼律师", "简历", _focus_payload())
    quality = profile_quality(whitelist)
    assert quality["guard_mode"] == "whitelist"
    assert quality["focus_key_available"] is True
    assert quality["focus_count"] == 2 and quality["weak_focus_count"] == 0
    assert quality["out_of_scope_ok"] is True

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "assessment_focus": ["法律实务"],
            "forbidden_topics": [],
            "confidence": 0.9,
        },
    )
    legacy_quality = profile_quality(legacy)
    assert legacy_quality["guard_mode"] == "legacy"
    assert legacy_quality["focus_key_available"] is False
    assert legacy_quality["out_of_scope_ok"] is False


# ---------------------------------------------------------------------------
# 3.2.2：app/guard.py（白名单审计 + 批量仲裁 + 三票合成）
# ---------------------------------------------------------------------------


def _audit_question(
    question_id,
    content,
    *,
    focus_key="",
    skills=None,
    jd_ref="",
    project_ref="",
    research_ref="",
    question_type="scenario",
):
    return {
        "id": question_id,
        "content": content,
        "intent": "",
        "skills": list(skills or []),
        "focus_key": focus_key,
        "jd_ref": jd_ref,
        "project_ref": project_ref,
        "research_ref": research_ref,
        "question_type": question_type,
    }


def _guard_arbiter(payload=None, *, raises=False, cache=None, max_calls=1):
    from app.guard import BatchArbiter

    def judge_fn(messages):
        if raises:
            raise RuntimeError("arbiter unavailable")
        return json.dumps(payload or {"judgements": []}, ensure_ascii=False)

    return BatchArbiter(max_calls=max_calls, cache=cache, judge_fn=judge_fn)


def test_audit_whitelist_hit_keeps():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    questions = [
        _audit_question(1, "请说明合同主体资格审查的要点", focus_key="f1", skills=["合同审查"]),
        _audit_question(2, "举证期限与证据交换怎么把握", skills=["诉讼"]),
        _audit_question(3, "合同主体的资格审查流程要注意什么", skills=["合同审查"]),
    ]
    result = audit_questions(questions, profile)
    assert result["dropped"] == []
    assert len(result["kept"]) == 3
    stages = {record["question_id"]: record["stage"] for record in result["audit"]}
    assert stages[1] == "binding"
    assert stages[2] == "whitelist_hit"
    assert stages[3] == "whitelist_hit"
    assert result["arbiter_calls"] == 0


def test_audit_binding_repaired_keeps():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    result = audit_questions(
        [_audit_question(1, "合同主体资格审查要注意什么", focus_key="f9")], profile
    )
    record = result["audit"][0]
    assert record["binding_status"] == "repaired"
    assert record["focus_key"] == "f1"
    assert record["verdict"] == "keep"
    assert result["status"] == "ok"


def test_audit_unverified_arbiter_keep():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    arbiter = _guard_arbiter(
        {"judgements": [{"qid": 1, "verdict": "keep", "reason": "通用经历题", "confidence": 0.6}]}
    )
    result = audit_questions(
        [_audit_question(1, "聊聊你最近读的一本书")], profile, arbiter=arbiter
    )
    record = result["audit"][0]
    assert record["verdict"] == "kept_unverified"
    assert record["stage"] == "arbiter"
    assert arbiter.calls == 1 and result["arbiter_calls"] == 1
    assert result["summary"]["guard_uncertain_count"] == 1


def test_audit_arbiter_out_of_scope_drops():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    question = _audit_question(1, "请讲讲你在护理岗位上评估患者疼痛的流程", skills=["护理"])
    arbiter = _guard_arbiter(
        {
            "judgements": [
                {
                    "qid": 1,
                    "verdict": "out_of_scope",
                    "reason": "属于医疗职能域",
                    "evidence": "护理岗位",
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = audit_questions([question], profile, arbiter=arbiter)
    record = result["audit"][0]
    assert record["verdict"] == "drop" and len(result["dropped"]) == 1
    assert record["votes"] == {"D1": True, "D2": False, "D3": True}
    assert record["vote_evidence"]["D3"] == "护理岗位"
    assert "未绑定白名单" in record["vote_evidence"]["D1"]


def test_audit_arbiter_bad_evidence_downgrades_to_uncertain():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    arbiter = _guard_arbiter(
        {
            "judgements": [
                {
                    "qid": 1,
                    "verdict": "out_of_scope",
                    "reason": "看着像医疗题",
                    "evidence": "题目考查了护理技能",
                    "confidence": 0.8,
                }
            ]
        }
    )
    result = audit_questions(
        [_audit_question(1, "请讲讲你在护理岗位上评估患者疼痛的流程")],
        profile,
        arbiter=arbiter,
    )
    record = result["audit"][0]
    assert record["verdict"] == "kept_unverified"
    assert record["votes"]["D3"] is False
    assert record["raw"]["verdict"] == "uncertain"


def test_audit_d1_d2_short_circuit_drops_without_arbiter():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    arbiter = _guard_arbiter({"judgements": []})
    result = audit_questions(
        [_audit_question(1, "请写一段代码实现模型训练流水线", skills=["模型训练"])],
        profile,
        arbiter=arbiter,
    )
    record = result["audit"][0]
    assert len(result["dropped"]) == 1
    assert result["arbiter_calls"] == 0 and arbiter.calls == 0  # 短路省下预算
    assert record["stage"] == "strong_signal"
    assert record["votes"]["D1"] is True and record["votes"]["D2"] is True


def test_audit_conflict_uses_arbiter_as_third_vote():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    question = _audit_question(
        1, "合同审查里如何用模型训练做风险预测", focus_key="f1", skills=["合同审查"]
    )

    # 分支 1：绑定合法但命中强信号 → 仲裁确认跨域 → D2+D3 两票 → 丢弃
    arbiter_out = _guard_arbiter(
        {
            "judgements": [
                {
                    "qid": 1,
                    "verdict": "out_of_scope",
                    "reason": "模型训练不属于法律岗",
                    "evidence": "模型训练",
                    "confidence": 0.9,
                }
            ]
        }
    )
    result_out = audit_questions([question], profile, arbiter=arbiter_out)
    record_out = result_out["audit"][0]
    assert record_out["verdict"] == "drop"
    assert record_out["votes"] == {"D1": False, "D2": True, "D3": True}
    assert arbiter_out.calls == 1

    # 分支 2：仲裁不确定 → 只有 D2 一票 → 保留
    arbiter_uncertain = _guard_arbiter(
        {"judgements": [{"qid": 1, "verdict": "uncertain", "reason": "无法判断"}]}
    )
    result_keep = audit_questions([question], profile, arbiter=arbiter_uncertain)
    record_keep = result_keep["audit"][0]
    assert record_keep["verdict"] == "keep"
    assert record_keep["votes"]["D3"] is False
    assert arbiter_uncertain.calls == 1


def test_audit_arbiter_unavailable_keeps_all_and_marks_degraded():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    arbiter = _guard_arbiter(raises=True)
    result = audit_questions(
        [_audit_question(1, "聊聊你最近读的一本书")], profile, arbiter=arbiter
    )
    record = result["audit"][0]
    assert record["verdict"] == "kept_unverified"
    assert result["dropped"] == []
    assert result["status"] == "degraded"
    assert record["raw"]["source"] == "degraded"


def test_audit_weak_whitelist_never_drops():
    from app.guard import audit_questions
    from app.role import build_role_profile

    weak_profile = build_role_profile(
        "诉讼律师",
        "简历",
        _focus_payload(
            assessment_focus=[
                {
                    "key": "f1",
                    "name": "法律实务",
                    "subtopics": ["法律", "合同", "合规"],
                    "why_relevant": "",
                    "source_jd_ref": "",
                }
            ]
        ),
    )
    assert weak_profile["guard_mode"] == "weak_whitelist"
    result = audit_questions(
        [_audit_question(1, "请写一段代码实现模型训练流水线", skills=["模型训练"])],
        weak_profile,
    )
    assert result["dropped"] == []
    assert result["audit"][0]["verdict"] == "kept_unverified"
    assert result["status"] == "degraded"


def test_audit_all_weak_profile_never_drops():
    from app.guard import audit_questions
    from app.role import build_role_profile, profile_quality

    all_weak = build_role_profile(
        "诉讼律师",
        "简历",
        _focus_payload(
            assessment_focus=[
                {"key": "f1", "name": "维度一", "subtopics": ["法律"], "why_relevant": "", "source_jd_ref": ""},
                {"key": "f2", "name": "维度二", "subtopics": ["合同"], "why_relevant": "", "source_jd_ref": ""},
            ]
        ),
    )
    quality = profile_quality(all_weak)
    assert quality["guard_mode"] == "weak_whitelist"
    assert quality["weak_focus_count"] == 2
    result = audit_questions(
        [_audit_question(1, "请写一段代码实现模型训练流水线", skills=["模型训练"])],
        all_weak,
    )
    assert result["dropped"] == []
    assert result["summary"]["weak_focus_count"] == 2


def test_audit_legacy_llm_profile_drops_on_strong_signal():
    from app.guard import audit_questions
    from app.role import build_role_profile

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "domain": "法律-诉讼",
            "industry": "法律服务",
            "assessment_focus": ["法律实务"],
            "forbidden_topics": [],
            "confidence": 0.92,
        },
    )
    assert legacy["guard_mode"] == "legacy"
    result = audit_questions(
        [_audit_question(1, "你如何搭建用户增长模型？")], legacy
    )
    assert len(result["dropped"]) == 1
    assert result["status"] == "disabled"
    assert result["audit"][0]["stage"] == "legacy"


def test_audit_legacy_fallback_never_drops():
    from app.guard import audit_questions
    from app.role import build_role_profile

    fallback = build_role_profile("某岗位", "简历", {"confidence": 0.1})
    assert fallback["guard_mode"] == "legacy"
    result = audit_questions([_audit_question(1, "你如何搭建用户增长模型？")], fallback)
    assert result["dropped"] == []
    assert result["audit"][0]["verdict"] == "keep"


def test_arbiter_budget_hard_cap_one_call():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    arbiter = _guard_arbiter({"judgements": [{"qid": 1, "verdict": "uncertain"}]})

    first = audit_questions([_audit_question(1, "聊聊你最近读的一本书")], profile, arbiter=arbiter)
    assert arbiter.calls == 1 and first["arbiter_calls"] == 1

    second = audit_questions([_audit_question(1, "聊聊你的一个爱好")], profile, arbiter=arbiter)
    assert arbiter.calls == 1  # 预算耗尽，不再调用
    assert second["status"] == "degraded"
    assert second["audit"][0]["raw"]["source"] == "budget_exhausted"
    assert "预算" in second["audit"][0]["reason"]


def test_judgement_cache_reuses_without_llm():
    from app.guard import JudgementCache, audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    cache = JudgementCache()
    calls = {"count": 0}

    def judge_fn(messages):
        calls["count"] += 1
        return json.dumps(
            {"judgements": [{"qid": 1, "verdict": "keep", "reason": "通用经历"}]},
            ensure_ascii=False,
        )

    from app.guard import BatchArbiter

    first_arbiter = BatchArbiter(cache=cache, judge_fn=judge_fn)
    questions = [_audit_question(1, "聊聊你最近读的一本书")]
    audit_questions(questions, profile, arbiter=first_arbiter)
    assert calls["count"] == 1

    second_arbiter = BatchArbiter(cache=cache, judge_fn=judge_fn)
    result = audit_questions(questions, profile, arbiter=second_arbiter)
    assert calls["count"] == 1 and second_arbiter.calls == 0  # 命中缓存，不调用 LLM
    assert result["audit"][0]["raw"]["source"] == "cache"


def test_audit_follow_up_uses_strong_signal_only():
    from app.guard import audit_follow_up
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    keep = audit_follow_up("那你如何组织证据链？", profile)
    assert keep["verdict"] == "keep" and keep["hits"] == []

    drop = audit_follow_up("你能讲讲模型训练的流程吗？", profile)
    assert drop["verdict"] == "drop" and "模型训练" in drop["hits"]

    # 弱白名单与兜底画像不做硬丢弃
    weak = build_role_profile(
        "诉讼律师",
        "简历",
        _focus_payload(
            assessment_focus=[
                {"key": "f1", "name": "法律实务", "subtopics": ["法律", "合同", "合规"], "why_relevant": "", "source_jd_ref": ""}
            ]
        ),
    )
    assert audit_follow_up("你能讲讲模型训练的流程吗？", weak)["verdict"] == "keep"


def test_audit_summary_fields_match_report_contract():
    from app.guard import audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    arbiter = _guard_arbiter(
        {
            "judgements": [
                {
                    "qid": 1,
                    "verdict": "out_of_scope",
                    "reason": "医疗职能域",
                    "evidence": "护理岗位",
                }
            ]
        }
    )
    result = audit_questions(
        [
            _audit_question(1, "请讲讲你在护理岗位上评估患者疼痛的流程"),
            _audit_question(2, "聊聊你最近读的一本书"),
        ],
        profile,
        arbiter=arbiter,
    )
    summary = result["summary"]
    for key in (
        "guard_audit_status",
        "guard_uncertain_count",
        "guard_dropped_count",
        "guard_dropped_samples",
        "weak_focus_count",
        "guard_mode",
        "evidence_mismatch_count",
    ):
        assert key in summary, f"summary 缺少 {key}"
    assert summary["guard_audit_status"] == result["status"]
    assert summary["guard_dropped_count"] == len(result["dropped"])
    assert len(summary["guard_dropped_samples"]) <= 5


def test_arbiter_prompt_renders_whitelist_once():
    from app.guard import build_arbiter_messages
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    messages = build_arbiter_messages(
        profile, [_audit_question(1, "合同主体资格审查要注意什么")]
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "只输出 JSON" in system
    assert "拿不准一律 keep" in system
    assert "不得改写" in system
    assert "[f1] 合同审查实务" in user
    # subtopics 只在白名单块里渲染一次（题面里出现的那次不算重复渲染）
    whitelist_block = user.split("待判定题目：")[0]
    assert whitelist_block.count("合同主体资格审查") == 1


# ---------------------------------------------------------------------------
# 3.2.3：plan.py / ask.py 接入（含 3.2.0 的生成期绑定与来源约束）
# ---------------------------------------------------------------------------


def _plan_state_with_profile(profile, requirement="独立完成合同审核", jd_text="诉讼律师", resume_text="三年诉讼经验"):
    return {
        "jd_text": jd_text,
        "resume_text": resume_text,
        "job_title": str(profile.get("job_title", "") or ""),
        "role_profile": profile,
        "gap_report": {"missing_skills": [], "weak_skills": []},
        "jd_profile": {"requirements": [{"text": requirement}]},
        "resume_profile": {"projects": [], "concerns": []},
        "job_research": {"raw_notes": ""},
        "question_plan": [],
    }


def _plan_question(qid, content, *, focus_key="", jd_ref="", skills=None, question_type="scenario", category="scenario"):
    return {
        "id": qid,
        "category": category,
        "question_type": question_type,
        "difficulty": "medium",
        "depth_level": "application",
        "content": content,
        "skills": list(skills or []),
        "focus_key": focus_key,
        "jd_ref": jd_ref,
        "project_ref": "",
        "research_ref": "",
        "intent": "",
        "source_id": f"llm:{qid}",
        "source_type": "llm",
    }


def test_plan_prompt_renders_whitelist_and_sources():
    from app.prompts.templates import PLAN_PROMPT
    from app.role import MOTIVATION_QUOTA, build_role_profile, render_focus_whitelist

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    text = PLAN_PROMPT.format(
        job_title="诉讼律师",
        experience_level="junior",
        target=15,
        min_q=12,
        max_q=18,
        motivation_quota=MOTIVATION_QUOTA,
        focus_whitelist=render_focus_whitelist(profile),
        source_ids="research:job:1、research:job:2、qb:9",
        jd_ref_range="0..2",
        project_ref_range="0..1",
        gap_report="{}",
        jd_summary="JD 摘要",
        jd_requirements="[0] 要求",
        resume_projects="[0] 项目",
        role_policy="岗位政策",
        references="参考资料",
    )
    assert "[f1] 合同审查实务" in text
    assert "research:job:1" in text
    assert "0..2" in text and "0..1" in text
    assert "focus_key" in text and "research_ref" in text
    assert "不要为了填字段而啰嗦" in text


def test_normalize_plan_focus_key_and_research_ref():
    import app.nodes.plan as np
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    questions = np._normalize_plan(
        [
            {
                "id": 1,
                "category": "scenario",
                "content": "合同主体资格审查要点？",
                "skills": ["合同审查"],
                "focus_key": " F1 ",
                "research_ref": "  research:job:2 ",
            },
            {
                "id": 2,
                "category": "scenario",
                "content": "另一道题",
                "skills": [],
                "focus_key": "f99",
                "research_ref": "web:不存在",
            },
        ],
        [],
        profile,
    )
    assert questions[0]["focus_key"] == "f1"
    assert questions[0]["research_ref"] == "research:job:2"
    # 非法值原样保留，由审计给出 binding_status / evidence_status
    assert questions[1]["focus_key"] == "f99"
    assert questions[1]["research_ref"] == "web:不存在"


def test_plan_uses_guard_audit_for_legacy_profile(monkeypatch):
    import app.nodes.plan as np
    from app.role import build_role_profile

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "domain": "法律-诉讼",
            "industry": "法律服务",
            "assessment_focus": ["法律实务"],
            "forbidden_topics": [],
            "confidence": 0.92,
        },
    )
    questions = [
        _plan_question(1, "你如何搭建用户增长模型？"),
        _plan_question(2, "请说明合同审查的重点步骤？", skills=["合同审查"]),
    ]
    monkeypatch.setattr(np, "chat_with_usage", _plan_chat_stub(questions))
    result = np.plan(_plan_state_with_profile(legacy, requirement="合同审查"))
    contents = " ".join(str(item["content"]) for item in result["question_plan"])
    assert "用户增长模型" not in contents  # legacy：4 组命中即丢（单票）
    assert result["guard_dropped"]
    assert result["guard_dropped"][0]["hits"]
    assert "合同审查" in contents


def test_plan_audit_uses_whitelist_pipeline(monkeypatch):
    import app.nodes.plan as np
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    questions = [
        _plan_question(1, "合同主体资格审查的要点？", focus_key="f1", jd_ref="0", skills=["合同审查"]),
        _plan_question(2, "请写一段代码实现模型训练流水线", skills=["模型训练"]),
    ]
    monkeypatch.setattr(np, "chat_with_usage", _plan_chat_stub(questions))
    result = np.plan(_plan_state_with_profile(profile))
    contents = " ".join(str(item["content"]) for item in result["question_plan"])
    assert "模型训练" not in contents  # D1+D2 两票 → 丢弃
    assert "合同主体资格审查" in contents
    assert result["guard_dropped"]


def test_plan_regeneration_requires_low_keep(monkeypatch):
    """丢弃率 >30% 但保留题数仍达标 → 不重生成（省一次调用）。"""
    import app.nodes.plan as np
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    questions = [
        _plan_question(1, "你为什么选择这个岗位？", question_type="behavioral"),
        _plan_question(2, "入职后你打算如何补齐能力？", question_type="behavioral"),
    ]
    on_domain = [
        ("合同主体资格审查的要点？", ["合同主体资格"]),
        ("违约责任与违约金如何认定？", ["违约责任"]),
        ("争议解决条款怎么写？", ["争议解决"]),
        ("举证期限与证据交换怎么安排？", ["举证期限"]),
        ("庭审质证要点有哪些？", ["庭审质证"]),
        ("管辖与送达怎么判断？", ["管辖送达"]),
        ("合规风险如何识别？", ["合规风险"]),
    ]
    for index, (content, skills) in enumerate(on_domain, 3):
        questions.append(_plan_question(index, content, focus_key="f1", jd_ref="0", skills=skills))
    # 跨域题要用不同技能名：_improve_quality 对同一技能最多保留 2 道
    cross_skills = ["模型训练", "编程开发", "护理实务", "模型训练优化"]
    for offset, index in enumerate(range(10, 14)):
        questions.append(
            _plan_question(
                index,
                f"请写一段代码实现模型训练流水线（第{index}版）",
                skills=[cross_skills[offset]],
            )
        )

    calls = {"count": 0}

    def fake_chat(messages, temperature=0.5, max_tokens=None):
        calls["count"] += 1
        return json.dumps({"questions": questions}, ensure_ascii=False), USAGE

    monkeypatch.setattr(np, "chat_with_usage", fake_chat)
    result = np.plan(_plan_state_with_profile(profile))
    kept = len(result["question_plan"])
    dropped = len(result["guard_dropped"])
    assert dropped == 4 and kept >= 9, f"kept={kept} dropped={dropped}"
    assert calls["count"] == 1, "保留题数达标时不应该重生成"


def test_plan_regeneration_triggers_when_keep_low(monkeypatch):
    """丢弃率高且保留题数不足目标×0.6 → 重生成一次。"""
    import app.nodes.plan as np
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    questions = [
        _plan_question(1, "合同主体资格审查的要点？", focus_key="f1", jd_ref="0", skills=["合同主体资格"]),
        _plan_question(2, "违约责任与违约金如何认定？", focus_key="f1", jd_ref="0", skills=["违约责任"]),
    ]
    for index in range(3, 11):
        questions.append(
            _plan_question(
                index,
                f"请写一段代码实现模型训练流水线（第{index}版）",
                skills=[f"模型训练{index}"],  # 技能名互不相同，避免被同技能上限截掉
            )
        )

    calls = {"count": 0}

    def fake_chat(messages, temperature=0.5, max_tokens=None):
        calls["count"] += 1
        return json.dumps({"questions": questions}, ensure_ascii=False), USAGE

    monkeypatch.setattr(np, "chat_with_usage", fake_chat)
    result = np.plan(_plan_state_with_profile(profile))
    assert calls["count"] == 2, "保留题数不足时应重生成一次（共享预算）"
    assert len(result["guard_dropped"]) == 16  # 两轮各 8 条


def test_plan_llm_call_budget_hard_cap(monkeypatch):
    """调用预算 2：截断重试用掉 2 次后，重生成不再调用 LLM，直接落地兜底题。"""
    import app.nodes.plan as np
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    questions = [
        _plan_question(
            index,
            f"请写一段代码实现模型训练流水线（第{index}版）",
            skills=[f"模型训练{index}"],
        )
        for index in range(1, 10)
    ]
    calls = {"count": 0}

    def truncated_chat(messages, temperature=0.5, max_tokens=None):
        calls["count"] += 1
        usage = {**USAGE, "finish_reason": "length"}
        return json.dumps({"questions": questions}, ensure_ascii=False), usage

    monkeypatch.setattr(np, "chat_with_usage", truncated_chat)
    result = np.plan(_plan_state_with_profile(profile))
    assert calls["count"] == 2, "单会话出题调用硬上限为 2"
    # 跨域题全部被丢弃后，题单只剩 _improve_quality 追加的通用动机题（豁免保留）
    assert result["question_plan"]
    assert all(item["question_type"] == "behavioral" for item in result["question_plan"])


def test_fallback_question_is_exempt_from_d1():
    from app.guard import audit_questions
    from app.nodes.plan import _fallback_question
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    result = audit_questions([_fallback_question("诉讼律师")], profile)
    record = result["audit"][0]
    assert record["verdict"] == "keep" and record["stage"] == "exempt"
    assert result["dropped"] == []
    assert record["votes"]["D1"] is False


def test_research_job_fragments_ids(monkeypatch):
    import app.nodes.research_job as rj

    monkeypatch.setattr(
        rj,
        "web_search_handler",
        lambda args: {
            "success": True,
            "content": f"结果：{args.get('query', '')}",
            "source": "https://example.test",
        },
    )
    result = rj.research_job_node({"jd_text": "诉讼律师\n负责合同审查", "resume_text": "三年诉讼经验"})
    fragments = result["job_research"]["fragments"]
    assert [item["id"] for item in fragments] == [f"research:job:{i}" for i in range(1, 6)]
    assert all(item["content"] for item in fragments)
    assert result["job_research"]["raw_notes"]  # 兼容字段仍在


def test_plan_prompt_includes_research_fragment_ids(monkeypatch):
    import app.nodes.plan as np
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    captured = {}
    question = _plan_question(1, "合同主体资格审查的要点？", focus_key="f1", jd_ref="0", skills=["合同审查"])

    def fake_chat(messages, temperature=0.5, max_tokens=None):
        captured["prompt"] = " ".join(str(getattr(m, "content", m)) for m in messages)
        return json.dumps({"questions": [question]}, ensure_ascii=False), USAGE

    monkeypatch.setattr(np, "chat_with_usage", fake_chat)
    state = _plan_state_with_profile(profile)
    state["job_research"] = {
        "raw_notes": "旧笔记",
        "fragments": [
            {"id": f"research:job:{index}", "query": f"查询{index}", "content": f"片段{index}"}
            for index in range(1, 6)
        ],
    }
    np.plan(state)
    prompt = captured["prompt"]
    assert "research:job:1" in prompt and "research:job:5" in prompt


def test_ask_follow_up_uses_audit(monkeypatch):
    import app.nodes.ask as na
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    sequence = iter(["你能讲讲模型训练的流程吗？", "那你如何组织证据链与举证顺序？"])
    monkeypatch.setattr(
        na,
        "chat_with_usage",
        lambda messages, temperature=0.4, max_tokens=None: (next(sequence), USAGE),
    )
    state = {
        "jd_text": "诉讼律师",
        "resume_text": "简历",
        "job_title": "诉讼律师",
        "role_profile": profile,
        "question_plan": [_plan_question(1, "合同主体资格审查的要点？", focus_key="f1", jd_ref="0")],
        "current_question_index": 0,
        "assessments": [
            {
                "question_id": 1,
                "is_follow_up": False,
                "score": 6,
                "hint": "",
                "follow_up_reason": "证据组织讲得太浅",
                "is_relevant": True,
                "covered_aspects": [],
                "question": "合同主体资格审查的要点？",
            }
        ],
        "conversation_history": [{"role": "assistant", "content": "合同主体资格审查的要点？"}],
        "difficulty": "medium",
        "follow_up_count": 0,
        "max_follow_ups": 3,
    }
    out = na.ask_follow_up(state)
    assert out["current_question"] == "那你如何组织证据链与举证顺序？"


def test_ask_follow_up_escalates_on_second_hit(monkeypatch):
    import app.nodes.ask as na
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())
    monkeypatch.setattr(
        na,
        "chat_with_usage",
        lambda messages, temperature=0.4, max_tokens=None: ("模型训练的具体流程是什么？", USAGE),
    )
    state = {
        "jd_text": "诉讼律师",
        "resume_text": "简历",
        "job_title": "诉讼律师",
        "role_profile": profile,
        "question_plan": [_plan_question(1, "合同主体资格审查的要点？", focus_key="f1", jd_ref="0")],
        "current_question_index": 0,
        "assessments": [
            {
                "question_id": 1,
                "is_follow_up": False,
                "score": 6,
                "hint": "",
                "follow_up_reason": "细节不足",
                "is_relevant": True,
                "covered_aspects": [],
                "question": "合同主体资格审查的要点？",
            }
        ],
        "conversation_history": [{"role": "assistant", "content": "合同主体资格审查的要点？"}],
        "difficulty": "medium",
        "follow_up_count": 0,
        "max_follow_ups": 3,
    }
    out = na.ask_follow_up(state)
    assert out["current_question"] == "这道题我们先聊到这里，接下来换下一个话题。"
    assert out["follow_up_count"] == 3


def test_judgement_cache_key_includes_out_of_scope():
    from app.guard import JudgementCache
    from app.role import build_role_profile

    first = build_role_profile("诉讼律师", "简历", _focus_payload())
    second = build_role_profile(
        "诉讼律师",
        "简历",
        _focus_payload(
            out_of_scope_topics=["其他跨域话题"],
            forbidden_topics=["其他跨域话题"],
        ),
    )
    assert JudgementCache.key(first, "同一道题") != JudgementCache.key(second, "同一道题")
    assert JudgementCache.key(first, "同一道题") == JudgementCache.key(first, "同一道题")


# ---------------------------------------------------------------------------
# 3.2.4：状态与报告字段（guard_audit / guard_summary / token_by_node）
# ---------------------------------------------------------------------------


def test_guard_summary_maps_to_report_fields():
    from app.nodes.evaluate import evaluate

    state = _base_state(
        assessments=[{"question_id": 0, "score": 7, "question": "q"}],
        usage_records=[{"node": "assess", "total_tokens": 4}],
        conversation_history=[
            {"role": "assistant", "content": "question?"},
            {"role": "user", "content": "answer"},
        ],
        guard_summary={
            "guard_audit_status": "degraded",
            "guard_uncertain_count": 2,
            "guard_dropped_count": 3,
            "guard_dropped_samples": ["跨域题一", "跨域题二", "跨域题三"],
            "weak_focus_count": 1,
            "evidence_mismatch_count": 4,
            "guard_degrade_reason": "invalid_subtopics",
            "guard_domain_slug_missing": True,
        },
    )
    report = evaluate(state)["evaluation_report"]
    assert report["guard_audit_status"] == "degraded"
    assert report["guard_uncertain_count"] == 2
    assert report["guard_dropped_count"] == 3
    assert report["guard_dropped_samples"] == ["跨域题一", "跨域题二", "跨域题三"]
    assert report["weak_focus_count"] == 1
    assert report["guard_evidence_mismatch_count"] == 4
    assert report["guard_degrade_reason"] == "invalid_subtopics"
    assert report["guard_domain_slug_missing"] is True


def test_guard_dropped_count_dedup_by_text():
    from app.guard import summarize_audit

    dropped = [
        {"content": "请写一段代码实现模型训练流水线"},
        {"content": "请写一段代码实现模型训练流水线"},      # 重生成后的同一道题
        {"content": "  请写一段代码实现模型训练流水线   "},  # 前后空白视为同一道
        {"content": "另一道跨域题"},
    ]
    summary = summarize_audit([], dropped)
    assert summary["guard_dropped_count"] == 2
    assert summary["guard_dropped_samples"] == [
        "请写一段代码实现模型训练流水线",
        "另一道跨域题",
    ]
    assert len(summary["guard_dropped_samples"]) <= 5


def test_token_by_node_aggregates_by_node():
    from app.nodes.evaluate import _token_by_node

    records = [
        {"node": "ask", "total_tokens": 10},
        {"node": "ask", "total_tokens": 5},
        {"node": "plan_questions", "total_tokens": 100},
        {"node": "plan_questions_retry", "total_tokens": 80},
    ]
    totals = _token_by_node(records)
    assert totals["ask"] == 15                      # 同节点多条求和
    assert totals["plan_questions"] == 100          # 初版与重生成分开计
    assert totals["plan_questions_retry"] == 80
    assert _token_by_node([]) == {}                 # 空 usage_records → 空 dict


def test_token_by_node_skips_missing_nodes():
    from app.nodes.evaluate import _token_by_node

    totals = _token_by_node([{"node": "ask", "total_tokens": 7}])
    assert "compress" not in totals        # 本会话没触发压缩 → 不补 0
    assert "guard_arbiter" not in totals   # 没走仲裁 → 不补 0
    assert all(value > 0 for value in totals.values())


def test_guard_audit_raw_truncated_in_state():
    import app.nodes.plan as np

    long_raw = {"reason": "x" * 5000, "evidence": "y" * 1000}
    short_raw = {"reason": "ok"}
    trimmed = np._truncate_audit_records(
        [
            {"question_id": 1, "raw": long_raw},
            {"question_id": 2, "raw": short_raw},
            {"question_id": 3},
        ]
    )
    assert trimmed[0]["raw_truncated"] is True
    assert len(trimmed[0]["raw"]["text"]) <= np.GUARD_RAW_MAX_CHARS
    assert trimmed[1]["raw"] == short_raw
    assert "raw" not in trimmed[2]


def test_guard_arbiter_usage_accounting(monkeypatch):
    """仲裁记账：注入 judge_fn / 预算耗尽都不产生记录；真实调用产生 guard_arbiter 记录。"""
    import app.guard as guard
    from app.guard import BatchArbiter, audit_questions
    from app.role import build_role_profile

    profile = build_role_profile("诉讼律师", "简历", _focus_payload())

    injected = _guard_arbiter({"judgements": [{"qid": 1, "verdict": "keep"}]})
    audit_questions([_audit_question(1, "聊聊你最近读的一本书")], profile, arbiter=injected)
    assert injected.usage_records == []

    monkeypatch.setattr(
        guard,
        "chat_with_usage",
        lambda messages, temperature=0.1: (
            '{"judgements":[{"qid":1,"verdict":"keep"}]}',
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    )
    real = BatchArbiter()
    audit_questions([_audit_question(1, "聊聊你最近读的一本书")], profile, arbiter=real)
    assert real.usage_records == [
        {"node": "guard_arbiter", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    ]

    exhausted = BatchArbiter(max_calls=0)
    audit_questions([_audit_question(1, "聊聊你的一个爱好")], profile, arbiter=exhausted)
    assert exhausted.usage_records == []


# ---------------------------------------------------------------------------
# 3.2.5：评测集夹具与守卫的一致性（offline，零 LLM 调用）
# ---------------------------------------------------------------------------


def test_guard_cases_fixture_offline_consistency():
    """cases.json 的期望值必须与"不需要仲裁即可判定"的守卫行为完全一致。

    说明：仲裁相关的 case（CD-A 主体）在 offline 下会被标记 pending_live 并排除在指标外，
    由 3.2.5 的 live 模式另行验证。
    """
    import importlib
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    eval_guard = importlib.import_module("guard_benchmark")

    class _Args:
        mode = "offline"
        only = ""
        limit = 0
        cache_file = ""

    payload = eval_guard.evaluate_cases(_Args())
    metrics = eval_guard._metrics(payload["results"])

    assert metrics["judgeable"] > 0, "offline 可判定的 case 不应为 0"
    assert metrics["false_drop"] == [], "不允许误删合法题"
    assert metrics["missed"] == [], "不允许漏放跨域题（offline 可判定范围内）"
    assert metrics["stage_hit_rate"] == 1.0, "expected_stage 必须与实际机制一致"
    assert payload["arbiter_calls"] == 0, "offline 模式不得调用仲裁"
    # 夹具规模契约：23 份画像 + 46 同域 + 28 CD-A（含 5 条镜像）+ 23 CD-B + 6 易混淆
    assert len(payload["roles"]) == 23
    counts: dict[str, int] = {}
    for item in payload["results"]:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert counts["same_domain"] == 46
    assert counts["cross_domain_a"] == 28
    assert counts["cross_domain_b"] == 23
    assert counts["confusable"] == 6

    # 夹具里的 23 份画像必须全部合规：白名单模式 + out_of_scope 3-8 条复合词
    import json as _json
    from pathlib import Path as _Path2

    from app.role import build_role_profile, profile_quality

    fixture = _json.loads(
        (_Path2(__file__).resolve().parent / "fixtures" / "guard" / "cases.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(fixture["profiles"]) == 23
    for domain, payload in fixture["profiles"].items():
        profile = build_role_profile("", "", payload)
        quality = profile_quality(profile)
        assert quality["guard_mode"] == "whitelist", domain
        assert quality["out_of_scope_ok"] is True, domain
        assert 3 <= quality["out_of_scope_count"] <= 8, domain
        assert quality["invalid_out_of_scope"] == [], domain
        assert quality["weak_focus_count"] == 0, domain
        assert quality["invalid_subtopics"] == {}, domain
        # 收尾 4：每份画像必须带合法的 domain_slug（规范化岗位标识），且无降级成因
        assert quality["domain_slug"] and quality["domain_slug_source"] == "llm", domain
        assert quality["domain_slug_missing"] is False, domain
        assert quality["degrade_reason"] == "", domain


# ---------------------------------------------------------------------------
# 收尾 3：out_of_scope 必填 + 降级 + 画像 prompt 强化
# ---------------------------------------------------------------------------


def test_out_of_scope_missing_degrades_to_weak():
    """out_of_scope 缺失（必填但 LLM 没给）→ 降级 weak_whitelist：不硬丢弃、不退回 legacy。"""
    from app.guard import audit_questions
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(out_of_scope_topics=[], forbidden_topics=[])
    profile = build_role_profile("诉讼律师", "简历", payload)
    assert profile["guard_mode"] == "weak_whitelist"

    quality = profile_quality(profile)
    assert quality["out_of_scope_source"] == "missing"
    assert quality["out_of_scope_ok"] is False
    assert quality["out_of_scope_count"] == 0

    result = audit_questions(
        [_audit_question(1, "请写一段代码实现模型训练流水线", skills=["模型训练"])], profile
    )
    assert result["dropped"] == [], "弱白名单不做任何硬丢弃"
    assert result["status"] == "degraded"


def test_out_of_scope_short_terms_recorded_invalid():
    """out_of_scope 里的通用短词要被标记（只标记，不影响判定）。"""
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(
        out_of_scope_topics=["编程", "模型", "SQL"],
        forbidden_topics=["编程", "模型", "SQL"],
    )
    profile = build_role_profile("诉讼律师", "简历", payload)
    quality = profile_quality(profile)
    assert quality["out_of_scope_count"] == 3
    assert quality["invalid_out_of_scope"] == ["编程", "模型"]  # SQL 是 3 字符，属复合词
    assert quality["out_of_scope_ok"] is True  # 来源仍是 LLM（不是镜像）


def test_role_profile_prompt_formats_with_new_rules():
    from app.prompts.templates import ROLE_PROFILE_PROMPT

    text = ROLE_PROFILE_PROMPT.format(jd="JD 内容", resume="简历内容")
    assert "该字段不得为空" in text
    assert "必须明确写出 out_of_scope_topics" in text
    assert "白名单互不重叠" in text
    assert "同一行业的不同职能域" in text


def test_role_profile_prompt_examples_use_composite_out_of_scope():
    """示例本身就是范本：示例区块里不得出现裸的通用短词。

    例外说明：要求 5/6 的说明文字会以"（例如 编程、检索、对话、开发、接口、数据库）"的形式
    列举**反例**——那是必须保留的说明，因此本测试只检查"示例"之后的区块。
    """
    import re

    from app.prompts.templates import ROLE_PROFILE_PROMPT

    examples = ROLE_PROFILE_PROMPT.split("示例 1（")[1]
    allowed_contexts = (
        "编程开发",
        "模型训练",
        "模型预训练",
        "向量检索",
        "法律检索",
        "检索链路",
        "数据库检索",
        "检索增强",
    )
    for word in ("编程", "模型", "检索"):
        for match in re.finditer(word, examples):
            window = examples[max(0, match.start() - 3) : match.end() + 8]
            assert any(ctx in window for ctx in allowed_contexts), (
                f"示例里出现裸词「{word}」：…{window}…"
            )


# ---------------------------------------------------------------------------
# 收尾 4：降级成因（guard_degrade_reason）与 domain_slug 完整链路
# ---------------------------------------------------------------------------


def test_degrade_reason_invalid_subtopics():
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(
        assessment_focus=[
            {
                "key": "f1",
                "name": "法律实务",
                "subtopics": ["法律", "合同", "合规"],
                "why_relevant": "",
                "source_jd_ref": "",
            }
        ]
    )
    quality = profile_quality(build_role_profile("诉讼律师", "简历", payload))
    assert quality["guard_mode"] == "weak_whitelist"
    assert quality["degrade_reason"] == "invalid_subtopics"


def test_degrade_reason_out_of_scope_missing():
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(out_of_scope_topics=[], forbidden_topics=[])
    quality = profile_quality(build_role_profile("诉讼律师", "简历", payload))
    assert quality["guard_mode"] == "weak_whitelist"
    assert quality["degrade_reason"] == "out_of_scope_missing"


def test_degrade_reason_both():
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(
        assessment_focus=[
            {
                "key": "f1",
                "name": "法律实务",
                "subtopics": ["法律", "合同", "合规"],
                "why_relevant": "",
                "source_jd_ref": "",
            }
        ],
        out_of_scope_topics=[],
        forbidden_topics=[],
    )
    quality = profile_quality(build_role_profile("诉讼律师", "简历", payload))
    assert quality["degrade_reason"] == "invalid_subtopics+out_of_scope_missing"


def test_degrade_reason_fallback_legacy_and_normal():
    from app.role import build_role_profile, profile_quality

    fallback = build_role_profile("某岗位", "简历", {"confidence": 0.1})
    assert profile_quality(fallback)["degrade_reason"] == "fallback_profile"

    legacy = build_role_profile(
        "诉讼律师",
        "简历",
        {
            "job_title": "诉讼律师",
            "assessment_focus": ["法律实务"],
            "forbidden_topics": [],
            "confidence": 0.9,
        },
    )
    assert profile_quality(legacy)["degrade_reason"] == "legacy_profile"

    normal = build_role_profile("诉讼律师", "简历", _focus_payload())
    assert profile_quality(normal)["degrade_reason"] == ""


def test_domain_slug_generated_from_profile():
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload(domain="技术-后端开发", domain_slug="tech-backend")
    profile = build_role_profile("Java 后端", "简历", payload)
    assert profile["domain_slug"] == "tech-backend"
    quality = profile_quality(profile)
    assert quality["domain_slug_source"] == "llm"
    assert quality["domain_slug_missing"] is False
    assert quality["degrade_reason"] == ""


def test_domain_slug_validation():
    from app.role import build_role_profile, profile_quality

    for bad in ("tech backend", "技术-后端", "tech_backend", "ab", "x" * 31, "Tech/Backend"):
        profile = build_role_profile("某岗位", "简历", _focus_payload(domain_slug=bad))
        quality = profile_quality(profile)
        assert profile["domain_slug"] == "", f"非法 slug 应被拒绝：{bad}"
        assert quality["domain_slug_source"] == "invalid"
        assert quality["domain_slug_missing"] is True

    # 大小写与首尾空白会被归一化，不算非法
    normalized = build_role_profile("某岗位", "简历", _focus_payload(domain_slug="  Tech-Backend "))
    assert normalized["domain_slug"] == "tech-backend"


def test_domain_slug_missing_fallback():
    from app.guard import audit_questions
    from app.role import build_role_profile, profile_quality

    payload = _focus_payload()
    payload.pop("domain_slug", None)
    profile = build_role_profile("诉讼律师", "简历", payload)
    assert profile["domain_slug"] == ""
    quality = profile_quality(profile)
    assert quality["domain_slug_source"] == "missing"
    assert quality["domain_slug_missing"] is True

    result = audit_questions(
        [_audit_question(1, "合同主体资格审查的要点？", focus_key="f1", jd_ref="0")], profile
    )
    assert result["summary"]["guard_domain_slug_missing"] is True
    assert result["summary"]["guard_degrade_reason"] == ""


# ---------------------------------------------------------------------------
# 模块 6：跨职能域回归测试（非技术岗不出技术题、技术岗不串行业题）
# ---------------------------------------------------------------------------


def _plan_chat_stub(questions):
    """出题节点的桩：无论提示词是什么都返回给定题单。"""
    def fake(messages, temperature=0.5, max_tokens=None):
        return json.dumps({"questions": questions}, ensure_ascii=False), USAGE

    return fake


def _domain_plan_state(jd_text, resume_text, profile_payload, requirement):
    """构造一个带"LLM 画像"的 plan 入参（画像走真实的校验 + 守卫）。"""
    from app.role import build_role_profile

    profile = build_role_profile(jd_text, resume_text, profile_payload)
    assert profile["profile_source"] == "llm", "回归测试需要 LLM 画像，守卫才会生效"
    return {
        "jd_text": jd_text,
        "resume_text": resume_text,
        "job_title": profile["job_title"],
        "role_profile": profile,
        "gap_report": {"missing_skills": [], "weak_skills": []},
        "jd_profile": {"requirements": [{"text": requirement}]},
        "resume_profile": {"projects": [], "concerns": []},
        "job_research": {"raw_notes": ""},
        "question_plan": [],
    }


def _question(question_id, content, skills, category="scenario"):
    return {
        "id": question_id,
        "category": category,
        "difficulty": "medium",
        "depth_level": "application",
        "project_ref": "",
        "jd_ref": "0",
        "intent": "考察点",
        "content": content,
        "skills": skills,
        "source_id": f"llm:{question_id}",
        "source_type": "llm",
    }


def _assert_no_forbidden_topics(questions, profile):
    from app.role import guard_forbidden_topics, text_hits_topics

    forbidden = guard_forbidden_topics(profile)
    for question in questions:
        blob = " ".join(
            [
                str(question.get("content", "")),
                str(question.get("intent", "")),
                " ".join(str(item) for item in (question.get("skills") or [])),
            ]
        )
        hits = text_hits_topics(blob, forbidden)
        assert not hits, f"题单里出现了禁止话题 {hits}：{blob}"


def test_non_tech_jd_no_cross_domain(monkeypatch):
    """律师 / 运营 / 护士三类非技术岗：跨职能域题目必须被守卫丢弃。"""
    import app.nodes.plan as np

    cases = [
        {
            "jd": "诉讼律师：负责民商事诉讼与合同审查",
            "resume": "三年诉讼经验，负责合同审查与证据组织",
            "requirement": "合同审查",
            "profile": {
                "job_title": "诉讼律师",
                "domain": "法律-诉讼",
                "industry": "法律服务",
                "core_responsibilities": ["案件梳理", "文书起草"],
                "key_skills": ["民商法", "证据规则"],
                "assessment_focus": ["法律实务", "案例分析", "合规意识"],
                "forbidden_topics": [],
                "confidence": 0.92,
            },
            "questions": [
                _question(1, "请说明你审核合同时的重点步骤？", ["合同审查"]),
                _question(2, "写一段 Python 代码，实现法律文书的批量解析？", ["Python"]),
                _question(3, "你会用大模型来做合同审查吗？", ["大模型"]),
            ],
            "must_drop": ["Python", "大模型"],
        },
        {
            "jd": "新媒体运营：负责抖音、小红书账号内容运营",
            "resume": "两年内容运营经验，负责选题策划与数据复盘",
            "requirement": "内容运营",
            "profile": {
                "job_title": "新媒体运营",
                "domain": "互联网-用户运营",
                "industry": "互联网",
                "core_responsibilities": ["内容选题与生产", "账号数据复盘"],
                "key_skills": ["选题策划", "文案撰写"],
                "assessment_focus": ["内容运营", "选题策划", "数据分析"],
                "forbidden_topics": [],
                "confidence": 0.9,
            },
            "questions": [
                _question(1, "你会如何为一款新品做内容选题策划？", ["选题策划"]),
                _question(2, "写一段 Python 代码实现 LRU 缓存。", ["Python"]),
                _question(3, "护理中如何评估患者的疼痛程度？", ["护理"]),
            ],
            "must_drop": ["Python", "护理"],
        },
        {
            "jd": "临床护士：负责病房护理与医嘱执行",
            "resume": "三年三甲医院护理经验，负责病房巡视与护理文书",
            "requirement": "护理流程",
            "profile": {
                "job_title": "临床护士",
                "domain": "医疗-临床护理",
                "industry": "医疗健康",
                "core_responsibilities": ["病房巡视", "医嘱执行"],
                "key_skills": ["基础护理操作", "病情观察"],
                "assessment_focus": ["护理流程", "临床判断", "医患沟通"],
                "forbidden_topics": [],
                "confidence": 0.93,
            },
            "questions": [
                _question(1, "夜班时患者生命体征异常，你的处理顺序是什么？", ["临床判断"]),
                _question(2, "合同违约的诉讼流程一般怎么走？", ["法律实务"]),
                _question(3, "写一段 SQL 统计接口的日活数据？", ["SQL"]),
            ],
            "must_drop": ["诉讼流程", "SQL"],
        },
    ]

    for case in cases:
        monkeypatch.setattr(np, "chat_with_usage", _plan_chat_stub(case["questions"]))
        state = _domain_plan_state(
            case["jd"], case["resume"], case["profile"], case["requirement"]
        )
        result = np.plan(state)
        questions = result["question_plan"]
        profile = result["role_profile"]

        # 守卫确实生效并留下了记录（避免"因为没出题所以通过"的假阳性）
        assert result["guard_dropped"], f"{case['jd']}：应该有跨域题被丢弃"
        dropped_topics = {
            topic for item in result["guard_dropped"] for topic in item["hits"]
        }
        for expected in case["must_drop"]:
            assert expected in dropped_topics, f"{case['jd']}：{expected} 未被识别"

        # 最终题单里不得出现任何禁止话题的词
        _assert_no_forbidden_topics(questions, profile)
        # 同域题与通用追问都还在，题单没有被删空
        assert questions, f"{case['jd']}：题单不应为空"


def test_ai_jd_allows_ai_topics(monkeypatch):
    """AI 应用工程师：RAG / Agent 属于本岗位考察重心，必须保留。"""
    import app.nodes.plan as np

    questions = [
        _question(1, "介绍一下你搭建的 RAG 检索链路？", ["RAG"], category="project"),
        _question(2, "你的 Agent 工具调用是怎么编排和兜底的？", ["Agent"]),
        _question(3, "你如何评测提示词效果？", ["提示词工程"]),
    ]
    profile_payload = {
        "job_title": "AI 应用工程师",
        "domain": "技术-AI应用",
        "industry": "互联网",
        "core_responsibilities": ["RAG 检索链路开发与调优", "Agent 编排"],
        "key_skills": ["Python", "LangChain"],
        "assessment_focus": ["RAG 全流程", "Agent/工具调用", "提示词工程", "工程落地"],
        "forbidden_topics": [],
        "confidence": 0.9,
    }
    monkeypatch.setattr(np, "chat_with_usage", _plan_chat_stub(questions))
    state = _domain_plan_state(
        "AI 应用工程师：负责 RAG 与 Agent 应用落地",
        "三年 Python 开发，做过 RAG 问答系统",
        profile_payload,
        "RAG 检索链路",
    )

    result = np.plan(state)
    contents = " ".join(str(item["content"]) for item in result["question_plan"])
    assert "RAG" in contents and "Agent" in contents, "AI 岗位应保留 RAG/Agent 题目"
    assert result["guard_dropped"] == [], "AI 岗位不该丢弃自己的考察重心题目"


def test_backend_jd_no_medical(monkeypatch):
    """Java 后端：医疗类话题必须被丢弃，技术类话题必须保留。"""
    import app.nodes.plan as np

    questions = [
        _question(1, "请讲讲你在 Java 服务端做过的系统设计？", ["系统设计"]),
        _question(2, "护理中如何评估患者的疼痛程度？", ["护理"]),
        _question(3, "临床诊断的常见误区有哪些？", ["临床诊断"]),
    ]
    profile_payload = {
        "job_title": "Java 后端开发工程师",
        "domain": "技术-后端开发",
        "industry": "互联网",
        "core_responsibilities": ["服务端开发", "接口设计"],
        "key_skills": ["Java", "MySQL"],
        "assessment_focus": ["Java 并发与 JVM", "数据库与 SQL 优化", "系统设计"],
        "forbidden_topics": [],
        "confidence": 0.88,
    }
    monkeypatch.setattr(np, "chat_with_usage", _plan_chat_stub(questions))
    state = _domain_plan_state(
        "Java 后端开发工程师：负责服务端接口与数据库优化",
        "五年 Java 服务端开发经验",
        profile_payload,
        "系统设计",
    )

    result = np.plan(state)
    contents = " ".join(str(item["content"]) for item in result["question_plan"])
    assert "护理" not in contents and "临床" not in contents, "后端岗不该出现医疗题"
    assert "Java 服务端" in contents, "技术题应保留"

    dropped_topics = {topic for item in result["guard_dropped"] for topic in item["hits"]}
    # 加宽后自然表述（护理/患者/临床）同样能被拦住
    assert {"护理", "患者", "临床", "诊断"} <= dropped_topics


def test_stop_button_path_still_works_after_m2(monkeypatch, tmp_path):
    """M2 验证：前端"结束并评估"按钮走的是 stop_session，与关键词规则无关。

    对应调用链：ChatView.tsx -> client.stop() -> POST /sessions/{id}/stop
    -> service.stop_session() -> graph.update_state(end_requested=True)。
    """

    import app.nodes.research_job as rj
    from app.config import settings
    from app.service import InterviewService

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    rj.web_search_handler = lambda args: {
        "success": True,
        "content": "后端开发 高频面试：并发、数据库索引",
        "source": "https://example.test",
    }

    service = InterviewService(owner="unit_stop_" + uuid.uuid4().hex[:6])
    started = service.start_session("后端开发工程师 JD", "简历：Python 后端三年")
    assert started["question"]

    result = service.stop_session(started["session_id"])
    assert result["done"] is True
    assert result["report"] is not None
    assert result["report"]["grade"] in {"A", "B", "C", "D", "—"}


def test_m13_graph_replay_follow_ups_share_plan_index(monkeypatch, tmp_path):
    """M13 验证（图级重放）：强制前 3 轮追问，断言同一题的记录共享题单下标。

    这是"新代码是否真的写对"的端到端证据——单测只验证了函数，这里跑的是真实图。
    """

    import app.nodes.assess as ns
    import app.nodes.research_job as rj
    from app.config import settings
    from app.service import InterviewService

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    rj.web_search_handler = lambda args: {
        "success": True,
        "content": "后端开发 高频面试：并发、数据库索引、缓存",
        "source": "https://example.test",
    }

    calls = {"n": 0}
    # 每轮用**不同**的遗漏点：若沿用同一个 reason，edges 的同知识点去重
    # 会判定"重复追问同一知识点"并强制换题（这是系统既有设计，不是缺陷）
    reasons = ["流程没讲清", "缺少量化指标", "没有说明取舍"]

    def _assess_stub(messages, temperature=0.7, max_tokens=None):
        calls["n"] += 1
        if calls["n"] <= 3:  # 前三轮强制追问
            return (
                json.dumps(
                    {
                        # M18：档位 2 档 → 4.0 分（低分，触发追问）
                        # 覆盖三种岗位画像可能的所有维度 key，避免因 rubric 分支不同而落空
                        "dimension_levels": {
                            key: 2
                            for key in (
                                "law1", "law2", "law3", "law4",
                                "ai1", "ai2", "ai3", "ai4",
                                "job1", "job2", "job3", "job4", "m1", "m2",
                            )
                        },
                        "should_follow_up": True,
                        "missed_points": [reasons[calls["n"] - 1]],
                        "follow_up_reason": reasons[calls["n"] - 1],
                        "next_action": "follow_up",
                        "candidate_intent": "answer",
                    },
                    ensure_ascii=False,
                ),
                USAGE,
            )
        return (
            json.dumps(
                {
                    # M18：4 档 → 8.0 分（高分，换下一题）
                    "dimension_levels": {
                        key: 4
                        for key in (
                            "law1", "law2", "law3", "law4",
                            "ai1", "ai2", "ai3", "ai4",
                            "job1", "job2", "job3", "job4", "m1", "m2",
                        )
                    },
                    "should_follow_up": False,
                    "covered_points": ["讲清楚了"],
                    "next_action": "next_question",
                    "candidate_intent": "answer",
                },
                ensure_ascii=False,
            ),
            USAGE,
        )

    monkeypatch.setattr(ns, "chat_with_usage", _assess_stub)

    service = InterviewService(owner="unit_m13_" + uuid.uuid4().hex[:6])
    started = service.start_session("后端开发工程师 JD", "简历：Python 后端三年")
    session_id = started["session_id"]
    # This replay intentionally covers a pre-protocol checkpoint with legacy stubs.
    service.graph.update_state(service._config(session_id), {'assessment_protocol_version': 'legacy'})

    state = started["state"]
    for _ in range(4):
        outcome = service.submit_answer(session_id, "我负责把接口 P99 从 800ms 降到 200ms。")
        state = outcome["state"]
        if outcome.get("report"):
            break

    assessments = state.get("assessments") or []
    assert len(assessments) >= 4, f"应至少产生 1 主 + 3 追问，实际 {len(assessments)}"

    first_four = assessments[:4]
    assert [item.get("is_follow_up") for item in first_four] == [False, True, True, True]
    indices = [item.get("plan_question_index") for item in first_four]
    assert indices == [0, 0, 0, 0], f"同一题的追问必须共享下标，实际 {indices}"
    assert all(isinstance(i, int) and i >= 0 for i in indices)

    # 全量记录：下标必须单调不减，且追问不推进下标
    previous = -1
    for item in assessments:
        index = item.get("plan_question_index")
        assert isinstance(index, int) and index >= 0
        assert index >= previous, "题单下标不允许回退"
        previous = index
