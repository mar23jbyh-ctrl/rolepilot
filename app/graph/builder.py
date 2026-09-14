from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph import edges
from app.graph.runtime import wrap_edge, wrap_node
from app.graph.state import InterviewState
from app.nodes import (
    advance,
    analyze,
    ask,
    assess,
    evaluate,
    maybe_compress,
    plan,
    research_job,
    self_check,
)


def build_graph(checkpointer=None):
    graph = StateGraph(InterviewState)
    graph.add_node("research_job", wrap_node(research_job.research_job_node, "research_job"))
    graph.add_node("analyze", wrap_node(analyze.analyze, "analyze"))
    graph.add_node("plan", wrap_node(plan.plan, "plan"))
    graph.add_node("ask", wrap_node(ask.ask, "ask"))
    graph.add_node("ask_follow_up", wrap_node(ask.ask_follow_up, "ask_follow_up"))
    graph.add_node("ask_explain", wrap_node(ask.ask_explain, "ask_explain"))
    graph.add_node("assess", wrap_node(assess.assess, "assess"))
    graph.add_node("maybe_compress", wrap_node(maybe_compress.maybe_compress, "maybe_compress"))
    graph.add_node("advance", wrap_node(advance.advance, "advance"))
    graph.add_node("evaluate", wrap_node(evaluate.evaluate, "evaluate"))
    graph.add_node("self_check", wrap_node(self_check.self_check, "self_check"))

    # 先从完整材料分析岗位，再用结构化岗位画像生成搜索查询。
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "research_job")
    graph.add_edge("research_job", "plan")
    graph.add_edge("plan", "ask")
    graph.add_conditional_edges(
        "ask",
        wrap_edge(edges.route_after_ask),
        {
            "assess": "assess",
            "evaluate": "evaluate",
        },
    )
    graph.add_conditional_edges("ask_follow_up", wrap_edge(edges.route_after_follow_up),
                                {"assess": "assess", "advance": "advance", "evaluate": "evaluate"})
    graph.add_edge("assess", "maybe_compress")

    graph.add_conditional_edges(
        "maybe_compress",
        wrap_edge(edges.route_after_assessment),
        {
            "ask_follow_up": "ask_follow_up",
            "ask_explain": "ask_explain",
            "advance": "advance",
            "evaluate": "evaluate",
        },
    )
    graph.add_conditional_edges(
        "ask_explain",
        wrap_edge(edges.after_explain),
        {
            "advance": "advance",
            "evaluate": "evaluate",
        },
    )
    graph.add_edge("advance", "ask")
    graph.add_edge("evaluate", "self_check")
    graph.add_edge("self_check", END)

    return graph.compile(interrupt_before=["assess"], checkpointer=checkpointer)
