"""B2（M3 / M2）的验证用例：意图识别改用模型判断 + 结束规则收敛为显式命令。

不需要 LLM 的部分放在这里；需要完整图的"结束按钮"用例放在 test_nodes.py
（那里有全局的 LLM 打桩 fixture）。
"""

from __future__ import annotations

from app.graph.edges import route_after_assessment
from app.nodes.assess import CANDIDATE_INTENTS
from app.service import _is_explicit_stop_command


# ---------------- M2：显式结束命令 ----------------

def test_m2_explicit_commands_are_recognized():
    for text in ("结束面试", "结束面试。", " 结束面试 ", "不想面了", "就到这里", "先这样", "退出面试"):
        assert _is_explicit_stop_command(text) is True, text


def test_m2_historical_false_positives_are_not_commands():
    """实测：旧子串规则在这 6 条真实回答上 100% 误判，新规则必须全部放行。"""

    answers = [
        "Agent主要5个核心组件：1. LLM大模型大脑；2. 规划模块；3. 工具系统。",
        "面对步骤多、分支多的复杂任务，我会采用先规划再执行、每步反思校验的思路。",
        "立案受理 → 送达、举证答辩 → 开庭审理 → 合议裁判 → 送达判决书。",
        "训练后量化（PTQ）：模型训练完成之后再做量化，仅用少量校准数据统计数值范围。",
        "先核对医嘱，评估患者病情、过敏史、穿刺部位皮肤与血管条件。",
        "我会准备两个版本供律师选择：一旦信息公开，保密义务自动终止。",
    ]
    for text in answers:
        assert _is_explicit_stop_command(text) is False, text


def test_m2_long_text_containing_command_word_is_not_a_command():
    text = "结束面试之后我会复盘，但先把这个项目的技术选型讲完。"
    assert _is_explicit_stop_command(text) is False


def test_m2_empty_answer_is_not_a_command():
    assert _is_explicit_stop_command("") is False
    assert _is_explicit_stop_command("   ") is False


# ---------------- M3：模型意图驱动的路由 ----------------

def _state(**overrides) -> dict:
    state = {
        "assessments": [
            {
                "question_id": 1,
                "score": 6.0,
                "is_follow_up": False,
                "next_action": "next_question",
            }
        ],
        "current_question_index": 0,
        "question_plan": [{"id": 1}, {"id": 2}],
        "follow_up_count": 0,
        "max_follow_ups": 3,
    }
    state.update(overrides)
    return state


def test_m3_stop_suggested_routes_to_evaluate():
    state = _state()
    state["assessments"][0]["stop_suggested"] = True
    assert route_after_assessment(state) == "evaluate"


def test_m3_normal_answer_does_not_stop():
    assert route_after_assessment(_state()) == "advance"


def test_m3_clarification_does_not_trigger_explain_path():
    """要求重复题目走正常评分路径，不进入讲解分支。"""

    state = _state()
    state["assessments"][0]["candidate_intent"] = "request_clarification"
    assert route_after_assessment(state) == "advance"


def test_m3_intent_whitelist_covers_all_five_categories():
    assert set(CANDIDATE_INTENTS) == {
        "answer",
        "request_explanation",
        "request_clarification",
        "request_stop",
        "off_topic",
    }
