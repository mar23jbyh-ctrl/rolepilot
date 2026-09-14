from __future__ import annotations

from app.graph.edges import recent_scores

DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def _adjust_difficulty(state) -> tuple[str, str]:
    scores = recent_scores(state.get("assessments", []) or [], 3)
    current = str(state.get("difficulty", "medium"))
    if len(scores) < 3:
        return current, "normal"
    average = sum(scores) / len(scores)
    if average >= 8.0:
        position = min(len(DIFFICULTY_ORDER) - 1, DIFFICULTY_ORDER.index(current) + 1)
        return DIFFICULTY_ORDER[position], "high"
    if average < 5.0:
        position = max(0, DIFFICULTY_ORDER.index(current) - 1)
        return DIFFICULTY_ORDER[position], "low"
    return current, "normal"


def advance(state) -> dict:
    plan = list(state.get("question_plan", []) or [])
    index = int(state.get("current_question_index", 0)) + 1
    # difficulty_events uses an append reducer: return only this node's delta.
    difficulty_events = []

    new_difficulty, _ = _adjust_difficulty(state)
    if new_difficulty != state.get("difficulty", "medium"):
        difficulty_events.append(
            {
                "question_index": index,
                "previous_difficulty": str(state.get("difficulty", "medium")),
                "new_difficulty": new_difficulty,
                "reason": "前3题平均分触发难度调整",
            }
        )

    result = {
        "current_question_index": index,
        "follow_up_count": 0,
        "difficulty": new_difficulty,
        "question_plan": plan,
    }
    if difficulty_events:
        result["difficulty_events"] = difficulty_events
    return result
