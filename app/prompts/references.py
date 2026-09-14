"""Format web-research excerpts for prompts; no local retrieval or indexing."""


def research_references_block(results: list[dict], limit: int = 1200) -> str:
    lines = []
    for index, item in enumerate(results[:8], 1):
        answer = str(item.get("answer", ""))[:limit]
        lines.append(
            f"[{index}] source_id={item.get('source_id')} "
            f"category={item.get('category')} difficulty={item.get('difficulty')}\n"
            f"Q: {item.get('question')}\nA: {answer}"
        )
    return "\n\n".join(lines)
