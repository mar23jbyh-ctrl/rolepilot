"""Web search tool wrapped with the LangChain @tool decorator."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for real job duties, interview questions and industry
    pain points. Returns cleaned snippets plus source URLs.

    The result must be stored in graph state (job_research), never appended
    directly to the message chain as an OpenAI-native object.
    """
    from app.tools.executor import web_search_handler

    outcome = web_search_handler({"query": query, "max_results": 6})
    if outcome is None:
        return "[web_search] no result"
    if not outcome.get("success"):
        return "[web_search] no result: " + str(outcome.get("content", ""))
    return str(outcome.get("content", ""))


def web_search_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": web_search.__doc__,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
