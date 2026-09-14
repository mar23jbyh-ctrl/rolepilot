from __future__ import annotations

from app.models.schemas import (
    CodeExplainerArgs,
    DynamicQuestionArgs,
    WebSearchArgs,
)


def openai_tool_schema(model) -> dict:
    description = model.__doc__
    if not description or not str(description).strip():
        description = f"Parameters for the {model.__name__} tool."
    return {
        "type": "function",
        "function": {
            "name": model.__name__.removesuffix("Args"),
            "description": str(description).strip(),
            "parameters": model.model_json_schema(),
        },
    }


def get_tool_schemas() -> list[dict]:
    return [
        openai_tool_schema(WebSearchArgs),
        openai_tool_schema(CodeExplainerArgs),
        openai_tool_schema(DynamicQuestionArgs),
    ]


ARG_MODELS = {
    "WebSearch": WebSearchArgs,
    "CodeExplainer": CodeExplainerArgs,
    "DynamicQuestion": DynamicQuestionArgs,
}
