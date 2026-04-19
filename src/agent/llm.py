"""OpenAI chat helpers for JSON agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*", re.I)
_CODE_FENCE_END = re.compile(r"\s*```\s*$", re.I)


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    text = _CODE_FENCE.sub("", text)
    text = _CODE_FENCE_END.sub("", text)
    return json.loads(text)


def make_json_chat_fn(model: str, api_key: str | None):
    """Returns invoke_json(system_prompt, user_prompt) -> dict."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    def invoke_json(system: str, user: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=system + "\nAlways respond with a single JSON object."),
            HumanMessage(content=user),
        ]
        resp = llm.invoke(messages)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        try:
            return _parse_json_loose(raw)
        except json.JSONDecodeError:
            logger.warning("JSON repair retry for model output")
            repair = ChatOpenAI(model=model, temperature=0, api_key=api_key).invoke(
                [
                    SystemMessage(
                        content="Fix the following into a single valid JSON object only. No markdown."
                    ),
                    HumanMessage(content=raw[:120000]),
                ]
            )
            fixed = repair.content if isinstance(repair.content, str) else str(repair.content)
            return _parse_json_loose(fixed)

    return invoke_json


def make_structured_chat_fn(model: str, api_key: str | None, schema: type[TModel]):
    """
    Tool-style structured output validated against a Pydantic model.

    Uses function_calling (not json_schema strict mode): OpenAI requires additionalProperties:
    false on every object in strict schema mode, which our Pydantic models do not emit.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
    )
    structured = llm.with_structured_output(
        schema,
        method="function_calling",
    )

    def invoke_structured(system: str, user: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        result = structured.invoke(messages)
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json", exclude_none=False)
        if isinstance(result, dict):
            return schema.model_validate(result).model_dump(mode="json", exclude_none=False)
        raise TypeError(f"Unexpected structured output type: {type(result)}")

    return invoke_structured


def make_plain_chat_fn(model: str, api_key: str | None):
    """No JSON mode (plain chat for repair / fallback)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    def invoke(system: str, user: str) -> str:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return invoke
