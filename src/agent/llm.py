"""OpenAI chat helpers for JSON agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

# Module-level cache: (model, api_key, json_mode) → ChatOpenAI instance
_LLM_CACHE: dict[tuple, Any] = {}
# Module-level cache: (model, api_key, schema) → structured runnable
_STRUCTURED_CACHE: dict[tuple, Any] = {}


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip ``` or ```json fences robustly — handle trailing whitespace on fence lines
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.I)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.I)
    return json.loads(text.strip())


def _is_retryable(exc: Exception) -> bool:
    """True for transient OpenAI / LangChain API errors worth retrying."""
    name = type(exc).__name__
    module = getattr(type(exc), "__module__", "") or ""
    return (
        "RateLimitError" in name
        or "APIConnectionError" in name
        or "APITimeoutError" in name
        or "ServiceUnavailableError" in name
        or "InternalServerError" in name
        or ("openai" in module and "APIError" in name)
    )


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_with_retry(llm: Any, messages: list) -> Any:
    """Call llm.invoke with exponential back-off on transient API errors."""
    return llm.invoke(messages)


def _get_llm(model: str, api_key: str | None, json_mode: bool = False) -> Any:
    """Return a cached ChatOpenAI instance keyed by (model, api_key, json_mode)."""
    from langchain_openai import ChatOpenAI

    cache_key = (model, api_key, json_mode)
    if cache_key not in _LLM_CACHE:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        _LLM_CACHE[cache_key] = ChatOpenAI(model=model, temperature=0, api_key=api_key, **kwargs)
    return _LLM_CACHE[cache_key]


def _get_structured(model: str, api_key: str | None, schema: type) -> Any:
    """Return a cached structured-output runnable keyed by (model, api_key, schema)."""
    cache_key = (model, api_key, schema)
    if cache_key not in _STRUCTURED_CACHE:
        llm = _get_llm(model, api_key, json_mode=False)
        _STRUCTURED_CACHE[cache_key] = llm.with_structured_output(
            schema, method="function_calling"
        )
    return _STRUCTURED_CACHE[cache_key]


def make_json_chat_fn(model: str, api_key: str | None):
    """Returns invoke_json(system_prompt, user_prompt) -> dict."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm(model, api_key, json_mode=True)

    def invoke_json(system: str, user: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=system + "\nAlways respond with a single JSON object."),
            HumanMessage(content=user),
        ]
        resp = _invoke_with_retry(llm, messages)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        try:
            return _parse_json_loose(raw)
        except json.JSONDecodeError:
            logger.warning("JSON repair retry for model output")
            repair_messages = [
                SystemMessage(
                    content="Fix the following into a single valid JSON object only. No markdown."
                ),
                HumanMessage(content=raw[:120000]),
            ]
            fixed_resp = _invoke_with_retry(llm, repair_messages)
            fixed = fixed_resp.content if isinstance(fixed_resp.content, str) else str(fixed_resp.content)
            return _parse_json_loose(fixed)

    return invoke_json


def make_structured_chat_fn(model: str, api_key: str | None, schema: type[TModel]):
    """
    Tool-style structured output validated against a Pydantic model.

    Uses function_calling (not json_schema strict mode): OpenAI requires additionalProperties:
    false on every object in strict schema mode, which our Pydantic models do not emit.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = _get_structured(model, api_key, schema)

    def invoke_structured(system: str, user: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        result = _invoke_with_retry(structured, messages)
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json", exclude_none=False)
        if isinstance(result, dict):
            return schema.model_validate(result).model_dump(mode="json", exclude_none=False)
        raise TypeError(f"Unexpected structured output type: {type(result)}")

    return invoke_structured


def make_plain_chat_fn(model: str, api_key: str | None):
    """No JSON mode (plain chat for repair / fallback)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm(model, api_key, json_mode=False)

    def invoke(system: str, user: str) -> str:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        resp = _invoke_with_retry(llm, messages)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    return invoke
