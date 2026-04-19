"""Per-PDF relevance: contract vs bank statement / invoice-only, etc."""

from __future__ import annotations

import logging
import re
from typing import Any

from . import prompts

logger = logging.getLogger(__name__)

# Strong signals for commercial agreements
_CONTRACT_TERMS = re.compile(
    r"\b("
    r"master\s+services?\s+agreement|subscription\s+agreement|license\s+agreement|"
    r"statement\s+of\s+work|order\s+form|amendment|exhibit\s+[a-z0-9]|"
    r"effective\s+date|governing\s+law|termination|indemnif|confidential|"
    r"saas|software\s+as\s+a\s+service|data\s+processing|"
    r"service\s+level|mSA|non-?disclosure"
    r")\b",
    re.I,
)

# Strong signals for non-contract transactional docs
_NON_CONTRACT_TERMS = re.compile(
    r"\b("
    r"bank\s+statement|account\s+summary|routing\s+number|account\s+balance|"
    r"opening\s+balance|closing\s+balance|transactions\s+in\s+this\s+period|"
    r"remittance\s+advice|pay\s+stub|payroll|w-2|1099|"
    r"deposit\s+slip|check\s+image|wire\s+transfer\s+confirmation"
    r")\b",
    re.I,
)

# Invoice-only (often not a full contract)
_INVOICE_TERMS = re.compile(
    r"\b(invoice\s*#|invoice\s+number|amount\s+due|please\s+remit|"
    r"payment\s+terms:\s*net\s+\d+)\b",
    re.I,
)


def _score(text: str) -> tuple[int, int, int]:
    c = len(_CONTRACT_TERMS.findall(text[:20000]))
    n = len(_NON_CONTRACT_TERMS.findall(text[:20000]))
    inv = len(_INVOICE_TERMS.findall(text[:20000]))
    return c, n, inv


def heuristic_relevant(sample: str) -> tuple[bool | None, str]:
    """
    Returns (verdict or None if uncertain, reason).
    """
    if len(sample.strip()) < 80:
        return None, "too little text to classify heuristically"

    c, n, inv = _score(sample)
    if n >= 2 and c == 0:
        return False, "matches bank/transactional document patterns with no contract language"
    if n >= 1 and c == 0 and inv >= 3:
        return False, "looks like invoice or payment document without agreement terms"
    if c >= 2:
        return True, "contractual language detected"
    if c == 1 and n == 0:
        return True, "some contractual language and no strong non-contract signals"
    if c == 0 and n == 0 and inv >= 5:
        return False, "repeated invoice-only cues without MSA/order-form structure"
    return None, f"ambiguous (contract_hints={c}, non_contract_hints={n}, invoice_hints={inv})"


def llm_relevant(sample: str, invoke_json: Any) -> tuple[bool, str]:
    """invoke_json: callable taking (system, user) -> parsed dict."""
    user = (
        "Classify this PDF text excerpt (first portion of the document):\n\n"
        f"{sample[:12000]}"
    )
    data = invoke_json(prompts.RELEVANCE_SYSTEM, user)
    rel = bool(data.get("relevant"))
    reason = str(data.get("reason") or "model classification")
    return rel, reason


def classify_pdf(
    full_text: str,
    invoke_json: Any | None,
) -> tuple[bool, str]:
    """
    Decide if this PDF should participate in contract extraction.
    Uses heuristics first; optional LLM for borderline cases.
    """
    sample = full_text[:25000]
    h_res, h_reason = heuristic_relevant(sample)
    if h_res is True:
        return True, h_reason
    if h_res is False:
        return False, h_reason

    if invoke_json is None:
        logger.warning("Relevance uncertain and no LLM; accepting document: %s", h_reason)
        return True, "uncertain without LLM; default accept"

    try:
        return llm_relevant(sample, invoke_json)
    except Exception as exc:
        logger.warning("LLM relevance failed (%s); falling back to heuristic accept", exc)
        return True, f"LLM relevance error; default accept: {exc}"
