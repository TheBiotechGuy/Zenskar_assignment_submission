"""Shared LangGraph state for contract extraction."""

from __future__ import annotations

from typing import Any, TypedDict


class PdfRecord(TypedDict, total=False):
    path: str
    accepted: bool
    reject_reason: str
    page_count: int
    parse_error: str | None
    full_text: str  # set when parse succeeds (used for accepted documents)


class ContractAgentState(TypedDict, total=False):
    input_path: str
    recursive_pdf: bool
    pdf_paths: list[str]
    pdf_records: list[PdfRecord]
    corpus: str
    corpus_meta: str
    customer: dict[str, Any]
    commercial_core: dict[str, Any]
    phases: list[dict[str, Any]]
    validated_json: dict[str, Any]
    missing_fields: list[str]
    extraction_notes: list[dict[str, Any]]
    fatal_error: str | None
