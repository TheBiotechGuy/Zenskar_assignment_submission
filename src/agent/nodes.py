"""LangGraph node functions: ingest, parse, relevance, multi-agent extraction, assembly."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from src.parsers.pdf_parser import PDFParser, ParsedDocument
from src.agent import prompts
from src.agent.llm import make_json_chat_fn, make_structured_chat_fn
from src.agent.relevance import classify_pdf
from src.agent.state import ContractAgentState, PdfRecord
from src.models.contract_v2 import (
    CommercialCore,
    ContractCustomer,
    NotesExtraction,
    PhasesExtraction,
    build_contract_payload_from_extraction,
    compute_missing_field_keys,
    contract_to_validated_json,
    resolve_customer_id_from_extraction,
    validate_contract_payload,
)

logger = logging.getLogger(__name__)

PER_PDF_TIMEOUT_SEC = int(os.getenv("ZENSKAR_PDF_TIMEOUT_SEC", "360"))
MAX_CORPUS_CHARS = int(os.getenv("ZENSKAR_MAX_CORPUS_CHARS", "100000"))


def _rank_pdf(name: str) -> tuple[int, str]:
    n = name.lower()
    ordered = [
        ("master", 0),
        ("msa", 0),
        ("framework", 0),
        ("agreement", 1),
        ("subscription", 1),
        ("license", 1),
        ("order", 2),
        ("statement_of_work", 2),
        ("sow", 2),
        ("order_form", 2),
        ("amendment", 3),
        ("addendum", 3),
        ("exhibit", 4),
        ("schedule", 4),
        ("appendix", 4),
    ]
    for kw, rank in ordered:
        if kw in n.replace(" ", "_"):
            return rank, name
    return 10, name


def collect_pdf_paths(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {root}")
        return [root.resolve()]
    if not root.is_dir():
        raise ValueError(f"Path not found: {root}")
    if recursive:
        paths = sorted(root.rglob("*.pdf"))
    else:
        paths = sorted(root.glob("*.pdf"))
    return [p.resolve() for p in paths]


def node_gather(state: ContractAgentState) -> dict[str, Any]:
    inp = Path(state["input_path"]).expanduser().resolve()
    recursive = bool(state.get("recursive_pdf", False))
    paths = collect_pdf_paths(inp, recursive)
    if not paths:
        return {"fatal_error": f"No PDF files found under {inp}", "pdf_paths": []}
    ranked = sorted(paths, key=lambda p: (_rank_pdf(p.name), str(p)))
    return {"pdf_paths": [str(p) for p in ranked], "fatal_error": None}


def _parse_one(parser: PDFParser, path: str) -> ParsedDocument:
    return parser.parse(path)


def node_parse_and_screen(state: ContractAgentState) -> dict[str, Any]:
    if state.get("fatal_error"):
        return {}
    api_key = os.getenv("OPENAI_API_KEY")
    relevance_model = os.getenv("ZENSKAR_RELEVANCE_MODEL", "gpt-5.4-mini")
    invoke_json: Callable[..., dict[str, Any]] | None = None
    if api_key:
        invoke_json = make_json_chat_fn(relevance_model, api_key)

    parser = PDFParser(openai_api_key=api_key)
    records: list[PdfRecord] = []

    for path in state["pdf_paths"]:
        rec: PdfRecord = {"path": path, "accepted": False, "reject_reason": "", "page_count": 0, "parse_error": None}
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_parse_one, parser, path)
                doc = fut.result(timeout=PER_PDF_TIMEOUT_SEC)
        except FuturesTimeout:
            rec["accepted"] = False
            rec["parse_error"] = f"parse exceeded {PER_PDF_TIMEOUT_SEC}s"
            logger.error("%s", rec["parse_error"])
            records.append(rec)
            continue
        except Exception as exc:
            rec["accepted"] = False
            rec["parse_error"] = str(exc)
            logger.exception("Parse failed for %s", path)
            records.append(rec)
            continue

        rec["page_count"] = doc.page_count
        rec["full_text"] = doc.full_text
        text = doc.full_text
        ok, reason = classify_pdf(text, invoke_json)
        rec["accepted"] = ok
        rec["reject_reason"] = reason if not ok else ""
        records.append(rec)

    return {"pdf_records": records}


def node_merge_corpus(state: ContractAgentState) -> dict[str, Any]:
    if state.get("fatal_error"):
        return {}
    accepted = [r for r in state["pdf_records"] if r.get("accepted") and r.get("full_text")]
    if not accepted:
        return {
            "fatal_error": "All PDFs were rejected as non-contract documents or failed to parse.",
            "corpus": "",
            "corpus_meta": "",
        }

    parts: list[str] = []
    meta_lines: list[str] = []

    for r in accepted:
        path = r["path"]
        text = r.get("full_text") or ""
        meta_lines.append(f"- {path} ({r.get('page_count', '?')} pages)")
        parts.append(f"\n\n### SOURCE FILE: {Path(path).name}\n{text}\n")

    corpus = "\n".join(parts)
    if len(corpus) > MAX_CORPUS_CHARS:
        corpus = corpus[:MAX_CORPUS_CHARS] + "\n\n[TRUNCATED - corpus cap for model context]\n"
    meta = "Merged corpus from:\n" + "\n".join(meta_lines)
    return {"corpus": corpus, "corpus_meta": meta, "fatal_error": None}


def node_extract_customer(state: ContractAgentState) -> dict[str, Any]:
    if state.get("fatal_error") or not state.get("corpus"):
        return {"customer": {}}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"customer": {}}
    model = os.getenv("ZENSKAR_MODEL", "gpt-5.4")
    user_msg = f"CONTRACT TEXT:\n{state['corpus']}"
    try:
        invoke = make_structured_chat_fn(model, api_key, ContractCustomer)
        data = invoke(prompts.CUSTOMER_AGENT, user_msg)
    except Exception as exc:
        logger.warning("Structured customer extraction failed (%s); falling back to JSON + validate", exc)
        invoke = make_json_chat_fn(model, api_key)
        raw = invoke(prompts.CUSTOMER_AGENT, user_msg)
        data = ContractCustomer.model_validate(raw).model_dump(mode="json", exclude_none=False)
    return {"customer": {k: v for k, v in data.items() if not str(k).startswith("_")}}


def node_extract_commercial(state: ContractAgentState) -> dict[str, Any]:
    if state.get("fatal_error") or not state.get("corpus"):
        return {"commercial_core": {}}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"commercial_core": {}}
    model = os.getenv("ZENSKAR_MODEL", "gpt-5.4")
    user_msg = f"CONTRACT TEXT:\n{state['corpus']}"
    try:
        invoke = make_structured_chat_fn(model, api_key, CommercialCore)
        data = invoke(prompts.COMMERCIAL_CORE_AGENT, user_msg)
    except Exception as exc:
        logger.warning("Structured commercial extraction failed (%s); falling back to JSON + validate", exc)
        invoke = make_json_chat_fn(model, api_key)
        raw = invoke(prompts.COMMERCIAL_CORE_AGENT, user_msg)
        data = CommercialCore.model_validate(raw).model_dump(mode="json", exclude_none=False)
    return {"commercial_core": {k: v for k, v in data.items() if not str(k).startswith("_")}}


def node_extract_phases(state: ContractAgentState) -> dict[str, Any]:
    if state.get("fatal_error") or not state.get("corpus"):
        return {"phases": []}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"phases": []}
    model = os.getenv("ZENSKAR_MODEL", "gpt-5.4")
    user_msg = f"CONTRACT TEXT:\n{state['corpus']}"
    try:
        invoke = make_structured_chat_fn(model, api_key, PhasesExtraction)
        data = invoke(prompts.PHASES_AGENT, user_msg)
        phases = data.get("phases") or []
    except Exception as exc:
        logger.warning("Structured phases extraction failed (%s); falling back to JSON + validate", exc)
        invoke = make_json_chat_fn(model, api_key)
        raw = invoke(prompts.PHASES_AGENT, user_msg)
        phases = PhasesExtraction.model_validate(raw).model_dump(mode="json", exclude_none=False)["phases"]
    if not isinstance(phases, list):
        phases = []
    return {"phases": phases}


def node_extraction_notes(state: ContractAgentState) -> dict[str, Any]:
    if state.get("fatal_error") or not state.get("corpus"):
        return {"extraction_notes": []}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"extraction_notes": []}
    notes_model = os.getenv("ZENSKAR_NOTES_MODEL", "gpt-5.4")
    draft = {
        "customer": state.get("customer"),
        "commercial_core": state.get("commercial_core"),
        "phases": state.get("phases"),
    }
    user_msg = (
        f"CONTRACT TEXT (for quoting only):\n{state['corpus'][:80000]}\n\n"
        f"DRAFT JSON SUMMARY:\n{json.dumps(draft, ensure_ascii=False)[:20000]}"
    )
    try:
        invoke = make_structured_chat_fn(notes_model, api_key, NotesExtraction)
        data = invoke(prompts.NOTES_AGENT, user_msg)
        notes = data.get("extraction_notes") or []
    except Exception as exc:
        logger.warning("Structured notes extraction failed (%s); falling back to JSON + validate", exc)
        invoke = make_json_chat_fn(notes_model, api_key)
        raw = invoke(prompts.NOTES_AGENT, user_msg)
        notes = NotesExtraction.model_validate(raw).model_dump(mode="json", exclude_none=False)["extraction_notes"]
    if not isinstance(notes, list):
        notes = []
    return {"extraction_notes": notes}


def _strip_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(x) for x in obj if x is not None]
    return obj


def _json_safe(obj: Any) -> Any:
    """Recursively convert UUID and other non-JSON-native values for fallback dumps."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    return obj


def node_assemble(state: ContractAgentState) -> dict[str, Any]:
    notes = list(state.get("extraction_notes") or [])
    missing: list[str] = []

    if state.get("fatal_error"):
        return {
            "validated_json": {},
            "missing_fields": [f"error:{state['fatal_error']}"],
            "extraction_notes": notes,
        }

    commercial = state.get("commercial_core") or {}
    phases = state.get("phases") or []
    cust = state.get("customer") or {}

    customer_id = resolve_customer_id_from_extraction(cust)
    missing.extend(compute_missing_field_keys(commercial, phases, customer_id))

    payload = build_contract_payload_from_extraction(
        commercial,
        phases,
        cust,
        state.get("corpus_meta", ""),
    )

    contract, validation_paths = validate_contract_payload(payload)
    if contract is not None:
        validated = contract_to_validated_json(contract)
    else:
        missing.extend(validation_paths)
        validated = _json_safe(_strip_none(payload))

    for r in state.get("pdf_records") or []:
        if r.get("parse_error"):
            notes.append(
                {
                    "field": f"parse_error:{Path(r['path']).name}",
                    "quote_or_span": r["parse_error"],
                }
            )
        elif not r.get("accepted"):
            notes.append(
                {
                    "field": f"rejected_pdf:{Path(r['path']).name}",
                    "quote_or_span": r.get("reject_reason") or "not a contract document",
                }
            )

    return {
        "validated_json": validated,
        "missing_fields": sorted(set(missing)),
        "extraction_notes": notes,
    }
