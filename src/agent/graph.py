"""LangGraph: gather PDFs -> parse & relevance -> merge corpus -> multi-agent extraction -> assemble."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    node_assemble,
    node_extract_commercial,
    node_extract_customer,
    node_extraction_notes,
    node_extract_phases,
    node_gather,
    node_merge_corpus,
    node_parse_and_screen,
)
from src.agent.step_audit import (
    StepAuditLogger,
    get_audit,
    merge_state_shallow,
    run_audit_session,
)
from src.agent.state import ContractAgentState
from src.models.contract_v2 import build_submission_envelope


def _wrap_step(
    step_name: str,
    fn: Callable[[ContractAgentState], dict[str, Any]],
) -> Callable[[ContractAgentState], dict[str, Any]]:
    """Log sanitized state before / node return / merged state after when audit is active."""

    def wrapped(state: ContractAgentState) -> dict[str, Any]:
        audit = get_audit()
        before = dict(state)
        update = fn(state)
        if update is None:
            update = {}
        after = merge_state_shallow(before, update)
        if audit is not None:
            audit.step(step_name, before, update, after)
        return update

    return wrapped


def build_graph() -> StateGraph:
    g: StateGraph = StateGraph(ContractAgentState)
    g.add_node("gather", _wrap_step("gather", node_gather))
    g.add_node("parse", _wrap_step("parse", node_parse_and_screen))
    g.add_node("merge", _wrap_step("merge", node_merge_corpus))
    g.add_node("customer", _wrap_step("customer", node_extract_customer))
    g.add_node("commercial", _wrap_step("commercial", node_extract_commercial))
    g.add_node("phases", _wrap_step("phases", node_extract_phases))
    g.add_node("notes", _wrap_step("notes", node_extraction_notes))
    g.add_node("assemble", _wrap_step("assemble", node_assemble))

    g.set_entry_point("gather")
    g.add_edge("gather", "parse")
    g.add_edge("parse", "merge")
    g.add_edge("merge", "customer")
    g.add_edge("customer", "commercial")
    g.add_edge("commercial", "phases")
    g.add_edge("phases", "notes")
    g.add_edge("notes", "assemble")
    g.add_edge("assemble", END)
    return g


def run_contract_agent(
    input_path: str,
    *,
    recursive_pdf: bool = False,
    audit_log_path: str | None = None,
) -> dict[str, Any]:
    """Run the compiled graph and return the submission envelope."""
    graph = build_graph()
    app = graph.compile()
    initial: ContractAgentState = {
        "input_path": input_path,
        "recursive_pdf": recursive_pdf,
    }

    if audit_log_path:
        audit = StepAuditLogger(Path(audit_log_path))
        try:
            with run_audit_session(audit):
                audit.write_run_meta(
                    input_path=input_path,
                    recursive_pdf=recursive_pdf,
                    audit_log_path=str(Path(audit_log_path).resolve()),
                )
                audit.attach_root_logging()
                final = app.invoke(initial)
        finally:
            audit.close()
        return submission_envelope(final)

    final = app.invoke(initial)
    return submission_envelope(final)


def submission_envelope(state: dict[str, Any]) -> dict[str, Any]:
    env = build_submission_envelope(
        customer=state.get("customer") or {},
        validated_json=state.get("validated_json") or {},
        missing_fields=list(state.get("missing_fields") or []),
        extraction_notes=list(state.get("extraction_notes") or []),
    )
    return env.model_dump(mode="json", exclude_none=False)
