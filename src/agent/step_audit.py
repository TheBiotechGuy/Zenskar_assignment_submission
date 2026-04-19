"""Per-run step audit: before/after state snapshots written to a log file."""

from __future__ import annotations

import copy
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_CORPUS_LOG = 12_000
_MAX_FULL_TEXT_SNIPPET = 800
_MAX_JSON_INDENT = 2

_audit: ContextVar["StepAuditLogger | None"] = ContextVar("step_audit", default=None)


def get_audit() -> "StepAuditLogger | None":
    return _audit.get()


@contextmanager
def run_audit_session(audit: "StepAuditLogger"):
    token = _audit.set(audit)
    try:
        yield audit
    finally:
        _audit.reset(token)


def _sanitize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy state and truncate huge fields for logging."""
    s = copy.deepcopy(state)
    corpus = s.get("corpus")
    if isinstance(corpus, str) and len(corpus) > _MAX_CORPUS_LOG:
        total = len(corpus)
        s["corpus"] = corpus[:_MAX_CORPUS_LOG] + f"\n... [truncated, {total} chars total]"

    pr = s.get("pdf_records")
    if isinstance(pr, list):
        s["pdf_records"] = [_sanitize_pdf_record(r) for r in pr]

    return s


def _sanitize_pdf_record(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    ft = out.get("full_text")
    if isinstance(ft, str) and len(ft) > _MAX_FULL_TEXT_SNIPPET:
        out["full_text"] = (
            ft[:_MAX_FULL_TEXT_SNIPPET]
            + f"... [truncated, {len(ft)} chars total]"
        )
    return out


def _json_block(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=_MAX_JSON_INDENT, ensure_ascii=False, default=str)
    except TypeError:
        return repr(obj)


class StepAuditLogger:
    """Run log: structured step before/after plus optional mirrored INFO logging."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._log_handler: logging.Handler | None = None
        self._closed = False
        self._write_header()

    def attach_root_logging(self, level: int = logging.INFO) -> None:
        """Mirror INFO+ log records into this file (interleaved with step blocks)."""
        if self._log_handler is not None:
            return

        class _MirrorHandler(logging.Handler):
            def __init__(self, stream):
                super().__init__(level)
                self.stream = stream

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    self.stream.write(msg + "\n")
                    self.stream.flush()
                except Exception:
                    self.handleError(record)

        h = _MirrorHandler(self._fh)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(min(root.level or logging.WARNING, level))
        self._log_handler = h

    def _detach_root_logging(self) -> None:
        if self._log_handler is None:
            return
        logging.getLogger().removeHandler(self._log_handler)
        self._log_handler = None

    def _write_header(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._fh.write("Zenskar contract agent - step audit log\n")
        self._fh.write(f"started_utc={ts}\n")
        self._fh.write("=" * 80 + "\n\n")
        self._fh.flush()

    def write_run_meta(self, **kwargs: Any) -> None:
        self._fh.write("RUN_META\n")
        self._fh.write(_json_block(kwargs) + "\n\n")
        self._fh.flush()

    def step(
        self,
        step_name: str,
        state_before: dict[str, Any],
        node_return: dict[str, Any],
        state_after: dict[str, Any],
    ) -> None:
        self._fh.write("-" * 80 + "\n")
        self._fh.write(f"STEP: {step_name}\n")
        self._fh.write(f"ts_utc={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        self._fh.write("\n--- BEFORE (state entering step) ---\n")
        self._fh.write(_json_block(_sanitize_state(state_before)) + "\n")
        self._fh.write("\n--- NODE_RETURN (delta merged by LangGraph) ---\n")
        self._fh.write(_json_block(_sanitize_state(dict(node_return))) + "\n")
        self._fh.write("\n--- AFTER (merged state leaving step) ---\n")
        self._fh.write(_json_block(_sanitize_state(state_after)) + "\n")
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._detach_root_logging()
            self._fh.write("=" * 80 + "\n")
            self._fh.write(f"ended_utc={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        finally:
            self._fh.close()

    def __enter__(self) -> StepAuditLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def merge_state_shallow(
    base: dict[str, Any],
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """Approximate LangGraph default dict merge: shallow copy + update keys."""
    out = dict(base)
    if update:
        out.update(update)
    return out
