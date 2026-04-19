#!/usr/bin/env python3
"""
Zenskar Contract Agent - LangGraph multi-agent submission entrypoint.

Usage:
  python run_submission.py --input <pdf_or_directory> --output <json_file>
  python run_submission.py ... --output out.json --log custom_audit.log
  python run_submission.py ... --output out.json --no-audit-log

By default writes <output_stem>_run_audit.log with per-step before/after state.

Environment:
  OPENAI_API_KEY   Required for relevance (optional fallback) and extraction agents.
  ZENSKAR_MODEL    OpenAI model for extraction (default: gpt-5.4).
  ZENSKAR_PDF_TIMEOUT_SEC  Per-PDF parse timeout (default: 360).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Extract ContractV2 JSON from contract PDF(s).")
    ap.add_argument(
        "--input",
        required=True,
        help="Path to a PDF file or a directory containing PDFs (standalone or pack folder).",
    )
    ap.add_argument("--output", required=True, help="Write JSON result to this path.")
    ap.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="Per-step audit log (before/after state + INFO logs). "
        "Default: <output_stem>_run_audit.log next to --output.",
    )
    ap.add_argument(
        "--no-audit-log",
        action="store_true",
        help="Disable writing the step audit file.",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, include PDFs in subfolders (e.g. pdf/).",
    )
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    from src.agent.graph import run_contract_agent

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        print(f"Input path does not exist: {inp}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audit_log_path: str | None = None
    if not args.no_audit_log:
        if args.log:
            audit_log_path = str(Path(args.log).expanduser().resolve())
        else:
            audit_log_path = str(out_path.parent / f"{out_path.stem}_run_audit.log")

    result = run_contract_agent(
        str(inp),
        recursive_pdf=args.recursive,
        audit_log_path=audit_log_path,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    if audit_log_path:
        print(f"Step audit log: {audit_log_path}")


if __name__ == "__main__":
    main()
