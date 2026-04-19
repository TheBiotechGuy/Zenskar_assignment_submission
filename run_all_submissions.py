#!/usr/bin/env python3
"""
Run run_submission.py once per pack folder (pdf/packs/pack_*) and once per standalone PDF
(pdf/standalone/*.pdf), in sorted order.

Usage (from repo root):
  python run_all_submissions.py

Same environment as your normal runs (activate .venv first if you use one).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def _format_duration(seconds: float) -> str:
    """Seconds with two decimals; add H:MM:SS when >= 1 minute for readability."""
    s = f"{seconds:.2f}s"
    if seconds >= 60:
        h, rem = divmod(int(seconds), 3600)
        m, sec_part = divmod(rem, 60)
        if h:
            s += f" ({h:d}:{m:02d}:{sec_part:02d})"
        else:
            s += f" ({m:d}m {sec_part:02d}s)"
    return s


def _print_latency_summary(rows: list[tuple[str, float, int]]) -> None:
    total = sum(r[1] for r in rows)
    n = len(rows)
    avg = total / n if n else 0.0
    n_ok = sum(1 for r in rows if r[2] == 0)

    print("\n" + "=" * 72, flush=True)
    print("Latency summary (per subprocess run)", flush=True)
    print("=" * 72, flush=True)
    for label, sec, code in rows:
        status = "ok" if code == 0 else f"exit {code}"
        print(f"  {label:42s} {_format_duration(sec):>22s}  [{status}]", flush=True)
    print("-" * 72, flush=True)
    print(f"  Runs recorded:     {n}  (successful: {n_ok})", flush=True)
    print(f"  Total (sum runs):  {_format_duration(total)}", flush=True)
    print(f"  Average per run:   {_format_duration(avg)}", flush=True)
    print("=" * 72, flush=True)


def main() -> int:
    root = Path(__file__).resolve().parent
    runner = root / "run_submission.py"
    pdf_root = root / "pdf"
    out_dir = root / "output"

    if not runner.is_file():
        print(f"Missing {runner}", file=sys.stderr)
        return 1
    if not pdf_root.is_dir():
        print(f"Missing {pdf_root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, Path]] = []

    packs_dir = pdf_root / "packs"
    if packs_dir.is_dir():
        for pack in sorted(packs_dir.glob("pack_*")):
            if pack.is_dir():
                jobs.append((f"pack/{pack.name}", pack, out_dir / f"{pack.name}.json"))

    standalone_dir = pdf_root / "standalone"
    if standalone_dir.is_dir():
        for pdf in sorted(standalone_dir.glob("*.pdf")):
            jobs.append(
                (
                    f"standalone/{pdf.name}",
                    pdf,
                    out_dir / f"{pdf.stem}.json",
                )
            )

    if not jobs:
        print("No pack_* folders or standalone/*.pdf found under pdf/", file=sys.stderr)
        return 1

    batch_t0 = time.perf_counter()
    results: list[tuple[str, float, int]] = []

    for label, input_path, output_path in jobs:
        cmd = [
            sys.executable,
            str(runner),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        print(f"\n=== {label} -> {output_path.relative_to(root)} ===\n", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=root)
        elapsed = time.perf_counter() - t0
        results.append((label, elapsed, proc.returncode))
        print(f"\n  Run wall time: {_format_duration(elapsed)}", flush=True)
        if proc.returncode != 0:
            print(f"FAILED (exit {proc.returncode}): {label}", file=sys.stderr)
            _print_latency_summary(results)
            return proc.returncode

    batch_elapsed = time.perf_counter() - batch_t0
    print("\nAll submission runs finished.", flush=True)
    _print_latency_summary(results)
    print(f"\nEnd-to-end batch time (including overhead): {_format_duration(batch_elapsed)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
