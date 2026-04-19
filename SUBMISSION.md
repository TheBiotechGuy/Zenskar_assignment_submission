# Submission: Zenskar Contract Agent (LangGraph)

## What this is

A **LangGraph** pipeline that:

1. Resolves `--input` to one or more PDF files (single file, a folder of PDFs, or recursively with `--recursive`).
2. **Parses** each PDF with `src/parsers/pdf_parser.py` (pdfplumber, then pdftotext, then OCR, then optional gpt-5.4 vision), with a **per-document timeout** (default 360s, configurable).
3. **Screens each PDF for relevance** at document level: commercial agreements / order forms / MSAs / amendments vs bank statements, invoice-only documents, etc. Heuristics run first; **gpt-5.4-mini** JSON classification is used for ambiguous cases when `OPENAI_API_KEY` is set. Non-contract PDFs are **excluded** from the merged text but listed in `extraction_notes`.
4. **Merges** accepted documents in a stable order (master/MSA, agreement, order form, amendment, exhibit).
5. Runs **specialized agents** (separate JSON-mode LLM calls with focused system prompts):
   - Customer agent  
   - Commercial core agent (dates, currency, status, renewal, etc.)  
   - Phases & pricing agent (ContractV2-oriented `phases` with inline `product` / `pricing`)  
   - Notes agent (verbatim quotes for audit)
6. **Assembles** the README envelope: `customer`, `validated_json`, `missing_fields`, `extraction_notes`.

## System prerequisites (before Python)

The PDF parser shells out to **Poppler** (`pdftoppm`, `pdftotext`, `pdfinfo`) and **Tesseract** (OCR). Without them, text extraction is degraded and rasterisation for OCR/vision is skipped (the code no longer crashes if `pdftoppm` is missing, but you should install these for real contracts).

**macOS (Homebrew):**

```bash
brew install poppler tesseract
```

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install -y poppler-utils tesseract-ocr
```

**Windows:** Install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add its `bin` directory to `PATH`. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH`.

**Verify** (expect paths or version output):

```bash
which pdftoppm pdftotext pdfinfo
pdftoppm -v
tesseract --version
```

## Requirements

- Python 3.10+
- Poppler and Tesseract as above (required for full parsing quality).

## Install

```bash
cd /path/to/Zenskar_assignment_submission
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** for full extraction | OpenAI API key for JSON agents and optional vision/OCR escalation in the PDF parser. |
| `ZENSKAR_MODEL` | No | Default `gpt-5.4`. Used for customer, commercial, phases extraction (and notes use `gpt-5.4` in code). |
| `ZENSKAR_PDF_TIMEOUT_SEC` | No | Default `360` (6 minutes per PDF, per README). |
| `ZENSKAR_MAX_CORPUS_CHARS` | No | Default `100000` - caps merged text for model context. |

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (do not commit real keys).

LLM vision in the PDF parser uses the official **`openai`** Python package for API calls (same TLS behavior as other OpenAI clients).

## Run

Each invocation writes the JSON envelope to the path given by `--output` (and the step audit log beside it unless `--no-audit-log`). **For the full local benchmark, all generated JSON outputs (and their audit logs when enabled) are kept under the repository `output/` folder** � for example `output/pack_001.json`, `output/benchmark_0030.json`, and `output/pack_001_run_audit.log`.

Single PDF:

```bash
python run_submission.py --input pdf/standalone/some_contract.pdf --output out.json
```

Pack (directory with multiple related PDFs):

```bash
python run_submission.py --input pdf/packs/pack_001 --output out.json
```

All PDFs under `pdf/` (recursive):

```bash
python run_submission.py --input pdf --output out.json --recursive
```

### Batch runner (`run_all_submissions.py`)

```bash
python run_all_submissions.py
```

Runs each `pdf/packs/pack_*` folder and each `pdf/standalone/*.pdf` one after another and prints per-run and aggregate latency at the end.

**Sample latency summary** (all runs successful; times depend on machine, model, and API):

```
========================================================================
Latency summary (per subprocess run)
========================================================================
  pack/pack_001                                              38.06s  [ok]
  pack/pack_002                                     68.22s (1m 08s)  [ok]
  pack/pack_003                                              39.39s  [ok]
  pack/pack_004                                              59.25s  [ok]
  pack/pack_005                                              45.12s  [ok]
  pack/pack_006                                     63.45s (1m 03s)  [ok]
  pack/pack_007                                              35.32s  [ok]
  pack/pack_008                                     69.42s (1m 09s)  [ok]
  pack/pack_009                                              35.11s  [ok]
  pack/pack_010                                     65.25s (1m 05s)  [ok]
  standalone/benchmark_0001.pdf                               4.02s  [ok]
  standalone/benchmark_0003.pdf                              16.99s  [ok]
  standalone/benchmark_0005.pdf                              29.47s  [ok]
  standalone/benchmark_0007.pdf                              18.67s  [ok]
  standalone/benchmark_0009.pdf                              40.24s  [ok]
  standalone/benchmark_0018.pdf                               6.06s  [ok]
  standalone/benchmark_0020.pdf                              13.68s  [ok]
  standalone/benchmark_0022.pdf                              23.94s  [ok]
  standalone/benchmark_0030.pdf                              53.63s  [ok]
  standalone/benchmark_0032.pdf                              35.54s  [ok]
  standalone/benchmark_0034.pdf                              39.16s  [ok]
  standalone/benchmark_0036.pdf                              41.67s  [ok]
  standalone/benchmark_0044.pdf                              17.33s  [ok]
  standalone/benchmark_0046.pdf                              25.97s  [ok]
  standalone/benchmark_0048.pdf                              25.91s  [ok]
  standalone/benchmark_0050.pdf                              23.65s  [ok]
  standalone/benchmark_0059.pdf                              53.76s  [ok]
  standalone/benchmark_0061.pdf                              29.82s  [ok]
  standalone/benchmark_0063.pdf                              25.31s  [ok]
  standalone/benchmark_0065.pdf                              23.78s  [ok]
  standalone/benchmark_0073.pdf                              40.95s  [ok]
  standalone/benchmark_0075.pdf                              36.07s  [ok]
  standalone/benchmark_0077.pdf                              28.76s  [ok]
  standalone/benchmark_0086.pdf                              21.14s  [ok]
  standalone/benchmark_0088.pdf                              21.93s  [ok]
  standalone/benchmark_0090.pdf                              56.87s  [ok]
  standalone/benchmark_0092.pdf                              32.54s  [ok]
  standalone/benchmark_0100.pdf                              10.17s  [ok]
------------------------------------------------------------------------
  Runs recorded:     38  (successful: 38)
  Total (sum runs):  1315.61s (21m 55s)
  Average per run:   34.62s
========================================================================

End-to-end batch time (including overhead): 1315.62s (21m 55s)
```

### Step audit log (default on)

Each run writes a **step audit file** next to the JSON output unless disabled:

- **Default path:** `<output_basename>_run_audit.log` in the same directory as `--output` (e.g. `out.json` ? `out_run_audit.log`).
- **Custom path:** `--log /path/to/audit.log`
- **Disable:** `--no-audit-log`

The file contains, for each LangGraph step (`gather`, `parse`, `merge`, `customer`, `commercial`, `phases`, `notes`, `assemble`):

1. **BEFORE** � state entering the step (large fields like `corpus` and PDF `full_text` are truncated).
2. **NODE_RETURN** � the update dict returned by that node.
3. **AFTER** � merged state after the step.

It also mirrors **INFO** (and above) log lines from libraries (e.g. PDF parsing) into the same file.

## Output shape

Matches the assignment README:

```json
{
  "customer": {},
  "validated_json": {},
  "missing_fields": [],
  "extraction_notes": []
}
```

## Limitations / honesty

- **No open internet** at evaluation time beyond declared tools - this submission uses OpenAI APIs only (and local PDF/OCR).
- **Prefer missing fields over hallucinations**; agents are instructed accordingly; `missing_fields` lists gaps.
- **ContractV2** is large; `validated_json` follows **create-contract** style fields (`customer_id`, `phases`, inline `product`/`pricing` where applicable). Strict API validation against the full OpenAPI schema may require additional normalization for edge-case pricing shapes.

## Reproducibility

- `requirements.txt` pins minimum versions; for stricter reproducibility, run `pip freeze > requirements.lock` after install and use that lockfile in CI.
