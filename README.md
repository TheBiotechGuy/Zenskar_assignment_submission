# Zenskar Contract Agent Challenge

Build an agent that reads messy contract PDFs and outputs Zenskar `ContractV2` JSON.

**Start here:** read https://docs.zenskar.com to understand the `ContractV2` schema, the Zenskar contract model, and what fields your agent needs to extract. You will not be able to solve this challenge without understanding the schema first.

## What's in this repo

60 contract PDFs for you to practice on.

```
pdf/
  standalone/    28 single-document contracts
  packs/         10 multi-document packets (3-4 related PDFs each)
```

The PDFs are intentionally messy — bad scans, skewed pages, OCR noise, table-heavy pricing, handwritten notes, redacted sections. This is what real contract ingestion looks like.

Each pack contains related documents (master agreement + order form + amendment, sometimes an exhibit). Your agent must read all documents in a pack together and produce one unified output.

## What your agent returns

```json
{
  "customer": {},
  "validated_json": {},
  "missing_fields": ["start_date", "phases[0].pricings[1].quantity.aggregate_id"],
  "extraction_notes": [
    {"field": "start_date", "quote_or_span": "Effective Date: July 16, 2025"}
  ]
}
```

- `customer` — who the contract is with
- `validated_json` — the extracted contract as `ContractV2` JSON (schema at https://docs.zenskar.com)
- `missing_fields` — anything you couldn't find and chose not to hallucinate (array of field paths)
- `extraction_notes` — quotes from the PDF that justify your extractions

If a value isn't in the document, leave it out and list it in `missing_fields`. Do not invent data.

## Why this is hard

These are not clean digital contracts. Expect:

- Scanned pages with noise, skew, borders, and JPEG artifacts
- Pricing tables with tiered/volume/per-unit/overage/bundle structures
- Amendments that override pricing or terms from earlier documents
- References to exhibits that may be missing or incomplete
- Redacted sections, mixed languages, handwritten annotations
- Some inputs that aren't contracts at all (bank statements, invoices) — your agent should reject these

## How to submit

Expose **one** of:
- `python run_submission.py --input <pdf> --output <json>`
- A single HTTP endpoint documented in `SUBMISSION.md`

You may use any model, OCR tool, MCP server, or multi-agent setup. Just make sure:
- Everything is documented in your `SUBMISSION.md`
- It runs without manual intervention
- The evaluator can reproduce it

**Include:** runnable entrypoint, dependency lockfile, `SUBMISSION.md` with run instructions and required env vars.

## How we score

We evaluate on a **hidden pool larger than this public set**. You will not see the test cases in advance.

What matters (in order):
1. Did you extract what's actually in the contract? (faithfulness)
2. Is the JSON valid against the `ContractV2` schema?
3. Did you get pricing right? (model, amounts, tiers, phases)
4. Did you avoid making things up? (hallucination penalty is heavy)
5. Can we run it again and get the same result? (reproducibility)
6. How fast and how cheap? (tie-breaker only)

**A slow agent that gets it right beats a fast one that guesses.**

## Rules

- Per-document timeout: **6 minutes**. Full-run timeout: **60 minutes**.
- No hardcoded answers. No manually generated outputs.
- No open internet during evaluation — only your declared tools.
- If your tools fail, fail fast. Hanging the evaluator = disqualification.
- Prefer missing fields over hallucinated fields. Always.

## Links

- **ContractV2 schema and Zenskar docs:** https://docs.zenskar.com
- **Practice PDFs:** `pdf/` in this repo
