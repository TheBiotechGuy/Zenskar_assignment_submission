"""FastAPI HTTP endpoint for contract extraction.

Usage
-----
    uvicorn src.api:app --reload

POST /extract
    Upload one PDF (standalone contract) or multiple PDFs (document pack).
    Returns the SubmissionEnvelope JSON.

GET /health
    Liveness check.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.agent.graph import run_contract_agent

app = FastAPI(
    title="Zenskar Contract Agent API",
    description=(
        "Extract structured ContractV2 JSON from one or more contract PDF files. "
        "Upload a single PDF for a standalone contract, or multiple PDFs for a document pack."
    ),
    version="1.0.0",
)


@app.post(
    "/extract",
    summary="Extract contract data from uploaded PDF(s)",
    response_description="SubmissionEnvelope with validated_json, customer, missing_fields, extraction_notes",
)
async def extract(files: list[UploadFile] = File(...)) -> JSONResponse:
    """
    Upload one or more PDF files. Returns a SubmissionEnvelope JSON object.

    - **Single file**: treated as a standalone contract.
    - **Multiple files**: treated as a document pack (e.g., MSA + order form + amendment).
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    for upload in files:
        fname = upload.filename or ""
        if not fname.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail=f"File '{fname}' is not a PDF. Only .pdf files are accepted.",
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for upload in files:
            dest = tmp / (upload.filename or "upload.pdf")
            dest.write_bytes(await upload.read())

        if len(files) == 1:
            input_path = str(tmp / (files[0].filename or "upload.pdf"))
        else:
            input_path = str(tmp)

        result = run_contract_agent(input_path)

    return JSONResponse(content=result)


@app.get("/health", summary="Liveness check")
async def health() -> dict[str, str]:
    return {"status": "ok"}
