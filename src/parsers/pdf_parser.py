"""
pdf_parser.py
=============
Production-grade PDF parser for messy contract documents.

Strategy (per page, in order of cheapness):
  1. pdfplumber  — native text + table extraction
  2. pdftotext   — layout-preserving fallback for garbled pdfplumber output
  3. OCR         — pytesseract on rasterised page image (scanned / image-only pages)
  4. LLM Vision  — OpenAI gpt-5.4 vision for truly unreadable pages
                   (skewed scans, overlays, handwriting, bad OCR confidence)

The class returns a single structured ParsedDocument with per-page results
so callers can combine multi-document packs cleanly.

Usage
-----
    parser = PDFParser(openai_api_key="sk-...")
    doc    = parser.parse("contract.pdf")
    print(doc.full_text)          # all text concatenated
    print(doc.pages[2].tables)    # tables on page 3
    print(doc.pages[2].method)    # how page 3 was read

Dependencies
------------
  pdfplumber, pypdf, pytesseract, Pillow, opencv-python, numpy, openai
  poppler-utils (pdftotext, pdftoppm, pdfinfo) via system PATH
  OpenAI API key (optional — only needed for LLM vision fallback, set OPENAI_API_KEY)

Large pages: optional env ``ZENSKAR_MAX_IMAGE_PIXELS`` raises Pillow's decompression limit
(default 500M pixels; use ``none`` / ``unlimited`` / ``off`` to disable the cap for trusted PDFs only).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pypdf import PdfReader

# Pillow's default ~89M pixel cap triggers DecompressionBombWarning on large but legitimate
# rasterised pages (high DPI / big sheets). Raise or disable via ZENSKAR_MAX_IMAGE_PIXELS.
_mp = os.getenv("ZENSKAR_MAX_IMAGE_PIXELS", "").strip().lower()
if _mp in ("", "default"):
    Image.MAX_IMAGE_PIXELS = 500_000_000
elif _mp in ("none", "unlimited", "off"):
    Image.MAX_IMAGE_PIXELS = None  # no limit — only for trusted local PDFs
else:
    Image.MAX_IMAGE_PIXELS = int(_mp)

logger = logging.getLogger(__name__)

# Log once if Poppler's pdftoppm is missing (rasterisation for OCR / LLM vision).
_pdftoppm_missing_logged = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ExtractionMethod(str, Enum):
    PDFPLUMBER   = "pdfplumber"
    PDFTOTEXT    = "pdftotext"
    OCR          = "ocr"
    LLM_VISION   = "llm_vision"
    EMPTY        = "empty"          # truly blank / non-contract page


@dataclass
class PageResult:
    page_num: int                   # 1-based
    method:   ExtractionMethod
    text:     str
    tables:   list[list[list[str]]] = field(default_factory=list)
    images:   list[bytes]           = field(default_factory=list)   # raw PNG bytes per embedded image
    ocr_confidence: float | None    = None
    warnings: list[str]             = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_path:  str
    page_count:   int
    pages:        list[PageResult]
    metadata:     dict[str, Any]    = field(default_factory=dict)
    form_fields:  dict[str, str]    = field(default_factory=dict)
    attachments:  list[str]         = field(default_factory=list)   # extracted attachment paths
    warnings:     list[str]         = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """All page text concatenated with page separators."""
        parts = []
        for p in self.pages:
            if p.text.strip():
                parts.append(f"\n--- Page {p.page_num} [{p.method.value}] ---\n{p.text}")
        return "\n".join(parts)

    @property
    def all_tables(self) -> list[tuple[int, list[list[str]]]]:
        """Flat list of (page_num, table) tuples across the document."""
        out = []
        for p in self.pages:
            for t in p.tables:
                out.append((p.page_num, t))
        return out


# ---------------------------------------------------------------------------
# Image pre-processing helpers
# ---------------------------------------------------------------------------

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(mat: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Apply a pipeline of image corrections to maximise OCR accuracy:
      1. Upscale small images to at least 300 DPI equivalent
      2. Convert to grayscale
      3. Deskew (rotate to align text horizontally)
      4. Remove noise (median blur)
      5. Adaptive thresholding → binary image
      6. Mild sharpening
    """
    # --- 1. Upscale if too small ------------------------------------------
    MIN_WIDTH = 1500
    w, h = img.size
    if w < MIN_WIDTH:
        scale = MIN_WIDTH / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    mat = _pil_to_cv(img)

    # --- 2. Grayscale --------------------------------------------------------
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    # --- 3. Deskew -----------------------------------------------------------
    gray = _deskew(gray)

    # --- 4. Denoise ----------------------------------------------------------
    gray = cv2.medianBlur(gray, 3)

    # --- 5. Adaptive threshold -----------------------------------------------
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # --- 6. Sharpen ----------------------------------------------------------
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    binary = cv2.filter2D(binary, -1, kernel)
    binary = np.clip(binary, 0, 255).astype(np.uint8)

    result = Image.fromarray(binary).convert("RGB")
    return result


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew angle using Hough line transform.
    Handles up to ±45° rotation. Safe — returns input unchanged on failure.
    """
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return gray

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:   # sub-pixel skew — not worth rotating
            return gray

        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Deskewed by %.2f°", median_angle)
        return rotated
    except Exception as exc:
        logger.debug("Deskew failed: %s", exc)
        return gray


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

_TESSERACT_CONFIG = "--oem 3 --psm 6 -l eng"   # LSTM engine, assume uniform text block


def _ocr_image(img: Image.Image) -> tuple[str, float]:
    """
    Run Tesseract on a pre-processed PIL image.
    Returns (text, mean_confidence). confidence is 0-100.
    """
    processed = preprocess_for_ocr(img)
    data = pytesseract.image_to_data(
        processed,
        config=_TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )
    words = [w for w, c in zip(data["text"], data["conf"])
             if w.strip() and int(c) > 0]
    confs = [int(c) for c in data["conf"] if int(c) > 0]
    text  = pytesseract.image_to_string(processed, config=_TESSERACT_CONFIG)
    confidence = float(np.mean(confs)) if confs else 0.0
    return text, confidence


def _page_to_pil(pdf_path: str, page_num: int, dpi: int = 250) -> Image.Image | None:
    """
    Rasterise a single PDF page to a PIL image using pdftoppm.
    page_num is 1-based.
    """
    global _pdftoppm_missing_logged
    if not shutil.which("pdftoppm"):
        if not _pdftoppm_missing_logged:
            _pdftoppm_missing_logged = True
            logger.warning(
                "pdftoppm not on PATH (install Poppler: "
                "macOS: brew install poppler | "
                "Ubuntu/Debian: sudo apt install poppler-utils | "
                "Windows: add Poppler bin to PATH). "
                "OCR and LLM vision rasterisation are skipped for affected pages."
            )
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "pg")
        cmd = [
            "pdftoppm", "-jpeg",
            "-r", str(dpi),
            "-f", str(page_num),
            "-l", str(page_num),
            pdf_path, prefix,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True)
        except FileNotFoundError:
            if not _pdftoppm_missing_logged:
                _pdftoppm_missing_logged = True
                logger.warning(
                    "pdftoppm could not be executed. Install Poppler so it is on PATH."
                )
            return None
        if result.returncode != 0:
            logger.warning("pdftoppm failed for page %d: %s",
                           page_num, result.stderr.decode())
            return None
        files = sorted(Path(tmpdir).glob("pg-*.jpg")) or \
                sorted(Path(tmpdir).glob("pg-*.jpeg"))
        if not files:
            return None
        return Image.open(files[0]).copy()


# ---------------------------------------------------------------------------
# LLM Vision fallback
# ---------------------------------------------------------------------------

_VISION_MODEL = "gpt-5.4"

_VISION_PROMPT = """You are extracting text from a scanned contract page.

Your task:
1. Transcribe ALL visible text faithfully, preserving structure and line breaks.
2. Represent tables using pipe-delimited rows: | col1 | col2 | col3 |
3. If there are handwritten annotations, prefix them with [HANDWRITTEN]: 
4. If a section is redacted, write [REDACTED].
5. Do NOT summarise or interpret — output verbatim text only.
6. Start directly with the extracted text. No preamble."""


def _call_openai_vision(image_bytes: bytes, api_key: str) -> str:
    """Call gpt-5.4 vision via the official OpenAI Python SDK (handles HTTPS/TLS)."""
    from openai import OpenAI

    b64 = base64.standard_b64encode(image_bytes).decode()
    client = OpenAI(api_key=api_key, timeout=120.0)
    try:
        resp = client.chat.completions.create(
            model=_VISION_MODEL,
            max_completion_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": _VISION_PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI vision API error: {exc}") from exc
    text = resp.choices[0].message.content
    return (text or "").strip()


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Text quality heuristics
# ---------------------------------------------------------------------------

def _text_quality(text: str) -> float:
    """
    Rough 0-1 score of how 'readable' extracted text is.
    Based on ratio of printable ASCII words vs. garbage characters.
    """
    if not text.strip():
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    good = sum(1 for w in words if re.fullmatch(r"[A-Za-z0-9,./:;$%@()\-\+'\"&]+", w))
    return good / len(words)


_MIN_TEXT_QUALITY   = 0.35   # below this → try next strategy
_MIN_OCR_CONFIDENCE = 60   # below this → escalate to LLM vision
_MIN_TEXT_LENGTH    = 30     # characters — fewer than this on a "real" page → likely scan


def _format_tables_block(tables: list[list[list[str]]]) -> str:
    """Render extracted tables as pipe-separated rows (one row per line)."""
    parts: list[str] = []
    for idx, tbl in enumerate(tables, start=1):
        row_lines = [" | ".join(row) for row in tbl]
        parts.append(f"Table {idx}:\n" + "\n".join(row_lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Multi-strategy PDF parser for messy contract documents.

    Parameters
    ----------
    openai_api_key : str | None
        OpenAI API key. When provided, LLM vision (gpt-5.4) is used as final
        fallback for unreadable pages. If None, OCR is the final fallback.
    ocr_confidence_threshold : float
        Pages whose OCR confidence falls below this value are escalated to
        LLM vision (default 45.0).
    llm_vision_dpi : int
        DPI used when rasterising pages for LLM vision (default 200).
    ocr_dpi : int
        DPI used when rasterising pages for OCR (default 250).
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        ocr_confidence_threshold: float = _MIN_OCR_CONFIDENCE,
        llm_vision_dpi: int = 200,
        ocr_dpi: int = 250,
    ) -> None:
        self.api_key                  = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.llm_vision_dpi           = llm_vision_dpi
        self.ocr_dpi                  = ocr_dpi

    def _merge_page_text(self, text: str, tables: list[list[list[str]]]) -> str:
        """Append structured table rows to body text when tables are not already redundant."""
        if not tables:
            return text
        if self._tables_from_text(text) == tables:
            return text
        block = _format_tables_block(tables)
        base = text.rstrip()
        if base:
            return f"{base}\n\n--- Extracted tables ---\n{block}"
        return f"--- Extracted tables ---\n{block}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF and return a ParsedDocument with per-page results.
        Automatically selects the best extraction strategy per page.
        """
        pdf_path = str(pdf_path)
        logger.info("Parsing: %s", pdf_path)

        metadata    = self._extract_metadata(pdf_path)
        form_fields = self._extract_form_fields(pdf_path)
        attachments = self._extract_attachments(pdf_path)
        pages       = self._parse_pages(pdf_path)

        doc = ParsedDocument(
            source_path  = pdf_path,
            page_count   = len(pages),
            pages        = pages,
            metadata     = metadata,
            form_fields  = form_fields,
            attachments  = attachments,
        )
        logger.info(
            "Done: %d pages | methods=%s",
            len(pages),
            {m.value: sum(1 for p in pages if p.method == m) for m in ExtractionMethod},
        )
        return doc

    def parse_many(self, pdf_paths: list[str | Path]) -> list[ParsedDocument]:
        """Parse multiple PDFs (e.g. a document pack) and return all results."""
        return [self.parse(p) for p in pdf_paths]

    # ------------------------------------------------------------------
    # Metadata / form fields / attachments
    # ------------------------------------------------------------------

    def _extract_metadata(self, pdf_path: str) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        try:
            result = subprocess.run(
                ["pdfinfo", pdf_path], capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
        except Exception as exc:
            logger.debug("pdfinfo failed: %s", exc)
        # Also grab pypdf metadata (richer)
        try:
            reader = PdfReader(pdf_path)
            if reader.metadata:
                for k, v in reader.metadata.items():
                    meta[k.lstrip("/")] = str(v)
        except Exception:
            pass
        return meta

    def _extract_form_fields(self, pdf_path: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        try:
            reader = PdfReader(pdf_path)
            raw = reader.get_fields() or {}
            for name, field in raw.items():
                val = field.get("/V", "")
                fields[name] = str(val) if val else ""
        except Exception as exc:
            logger.debug("Form field extraction failed: %s", exc)
        return fields

    def _extract_attachments(self, pdf_path: str) -> list[str]:
        """Extract embedded attachments to a temp dir, return their paths."""
        paths: list[str] = []
        try:
            result = subprocess.run(
                ["pdfdetach", "-list", pdf_path],
                capture_output=True, text=True,
            )
            if not result.stdout.strip() or "0 embedded" in result.stdout:
                return paths
            out_dir = tempfile.mkdtemp(prefix="pdf_attachments_")
            subprocess.run(
                ["pdfdetach", "-saveall", "-o", out_dir, pdf_path],
                capture_output=True,
            )
            paths = [str(p) for p in Path(out_dir).iterdir() if p.is_file()]
        except Exception as exc:
            logger.debug("Attachment extraction failed: %s", exc)
        return paths

    # ------------------------------------------------------------------
    # Per-page extraction
    # ------------------------------------------------------------------

    def _parse_pages(self, pdf_path: str) -> list[PageResult]:
        results: list[PageResult] = []

        with pdfplumber.open(pdf_path) as plumber_pdf:
            total_pages = len(plumber_pdf.pages)
            for page_num in range(1, total_pages + 1):
                logger.info("  Page %d/%d", page_num, total_pages)
                plumber_page = plumber_pdf.pages[page_num - 1]
                result = self._extract_page(pdf_path, page_num, plumber_page)
                results.append(result)

        return results

    def _extract_page(
        self,
        pdf_path: str,
        page_num: int,
        plumber_page: Any,
    ) -> PageResult:
        warnings: list[str] = []

        # ---- Strategy 1: pdfplumber ----------------------------------------
        plumber_text, tables = self._try_pdfplumber(plumber_page, warnings)
        quality = _text_quality(plumber_text)

        if quality >= _MIN_TEXT_QUALITY and len(plumber_text.strip()) >= _MIN_TEXT_LENGTH:
            logger.debug("    Page %d → pdfplumber (quality=%.2f)", page_num, quality)
            return PageResult(
                page_num = page_num,
                method   = ExtractionMethod.PDFPLUMBER,
                text     = self._merge_page_text(plumber_text, tables),
                tables   = tables,
                warnings = warnings,
            )

        # ---- Strategy 2: pdftotext (layout) --------------------------------
        pdftotext_text = self._try_pdftotext(pdf_path, page_num)
        qt2 = _text_quality(pdftotext_text)

        if qt2 >= _MIN_TEXT_QUALITY and len(pdftotext_text.strip()) >= _MIN_TEXT_LENGTH:
            logger.debug("    Page %d → pdftotext (quality=%.2f)", page_num, qt2)
            # Still use pdfplumber tables if any were found
            return PageResult(
                page_num = page_num,
                method   = ExtractionMethod.PDFTOTEXT,
                text     = self._merge_page_text(pdftotext_text, tables),
                tables   = tables,
                warnings = warnings,
            )

        # ---- Strategy 3: OCR -----------------------------------------------
        img = _page_to_pil(pdf_path, page_num, dpi=self.ocr_dpi)
        if img is None:
            warnings.append(f"Page {page_num}: rasterisation failed")
            fallback_txt = plumber_text or pdftotext_text
            return PageResult(
                page_num = page_num,
                method   = ExtractionMethod.EMPTY,
                text     = self._merge_page_text(fallback_txt, tables),
                tables   = tables,
                warnings = warnings,
            )

        ocr_text, ocr_conf = _ocr_image(img)
        logger.debug("    Page %d → OCR (conf=%.1f)", page_num, ocr_conf)

        if ocr_conf >= self.ocr_confidence_threshold and \
                len(ocr_text.strip()) >= _MIN_TEXT_LENGTH:
            ocr_tables = tables or self._tables_from_text(ocr_text)
            return PageResult(
                page_num       = page_num,
                method         = ExtractionMethod.OCR,
                text           = self._merge_page_text(ocr_text, ocr_tables),
                tables         = ocr_tables,
                ocr_confidence = ocr_conf,
                warnings       = warnings,
            )

        # ---- Strategy 4: LLM Vision ----------------------------------------
        if self.api_key:
            logger.info("    Page %d → LLM vision (ocr_conf=%.1f)", page_num, ocr_conf)
            # Re-rasterise at vision DPI (can be lower — LLM handles noise better)
            vis_img = _page_to_pil(pdf_path, page_num, dpi=self.llm_vision_dpi)
            if vis_img is not None:
                try:
                    img_bytes  = _pil_to_jpeg_bytes(vis_img, quality=92)
                    llm_text   = _call_openai_vision(img_bytes, self.api_key)
                    llm_tables = tables or self._tables_from_text(llm_text)
                    return PageResult(
                        page_num       = page_num,
                        method         = ExtractionMethod.LLM_VISION,
                        text           = self._merge_page_text(llm_text, llm_tables),
                        tables         = llm_tables,
                        ocr_confidence = ocr_conf,
                        warnings       = warnings,
                    )
                except Exception as exc:
                    warnings.append(f"Page {page_num}: LLM vision failed: {exc}")
                    logger.warning("    LLM vision failed for page %d: %s", page_num, exc)

        # ---- Fallback: best available text ----------------------------------
        best_text = max([plumber_text, pdftotext_text, ocr_text],
                        key=lambda t: _text_quality(t) * len(t))
        method = ExtractionMethod.OCR if best_text is ocr_text else \
                 ExtractionMethod.PDFTOTEXT if best_text is pdftotext_text else \
                 ExtractionMethod.PDFPLUMBER
        warnings.append(f"Page {page_num}: low-confidence fallback (ocr_conf={ocr_conf:.1f})")
        fb_tables = tables or self._tables_from_text(best_text)
        return PageResult(
            page_num       = page_num,
            method         = method,
            text           = self._merge_page_text(best_text, fb_tables),
            tables         = fb_tables,
            ocr_confidence = ocr_conf,
            warnings       = warnings,
        )

    # ------------------------------------------------------------------
    # Individual strategy implementations
    # ------------------------------------------------------------------

    def _try_pdfplumber(
        self,
        page: Any,
        warnings: list[str],
    ) -> tuple[str, list[list[list[str]]]]:
        """Extract text and tables using pdfplumber. Returns (text, tables)."""
        text   = ""
        tables: list[list[list[str]]] = []
        try:
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
        except Exception as exc:
            warnings.append(f"pdfplumber text extraction error: {exc}")

        try:
            raw_tables = page.extract_tables({
                "vertical_strategy":   "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance":       3,
                "join_tolerance":       3,
                "edge_min_length":     10,
            }) or []

            if not raw_tables:
                # Fall back to text-based table detection
                raw_tables = page.extract_tables({
                    "vertical_strategy":   "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance":       5,
                }) or []

            for tbl in raw_tables:
                cleaned = [
                    [str(cell).strip() if cell else "" for cell in row]
                    for row in tbl
                ]
                tables.append(cleaned)
        except Exception as exc:
            warnings.append(f"pdfplumber table extraction error: {exc}")

        return text, tables

    def _try_pdftotext(self, pdf_path: str, page_num: int) -> str:
        """Run pdftotext with layout mode on a single page."""
        try:
            result = subprocess.run(
                [
                    "pdftotext", "-layout",
                    "-f", str(page_num),
                    "-l", str(page_num),
                    pdf_path, "-",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout
        except Exception as exc:
            logger.debug("pdftotext failed page %d: %s", page_num, exc)
            return ""

    # ------------------------------------------------------------------
    # Table extraction from raw text (OCR / LLM output)
    # ------------------------------------------------------------------

    def _tables_from_text(self, text: str) -> list[list[list[str]]]:
        """
        Detect pipe-delimited tables in raw text (as produced by LLM vision
        or sometimes OCR). Returns list of tables (each table = list of rows).
        """
        lines  = text.splitlines()
        tables: list[list[list[str]]] = []
        current: list[list[str]]      = []

        for line in lines:
            stripped = line.strip()
            if "|" in stripped and not stripped.startswith("#"):
                cells = [c.strip() for c in stripped.split("|")]
                # Remove leading/trailing empty cells from split
                if cells and cells[0]  == "": cells  = cells[1:]
                if cells and cells[-1] == "": cells  = cells[:-1]
                if cells:
                    current.append(cells)
            else:
                if len(current) >= 2:    # at least header + one data row
                    tables.append(current)
                current = []

        if len(current) >= 2:
            tables.append(current)

        return tables


# ---------------------------------------------------------------------------
# CLI convenience wrapper
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Parse a PDF and print extracted text.")
    ap.add_argument("pdf",  help="Path to PDF file")
    ap.add_argument("--json", action="store_true", help="Output structured JSON")
    ap.add_argument("--api-key", default=None, help="OpenAI API key for LLM vision fallback (or set OPENAI_API_KEY)")
    args = ap.parse_args()

    parser = PDFParser(openai_api_key=args.api_key)
    doc    = parser.parse(args.pdf)

    if args.json:
        out = {
            "source":     doc.source_path,
            "page_count": doc.page_count,
            "metadata":   doc.metadata,
            "form_fields": doc.form_fields,
            "pages": [
                {
                    "page_num":       p.page_num,
                    "method":         p.method.value,
                    "text":           p.text,
                    "tables":         p.tables,
                    "ocr_confidence": p.ocr_confidence,
                    "warnings":       p.warnings,
                }
                for p in doc.pages
            ],
        }
        json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    else:
        print(doc.full_text)

    if doc.warnings:
        print("\n=== Document Warnings ===", file=sys.stderr)
        for w in doc.warnings:
            print(" •", w, file=sys.stderr)


if __name__ == "__main__":
    main()
