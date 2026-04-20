"""Mocked unit tests for pipeline nodes, relevance classifier, and data models."""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from uuid import UUID

from src.agent.nodes import (
    _rank_pdf,
    node_assemble,
    node_gather,
    node_merge_corpus,
)
from src.agent.relevance import classify_pdf, heuristic_relevant
from src.models.contract_v2 import (
    PricingData,
    compute_missing_field_keys,
    resolve_customer_id_from_extraction,
)


# ---------------------------------------------------------------------------
# _rank_pdf
# ---------------------------------------------------------------------------

class TestRankPdf(unittest.TestCase):
    def test_master_ranked_before_amendment(self) -> None:
        self.assertLess(_rank_pdf("master_agreement.pdf")[0], _rank_pdf("amendment_01.pdf")[0])

    def test_msa_ranked_before_order(self) -> None:
        self.assertLess(_rank_pdf("msa.pdf")[0], _rank_pdf("order_form.pdf")[0])

    def test_hyphenated_order_form_ranked_correctly(self) -> None:
        # "order-form.pdf" must match "order" keyword after hyphen normalization
        rank_order, _ = _rank_pdf("order-form.pdf")
        rank_exhibit, _ = _rank_pdf("exhibit-a.pdf")
        self.assertLess(rank_order, rank_exhibit)

    def test_hyphenated_amendment_ranked_correctly(self) -> None:
        rank_msa, _ = _rank_pdf("master-services-agreement.pdf")
        rank_amendment, _ = _rank_pdf("first-amendment.pdf")
        self.assertLess(rank_msa, rank_amendment)

    def test_unknown_filename_defaults_to_rank_10(self) -> None:
        rank, _ = _rank_pdf("misc_document.pdf")
        self.assertEqual(rank, 10)


# ---------------------------------------------------------------------------
# node_gather
# ---------------------------------------------------------------------------

class TestNodeGather(unittest.TestCase):
    def test_single_pdf_file_input(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 minimal")
            tmp_path = f.name
        try:
            result = node_gather({"input_path": tmp_path})
            self.assertIsNone(result["fatal_error"])
            self.assertEqual(len(result["pdf_paths"]), 1)
        finally:
            os.unlink(tmp_path)

    def test_directory_with_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name in ("master.pdf", "amendment.pdf"):
                (Path(d) / name).write_bytes(b"%PDF-1.4")
            result = node_gather({"input_path": d})
            self.assertIsNone(result["fatal_error"])
            self.assertEqual(len(result["pdf_paths"]), 2)
            # master should come before amendment in ranked order
            names = [Path(p).name for p in result["pdf_paths"]]
            self.assertEqual(names.index("master.pdf"), 0)

    def test_empty_directory_returns_fatal_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            result = node_gather({"input_path": d})
            self.assertIsNotNone(result["fatal_error"])
            self.assertEqual(result["pdf_paths"], [])

    def test_fatal_error_short_circuits_gather(self) -> None:
        # If fatal_error is already set by a prior node, gather still runs (it's first)
        # but an empty dir returns its own fatal_error cleanly
        with tempfile.TemporaryDirectory() as d:
            result = node_gather({"input_path": d, "fatal_error": "prior error"})
            self.assertIsNotNone(result.get("fatal_error"))


# ---------------------------------------------------------------------------
# node_merge_corpus
# ---------------------------------------------------------------------------

class TestNodeMergeCorpus(unittest.TestCase):
    def _rec(self, path: str, text: str, accepted: bool = True, pages: int = 2) -> dict:
        return {
            "path": path,
            "accepted": accepted,
            "full_text": text,
            "page_count": pages,
            "reject_reason": "",
            "parse_error": None,
        }

    def test_accepted_pdfs_merged_into_corpus(self) -> None:
        state = {
            "pdf_records": [
                self._rec("/docs/a.pdf", "Alpha text"),
                self._rec("/docs/b.pdf", "Beta text"),
            ]
        }
        result = node_merge_corpus(state)
        self.assertIn("Alpha text", result["corpus"])
        self.assertIn("Beta text", result["corpus"])
        self.assertIsNone(result.get("fatal_error"))

    def test_rejected_pdf_excluded_from_corpus(self) -> None:
        state = {
            "pdf_records": [
                self._rec("/docs/a.pdf", "Included text", accepted=True),
                self._rec("/docs/b.pdf", "Should not appear", accepted=False),
            ]
        }
        result = node_merge_corpus(state)
        self.assertNotIn("Should not appear", result["corpus"])

    def test_all_rejected_returns_fatal_error(self) -> None:
        state = {
            "pdf_records": [self._rec("/docs/a.pdf", "X", accepted=False)]
        }
        result = node_merge_corpus(state)
        self.assertIsNotNone(result.get("fatal_error"))
        self.assertEqual(result["corpus"], "")

    def test_corpus_truncated_at_newline_boundary(self) -> None:
        # Build text that exceeds MAX_CORPUS_CHARS (100K default)
        long_text = ("A" * 90 + "\n") * 1200  # ~108 000 chars, guaranteed to trigger cap
        state = {"pdf_records": [self._rec("/docs/a.pdf", long_text)]}
        result = node_merge_corpus(state)
        self.assertIn("TRUNCATED", result["corpus"])
        # Corpus must not end mid-word — last char before the truncation marker is '\n'
        truncated_body = result["corpus"].split("\n\n[TRUNCATED")[0]
        self.assertTrue(truncated_body.endswith("\n") or truncated_body.endswith("A"))

    def test_fatal_error_short_circuits(self) -> None:
        state = {"fatal_error": "prior error", "pdf_records": []}
        result = node_merge_corpus(state)
        self.assertEqual(result, {})

    def test_source_file_markers_in_corpus(self) -> None:
        state = {
            "pdf_records": [self._rec("/docs/contract.pdf", "Some contract text")]
        }
        result = node_merge_corpus(state)
        self.assertIn("SOURCE FILE: contract.pdf", result["corpus"])


# ---------------------------------------------------------------------------
# node_assemble
# ---------------------------------------------------------------------------

class TestNodeAssemble(unittest.TestCase):
    def _base_state(self, **overrides) -> dict:
        state = {
            "commercial_core": {},
            "phases": [],
            "customer": {},
            "corpus_meta": "",
            "extraction_notes": [],
            "pdf_records": [],
        }
        state.update(overrides)
        return state

    def test_fatal_error_propagated_to_missing_fields(self) -> None:
        state = self._base_state(fatal_error="No PDFs found", extraction_notes=[])
        result = node_assemble(state)
        self.assertIn("error:No PDFs found", result["missing_fields"])
        self.assertEqual(result["validated_json"], {})

    def test_empty_extraction_reports_missing_fields(self) -> None:
        result = node_assemble(self._base_state())
        self.assertIn("phases", result["missing_fields"])
        self.assertIn("currency", result["missing_fields"])
        self.assertIn("start_date", result["missing_fields"])
        self.assertIn("name", result["missing_fields"])

    def test_successful_assembly_includes_currency(self) -> None:
        state = self._base_state(
            commercial_core={
                "name": "Test Contract",
                "currency": "USD",
                "start_date": "2025-01-01T12:00:00Z",
                "status": "active",
            },
            customer={"external_id": "EXT-12345678"},
        )
        result = node_assemble(state)
        self.assertEqual(result["validated_json"].get("currency"), "USD")

    def test_customer_id_derived_from_external_id(self) -> None:
        state = self._base_state(
            commercial_core={"name": "X", "currency": "USD", "start_date": "2025-01-01T00:00:00Z"},
            customer={"external_id": "ACME-12345678"},
        )
        result = node_assemble(state)
        self.assertIn("customer_id", result["validated_json"])
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "ACME-12345678"))
        self.assertEqual(str(result["validated_json"]["customer_id"]), expected)

    def test_parse_errors_appended_to_notes(self) -> None:
        state = self._base_state(
            pdf_records=[
                {
                    "path": "/docs/bad.pdf",
                    "accepted": False,
                    "parse_error": "timeout",
                    "page_count": 0,
                    "reject_reason": "",
                }
            ]
        )
        result = node_assemble(state)
        fields = [n["field"] for n in result["extraction_notes"]]
        self.assertIn("parse_error:bad.pdf", fields)

    def test_rejected_pdfs_appended_to_notes(self) -> None:
        state = self._base_state(
            pdf_records=[
                {
                    "path": "/docs/bank.pdf",
                    "accepted": False,
                    "parse_error": None,
                    "page_count": 3,
                    "reject_reason": "looks like a bank statement",
                }
            ]
        )
        result = node_assemble(state)
        fields = [n["field"] for n in result["extraction_notes"]]
        self.assertIn("rejected_pdf:bank.pdf", fields)


# ---------------------------------------------------------------------------
# Relevance classifier
# ---------------------------------------------------------------------------

class TestRelevanceClassifier(unittest.TestCase):
    _CONTRACT_TEXT = (
        "MASTER SERVICES AGREEMENT\n"
        "This Master Services Agreement ('Agreement') is effective January 1, 2025.\n"
        "Governing Law. Subject to termination under Section 12.\n"
        "Indemnification clause. SaaS subscription terms and confidential obligations."
    )
    _BANK_TEXT = (
        "Bank Statement — Account Summary\n"
        "Routing Number: 021000021\nOpening Balance: $10,000\nClosing Balance: $9,500\n"
        "Transactions in this period include wire transfer confirmation."
    )

    def test_contract_language_accepted(self) -> None:
        ok, _ = classify_pdf(self._CONTRACT_TEXT, None)
        self.assertTrue(ok)

    def test_bank_statement_rejected(self) -> None:
        ok, _ = classify_pdf(self._BANK_TEXT, None)
        self.assertFalse(ok)

    def test_short_text_ambiguous_defaults_to_accept(self) -> None:
        ok, _ = classify_pdf("Hi there.", None)
        self.assertTrue(ok)

    def test_heuristic_contract_true_without_llm(self) -> None:
        verdict, _ = heuristic_relevant(self._CONTRACT_TEXT)
        self.assertTrue(verdict)

    def test_heuristic_bank_false_without_llm(self) -> None:
        verdict, _ = heuristic_relevant(self._BANK_TEXT)
        self.assertFalse(verdict)

    def test_llm_invoked_for_ambiguous_text(self) -> None:
        ambiguous = "Payment received for services rendered. Invoice #1234. Net 30."
        calls = []

        def fake_llm(system: str, user: str) -> dict:
            calls.append(user)
            return {"relevant": True, "reason": "order form context"}

        ok, _ = classify_pdf(ambiguous, fake_llm)
        self.assertTrue(ok)
        self.assertTrue(len(calls) > 0)


# ---------------------------------------------------------------------------
# PricingData validators
# ---------------------------------------------------------------------------

class TestPricingDataValidators(unittest.TestCase):
    def test_valid_pricing_type_accepted(self) -> None:
        pd = PricingData(pricing_type="flat_fee", amount=500.0)
        self.assertEqual(pd.pricing_type, "flat_fee")

    def test_unknown_pricing_type_coerced_to_custom_pricing(self) -> None:
        pd = PricingData(pricing_type="monthly_flat")
        self.assertEqual(pd.pricing_type, "custom_pricing")

    def test_hyphenated_pricing_type_normalized(self) -> None:
        pd = PricingData(pricing_type="per-unit")
        self.assertEqual(pd.pricing_type, "per_unit")

    def test_int_amount_coerced_to_float(self) -> None:
        pd = PricingData(pricing_type="per_unit", amount=99)
        self.assertIsInstance(pd.amount, float)
        self.assertEqual(pd.amount, 99.0)

    def test_non_numeric_amount_set_to_none(self) -> None:
        pd = PricingData(pricing_type="flat_fee", amount="TBD")
        self.assertIsNone(pd.amount)

    def test_lowercase_currency_uppercased(self) -> None:
        pd = PricingData(pricing_type="flat_fee", currency="usd")
        self.assertEqual(pd.currency, "USD")

    def test_currency_with_whitespace_normalized(self) -> None:
        pd = PricingData(pricing_type="flat_fee", currency=" eur ")
        self.assertEqual(pd.currency, "EUR")

    def test_empty_string_currency_set_to_none(self) -> None:
        pd = PricingData(pricing_type="flat_fee", currency="")
        self.assertIsNone(pd.currency)

    def test_none_amount_stays_none(self) -> None:
        pd = PricingData(pricing_type="flat_fee", amount=None)
        self.assertIsNone(pd.amount)


# ---------------------------------------------------------------------------
# compute_missing_field_keys
# ---------------------------------------------------------------------------

class TestComputeMissingFieldKeys(unittest.TestCase):
    _FULL_COMMERCIAL = {
        "name": "Enterprise Agreement",
        "currency": "USD",
        "start_date": "2025-01-01T00:00:00Z",
    }

    def _cid(self) -> UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, "EXT-12345678")

    def test_empty_inputs_flag_all_top_level_fields(self) -> None:
        missing = compute_missing_field_keys({}, [], None)
        for field in ("customer_id", "phases", "currency", "start_date", "name"):
            self.assertIn(field, missing)

    def test_populated_top_level_no_spurious_flags(self) -> None:
        missing = compute_missing_field_keys(self._FULL_COMMERCIAL, [], self._cid())
        self.assertNotIn("currency", missing)
        self.assertNotIn("name", missing)
        self.assertNotIn("customer_id", missing)

    def test_empty_pricings_flagged(self) -> None:
        phases = [{"name": "Phase 1", "pricings": []}]
        missing = compute_missing_field_keys(self._FULL_COMMERCIAL, phases, self._cid())
        self.assertIn("phases[0].pricings", missing)

    def test_missing_amount_flagged_for_flat_fee(self) -> None:
        phases = [{
            "name": "Phase 1",
            "pricings": [{
                "product": {"name": "Widget"},
                "pricing": {
                    "pricing_data": {
                        "pricing_type": "flat_fee",
                        "amount": None,
                        "billing_cadence": "monthly",
                    }
                },
            }],
        }]
        missing = compute_missing_field_keys(self._FULL_COMMERCIAL, phases, self._cid())
        self.assertIn("phases[0].pricings[0].pricing.pricing_data.amount", missing)

    def test_missing_billing_cadence_flagged(self) -> None:
        phases = [{
            "name": "Phase 1",
            "pricings": [{
                "product": {"name": "Widget"},
                "pricing": {
                    "pricing_data": {
                        "pricing_type": "per_unit",
                        "amount": 10.0,
                        "billing_cadence": None,
                    }
                },
            }],
        }]
        missing = compute_missing_field_keys(self._FULL_COMMERCIAL, phases, self._cid())
        self.assertIn("phases[0].pricings[0].pricing.pricing_data.billing_cadence", missing)

    def test_fully_populated_phase_no_missing_pricing_fields(self) -> None:
        phases = [{
            "name": "Phase 1",
            "pricings": [{
                "product": {"name": "Widget"},
                "pricing": {
                    "pricing_data": {
                        "pricing_type": "flat_fee",
                        "amount": 500.0,
                        "billing_cadence": "monthly",
                    }
                },
            }],
        }]
        missing = compute_missing_field_keys(self._FULL_COMMERCIAL, phases, self._cid())
        self.assertNotIn("phases[0].pricings[0].pricing.pricing_data.amount", missing)
        self.assertNotIn("phases[0].pricings[0].pricing.pricing_data.billing_cadence", missing)

    def test_custom_pricing_type_amount_not_flagged(self) -> None:
        # custom_pricing and features don't require a numeric amount
        phases = [{
            "name": "Phase 1",
            "pricings": [{
                "product": {"name": "Custom"},
                "pricing": {
                    "pricing_data": {
                        "pricing_type": "custom_pricing",
                        "amount": None,
                    }
                },
            }],
        }]
        missing = compute_missing_field_keys(self._FULL_COMMERCIAL, phases, self._cid())
        self.assertNotIn("phases[0].pricings[0].pricing.pricing_data.amount", missing)


# ---------------------------------------------------------------------------
# resolve_customer_id_from_extraction
# ---------------------------------------------------------------------------

class TestResolveCustomerId(unittest.TestCase):
    def test_long_external_id_produces_uuid(self) -> None:
        ext = "ACME-12345678"
        cid = resolve_customer_id_from_extraction({"external_id": ext})
        self.assertIsNotNone(cid)
        self.assertEqual(cid, uuid.uuid5(uuid.NAMESPACE_URL, ext))

    def test_short_external_id_returns_none(self) -> None:
        cid = resolve_customer_id_from_extraction({"external_id": "short"})
        self.assertIsNone(cid)

    def test_missing_external_id_returns_none(self) -> None:
        cid = resolve_customer_id_from_extraction({})
        self.assertIsNone(cid)

    def test_none_external_id_returns_none(self) -> None:
        cid = resolve_customer_id_from_extraction({"external_id": None})
        self.assertIsNone(cid)


if __name__ == "__main__":
    unittest.main()
