"""ContractV2 Pydantic model tests (stdlib unittest)."""

from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from uuid import UUID

from src.models.contract_v2 import (
    CommercialCore,
    ContractV2,
    NotesExtraction,
    PhaseLine,
    PhasesExtraction,
    SubmissionEnvelope,
    build_contract_payload_from_extraction,
    build_submission_envelope,
    contract_to_validated_json,
    validate_contract_payload,
)

_ROOT = Path(__file__).resolve().parents[1]


class TestContractV2(unittest.TestCase):
    def test_contract_v2_inherits_commercial_core(self) -> None:
        self.assertTrue(issubclass(ContractV2, CommercialCore))
        cc = CommercialCore(name="X", currency="USD")
        self.assertEqual(cc.name, "X")

    def test_submission_envelope_and_build_payload(self) -> None:
        env = build_submission_envelope(
            customer={"customer_name": "Acme"},
            validated_json={"name": "Deal"},
            missing_fields=["customer_id", "customer_id"],
            extraction_notes=[{"field": "name", "quote_or_span": "Acme"}],
        )
        self.assertIsInstance(env, SubmissionEnvelope)
        self.assertEqual(env.missing_fields, ["customer_id"])
        payload = build_contract_payload_from_extraction(
            {
                "name": "N",
                "currency": "USD",
                "start_date": "2025-01-01T00:00:00Z",
            },
            [],
            {},
            "meta",
        )
        c, errs = validate_contract_payload(payload)
        self.assertEqual(errs, [])
        assert c is not None

    def test_pack_001_validated_json_round_trip(self) -> None:
        path = _ROOT / "output" / "pack_001.json"
        if not path.exists():
            self.skipTest(f"Missing fixture {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        vj = data["validated_json"]
        contract, errs = validate_contract_payload(vj)
        self.assertEqual(errs, [])
        assert contract is not None
        self.assertIsNotNone(contract.name)
        self.assertGreater(len(contract.phases), 0)
        dumped = contract_to_validated_json(contract)
        self.assertEqual(dumped["currency"], "USD")
        # customer_id is only present when external_id is ≥8 chars
        if "customer_id" in dumped:
            UUID(str(dumped["customer_id"]))

    def test_customer_id_optional_when_absent(self) -> None:
        payload = {
            "name": "Test",
            "status": "draft",
            "currency": "USD",
            "start_date": "2025-01-01T00:00:00Z",
            "customer_id": None,
            "phases": [{"name": "P1", "pricings": []}],
        }
        c, errs = validate_contract_payload(payload)
        self.assertEqual(errs, [])
        assert c is not None
        self.assertIsNone(c.customer_id)
        out = contract_to_validated_json(c)
        self.assertNotIn("customer_id", out)

    def test_external_id_derived_uuid_validates(self) -> None:
        ext = "ACCOUNT-12345678"
        cid = str(uuid.uuid5(uuid.NAMESPACE_URL, ext))
        payload = {
            "name": "N",
            "status": "active",
            "currency": "EUR",
            "start_date": "2025-06-01T12:00:00Z",
            "customer_id": cid,
            "customer": {"id": cid, "customer_name": "Acme"},
            "phases": [{"name": "Phase A", "pricings": []}],
        }
        c, errs = validate_contract_payload(payload)
        self.assertEqual(errs, [])
        assert c is not None
        self.assertEqual(str(c.customer_id), cid)

    def test_test_0030_paid_phase_line_shape(self) -> None:
        """Paid Phase from benchmark_0030: product + pricing.pricing_data (flat_fee + currency)."""
        path = _ROOT / "output" / "test_0030.json"
        if not path.exists():
            self.skipTest(f"Missing fixture {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        vj = data["validated_json"]
        contract, errs = validate_contract_payload(vj)
        self.assertEqual(errs, [])
        assert contract is not None
        paid = contract.phases[1]
        assert paid.name == "Paid Phase"
        self.assertEqual(len(paid.pricings), 1)
        line = paid.pricings[0]
        assert isinstance(line, PhaseLine)
        assert line.product is not None
        if isinstance(line.product, dict):
            self.fail("expected PhaseLineProduct, got dict")
        self.assertEqual(line.product.name, "Mixed Form")
        self.assertEqual(line.product.type, "product")
        assert line.pricing is not None
        pd = line.pricing.pricing_data
        if isinstance(pd, dict):
            self.fail("expected PricingData, got dict")
        self.assertEqual(pd.pricing_type, "flat_fee")
        self.assertEqual(pd.amount, 7000.0)
        self.assertEqual(pd.currency, "GBP")
        # billing_cadence may be absent depending on extraction run / fixture version

    def test_phases_extraction_from_test_0030(self) -> None:
        path = _ROOT / "output" / "test_0030.json"
        if not path.exists():
            self.skipTest(f"Missing fixture {path}")
        phases = json.loads(path.read_text(encoding="utf-8"))["validated_json"]["phases"]
        pe = PhasesExtraction.model_validate({"phases": phases})
        self.assertEqual(len(pe.phases), 2)
        self.assertEqual(pe.phases[0].name, "Trial Phase")

    def test_notes_extraction_from_test_0030(self) -> None:
        path = _ROOT / "output" / "test_0030.json"
        if not path.exists():
            self.skipTest(f"Missing fixture {path}")
        notes = json.loads(path.read_text(encoding="utf-8"))["extraction_notes"]
        ne = NotesExtraction.model_validate({"extraction_notes": notes})
        self.assertGreater(len(ne.extraction_notes), 0)
        self.assertIsInstance(ne.extraction_notes[0].field, str)


if __name__ == "__main__":
    unittest.main()
