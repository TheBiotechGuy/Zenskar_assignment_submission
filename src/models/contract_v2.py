"""Pydantic v2 models for ContractV2 create-style payloads (validated_json)."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


def format_validation_error_paths(exc: ValidationError) -> list[str]:
    """Turn Pydantic errors into dotted paths like phases[0].pricings[1].product.name."""
    out: list[str] = []
    for err in exc.errors():
        loc = err.get("loc") or ()
        parts: list[str] = []
        for p in loc:
            if isinstance(p, int):
                if parts:
                    parts[-1] = f"{parts[-1]}[{p}]"
                else:
                    parts.append(f"[{p}]")
            else:
                parts.append(str(p))
        path = ".".join(parts) if parts else "root"
        out.append(f"validation:{path}")
    return out


class Address(BaseModel):
    model_config = ConfigDict(extra="allow")

    line1: str | None = None
    line2: str | None = None
    city: str | None = None
    state: str | None = None
    zipCode: str | None = None
    country: str | None = None
    country_code: str | None = None


class TaxInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    tax_id_type: str | None = None
    tax_id_value: str | None = None


class ContractCustomer(BaseModel):
    """
    Customer on a contract: LLM extraction (leave id null) and assembly (set id when customer_id is grounded).
    """

    model_config = ConfigDict(extra="ignore")

    id: UUID | None = Field(default=None, description="Zenskar customer UUID; set at assembly, not by extraction.")
    external_id: str | None = None
    customer_name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    address: Address | dict[str, Any] | None = None
    tax_info: list[TaxInfo | dict[str, Any]] | None = None
    custom_attributes: dict[str, Any] | None = None


class PricingData(BaseModel):
    """Zenskar pricing_data discriminator + common fields; extra keys for tiered/volume/etc."""

    model_config = ConfigDict(extra="allow")

    pricing_type: str
    amount: float | int | None = None
    currency: str | None = None
    billing_cadence: str | None = None
    


class PhaseLineProduct(BaseModel):
    """Product block under each phase pricing line."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    type: str = "product"
    sku: str | None = None


class PhaseLinePricingBlock(BaseModel):
    """pricing wrapper with nested pricing_data."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    pricing_data: PricingData | dict[str, Any]


class PhaseLine(BaseModel):
    """One entry in phase.pricings (matches e.g. test_0030 Paid Phase lines)."""

    model_config = ConfigDict(extra="allow")

    description: str | None = None
    product: PhaseLineProduct | dict[str, Any]
    pricing: PhaseLinePricingBlock | dict[str, Any]


class Phase(BaseModel):
    """Contract phase with line-item pricings."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    phase_type: str | None = None
    pricings: list[PhaseLine | dict[str, Any]] = Field(default_factory=list)

    @field_validator("pricings", mode="before")
    @classmethod
    def _pricings_none_to_empty(cls, v: Any) -> Any:
        if v is None:
            return []
        return v


class PhasesExtraction(BaseModel):
    """Top-level JSON from the phases & pricing LLM agent: {\"phases\": [...]}."""

    model_config = ConfigDict(extra="ignore")

    phases: list[Phase] = Field(default_factory=list)

    @field_validator("phases", mode="before")
    @classmethod
    def _phases_none_to_empty(cls, v: Any) -> Any:
        if v is None:
            return []
        return v


class ExtractionNoteItem(BaseModel):
    """One audit line: field path + verbatim quote from the contract text."""

    model_config = ConfigDict(extra="ignore")

    field: str = Field(
        description='JSON-path style key, e.g. "start_date" or "phases[0].pricings[0].pricing.pricing_data.amount".',
    )
    quote_or_span: str = Field(description="Verbatim substring copied from the contract text.")


class NotesExtraction(BaseModel):
    """Top-level JSON from the extraction-notes LLM agent."""

    model_config = ConfigDict(extra="ignore")

    extraction_notes: list[ExtractionNoteItem] = Field(default_factory=list)

    @field_validator("extraction_notes", mode="before")
    @classmethod
    def _notes_none_to_empty(cls, v: Any) -> Any:
        if v is None:
            return []
        return v


class IngestionSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    ingestion: str = "zenskar_assignment_langgraph"
    corpus_meta: str = ""


class CommercialCore(BaseModel):
    """
    High-level commercial fields from contract text (LLM extraction + ContractV2 base).
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str | None = None
    description: str | None = None
    status: str = Field(default="draft", description="draft | active per extraction")
    currency: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    anchor_date: str | None = None
    renewal_policy: str | None = None
    tags: list[str] | None = None
    custom_attributes: dict[str, Any] | None = None
    contract_link: str | None = None


class ContractV2(CommercialCore):
    """
    Create-oriented contract payload for validated_json.
    Response-only fields (id, created_at, updated_at) are omitted.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    customer_id: Annotated[
        UUID | None,
        Field(default=None, description="Set only when grounded (external_id)."),
    ]
    customer: ContractCustomer | None = None

    phases: list[Phase] = Field(default_factory=list)

    source: IngestionSource | None = None

    @field_validator("customer_id", mode="before")
    @classmethod
    def coerce_customer_id(cls, v: Any) -> UUID | str | None:
        if v is None or v == "":
            return None
        return v


def contract_to_validated_json(contract: ContractV2) -> dict[str, Any]:
    """Serialize for submission JSON (no None values)."""
    return contract.model_dump(mode="json", exclude_none=True)


def validate_contract_payload(data: dict[str, Any]) -> tuple[ContractV2 | None, list[str]]:
    """
    Validate extraction payload. Returns (model, []) on success;
    on ValidationError returns (None, validation path strings).
    """
    try:
        return ContractV2.model_validate(data), []
    except ValidationError as e:
        return None, format_validation_error_paths(e)


class SubmissionEnvelope(BaseModel):
    """
    Final runner output matching the assignment README:
    customer, validated_json (ContractV2), missing_fields, extraction_notes.
    """

    model_config = ConfigDict(extra="forbid")

    customer: dict[str, Any] = Field(default_factory=dict)
    validated_json: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    extraction_notes: list[dict[str, Any]] = Field(default_factory=list)


def resolve_customer_id_from_extraction(customer: dict[str, Any]) -> UUID | None:
    """Derive customer_id only from document-grounded external_id (length >= 8)."""
    ext = customer.get("external_id")
    if isinstance(ext, str) and len(ext) >= 8:
        return uuid.uuid5(uuid.NAMESPACE_URL, ext)
    return None


def build_embedded_customer_dict(
    customer: dict[str, Any],
    customer_id: UUID | None,
) -> dict[str, Any] | None:
    """Nested customer for validated_json; includes id when customer_id is grounded."""
    base = {k: v for k, v in customer.items() if not str(k).startswith("_")}
    if not base and customer_id is None:
        return None
    out: dict[str, Any] = dict(base)
    if customer_id is not None:
        out["id"] = customer_id
    return out


def build_contract_payload_from_extraction(
    commercial_core: dict[str, Any],
    phases: list[Any],
    customer: dict[str, Any],
    corpus_meta: str,
) -> dict[str, Any]:
    """
    Merge customer, commercial_core, and phases node outputs into a dict for ContractV2.model_validate.
    """
    customer_id = resolve_customer_id_from_extraction(customer)
    core = CommercialCore.model_validate(commercial_core or {})
    payload: dict[str, Any] = core.model_dump(mode="json", exclude_none=False)
    if payload.get("currency") == "":
        payload["currency"] = None
    payload["status"] = payload.get("status") or "draft"
    payload["customer_id"] = customer_id
    payload["phases"] = phases
    payload["source"] = {
        "ingestion": "zenskar_assignment_langgraph",
        "corpus_meta": corpus_meta,
    }
    emb = build_embedded_customer_dict(customer, customer_id)
    if emb is not None:
        payload["customer"] = emb
    return payload


def compute_missing_field_keys(
    commercial_core: dict[str, Any],
    phases: list[Any],
    customer_id: UUID | None,
) -> list[str]:
    """Heuristic missing_fields from extracted commercial/phases/customer_id."""
    missing: list[str] = []
    if customer_id is None:
        missing.append("customer_id")
    if commercial_core.get("start_date") in (None, ""):
        missing.append("start_date")
    if commercial_core.get("currency") in (None, ""):
        missing.append("currency")
    if not phases:
        missing.append("phases")
    if commercial_core.get("name") in (None, ""):
        missing.append("name")
    return missing


def build_submission_envelope(
    *,
    customer: dict[str, Any],
    validated_json: dict[str, Any],
    missing_fields: list[str],
    extraction_notes: list[dict[str, Any]],
) -> SubmissionEnvelope:
    """Normalize and validate the final API response shape."""
    return SubmissionEnvelope(
        customer=dict(customer or {}),
        validated_json=dict(validated_json or {}),
        missing_fields=sorted(set(missing_fields)),
        extraction_notes=list(extraction_notes or []),
    )
