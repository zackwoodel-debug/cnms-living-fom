"""Request and response shapes for the pySEA router.

Deliberately loose about the envelope itself. ``PySeaImportRequest.envelope`` is a
plain dict rather than a modelled schema, because the canonical contract is
provisional and a Pydantic model would have to be rewritten alongside every
mapping change. ``pysea.validate`` does the checking, and it reports rather than
raises, which is the behaviour a caller needs when the container is close but not
right.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PySeaImportRequest(BaseModel):
    """One container to import, inline or by path."""

    envelope: dict[str, Any] | None = Field(
        default=None, description="The canonical pySEA envelope, inline."
    )
    path: str | None = Field(
        default=None,
        description="Path to a JSON envelope on a filesystem the server can read.",
    )
    source_filename: str | None = Field(
        default=None, description="Recorded on the row so the container stays identifiable."
    )
    unit_hints: dict[str, str] | None = Field(
        default=None,
        description=(
            "Units the caller asserts for an axis kind, applied only where the container "
            "is silent. Each axis records which source won."
        ),
    )

    @field_validator("path")
    @classmethod
    def _path_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("path was given but is empty.")
        return value


class PySeaIssueOut(BaseModel):
    code: str
    severity: str
    field_path: str
    message: str
    remediation: str | None = None


class PySeaImportResponse(BaseModel):
    id: int
    record_id: str
    record_kind: str
    sample_id: str
    instrument_id: str | None = None
    datafed_record_id: str | None = None
    content_sha256: str
    validation_status: str
    issues: list[PySeaIssueOut] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    n_signals: int
    n_scalars: int
    created: bool = Field(
        description="False when the content hash matched an existing row: import is idempotent."
    )


class PySeaRecordSummary(BaseModel):
    id: int
    record_id: str
    record_kind: str
    sample_id: str
    instrument_id: str | None = None
    datafed_record_id: str | None = None
    validation_status: str
    n_signals: int
    n_scalars: int


class PySeaPromoteRequest(BaseModel):
    """Promotion needs an identity and an explicit decision to write."""

    material_id: int = Field(
        description=(
            "Required and never inferred. Identity is composition + polymorph + specimen "
            "form; a sample label carries none of the three."
        )
    )
    scalar_ids: list[int] | None = Field(
        default=None, description="Narrow to a subset. Omitted, every eligible scalar."
    )
    dry_run: bool = Field(
        default=True,
        description="True previews. Promotion writes into the tables the FOM is computed from.",
    )
    operator: str | None = Field(default=None, description="Who asked for the promotion.")


class PySeaCompareRequest(BaseModel):
    sample_id: str = Field(min_length=1)
    quantity: str = Field(
        min_length=1, description="A registry property key, such as eps_inf."
    )

    @field_validator("sample_id", "quantity")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class PySeaDeterminationOut(BaseModel):
    platform: str
    label: str
    value: float
    uncertainty: float | None = None
    units: str | None = None
    method: str
    provenance_tier: str
    source: str


class PySeaCompareResponse(BaseModel):
    sample_id: str
    quantity: str
    platforms: list[str]
    n_determinations: int
    determinations: list[PySeaDeterminationOut]
    verdict: str
    summary: str
    note: str
