"""Schemas for the /materials router."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from cnms_fom.db.enums import ProvenanceTier, SpecimenForm


class DescriptorValueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    descriptor_key: str
    value: float | None
    units: str | None = None
    uncertainty: float | None = None
    method: str | None = None
    provenance_tier: ProvenanceTier
    reduction_rule: str | None = None


class PropertyValueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    property_key: str
    value: float | None
    units: str | None = None
    uncertainty: float | None = None
    tensor_component: str | None = None
    direction: list[float] | None = None
    temperature_k: float | None = None
    frequency_hz: float | None = None
    thickness_nm: float | None = None
    electrode: str | None = None
    substrate: str | None = None
    method: str | None = None
    xc_functional: str | None = None
    doi: str | None = None
    provenance_tier: ProvenanceTier


class PropertyValueIn(BaseModel):
    """A new property value.

    Every context field the registry declares for this property must be present,
    or the value is rejected — see ``fom_engine.eligibility``. Supplying the
    context later is not equivalent: a value entered without it cannot be
    matched to the run that produced it.
    """

    property_key: str
    value: float | None = Field(
        default=None, description="None means NA. It is stored as NA and never imputed (Eq. 4)."
    )
    units: str | None = None
    uncertainty: float | None = None
    tensor_component: str | None = None
    direction: list[float] | None = None
    reduction_rule: str | None = None
    temperature_k: float | None = None
    frequency_hz: float | None = None
    field_amplitude_v_per_cm: float | None = None
    thickness_nm: float | None = None
    electrode: str | None = None
    substrate: str | None = None
    interface: str | None = None
    area_cm2: float | None = None
    failure_criterion: str | None = None
    processing_route: str | None = None
    method: str | None = None
    software: str | None = None
    xc_functional: str | None = None
    pseudopotential: str | None = None
    convergence: str | None = None
    doi: str | None = None
    source_url: str | None = None
    database_identifier: str | None = None
    source_locator: str | None = None
    provenance_tier: ProvenanceTier


class MaterialIn(BaseModel):
    """A material-context record (Eq. 3).

    ``polymorph`` is required. FOM_PROOF Sec. 2.1: a chemical formula is not a
    material identifier — "TiO2" may be rutile, anatase, brookite, amorphous, a
    doped film, or a ceramic, and those are different records.
    """

    formula: str
    polymorph: str = Field(min_length=1)
    specimen_form: SpecimenForm
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    source_database: str | None = None
    source_identifier: str | None = None
    notes: str | None = None


class MaterialOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    formula: str
    formula_reduced: str
    polymorph: str
    specimen_form: SpecimenForm
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    source_database: str | None = None
    source_identifier: str | None = None
    created_at: datetime


class MaterialDetailOut(MaterialOut):
    descriptors: list[DescriptorValueOut] = Field(default_factory=list)
    properties: list[PropertyValueOut] = Field(default_factory=list)


class StructureIn(BaseModel):
    cif: str
    method: str | None = None
    xc_functional: str | None = None
    provenance_tier: ProvenanceTier = ProvenanceTier.CALCULATED
    doi: str | None = None


class DescriptorComputeOut(BaseModel):
    material_id: int
    computed: list[DescriptorValueOut]
    not_derivable_from_structure: list[str] = Field(
        description="Descriptors requiring DFPT or spectroscopy; ingest with their own provenance."
    )
    warnings: list[str] = Field(default_factory=list)
