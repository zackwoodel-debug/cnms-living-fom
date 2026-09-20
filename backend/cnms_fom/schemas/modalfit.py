"""Schemas for the /modalfit router."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from cnms_fom.db.enums import FitAlgorithm, FitTechnique
from cnms_fom.modalfit.slab_model import LENGTH_UNITS


class FitDatasetIn(BaseModel):
    """What one technique's data contributed, when the export did not record it."""

    technique: FitTechnique
    source_filename: str | None = None
    loader: str | None = Field(
        default=None, description="Which loader read it: jaw_vase, rigaku_ras, imes_csv, generic."
    )
    datafed_record_id: str | None = None
    n_points: int | None = Field(default=None, ge=0)
    x_min: float | None = None
    x_max: float | None = None
    x_units: str | None = None
    chi2: float | None = Field(default=None, ge=0)
    weight: float | None = Field(default=None, ge=0)
    settings: dict | None = None


class FitImportRequest(BaseModel):
    """Import one ModalFit export, or a directory of them.

    ``techniques`` is required unless the export carries its own fit metadata.
    It is never inferred from which slab-model blocks are populated: ModalFit
    fills every block a technique *could* read, whether or not data was ever
    loaded for it, so inference here would invent measurements.
    """

    path: str = Field(description="Exported model JSON, or a directory of them, readable by the API.")
    techniques: list[FitTechnique] | None = Field(
        default=None,
        description="Techniques actually co-refined. Required when the export has no fit metadata.",
    )
    algorithm: FitAlgorithm | None = None
    chi2_total: float | None = Field(default=None, ge=0)
    chi2_by_technique: dict[str, float] | None = None
    technique_weights: dict[str, float] | None = None
    datasets: list[FitDatasetIn] | None = None
    length_units: str = Field(
        default="angstrom",
        description="Unit the export's thicknesses are written in. ModalFit's physics backends "
        "use angstroms; its bundled substrate library is written in nanometres. The format does "
        "not record it, so it is not guessed.",
    )
    sample_id: str | None = None
    operator: str | None = None
    notes: str | None = None
    datafed_record_id: str | None = None
    material_id: int | None = None
    experiment_id: int | None = None
    resolution_smearing_applied: bool = Field(
        default=False,
        description="ModalFit builds its refnx models with dq=0, so this is normally False. "
        "Recorded because it biases fringe contrast.",
    )
    roughness_applied_to_spr: bool = Field(
        default=False, description="ModalFit's SPR path ignores layer roughness."
    )

    @field_validator("length_units")
    @classmethod
    def _known_unit(cls, value: str) -> str:
        if value.strip().lower() not in LENGTH_UNITS:
            raise ValueError(
                f"Unknown length unit {value!r}. Expected one of {sorted(set(LENGTH_UNITS))}."
            )
        return value.strip().lower()


class FitImportResponse(BaseModel):
    imported: list[dict]
    total_fits: int
    warnings: list[str] = Field(
        default_factory=list,
        description="The part worth reading: clamped parameters, techniques with no matching "
        "slab-model block, placeholder optical constants, missing chi-squared.",
    )


class FitLayerOut(BaseModel):
    layer_index: int
    role: str
    label: str | None = None
    material: str | None = None
    formula: str | None = None
    thickness_ang: float | None = None
    roughness_ang: float | None = None
    density_g_cm3: float | None = None
    free_parameters: list[str] = Field(default_factory=list)
    bounds: dict | None = None
    uncertainties: dict | None = None
    parameters: dict | None = None


class FitDatasetOut(BaseModel):
    technique: str
    source_filename: str | None = None
    n_points: int | None = None
    x_min: float | None = None
    x_max: float | None = None
    x_units: str | None = None
    chi2: float | None = None
    weight: float | None = None


class FitRecordOut(BaseModel):
    fit_record_id: int
    sample_id: str | None = None
    stack_id: str | None = None
    techniques: list[str]
    algorithm: str | None = None
    chi2_total: float | None = None
    chi2_by_technique: dict | None = None
    n_free_parameters: int | None = None
    uses_placeholder_optical_constants: bool
    resolution_smearing_applied: bool | None = None
    roughness_applied_to_spr: bool | None = None
    fitted_at: str | None = None
    operator: str | None = None
    layers: list[FitLayerOut] = Field(default_factory=list)
    datasets: list[FitDatasetOut] = Field(default_factory=list)
    description: str = Field(
        description="Prose rendering of the fit, caveats included — the form the assistant reads."
    )


class SampleFitsResponse(BaseModel):
    sample_id: str
    n_fits: int
    fits: list[FitRecordOut] = Field(default_factory=list)


class CompareRequest(BaseModel):
    sample_id: str
    parameter: str = Field(
        default="thickness",
        description="thickness, roughness, density, sld_real, sld_imag, n, or k.",
    )
    layer_label: str | None = Field(
        default=None, description="Required when the stack has more than one film layer."
    )


class PromoteRequest(BaseModel):
    """Promote fitted values into the analysis tables.

    ``material_id`` is required and never inferred. A slab-model layer carries a
    formula and nothing else, and FOM_PROOF Eq. (3) makes identity composition +
    polymorph + specimen form — the polymorph has to come from someone who knows
    it (XRD, a growth record), not from the importer.
    """

    material_id: int = Field(description="The material these fitted values belong to.")
    layer_label: str | None = None
    temperature_k: float | None = Field(default=None, gt=0)
    substrate: str | None = None
    dry_run: bool = Field(
        default=True,
        description="True (the default) reports what would be written and what is refused, "
        "without writing. Read the refusals before setting this to False.",
    )
