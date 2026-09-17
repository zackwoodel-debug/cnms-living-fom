"""Schemas for the /fom router."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cnms_fom.db.enums import (
    Direction,
    ProvenanceTier,
    ScoreStatus,
    SpecimenForm,
    Transform,
)


class NormalizationSpecIn(BaseModel):
    lo: float
    hi: float
    direction: Direction = Direction.BENEFIT
    transform: Transform = Transform.NONE
    floor_eps: float = 1e-3


class FomDefinitionIn(BaseModel):
    """A new (or new version of a) figure of merit.

    Weights are an application-policy choice, not a measurement (Sec. 6.2), so
    ``approved``/``approved_by`` exist and default to unapproved.
    """

    name: str
    application: str
    version: int = 1
    description: str | None = None
    weights: dict[str, float]
    normalization: dict[str, NormalizationSpecIn]
    floor_eps: float = 1e-3
    bounds_basis: str | None = None
    eligible_set_query: str | None = None
    n_materials_in_bounds: int | None = None
    approved: bool = False
    approved_by: str | None = None

    @field_validator("weights")
    @classmethod
    def _weights_sum_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        total = sum(value.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1 (Eq. 29); got {total}.")
        if any(w < 0 for w in value.values()):
            raise ValueError("Weights must be non-negative (Eq. 29).")
        return value


class FomDefinitionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: int
    application: str
    description: str | None = None
    weights: dict
    normalization: dict
    floor_eps: float
    bounds_basis: str | None = None
    approved: bool
    approved_by: str | None = None
    frozen: bool


class ScoreRequest(BaseModel):
    """Score materials under one FOM definition."""

    fom_name: str
    fom_version: int | None = None
    material_ids: list[int] | None = Field(
        default=None, description="Omit to score every material in the eligible context."
    )
    specimen_forms: list[SpecimenForm] | None = None
    temperature_k: tuple[float, float] | None = None
    frequency_hz: tuple[float, float] | None = None
    provenance_tiers: list[ProvenanceTier] | None = Field(
        default=None,
        description="Which provenance tiers are eligible. Defaults to measured + calculated. "
        "Include 'modeled' to run the separate modeled-scenario analysis Sec. 2.3 permits — the "
        "results come back labelled ILLUSTRATIVE and must be reported separately from "
        "measurement-based results.",
    )
    persist: bool = Field(default=False, description="Write results to fom_scores.")


class ScoreOut(BaseModel):
    material_key: str
    material_id: int | None = None
    status: ScoreStatus
    value: float | None = None
    log_value: float | None = None
    missing_inputs: list[str] = Field(default_factory=list)
    out_of_bounds: list[str] = Field(default_factory=list)
    floored: list[str] = Field(default_factory=list)
    components: dict = Field(default_factory=dict)
    uses_modeled_inputs: bool = False
    note: str | None = None


class ScoreResponse(BaseModel):
    fom_name: str
    fom_version: int
    approved: bool
    n_scored: int
    n_not_scored: int
    n_illustrative: int = Field(
        default=0, description="Scores built on at least one modeled input (Sec. 2.3)."
    )
    results: list[ScoreOut]
    warnings: list[str] = Field(default_factory=list)


class CorrelationRequest(BaseModel):
    """Compute a correlation block (Eqs. 35-38) with permutation + FDR."""

    block: str = Field(default="R_SP", description="One of R_SS, R_SP, R_PP, R_FF, R_SF.")
    descriptor_keys: list[str] | None = None
    property_keys: list[str] | None = None
    specimen_forms: list[SpecimenForm] | None = None
    temperature_k: tuple[float, float] | None = None
    frequency_hz: tuple[float, float] | None = None
    provenance_tiers: list[ProvenanceTier] | None = Field(
        default=None,
        description="Which provenance tiers are eligible. Defaults to measured + calculated. "
        "Include 'modeled' to run the separate modeled-scenario analysis Sec. 2.3 permits — the "
        "results come back labelled ILLUSTRATIVE and must be reported separately from "
        "measurement-based results.",
    )
    permutations: int = Field(default=10_000, ge=99)
    seed: int | None = None
    alpha: float = 0.05
    persist: bool = False


class CorrelationCellOut(BaseModel):
    block: str
    x_key: str
    y_key: str
    x_transform: str
    y_transform: str
    n_complete: int
    pearson_r: float | None = None
    spearman_rho: float | None = None
    p_permutation: float | None = None
    q_fdr: float | None = None
    predicted_sign: str | None = None
    mechanism: str | None = None
    outcome: str | None = None
    material_ids: list[str] = Field(default_factory=list)


class CorrelationResponse(BaseModel):
    cells: list[CorrelationCellOut]
    coverage: dict[str, int] = Field(
        description="Non-NA count per column. There is deliberately no single run-level n (Sec. 7.3)."
    )
    exclusions: list[dict] = Field(default_factory=list)
    run_id: int | None = None
    provenance_warnings: list[str] = Field(default_factory=list)


class MediationRequest(BaseModel):
    """Compute M = B Gamma (Eq. 57) for one application."""

    fom_name: str
    fom_version: int | None = None
    source: str = Field(
        default="theory",
        description="'theory' uses the oscillator elasticities (Eqs. 50-53); "
        "'regression' estimates B from the data (Sec. 9.1).",
    )
    reference_properties: dict[str, float] | None = Field(
        default=None, description="Defaults to the population median of the eligible set."
    )
    reference_descriptors: dict[str, float] | None = None
    descriptor_keys: list[str] | None = None
    specimen_forms: list[SpecimenForm] | None = None
    provenance_tiers: list[ProvenanceTier] | None = Field(
        default=None,
        description="Which provenance tiers are eligible. Defaults to measured + calculated. "
        "Include 'modeled' to run the separate modeled-scenario analysis Sec. 2.3 permits — the "
        "results come back labelled ILLUSTRATIVE and must be reported separately from "
        "measurement-based results.",
    )
    persist: bool = False


class MediationResponse(BaseModel):
    application: str
    fom_name: str
    fom_version: int
    descriptors: list[str]
    properties: list[str]
    mediated_effect: dict[str, float]
    mediated_elasticity: dict[str, float]
    contributions: dict[str, dict[str, float]]
    dominant_property: dict[str, str]
    sensitivity_source: str
    reference_point: dict[str, float]
    notes: list[str] = Field(default_factory=list)
    run_id: int | None = None


class IntegrityRequest(BaseModel):
    fom_names: list[str] = Field(min_length=2, description="At least two applications to compare.")
    specimen_forms: list[SpecimenForm] | None = None
    provenance_tiers: list[ProvenanceTier] | None = Field(
        default=None,
        description="Which provenance tiers are eligible. Defaults to measured + calculated. "
        "Include 'modeled' to run the separate modeled-scenario analysis Sec. 2.3 permits — the "
        "results come back labelled ILLUSTRATIVE and must be reported separately from "
        "measurement-based results.",
    )


class IntegrityResponse(BaseModel):
    applications: list[str]
    observed_correlation: list[list[float | None]]
    null_correlation: list[list[float]] = Field(
        description="Eq. (60): correlation expected from weight overlap alone."
    )
    excess_correlation: list[list[float | None]] = Field(
        description="Eq. (61): observed minus null. This is the part worth interpreting."
    )
    reconstruction_ok: bool | None = None
    max_reconstruction_error: float | None = None
    leakage: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class HypothesisOut(BaseModel):
    x_key: str
    y_key: str
    expected_sign: str
    mechanism: str
    conditional_on: list[str] = Field(default_factory=list)
    scope: str = ""


class HypothesisRegistryOut(BaseModel):
    fingerprint: str = Field(
        description="SHA-256 of the pre-registered table, stamped onto every analysis run."
    )
    hypotheses: list[HypothesisOut]
