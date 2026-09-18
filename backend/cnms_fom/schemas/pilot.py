"""Schemas for the /pilot router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cnms_fom.db.enums import ProvenanceTier


class PilotRunRequest(BaseModel):
    """Create the HfO2-on-Si pilot campaign.

    Defaults describe a gate-stack study: film thickness and surface roughness
    over ranges a PLD or ALD tool can actually reach, plus a dopant fraction
    that trades permittivity against bandgap.
    """

    name: str = "HfO2 logic pilot"
    fom_name: str = Field(default="logic", description="Which FOM the loop maximises (as ln F).")
    instrument_id: str | None = Field(
        default="CNMS-PLD-01",
        description="Intersects the search space with the tool's envelope. Placeholder registry.",
    )
    thickness_ang: tuple[float, float] = (50.0, 400.0)
    roughness_ang: tuple[float, float] = (1.0, 10.0)
    dopant_fraction: tuple[float, float] | None = (0.0, 0.30)
    acquisition: str = "qLogEI"
    random_seed: int | None = None


class IterateRequest(BaseModel):
    q: int = Field(default=1, ge=1, le=8, description="Recipes to propose and evaluate.")
    persist: bool = Field(
        default=True,
        description="Log the simulated observations so the surrogate retrains. "
        "False makes the call a dry run.",
    )
    seed: int | None = None
    simulate_reflectivity: bool = Field(
        default=True, description="Run the XRR forward model. Off makes the loop faster in tests."
    )


class EvaluationOut(BaseModel):
    recipe: dict
    stack: dict
    xrr: dict
    properties: dict
    fom_value: float | None
    fom_log_value: float | None
    fom_status: str
    objective_value: float | None
    material_id: int | None = None
    analysis_run_id: int | None = None
    fom_score_id: int | None = None
    observation_id: int | None = None
    suggestion_id: int | None = None
    notes: list[str] = Field(default_factory=list)


class IterateResponse(BaseModel):
    run_id: int
    fom_name: str
    evaluations: list[EvaluationOut]
    n_observations_after: int
    disclaimer: str = (
        "Simulated properties are MODELED, so every score here is ILLUSTRATIVE. The loop "
        "exercises the machinery; it does not produce a ranking (FOM_PROOF Sec. 2.3)."
    )


class MeasuredProperty(BaseModel):
    """One measured value, with the context its property requires."""

    property_key: str
    value: float
    units: str | None = None
    uncertainty: float | None = None
    tensor_component: str | None = None
    temperature_k: float | None = None
    frequency_hz: float | None = None
    field_amplitude_v_per_cm: float | None = None
    thickness_nm: float | None = None
    electrode: str | None = None
    substrate: str | None = None
    interface: str | None = None
    area_cm2: float | None = None
    failure_criterion: str | None = None
    method: str | None = None
    xc_functional: str | None = None
    doi: str | None = None
    provenance_tier: ProvenanceTier = ProvenanceTier.MEASURED


class IngestExperimentRequest(BaseModel):
    """Close the loop with real data instead of a simulation."""

    recipe: dict = Field(description="The recipe that was actually run.")
    measurements: list[MeasuredProperty] = Field(min_length=1)
    external_experiment_id: str | None = None
    instrument_id: str | None = None
    proposal_id: str | None = None
    operator: str | None = None
    sample_id: str | None = None
    succeeded: bool = Field(
        default=True,
        description="False for a failed growth. Logged as infeasible, which drops it from the "
        "surrogate rather than modelling it as a poor outcome.",
    )
    notes: str | None = None


class IngestExperimentResponse(BaseModel):
    run_id: int
    experiment_id: int
    material_id: int
    properties_stored: int
    fom_name: str
    fom_status: str
    fom_value: float | None = None
    fom_log_value: float | None = None
    missing_inputs: list[str] = Field(default_factory=list)
    fom_score_id: int | None = None
    observation_id: int | None = None
    objective_value: float | None = None
    warnings: list[str] = Field(default_factory=list)


class StackExportResponse(BaseModel):
    recipe: dict
    stack: dict
    stack_json: str
    nk_csv: str
    nk_csv_rows: int
