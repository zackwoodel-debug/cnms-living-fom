"""Schemas for the /bo router."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ParameterSpecIn(BaseModel):
    name: str
    kind: Literal["continuous", "integer", "categorical"] = "continuous"
    units: str = ""
    lower: float | None = None
    upper: float | None = None
    choices: list[Any] = Field(default_factory=list)
    log_scale: bool = Field(
        default=False,
        description="Set for parameters spanning decades (chamber pressure). The GP then sees "
        "log10 of the value, which is where its length scale is meaningful.",
    )
    description: str = ""


class BoRunIn(BaseModel):
    name: str
    search_space: list[ParameterSpecIn]
    fom_name: str | None = Field(
        default=None, description="Objective is ln F of this FOM (Eq. 30). See bo_engine.surrogate."
    )
    instrument_id: str | None = Field(
        default=None, description="Intersects the search space with the tool's envelope."
    )
    constraints: dict | None = None
    acquisition: Literal["qLogEI", "qLogNEI", "qUCB"] = "qLogEI"
    objective_sense: Literal["max", "min"] = "max"
    random_seed: int | None = None


class BoRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    search_space: dict
    constraints: dict | None = None
    acquisition: str
    objective_sense: str
    status: str
    random_seed: int | None = None
    n_observations: int = 0
    n_suggestions: int = 0


class ObservationIn(BaseModel):
    parameters: dict
    objective_value: float | None = None
    objective_noise: float | None = None
    fom_score_id: int | None = None
    experiment_id: int | None = None
    is_feasible: bool = Field(
        default=True,
        description="False for a run that failed. Infeasible runs are dropped from the GP rather "
        "than modelled as poor outcomes — a failed growth says nothing about the objective there.",
    )


class SuggestRequest(BaseModel):
    q: int = Field(default=1, ge=1, le=16, description="How many recipes to propose.")
    acquisition: Literal["qLogEI", "qLogNEI", "qUCB"] | None = None
    seed: int | None = None
    persist: bool = True


class SuggestionOut(BaseModel):
    parameters: dict
    acquisition_value: float | None = None
    predicted_mean: float | None = None
    predicted_std: float | None = None
    strategy: str
    batch_index: int
    suggestion_id: int | None = None


class SuggestResponse(BaseModel):
    run_id: int
    suggestions: list[SuggestionOut]
    strategy: str
    n_observations: int
    objective_name: str
    notes: list[str] = Field(default_factory=list)


class InstrumentOut(BaseModel):
    instrument_id: str
    name: str
    technique: str
    location: str = ""
    capabilities: dict = Field(default_factory=dict)
    available: bool = True
    source: str = Field(
        description="'placeholder' until the CNMS instrument registry is wired in."
    )
