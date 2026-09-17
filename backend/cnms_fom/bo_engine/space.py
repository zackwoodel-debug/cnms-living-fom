"""Search-space definition for growth recipes.

A recipe parameter is not just a number with bounds — it has units and a tool
that has to be able to reach it.  Keeping units on the parameter means a
suggestion can be handed to an operator without a unit-conversion step, which is
a reliable source of wasted runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ParameterType = Literal["continuous", "integer", "categorical"]

#  Enumerating categorical combinations is exponential; refuse past this.
MAX_CATEGORICAL_COMBINATIONS = 64


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable knob of a growth recipe."""

    name: str
    kind: ParameterType
    units: str = ""
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()
    log_scale: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: a categorical parameter needs choices.")
            return
        if self.lower is None or self.upper is None:
            raise ValueError(f"{self.name}: a {self.kind} parameter needs lower and upper bounds.")
        if self.upper <= self.lower:
            raise ValueError(f"{self.name}: upper ({self.upper}) must exceed lower ({self.lower}).")
        if self.log_scale and self.lower <= 0:
            raise ValueError(f"{self.name}: a log-scale parameter needs a positive lower bound.")

    @property
    def is_numeric(self) -> bool:
        return self.kind in ("continuous", "integer")

    def to_model_space(self, value: float) -> float:
        """Map a physical value into the space the GP sees."""
        return float(np.log10(value)) if self.log_scale else float(value)

    def from_model_space(self, value: float) -> float | int:
        """Map back to physical units, rounding integers."""
        physical = float(10.0**value) if self.log_scale else float(value)
        if self.kind == "integer":
            return int(round(physical))
        return physical

    def model_bounds(self) -> tuple[float, float]:
        return self.to_model_space(self.lower), self.to_model_space(self.upper)  # type: ignore[arg-type]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "units": self.units,
            "lower": self.lower,
            "upper": self.upper,
            "choices": list(self.choices),
            "log_scale": self.log_scale,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParameterSpec:
        return cls(
            name=data["name"],
            kind=data.get("kind", "continuous"),
            units=data.get("units", ""),
            lower=data.get("lower"),
            upper=data.get("upper"),
            choices=tuple(data.get("choices", ())),
            log_scale=bool(data.get("log_scale", False)),
            description=data.get("description", ""),
        )


@dataclass
class SearchSpace:
    """An ordered set of parameters, plus conversion to and from GP coordinates."""

    parameters: list[ParameterSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate parameter names: {sorted(duplicates)}")

    @property
    def numeric(self) -> list[ParameterSpec]:
        return [p for p in self.parameters if p.is_numeric]

    @property
    def categorical(self) -> list[ParameterSpec]:
        return [p for p in self.parameters if p.kind == "categorical"]

    @property
    def dim(self) -> int:
        return len(self.numeric)

    def bounds_array(self) -> np.ndarray:
        """``(2, d)`` array of model-space bounds for the numeric dimensions."""
        if not self.numeric:
            raise ValueError("Search space has no numeric dimensions to optimize over.")
        lows, highs = zip(*(p.model_bounds() for p in self.numeric), strict=True)
        return np.asarray([lows, highs], dtype=float)

    def encode(self, recipe: dict) -> np.ndarray:
        """Physical recipe -> model-space vector over the numeric dimensions."""
        missing = [p.name for p in self.numeric if p.name not in recipe]
        if missing:
            raise KeyError(f"Recipe is missing numeric parameters {missing}.")
        return np.asarray(
            [p.to_model_space(float(recipe[p.name])) for p in self.numeric], dtype=float
        )

    def decode(self, vector, categorical_values: dict | None = None) -> dict:
        """Model-space vector -> physical recipe."""
        vector = np.asarray(vector, dtype=float).ravel()
        if vector.size != self.dim:
            raise ValueError(f"Expected {self.dim} numeric values, got {vector.size}.")
        recipe: dict = {
            p.name: p.from_model_space(v) for p, v in zip(self.numeric, vector, strict=True)
        }
        recipe.update(categorical_values or {})
        return recipe

    def categorical_combinations(self) -> list[dict]:
        """Every combination of categorical settings.

        BoTorch optimizes continuous dimensions; categoricals are handled by
        enumerating them and optimizing within each, which is exact for the small
        spaces a growth campaign actually has (substrate, target, carrier gas).
        """
        import itertools

        if not self.categorical:
            return [{}]
        total = int(np.prod([len(p.choices) for p in self.categorical]))
        if total > MAX_CATEGORICAL_COMBINATIONS:
            raise ValueError(
                f"{total} categorical combinations exceeds the cap of "
                f"{MAX_CATEGORICAL_COMBINATIONS}. Split the campaign, or model the categorical "
                "dimension with a proper mixed-variable kernel."
            )
        names = [p.name for p in self.categorical]
        return [
            dict(zip(names, values, strict=True))
            for values in itertools.product(*(p.choices for p in self.categorical))
        ]

    def as_dict(self) -> dict:
        return {"parameters": [p.as_dict() for p in self.parameters]}

    @classmethod
    def from_dict(cls, data: dict) -> SearchSpace:
        return cls(parameters=[ParameterSpec.from_dict(p) for p in data.get("parameters", [])])


def example_pld_space() -> SearchSpace:
    """A worked PLD oxide-growth space, for testing the loop end to end.

    TODO(CNMS): replace with the real envelope of the target chamber. The values
    below are typical of oxide PLD in the literature and are a placeholder for
    a tool-specific range pulled from the instrument registry.
    """
    return SearchSpace(
        parameters=[
            ParameterSpec(
                name="substrate_temp_c",
                kind="continuous",
                units="degC",
                lower=400.0,
                upper=850.0,
                description="Substrate heater setpoint.",
            ),
            ParameterSpec(
                name="o2_pressure_mtorr",
                kind="continuous",
                units="mTorr",
                lower=1e-2,
                upper=3e2,
                log_scale=True,  # spans four decades; a linear GP prior wastes its range
                description="Background oxygen partial pressure.",
            ),
            ParameterSpec(
                name="laser_fluence_j_cm2",
                kind="continuous",
                units="J/cm^2",
                lower=0.5,
                upper=3.0,
                description="On-target laser fluence.",
            ),
            ParameterSpec(
                name="repetition_rate_hz",
                kind="integer",
                units="Hz",
                lower=1,
                upper=20,
                description="Pulse repetition rate.",
            ),
            ParameterSpec(
                name="substrate",
                kind="categorical",
                choices=("SrTiO3(001)", "MgO(001)", "Al2O3(0001)", "Si(001)"),
                description="Substrate material and orientation.",
            ),
        ]
    )
