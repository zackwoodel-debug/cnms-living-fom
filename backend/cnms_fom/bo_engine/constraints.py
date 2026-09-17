"""Feasibility constraints on proposed recipes.

An optimizer will happily propose a point the tool cannot reach.  Three layers,
applied in order:

  1. **Instrument envelope** — the physical capability of the tool. Hard.
  2. **Safety / policy limits** — facility rules. Hard.
  3. **Campaign constraints** — the scientist's own restrictions for this study.

Constraints intersect the search space *before* optimization rather than
filtering suggestions afterwards. Filtering after the fact wastes the
acquisition budget on regions that were never available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .space import ParameterSpec, SearchSpace


@dataclass
class ConstraintViolation:
    parameter: str
    value: object
    reason: str


@dataclass
class ConstraintSet:
    """Bounds and rules layered on top of a search space."""

    #  {parameter: (lower, upper)} — tightens, never widens, the space.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #  {parameter: [allowed choices]} for categoricals.
    allowed_choices: dict[str, list] = field(default_factory=dict)
    #  Free-text rules a human must check. TODO(CNMS): formalise against the
    #  instrument registry once its schema is known.
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "allowed_choices": self.allowed_choices,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ConstraintSet:
        data = data or {}
        return cls(
            bounds={k: (float(v[0]), float(v[1])) for k, v in (data.get("bounds") or {}).items()},
            allowed_choices=dict(data.get("allowed_choices") or {}),
            notes=list(data.get("notes") or []),
        )


def from_instrument_capabilities(capabilities: dict | None) -> ConstraintSet:
    """Build a constraint set from an ``Instrument.capabilities`` blob.

    Expected shape: ``{parameter: {"min": x, "max": y, "units": "..."}}``.
    Units are *not* converted — a mismatch between the registry's units and the
    search space's is a configuration error that should surface loudly rather
    than be silently rescaled.
    """
    constraints = ConstraintSet()
    for name, envelope in (capabilities or {}).items():
        if not isinstance(envelope, dict):
            continue
        low, high = envelope.get("min"), envelope.get("max")
        if low is not None and high is not None:
            constraints.bounds[name] = (float(low), float(high))
        if envelope.get("choices"):
            constraints.allowed_choices[name] = list(envelope["choices"])
    return constraints


def apply(space: SearchSpace, constraints: ConstraintSet) -> SearchSpace:
    """Intersect a search space with a constraint set.

    Raises when the intersection is empty — an unsatisfiable campaign should stop
    here, not produce suggestions no tool can run.
    """
    tightened: list[ParameterSpec] = []

    for parameter in space.parameters:
        if parameter.kind == "categorical":
            allowed = constraints.allowed_choices.get(parameter.name)
            if allowed is None:
                tightened.append(parameter)
                continue
            choices = tuple(c for c in parameter.choices if c in allowed)
            if not choices:
                raise ValueError(
                    f"{parameter.name}: no overlap between the search space choices "
                    f"{list(parameter.choices)} and the allowed choices {allowed}."
                )
            tightened.append(ParameterSpec(**{**parameter.as_dict(), "choices": choices}))
            continue

        bounds = constraints.bounds.get(parameter.name)
        if bounds is None:
            tightened.append(parameter)
            continue

        low = max(float(parameter.lower), bounds[0])   # type: ignore[arg-type]
        high = min(float(parameter.upper), bounds[1])  # type: ignore[arg-type]
        if high <= low:
            raise ValueError(
                f"{parameter.name}: the instrument envelope [{bounds[0]}, {bounds[1]}] "
                f"{parameter.units} does not overlap the requested range "
                f"[{parameter.lower}, {parameter.upper}] {parameter.units}."
            )
        tightened.append(ParameterSpec(**{**parameter.as_dict(), "lower": low, "upper": high}))

    return SearchSpace(parameters=tightened)


def validate_recipe(recipe: dict, space: SearchSpace) -> list[ConstraintViolation]:
    """Check a concrete recipe against the space. Empty list means feasible."""
    violations: list[ConstraintViolation] = []
    by_name = {p.name: p for p in space.parameters}

    for name, value in recipe.items():
        parameter = by_name.get(name)
        if parameter is None:
            violations.append(ConstraintViolation(name, value, "not a parameter of this space"))
            continue
        if parameter.kind == "categorical":
            if value not in parameter.choices:
                violations.append(
                    ConstraintViolation(name, value, f"not among {list(parameter.choices)}")
                )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            violations.append(ConstraintViolation(name, value, "not numeric"))
            continue
        if not (float(parameter.lower) <= numeric <= float(parameter.upper)):  # type: ignore[arg-type]
            violations.append(
                ConstraintViolation(
                    name,
                    value,
                    f"outside [{parameter.lower}, {parameter.upper}] {parameter.units}".rstrip(),
                )
            )

    for parameter in space.parameters:
        if parameter.name not in recipe:
            violations.append(ConstraintViolation(parameter.name, None, "missing from the recipe"))
    return violations
