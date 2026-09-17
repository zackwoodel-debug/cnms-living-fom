"""The ask/tell loop: turn observations into the next recipes to run.

Mixed numeric/categorical spaces are handled the standard way — one GP over
[numeric dims | one-hot categorical dims], then the acquisition is optimized
once per categorical combination with those one-hot columns pinned. Because a
single model produces every candidate, acquisition values are directly
comparable across combinations, which is what makes "take the best q" valid.
Fitting a separate GP per combination would fragment the data and produce
incomparable scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .acquisition import build_acquisition, optimize
from .constraints import ConstraintSet
from .constraints import apply as apply_constraints
from .space import SearchSpace
from .surrogate import MIN_OBSERVATIONS_FOR_GP, fit_gp, posterior_summary, sobol_points

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """One evaluated recipe."""

    parameters: dict
    objective: float
    noise: float | None = None
    is_feasible: bool = True


@dataclass
class Suggestion:
    """A proposed recipe, with why it was proposed."""

    parameters: dict
    acquisition_value: float | None = None
    predicted_mean: float | None = None
    predicted_std: float | None = None
    strategy: str = "bayesian"
    batch_index: int = 0

    def as_dict(self) -> dict:
        return {
            "parameters": self.parameters,
            "acquisition_value": self.acquisition_value,
            "predicted_mean": self.predicted_mean,
            "predicted_std": self.predicted_std,
            "strategy": self.strategy,
            "batch_index": self.batch_index,
        }


@dataclass
class SuggestionBatch:
    suggestions: list[Suggestion] = field(default_factory=list)
    strategy: str = "bayesian"
    n_observations: int = 0
    objective_name: str = "log_fom"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "suggestions": [s.as_dict() for s in self.suggestions],
            "strategy": self.strategy,
            "n_observations": self.n_observations,
            "objective_name": self.objective_name,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Encoding: numeric dims + one-hot categorical dims
# ---------------------------------------------------------------------------


def _onehot_layout(space: SearchSpace) -> list[tuple[str, object, int]]:
    """``(parameter_name, choice, column_index)`` for every categorical level."""
    layout: list[tuple[str, object, int]] = []
    column = space.dim  # one-hot columns follow the numeric ones
    for parameter in space.categorical:
        for choice in parameter.choices:
            layout.append((parameter.name, choice, column))
            column += 1
    return layout


def _full_bounds(space: SearchSpace) -> np.ndarray:
    numeric = space.bounds_array()
    n_onehot = len(_onehot_layout(space))
    if not n_onehot:
        return numeric
    return np.column_stack([numeric, np.array([[0.0] * n_onehot, [1.0] * n_onehot])])


def _encode_full(space: SearchSpace, recipe: dict) -> np.ndarray:
    vector = list(space.encode(recipe))
    for name, choice, _ in _onehot_layout(space):
        vector.append(1.0 if recipe.get(name) == choice else 0.0)
    return np.asarray(vector, dtype=float)


def _decode_full(space: SearchSpace, vector: np.ndarray) -> dict:
    vector = np.asarray(vector, dtype=float).ravel()
    recipe = space.decode(vector[: space.dim])
    for parameter in space.categorical:
        columns = [
            (choice, index)
            for name, choice, index in _onehot_layout(space)
            if name == parameter.name
        ]
        best = max(columns, key=lambda item: vector[item[1]])
        recipe[parameter.name] = best[0]
    return recipe


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def suggest(
    space: SearchSpace,
    observations: list[Observation],
    *,
    q: int = 1,
    constraints: ConstraintSet | None = None,
    acquisition: str = "qLogEI",
    objective_sense: str = "max",
    objective_name: str = "log_fom",
    seed: int | None = None,
) -> SuggestionBatch:
    """Propose ``q`` recipes.

    ``objective_sense='min'`` is handled by negating the objective, so the
    acquisition machinery only ever maximises. Infeasible observations are
    dropped rather than penalised: a failed growth run carries no information
    about the objective surface, and assigning it a bad value would teach the GP
    something untrue about that region.

    TODO(CNMS): a failed run *does* carry information about feasibility. Model it
    with a separate feasibility classifier and a constrained acquisition once
    enough failures have accumulated to fit one.
    """
    if constraints is not None:
        space = apply_constraints(space, constraints)

    usable = [o for o in observations if o.is_feasible and np.isfinite(o.objective)]
    notes: list[str] = []
    if len(usable) < len(observations):
        notes.append(
            f"Dropped {len(observations) - len(usable)} infeasible or non-finite observation(s); "
            "they are not modelled as poor outcomes."
        )

    #  Cold start: a GP on 2-3 points is its prior wearing a disguise.
    if len(usable) < MIN_OBSERVATIONS_FOR_GP:
        return _sobol_batch(space, q, len(usable), objective_name, seed, notes)

    sign = 1.0 if objective_sense == "max" else -1.0
    x = np.vstack([_encode_full(space, o.parameters) for o in usable])
    y = np.asarray([sign * o.objective for o in usable], dtype=float)
    noise = (
        np.asarray([o.noise if o.noise is not None else 0.0 for o in usable], dtype=float)
        if any(o.noise is not None for o in usable)
        else None
    )

    bounds = _full_bounds(space)
    model = fit_gp(x, y, bounds, noise=noise)

    acq = build_acquisition(
        model,
        kind=acquisition,
        best_f=float(np.max(y)) if acquisition == "qLogEI" else None,
        x_baseline=x if acquisition == "qLogNEI" else None,
    )
    integer_dims = [i for i, p in enumerate(space.numeric) if p.kind == "integer"]
    layout = _onehot_layout(space)

    if not layout:
        candidates, values = optimize(
            acq, bounds, q=q, integer_dims=integer_dims
        )
        scored = [(float(v), c) for c, v in zip(candidates, np.resize(values, len(candidates)), strict=True)]
    else:
        scored = []
        for combination in space.categorical_combinations():
            fixed = {
                index: (1.0 if combination.get(name) == choice else 0.0)
                for name, choice, index in layout
            }
            candidate, value = optimize(
                acq, bounds, q=1, integer_dims=integer_dims, fixed_features=fixed
            )
            scored.append((float(value[0]), candidate[0]))
        notes.append(
            f"Optimized the acquisition separately over {len(scored)} categorical combination(s); "
            "values are comparable because a single GP produced them."
        )
        if q > 1:
            notes.append(
                "Batch selection takes the top-q across combinations. This is not a jointly "
                "optimized q-batch, so the suggestions may cluster; treat q > 1 here as a "
                "convenience rather than an optimal batch design."
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:q]

    suggestions: list[Suggestion] = []
    for index, (value, vector) in enumerate(selected):
        mean, std = posterior_summary(model, vector.reshape(1, -1))
        suggestions.append(
            Suggestion(
                parameters=_decode_full(space, vector),
                acquisition_value=value,
                predicted_mean=float(sign * mean[0]),  # back to the user's sense
                predicted_std=float(std[0]),
                strategy=acquisition,
                batch_index=index,
            )
        )

    return SuggestionBatch(
        suggestions=suggestions,
        strategy=acquisition,
        n_observations=len(usable),
        objective_name=objective_name,
        notes=notes,
    )


def _sobol_batch(
    space: SearchSpace,
    q: int,
    n_observations: int,
    objective_name: str,
    seed: int | None,
    notes: list[str],
) -> SuggestionBatch:
    """Space-filling cold start."""
    bounds = space.bounds_array()
    points = sobol_points(bounds, q, seed=seed)
    combinations = space.categorical_combinations()
    rng = np.random.default_rng(seed)

    suggestions = []
    for index, point in enumerate(points):
        vector = np.asarray(point, dtype=float)
        for dim, parameter in enumerate(space.numeric):
            if parameter.kind == "integer":
                vector[dim] = np.round(vector[dim])
        recipe = space.decode(vector, combinations[int(rng.integers(len(combinations)))])
        suggestions.append(Suggestion(parameters=recipe, strategy="sobol", batch_index=index))

    notes.append(
        f"Only {n_observations} usable observation(s) — fewer than the {MIN_OBSERVATIONS_FOR_GP} "
        "needed for a GP to say anything its prior did not. Using a scrambled Sobol design "
        "instead, which covers the space more evenly than random sampling at small n."
    )
    return SuggestionBatch(
        suggestions=suggestions,
        strategy="sobol",
        n_observations=n_observations,
        objective_name=objective_name,
        notes=notes,
    )
