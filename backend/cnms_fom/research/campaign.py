"""A deterministic snapshot of a BO campaign, and the warnings it earns.

Everything here is computed from stored data with no model involved, and that is
the point.  The warnings this module produces — an unapproved objective, a stalled
search, suggestions piled against a bound, a campaign that is mostly infeasible —
are exactly the ones a reader most needs and a language model is least reliable at
noticing.  A warning that fires only when the model remembers to mention it is not
a guardrail; it is a hope.

So :func:`snapshot` runs first, unconditionally, and its warnings are attached to
the brief whether or not the model refers to them.  The model's job is to explain
and to extract, not to audit.

Nothing here writes. A snapshot is a read.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#  A best-so-far that has not moved in this many evaluations is worth remarking on.
#  Not a convergence test — there is no such thing from inside a campaign — but the
#  point at which "is this still working?" becomes the right question.
STALL_THRESHOLD = 5

#  Fraction of a parameter's range within which a suggestion counts as sitting on
#  the boundary. A batch clustered here usually means the optimum is outside the
#  box rather than at its edge.
BOUNDARY_FRACTION = 0.02

#  Above this fraction of infeasible observations, the campaign has a constraint
#  problem rather than a search problem.
INFEASIBLE_FRACTION = 0.4

#  A predicted standard deviation above this fraction of the observed objective
#  range means the surrogate is still genuinely uncertain.
UNCERTAIN_STD_FRACTION = 0.25


@dataclass
class CampaignSnapshot:
    """Everything about a campaign that a brief needs, and nothing derived by a model."""

    bo_run_id: int
    name: str
    status: str
    acquisition: str
    objective_sense: str
    search_space: dict
    constraints: dict
    fom_definition: str | None = None
    fom_approved: bool | None = None
    fom_frozen: bool | None = None
    fom_weights: dict | None = None

    n_observations: int = 0
    n_infeasible: int = 0
    best_objective: float | None = None
    best_recipe: dict | None = None
    evaluations_since_best_improved: int = 0
    objective_min: float | None = None
    objective_max: float | None = None

    n_pending_suggestions: int = 0
    suggestions: list[dict] = field(default_factory=list)
    boundary_parameters: list[str] = field(default_factory=list)
    max_predicted_std: float | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def objective_range(self) -> float | None:
        if self.objective_min is None or self.objective_max is None:
            return None
        return self.objective_max - self.objective_min

    @property
    def is_stalled(self) -> bool:
        return self.evaluations_since_best_improved >= STALL_THRESHOLD

    @property
    def surrogate_is_uncertain(self) -> bool:
        """Whether the optimizer still thinks there is unexplored ground.

        Compared against the observed objective range rather than an absolute
        number, because ln F is dimensionless but its scale depends entirely on how
        bad the worst recipe was.
        """
        span = self.objective_range
        if self.max_predicted_std is None or not span:
            return False
        return self.max_predicted_std > UNCERTAIN_STD_FRACTION * span

    def fingerprint(self) -> str:
        """Hash of the configuration that decides what the optimizer may propose.

        Search space, constraints, acquisition, and objective sense — not the
        observations, which change with every run and would make the fingerprint
        useless for detecting a *configuration* change. This is what the context
        bridge records before and after an apply.
        """
        payload = {
            "search_space": self.search_space,
            "constraints": self.constraints,
            "acquisition": self.acquisition,
            "objective_sense": self.objective_sense,
            "fom_definition": self.fom_definition,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        return {
            "bo_run_id": self.bo_run_id,
            "name": self.name,
            "status": self.status,
            "acquisition": self.acquisition,
            "objective_sense": self.objective_sense,
            "search_space": self.search_space,
            "constraints": self.constraints,
            "fom": {
                "definition": self.fom_definition,
                "approved": self.fom_approved,
                "frozen": self.fom_frozen,
                "weights": self.fom_weights,
            },
            "history": {
                "n_observations": self.n_observations,
                "n_infeasible": self.n_infeasible,
                "best_objective": self.best_objective,
                "best_recipe": self.best_recipe,
                "evaluations_since_best_improved": self.evaluations_since_best_improved,
                "objective_min": self.objective_min,
                "objective_max": self.objective_max,
                "is_stalled": self.is_stalled,
            },
            "suggestions": {
                "n_pending": self.n_pending_suggestions,
                "max_predicted_std": self.max_predicted_std,
                "surrogate_is_uncertain": self.surrogate_is_uncertain,
                "boundary_parameters": self.boundary_parameters,
                "pending": self.suggestions,
            },
            "warnings": self.warnings,
            "fingerprint": self.fingerprint(),
            "objective_scale_note": (
                "The surrogate models ln F, not F (Eq. 30): F is a weighted geometric mean of "
                "terms in (0, 1], so it is strongly skewed and its residuals are nothing like "
                "the Gaussian a GP assumes. A difference of 0.7 in these numbers is a factor of "
                "two in F."
            ),
        }


def snapshot(db, bo_run_id: int) -> CampaignSnapshot:
    """Read one campaign and compute its warnings. No model, no writes."""

    from cnms_fom.db.models import BoObservation, BoRun, BoSuggestion, FomDefinition

    run = db.get(BoRun, int(bo_run_id))
    if run is None:
        raise LookupError(f"No BO campaign {bo_run_id}.")

    definition = db.get(FomDefinition, run.fom_definition_id) if run.fom_definition_id else None

    observations = (
        db.query(BoObservation)
        .filter(BoObservation.bo_run_id == run.id)
        .order_by(BoObservation.id)
        .all()
    )
    suggestions = (
        db.query(BoSuggestion)
        .filter(BoSuggestion.bo_run_id == run.id, BoSuggestion.status == "proposed")
        .order_by(BoSuggestion.id.desc())
        .all()
    )

    maximising = run.objective_sense == "max"
    best: float | None = None
    best_recipe: dict | None = None
    best_index = 0
    feasible_values: list[float] = []

    for index, observation in enumerate(observations, start=1):
        if not observation.is_feasible or observation.objective_value is None:
            continue
        value = float(observation.objective_value)
        feasible_values.append(value)
        if best is None or (value > best if maximising else value < best):
            best, best_recipe, best_index = value, observation.parameters, index

    stale_for = (len(observations) - best_index) if best is not None else 0

    stds = [s.predicted_std for s in suggestions if s.predicted_std is not None]
    snap = CampaignSnapshot(
        bo_run_id=run.id,
        name=run.name,
        status=run.status,
        acquisition=run.acquisition,
        objective_sense=run.objective_sense,
        search_space=run.search_space or {},
        constraints=run.constraints or {},
        fom_definition=(
            f"{definition.name} v{definition.version}" if definition else None
        ),
        fom_approved=definition.approved if definition else None,
        fom_frozen=definition.frozen if definition else None,
        fom_weights=definition.weights if definition else None,
        n_observations=len(observations),
        n_infeasible=sum(1 for o in observations if not o.is_feasible),
        best_objective=best,
        best_recipe=best_recipe,
        evaluations_since_best_improved=max(0, stale_for),
        objective_min=min(feasible_values) if feasible_values else None,
        objective_max=max(feasible_values) if feasible_values else None,
        n_pending_suggestions=len(suggestions),
        suggestions=[
            {
                "suggestion_id": s.id,
                "parameters": s.parameters,
                "acquisition_value": s.acquisition_value,
                "predicted_mean": s.predicted_mean,
                "predicted_std": s.predicted_std,
            }
            for s in suggestions
        ],
        max_predicted_std=max(stds) if stds else None,
    )
    snap.boundary_parameters = _parameters_on_boundary(snap)
    snap.warnings = _warnings_for(snap)
    return snap


def _parameters_on_boundary(snap: CampaignSnapshot) -> list[str]:
    """Numeric parameters whose pending suggestions all sit against a bound.

    *All*, not *any*: one suggestion at an edge is the optimizer probing, which is
    what it is for. Every suggestion in the batch at the same edge is the optimizer
    telling you the box is in the wrong place.
    """
    parameters = {
        spec.get("name"): spec
        for spec in (snap.search_space or {}).get("parameters", [])
        if spec.get("kind") in ("continuous", "integer")
    }
    if not parameters or not snap.suggestions:
        return []

    on_boundary: list[str] = []
    for name, spec in parameters.items():
        low, high = spec.get("lower"), spec.get("upper")
        if low is None or high is None:
            continue
        low, high = float(low), float(high)
        span = high - low
        if span <= 0:
            continue
        tolerance = BOUNDARY_FRACTION * span

        values = [
            s["parameters"].get(name)
            for s in snap.suggestions
            if isinstance(s.get("parameters"), dict) and s["parameters"].get(name) is not None
        ]
        if not values:
            continue
        if all(
            float(v) <= low + tolerance or float(v) >= high - tolerance for v in values
        ):
            on_boundary.append(name)
    return on_boundary


def _warnings_for(snap: CampaignSnapshot) -> list[str]:
    """Every warning the campaign's own state earns, in severity order."""
    warnings: list[str] = []

    if snap.fom_definition is None:
        warnings.append(
            "This campaign has no FOM definition attached, so its objective is not traceable to "
            "a versioned score. Whatever it is optimising, the result cannot be compared with a "
            "FOM-scored one."
        )
    elif snap.fom_approved is False:
        warnings.append(
            f"The objective '{snap.fom_definition}' is UNAPPROVED. Its weights are uniform "
            "placeholders with no named owner (FOM_PROOF Sec. 6.2: weights are an "
            "application-policy choice, not a measurement), so this campaign is optimising "
            "toward a decision nobody has made yet. Its ranking is not a result."
        )
    elif snap.fom_approved and not snap.fom_frozen:
        warnings.append(
            f"The objective '{snap.fom_definition}' is approved but not frozen, so its bounds and "
            "weights can still change underneath the observations already collected. Freeze it "
            "before treating the ranking as final (Sec. 5.3)."
        )

    if snap.n_observations == 0:
        warnings.append(
            "No observations yet. The first batch will be a space-filling Sobol design rather "
            "than a model-driven proposal — there is nothing to fit a surrogate on, so do not "
            "read the first suggestions as the optimizer's opinion."
        )

    if snap.n_observations and snap.n_infeasible / snap.n_observations > INFEASIBLE_FRACTION:
        fraction = snap.n_infeasible / snap.n_observations
        warnings.append(
            f"{fraction:.0%} of observations are infeasible ({snap.n_infeasible} of "
            f"{snap.n_observations}). That is a constraint problem, not a search problem: the "
            "search space and the instrument envelope disagree about what is reachable, and the "
            "acquisition budget is being spent discovering it."
        )

    if snap.is_stalled:
        detail = (
            f"The best objective has not improved in {snap.evaluations_since_best_improved} "
            "evaluations."
        )
        if snap.surrogate_is_uncertain:
            detail += (
                f" But the surrogate is still uncertain (predicted std up to "
                f"{snap.max_predicted_std:.3g} against an observed range of "
                f"{snap.objective_range:.3g}), so this is not convergence — it is a search that "
                "has not yet found what it is looking for."
            )
        else:
            detail += (
                " The surrogate is confident, which is consistent with convergence — but check "
                "the boundary warning before believing it."
            )
        warnings.append(detail)

    if snap.boundary_parameters:
        warnings.append(
            "Every pending suggestion sits on a bound for: "
            + ", ".join(snap.boundary_parameters)
            + ". A search space whose optimum lies outside its own bounds looks exactly like a "
            "converged campaign from the inside. Check whether the bound is a real instrument "
            "limit or a number somebody typed."
        )

    if snap.n_pending_suggestions == 0 and snap.n_observations:
        warnings.append(
            "No pending suggestions, so there is nothing queued to run. Any conclusion here is "
            "about the history, not about what the optimizer would do next."
        )

    return warnings
