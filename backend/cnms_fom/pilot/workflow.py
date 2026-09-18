"""The pilot loop: suggest → export → simulate → score → observe.

Each stage is a plain function over dataclasses, so any one can be replaced
without touching the others. Replacing ``simulate`` with a real measurement is
the entire point of the pilot, and ``ingest_experiment`` below is the same
pipeline with the simulation swapped out for measured data.

Provenance, which is the part that keeps the loop defensible:

* Every simulated property is stored tiered **MODELED**, with the recipe written
  into ``processing_route``. The recipe therefore becomes part of the
  measurement context (Eq. 3), and two recipes are two contexts on one material
  identity — which is exactly what they physically are: same composition,
  polymorph, and specimen form, different processing.
* Every score from those properties is **ILLUSTRATIVE**, enforced by
  ``score_material`` and by a database CHECK. A simulated loop exercises the
  machinery; it does not produce a ranking.
* Each simulated recipe gets its own ``AnalysisRun``, because it is genuinely a
  separate analysis over different inputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select

from cnms_fom.bo_engine.constraints import ConstraintSet
from cnms_fom.bo_engine.loop import Observation
from cnms_fom.bo_engine.loop import suggest as suggest_recipes
from cnms_fom.bo_engine.space import SearchSpace
from cnms_fom.cnms_integration.provenance import RunProvenance
from cnms_fom.db.enums import ProvenanceTier, ScoreStatus, SpecimenForm
from cnms_fom.db.models import (
    AnalysisRun,
    BoObservation,
    BoRun,
    BoSuggestion,
    FomDefinition,
    FomScore,
    Material,
    PropertyValue,
)
from cnms_fom.fom_engine.definitions import all_draft_foms
from cnms_fom.fom_engine.scores import FomSpec, score_material

from .properties import simulate_properties
from .simulate import simulate_xrr
from .stack import build_stack, stack_to_nk_csv

logger = logging.getLogger(__name__)

PILOT_FILM_FORMULA = "HfO2"
PILOT_POLYMORPH = "monoclinic"
PILOT_SPECIMEN_FORM = SpecimenForm.CRYSTALLINE_FILM
#  Bumped whenever the property model changes, so old and new simulated values
#  never share a measurement context.
PILOT_SIMULATION_VERSION = "pilot-sim-v1"


@dataclass
class PilotEvaluation:
    """One recipe taken all the way from suggestion to logged observation."""

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
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "recipe": self.recipe,
            "stack": self.stack,
            "xrr": self.xrr,
            "properties": self.properties,
            "fom_value": self.fom_value,
            "fom_log_value": self.fom_log_value,
            "fom_status": self.fom_status,
            "objective_value": self.objective_value,
            "material_id": self.material_id,
            "analysis_run_id": self.analysis_run_id,
            "fom_score_id": self.fom_score_id,
            "observation_id": self.observation_id,
            "suggestion_id": self.suggestion_id,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def ensure_pilot_material(session) -> Material:
    """The single material identity every pilot recipe is a processing variant of.

    One row, not one per recipe: composition, polymorph and specimen form are
    identical across the search space, and Eq. (3) puts processing in the
    *context*, not in the identity.
    """
    material = session.execute(
        select(Material).where(
            Material.formula_reduced == PILOT_FILM_FORMULA,
            Material.polymorph == PILOT_POLYMORPH,
            Material.specimen_form == PILOT_SPECIMEN_FORM,
        )
    ).scalar_one_or_none()
    if material is not None:
        return material

    material = Material(
        formula=PILOT_FILM_FORMULA,
        formula_reduced=PILOT_FILM_FORMULA,
        polymorph=PILOT_POLYMORPH,
        specimen_form=PILOT_SPECIMEN_FORM,
        notes=(
            "Pilot workflow material. Recipes are processing variants recorded in each "
            "property value's processing_route, not separate material identities."
        ),
    )
    session.add(material)
    session.flush()
    return material


def resolve_fom_spec(session, name: str) -> tuple[FomSpec, int | None]:
    """Find a FOM by name, falling back to the built-in draft."""
    row = (
        session.query(FomDefinition)
        .filter(FomDefinition.name == name)
        .order_by(FomDefinition.version.desc())
        .first()
    )
    if row is not None:
        return FomSpec.from_definition(row), row.id
    drafts = all_draft_foms()
    if name not in drafts:
        raise KeyError(f"Unknown FOM {name!r}; known: {sorted(drafts)}.")
    return drafts[name], None


# ---------------------------------------------------------------------------
# One recipe, end to end
# ---------------------------------------------------------------------------


def evaluate_recipe(
    session,
    recipe: dict,
    spec: FomSpec,
    *,
    fom_definition_id: int | None = None,
    material: Material | None = None,
    persist: bool = True,
    simulate_reflectivity: bool = True,
) -> PilotEvaluation:
    """Export a stack, simulate it, derive properties, and score them."""
    material = material or ensure_pilot_material(session)

    stack = build_stack(session, recipe, film_formula=PILOT_FILM_FORMULA)
    xrr = (
        simulate_xrr(stack).as_dict()
        if simulate_reflectivity
        else {"engine": "skipped"}
    )
    simulated = simulate_properties(recipe)

    result = score_material(
        simulated.properties,
        spec,
        material_key=f"{PILOT_FILM_FORMULA}|{PILOT_POLYMORPH}|{PILOT_SPECIMEN_FORM.value}",
        #  Everything the simulation produced is modeled, so the score comes
        #  back ILLUSTRATIVE rather than passing as a measurement-based result.
        provenance=dict.fromkeys(simulated.properties, ProvenanceTier.MODELED),
    )

    evaluation = PilotEvaluation(
        recipe=dict(recipe),
        stack=stack.as_dict(),
        xrr=xrr,
        properties=simulated.as_dict(),
        fom_value=result.value,
        fom_log_value=result.log_value,
        fom_status=result.status.value,
        #  ln F, per bo_engine.surrogate: the FOM is a geometric mean, and its
        #  log is the additive scale a GP should model on.
        objective_value=result.log_value,
        material_id=material.id,
        notes=[*simulated.notes, *([result.note] if result.note else [])],
    )

    if persist:
        _persist_evaluation(session, evaluation, recipe, spec, fom_definition_id, material)
    return evaluation


def _persist_evaluation(
    session,
    evaluation: PilotEvaluation,
    recipe: dict,
    spec: FomSpec,
    fom_definition_id: int | None,
    material: Material,
) -> None:
    """Write the simulated properties, the analysis run, and the score."""
    now = datetime.now(timezone.utc)
    #  The recipe *is* the processing context, so it goes in processing_route
    #  and thereby into the context digest: two recipes are two contexts.
    processing_route = json.dumps(
        {k: recipe[k] for k in sorted(recipe)}, separators=(",", ":")
    )
    method = f"{PILOT_SIMULATION_VERSION} (modeled; see pilot/properties.py)"

    provenance = RunProvenance(
        kind="pilot_simulation",
        eligible_material_ids=[material.id],
        params={"recipe": recipe, "fom": spec.name, "simulation": PILOT_SIMULATION_VERSION},
    )
    run = AnalysisRun(**provenance.as_run_kwargs(), status="complete")
    session.add(run)
    session.flush()
    evaluation.analysis_run_id = run.id

    for key, value in evaluation.properties["properties"].items():
        existing = session.execute(
            select(PropertyValue.id).where(
                PropertyValue.material_id == material.id,
                PropertyValue.property_key == key,
                PropertyValue.processing_route == processing_route,
                PropertyValue.method == method,
            )
        ).first()
        if existing:
            continue  # same recipe, same model version: already recorded
        session.add(
            PropertyValue(
                material_id=material.id,
                property_key=key,
                value=float(value),
                method=method,
                processing_route=processing_route,
                provenance_tier=ProvenanceTier.MODELED,
                temperature_k=300.0,
                thickness_nm=float(recipe["thickness_ang"]) / 10.0,
                ingested_at=now,
            )
        )

    if fom_definition_id is not None:
        score = FomScore(
            material_id=material.id,
            fom_definition_id=fom_definition_id,
            run_id=run.id,
            value=evaluation.fom_value,
            log_value=evaluation.fom_log_value,
            status=ScoreStatus(evaluation.fom_status),
            components={},
            uses_modeled_inputs=True,
        )
        session.add(score)
        session.flush()
        evaluation.fom_score_id = score.id
    else:
        evaluation.notes.append(
            f"FOM {spec.name!r} has no database row, so no FomScore was persisted. "
            "POST /fom/definitions/seed-drafts first if you want the score stored."
        )
    session.flush()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_pilot_iteration(
    session,
    bo_run: BoRun,
    *,
    q: int = 1,
    persist: bool = True,
    seed: int | None = None,
    simulate_reflectivity: bool = True,
) -> list[PilotEvaluation]:
    """Ask the optimizer for ``q`` recipes, evaluate them, and tell it the answers.

    The full cycle. With ``persist`` the observations go back into the campaign,
    so the surrogate retrains and the next call proposes something different —
    which is the behaviour the CI test asserts.
    """
    space = SearchSpace.from_dict(bo_run.search_space)
    constraints = ConstraintSet.from_dict(bo_run.constraints)

    observations = [
        Observation(
            parameters=parameters,
            objective=float(objective),
            noise=noise,
            is_feasible=feasible,
        )
        for parameters, objective, noise, feasible in session.execute(
            select(
                BoObservation.parameters,
                BoObservation.objective_value,
                BoObservation.objective_noise,
                BoObservation.is_feasible,
            )
            .where(
                BoObservation.bo_run_id == bo_run.id,
                BoObservation.objective_value.isnot(None),
            )
            .order_by(BoObservation.id)
        ).all()
    ]

    batch = suggest_recipes(
        space,
        observations,
        q=q,
        constraints=constraints,
        acquisition=bo_run.acquisition,
        objective_sense=bo_run.objective_sense,
        seed=seed if seed is not None else bo_run.random_seed,
    )

    spec, definition_id = resolve_fom_spec(
        session, _fom_name_for(session, bo_run)
    )
    material = ensure_pilot_material(session)

    evaluations: list[PilotEvaluation] = []
    for suggestion in batch.suggestions:
        evaluation = evaluate_recipe(
            session,
            suggestion.parameters,
            spec,
            fom_definition_id=definition_id,
            material=material,
            persist=persist,
            simulate_reflectivity=simulate_reflectivity,
        )
        evaluation.notes.extend(batch.notes)

        if persist:
            record = BoSuggestion(
                bo_run_id=bo_run.id,
                parameters=suggestion.parameters,
                acquisition_value=suggestion.acquisition_value,
                predicted_mean=suggestion.predicted_mean,
                predicted_std=suggestion.predicted_std,
                batch_index=suggestion.batch_index,
                status="evaluated",
            )
            session.add(record)
            session.flush()
            evaluation.suggestion_id = record.id

            observation = BoObservation(
                bo_run_id=bo_run.id,
                parameters=suggestion.parameters,
                objective_value=evaluation.objective_value,
                fom_score_id=evaluation.fom_score_id,
                is_feasible=evaluation.objective_value is not None,
            )
            session.add(observation)
            session.flush()
            evaluation.observation_id = observation.id

        evaluations.append(evaluation)

    return evaluations


def _fom_name_for(session, bo_run: BoRun) -> str:
    if bo_run.fom_definition_id is None:
        return "logic"
    definition = session.get(FomDefinition, bo_run.fom_definition_id)
    return definition.name if definition else "logic"


def export_stack_artifacts(session, recipe: dict) -> tuple[str, str]:
    """The stack JSON and its n,k CSV, for handing to an external simulator."""
    stack = build_stack(session, recipe, film_formula=PILOT_FILM_FORMULA)
    return stack.to_json(), stack_to_nk_csv(session, stack)
