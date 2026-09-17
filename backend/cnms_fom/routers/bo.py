"""/bo — Bayesian optimization campaigns over growth recipes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload

from cnms_fom.bo_engine.constraints import ConstraintSet, from_instrument_capabilities
from cnms_fom.bo_engine.loop import Observation
from cnms_fom.bo_engine.loop import suggest as suggest_recipes
from cnms_fom.bo_engine.space import ParameterSpec, SearchSpace
from cnms_fom.cnms_integration.instruments import get_instrument, list_instruments
from cnms_fom.config import get_settings
from cnms_fom.db.base import get_db
from cnms_fom.db.models import BoObservation, BoRun, BoSuggestion, FomDefinition
from cnms_fom.schemas.bo import (
    BoRunIn,
    BoRunOut,
    InstrumentOut,
    ObservationIn,
    SuggestRequest,
    SuggestResponse,
)

router = APIRouter(prefix="/bo", tags=["bo"])


@router.get("/instruments", response_model=list[InstrumentOut])
def instruments() -> list[dict]:
    """Available instruments and their capability envelopes.

    ``source`` is ``"placeholder"`` until the CNMS registry is wired in — see
    ``cnms_integration.instruments``.
    """
    return [record.as_dict() for record in list_instruments()]


@router.get("/runs", response_model=list[BoRunOut])
def list_runs(db: Session = Depends(get_db)) -> list[BoRunOut]:
    runs = (
        db.query(BoRun)
        .options(selectinload(BoRun.observations), selectinload(BoRun.suggestions))
        .order_by(BoRun.id.desc())
        .all()
    )
    return [
        BoRunOut(
            id=run.id,
            name=run.name,
            search_space=run.search_space,
            constraints=run.constraints,
            acquisition=run.acquisition,
            objective_sense=run.objective_sense,
            status=run.status,
            random_seed=run.random_seed,
            n_observations=len(run.observations),
            n_suggestions=len(run.suggestions),
        )
        for run in runs
    ]


@router.post("/run", response_model=BoRunOut, status_code=status.HTTP_201_CREATED)
def create_run(payload: BoRunIn, db: Session = Depends(get_db)) -> BoRunOut:
    """Start a campaign.

    When ``instrument_id`` is given, the search space is intersected with that
    tool's envelope up front. Intersecting before optimization (rather than
    filtering suggestions afterwards) means the acquisition budget is never
    spent on regions the tool cannot reach.
    """
    space = SearchSpace(
        parameters=[ParameterSpec.from_dict(p.model_dump()) for p in payload.search_space]
    )

    constraints = ConstraintSet.from_dict(payload.constraints)
    if payload.instrument_id:
        try:
            instrument = get_instrument(payload.instrument_id)
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        envelope = from_instrument_capabilities(instrument.capabilities)
        constraints.bounds = {**envelope.bounds, **constraints.bounds}
        constraints.allowed_choices = {**envelope.allowed_choices, **constraints.allowed_choices}
        constraints.notes.append(
            f"Envelope from {instrument.instrument_id} (source: {instrument.source})."
        )

    #  Fail now, not at the first suggest call, if the intersection is empty.
    try:
        from cnms_fom.bo_engine.constraints import apply as apply_constraints

        apply_constraints(space, constraints)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    fom_definition_id = None
    if payload.fom_name:
        definition = (
            db.query(FomDefinition)
            .filter(FomDefinition.name == payload.fom_name)
            .order_by(FomDefinition.version.desc())
            .first()
        )
        fom_definition_id = definition.id if definition else None

    run = BoRun(
        name=payload.name,
        fom_definition_id=fom_definition_id,
        search_space=space.as_dict(),
        constraints=constraints.as_dict(),
        acquisition=payload.acquisition,
        objective_sense=payload.objective_sense,
        random_seed=payload.random_seed or get_settings().random_seed,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return BoRunOut(
        id=run.id,
        name=run.name,
        search_space=run.search_space,
        constraints=run.constraints,
        acquisition=run.acquisition,
        objective_sense=run.objective_sense,
        status=run.status,
        random_seed=run.random_seed,
    )


@router.post("/run/{run_id}/observe", status_code=status.HTTP_201_CREATED)
def observe(run_id: int, payload: ObservationIn, db: Session = Depends(get_db)) -> dict:
    """Record an evaluated recipe.

    The objective is ln F of the campaign's FOM — see ``bo_engine.surrogate`` for
    why the log of a geometric score is the right scale to model on.
    """
    run = db.get(BoRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No BO run with id {run_id}.")

    space = SearchSpace.from_dict(run.search_space)
    from cnms_fom.bo_engine.constraints import validate_recipe

    violations = validate_recipe(payload.parameters, space)
    if violations:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {
                "error": "Recipe does not lie in this campaign's search space.",
                "violations": [
                    {"parameter": v.parameter, "value": v.value, "reason": v.reason}
                    for v in violations
                ],
            },
        )

    observation = BoObservation(
        bo_run_id=run_id,
        experiment_id=payload.experiment_id,
        parameters=payload.parameters,
        objective_value=payload.objective_value,
        objective_noise=payload.objective_noise,
        fom_score_id=payload.fom_score_id,
        is_feasible=payload.is_feasible,
    )
    db.add(observation)
    db.commit()
    db.refresh(observation)
    return {"observation_id": observation.id, "bo_run_id": run_id}


@router.post("/run/{run_id}/suggest", response_model=SuggestResponse)
def suggest(run_id: int, payload: SuggestRequest, db: Session = Depends(get_db)) -> SuggestResponse:
    """Propose the next ``q`` recipes.

    With fewer than a handful of usable observations this returns a Sobol design
    rather than GP suggestions — a GP fitted on three points is reporting its
    prior, and dressing that up as a recommendation wastes real growth runs.
    """
    run = (
        db.query(BoRun)
        .options(selectinload(BoRun.observations))
        .filter(BoRun.id == run_id)
        .one_or_none()
    )
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No BO run with id {run_id}.")

    space = SearchSpace.from_dict(run.search_space)
    constraints = ConstraintSet.from_dict(run.constraints)
    observations = [
        Observation(
            parameters=o.parameters,
            objective=float(o.objective_value),
            noise=o.objective_noise,
            is_feasible=o.is_feasible,
        )
        for o in run.observations
        if o.objective_value is not None
    ]

    try:
        batch = suggest_recipes(
            space,
            observations,
            q=payload.q,
            constraints=constraints,
            acquisition=payload.acquisition or run.acquisition,
            objective_sense=run.objective_sense,
            seed=payload.seed if payload.seed is not None else run.random_seed,
        )
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"BO extra not installed: {exc}. pip install -e '.[bo]'",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    suggestions_out = []
    for suggestion in batch.suggestions:
        record = None
        if payload.persist:
            record = BoSuggestion(
                bo_run_id=run_id,
                parameters=suggestion.parameters,
                acquisition_value=suggestion.acquisition_value,
                predicted_mean=suggestion.predicted_mean,
                predicted_std=suggestion.predicted_std,
                batch_index=suggestion.batch_index,
            )
            db.add(record)
        suggestions_out.append({**suggestion.as_dict(), "suggestion_id": None})

    if payload.persist:
        db.commit()

    return SuggestResponse(
        run_id=run_id,
        suggestions=suggestions_out,  # type: ignore[arg-type]
        strategy=batch.strategy,
        n_observations=batch.n_observations,
        objective_name=batch.objective_name,
        notes=batch.notes,
    )
