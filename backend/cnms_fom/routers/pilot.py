"""/pilot — one complete experimental loop for HfO2 on Si.

The template a real CNMS study is meant to be cut from:

    POST /pilot/hfo2_logic_run                      create the campaign
    POST /pilot/hfo2_logic_run/{id}/iterate         suggest → simulate → observe
    POST /pilot/hfo2_logic_run/{id}/ingest_experiment   the same, with real data
    GET  /pilot/hfo2_logic_run/{id}/stack           export a stack for an
                                                    external simulator

``iterate`` and ``ingest_experiment`` are deliberately the same pipeline with
one stage swapped: simulated properties (MODELED → ILLUSTRATIVE score) versus
measured ones (MEASURED → a real score). Moving from pilot to production means
calling the second instead of the first, not rewriting the loop.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cnms_fom.bo_engine.constraints import ConstraintSet, from_instrument_capabilities
from cnms_fom.bo_engine.space import ParameterSpec, SearchSpace
from cnms_fom.cnms_integration.instruments import get_instrument
from cnms_fom.config import get_settings
from cnms_fom.db.base import get_db
from cnms_fom.db.enums import ScoreStatus
from cnms_fom.db.models import (
    BoObservation,
    BoRun,
    Experiment,
    FomDefinition,
    FomScore,
    Instrument,
    PropertyValue,
)
from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES
from cnms_fom.fom_engine.scores import score_material
from cnms_fom.pilot import workflow
from cnms_fom.schemas.bo import BoRunOut
from cnms_fom.schemas.pilot import (
    IngestExperimentRequest,
    IngestExperimentResponse,
    IterateRequest,
    IterateResponse,
    PilotRunRequest,
    StackExportResponse,
)

router = APIRouter(prefix="/pilot", tags=["pilot"])

PILOT_RUN_PATH = "/hfo2_logic_run"


def _build_search_space(payload: PilotRunRequest) -> SearchSpace:
    parameters = [
        ParameterSpec(
            name="thickness_ang",
            kind="continuous",
            units="A",
            lower=payload.thickness_ang[0],
            upper=payload.thickness_ang[1],
            description="HfO2 film thickness.",
        ),
        ParameterSpec(
            name="roughness_ang",
            kind="continuous",
            units="A",
            lower=payload.roughness_ang[0],
            upper=payload.roughness_ang[1],
            description="RMS surface roughness of the film.",
        ),
    ]
    if payload.dopant_fraction is not None:
        parameters.append(
            ParameterSpec(
                name="dopant_fraction",
                kind="continuous",
                units="mol fraction",
                lower=payload.dopant_fraction[0],
                upper=payload.dopant_fraction[1],
                description="Al2O3 dopant fraction; trades permittivity against bandgap.",
            )
        )
    return SearchSpace(parameters=parameters)


@router.post(PILOT_RUN_PATH, response_model=BoRunOut, status_code=status.HTTP_201_CREATED)
def create_pilot_run(payload: PilotRunRequest, db: Session = Depends(get_db)) -> BoRunOut:
    """Create the HfO2-on-Si pilot campaign.

    Objective is ln F of the named FOM, maximised. The search space is
    intersected with the instrument envelope up front, so the acquisition budget
    is never spent on recipes the tool cannot run.
    """
    space = _build_search_space(payload)
    constraints = ConstraintSet()

    if payload.instrument_id:
        try:
            instrument = get_instrument(payload.instrument_id)
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        #  The pilot's parameters are film geometry; the placeholder registry
        #  describes process knobs. Only overlapping names constrain anything,
        #  which is honest: a thickness envelope has to come from the real
        #  registry, and TODO(CNMS) marks that.
        envelope = from_instrument_capabilities(instrument.capabilities)
        constraints.bounds.update(
            {k: v for k, v in envelope.bounds.items() if k in {p.name for p in space.parameters}}
        )
        constraints.notes.append(
            f"Envelope from {instrument.instrument_id} (source: {instrument.source}). "
            "TODO(CNMS): the real registry must supply thickness and roughness limits."
        )

    try:
        from cnms_fom.bo_engine.constraints import apply as apply_constraints

        apply_constraints(space, constraints)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    definition = (
        db.query(FomDefinition)
        .filter(FomDefinition.name == payload.fom_name)
        .order_by(FomDefinition.version.desc())
        .first()
    )

    run = BoRun(
        name=payload.name,
        fom_definition_id=definition.id if definition else None,
        search_space=space.as_dict(),
        constraints=constraints.as_dict(),
        acquisition=payload.acquisition,
        objective_sense="max",
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


@router.post(PILOT_RUN_PATH + "/{run_id}/iterate", response_model=IterateResponse)
def iterate(
    run_id: int, payload: IterateRequest, db: Session = Depends(get_db)
) -> IterateResponse:
    """One turn of the loop: suggest, export, simulate, score, observe.

    Every score returned is ILLUSTRATIVE — the properties behind it are modeled.
    With ``persist`` the observations are logged, so the next call sees them and
    the surrogate moves.
    """
    run = db.get(BoRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No BO run with id {run_id}.")

    try:
        evaluations = workflow.run_pilot_iteration(
            db,
            run,
            q=payload.q,
            persist=payload.persist,
            seed=payload.seed,
            simulate_reflectivity=payload.simulate_reflectivity,
        )
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"BO extra not installed: {exc}. pip install -e '.[bo]'",
        ) from exc
    except (KeyError, ValueError) as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    if payload.persist:
        db.commit()

    n_observations = db.execute(
        select(func.count(BoObservation.id)).where(BoObservation.bo_run_id == run_id)
    ).scalar()

    return IterateResponse(
        run_id=run_id,
        fom_name=workflow._fom_name_for(db, run),
        evaluations=[e.as_dict() for e in evaluations],  # type: ignore[arg-type]
        n_observations_after=int(n_observations or 0),
    )


@router.get(PILOT_RUN_PATH + "/stack", response_model=StackExportResponse)
def export_stack(
    thickness_ang: float = Query(gt=0),
    roughness_ang: float = Query(default=2.0, ge=0),
    dopant_fraction: float = Query(default=0.0, ge=0, le=1),
    db: Session = Depends(get_db),
) -> StackExportResponse:
    """Export a stack as JSON plus its n,k CSV, for an external simulator.

    SLDs come from the database when the import supplied them and are otherwise
    computed from composition and density; the source is recorded per layer.
    Layers with no ingested dispersion are omitted from the CSV rather than
    filled with a guess.
    """
    recipe = {
        "thickness_ang": thickness_ang,
        "roughness_ang": roughness_ang,
        "dopant_fraction": dopant_fraction,
    }
    stack = workflow.build_stack(db, recipe, film_formula=workflow.PILOT_FILM_FORMULA)
    csv_text = workflow.stack_to_nk_csv(db, stack)
    return StackExportResponse(
        recipe=recipe,
        stack=stack.as_dict(),
        stack_json=stack.to_json(),
        nk_csv=csv_text,
        nk_csv_rows=max(0, csv_text.count("\n") - 1),
    )


@router.post(
    PILOT_RUN_PATH + "/{run_id}/ingest_experiment",
    response_model=IngestExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_experiment(
    run_id: int, payload: IngestExperimentRequest, db: Session = Depends(get_db)
) -> IngestExperimentResponse:
    """Close the loop with measured data.

    The same pipeline as ``iterate`` with the simulation removed: measurements
    land as MEASURED property values, the FOM is recomputed from them, and the
    result is logged as a BO observation. A material missing a required input
    comes back ``not_scored`` and is logged as infeasible rather than given a
    manufactured objective.
    """
    run = db.get(BoRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No BO run with id {run_id}.")

    material = workflow.ensure_pilot_material(db)
    now = datetime.now(timezone.utc)
    warnings: list[str] = []

    instrument_pk = None
    if payload.instrument_id:
        instrument_pk = db.execute(
            select(Instrument.id).where(Instrument.instrument_id == payload.instrument_id)
        ).scalar()
        if instrument_pk is None:
            warnings.append(
                f"Instrument {payload.instrument_id!r} is not in the registry; "
                "the experiment is recorded without an instrument link."
            )

    experiment = Experiment(
        external_id=payload.external_experiment_id,
        instrument_id=instrument_pk,
        material_id=material.id,
        proposal_id=payload.proposal_id,
        operator=payload.operator,
        sample_id=payload.sample_id,
        recipe=payload.recipe,
        status="complete" if payload.succeeded else "failed",
        finished_at=now,
        notes=payload.notes,
    )
    db.add(experiment)
    db.flush()

    # --- store the measurements, rejecting any missing their context ---------
    import json

    processing_route = json.dumps(
        {k: payload.recipe[k] for k in sorted(payload.recipe)}, separators=(",", ":")
    )
    measured: dict[str, float] = {}
    stored = 0

    for measurement in payload.measurements:
        spec = PHYSICAL_PROPERTIES.get(measurement.property_key)
        if spec is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Unknown property {measurement.property_key!r}. "
                f"Known: {sorted(PHYSICAL_PROPERTIES)}.",
            )
        data = measurement.model_dump()
        absent = [field for field in spec.required_context if not data.get(field)]
        if absent:
            db.rollback()
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "error": f"Missing required context for {measurement.property_key!r}: {absent}.",
                    "why": spec.caveat
                    or "FOM_PROOF Sec. 16: context is what makes a value comparable.",
                },
            )

        db.add(
            PropertyValue(
                material_id=material.id,
                experiment_id=experiment.id,
                processing_route=processing_route,
                units=data.pop("units", None) or spec.units,
                ingested_at=now,
                **{k: v for k, v in data.items() if k != "units"},
            )
        )
        measured[measurement.property_key] = float(measurement.value)
        stored += 1

    db.flush()

    # --- rescore from the measured values ------------------------------------
    fom_name = workflow._fom_name_for(db, run)
    spec, definition_id = workflow.resolve_fom_spec(db, fom_name)
    result = score_material(
        {key: measured.get(key) for key in spec.required_properties},
        spec,
        material_key=f"{material.formula_reduced}|{material.polymorph}",
        provenance={
            m.property_key: m.provenance_tier for m in payload.measurements
        },
    )

    score_id = None
    if definition_id is not None:
        from cnms_fom.cnms_integration.provenance import RunProvenance
        from cnms_fom.db.models import AnalysisRun

        provenance = RunProvenance(
            kind="experiment_score",
            eligible_material_ids=[material.id],
            params={"experiment_id": experiment.id, "recipe": payload.recipe, "fom": fom_name},
        )
        analysis_run = AnalysisRun(**provenance.as_run_kwargs(), status="complete")
        db.add(analysis_run)
        db.flush()

        score = FomScore(
            material_id=material.id,
            fom_definition_id=definition_id,
            run_id=analysis_run.id,
            value=result.value,
            log_value=result.log_value,
            status=result.status,
            missing_inputs=result.missing_inputs,
            components=result.components,
            uses_modeled_inputs=result.uses_modeled_inputs,
        )
        db.add(score)
        db.flush()
        score_id = score.id
    else:
        warnings.append(
            f"FOM {fom_name!r} has no database row, so no score was persisted. "
            "POST /fom/definitions/seed-drafts first."
        )

    # --- log the observation --------------------------------------------------
    feasible = payload.succeeded and result.status is not ScoreStatus.NOT_SCORED
    if not feasible and result.status is ScoreStatus.NOT_SCORED:
        warnings.append(
            f"Not scored — missing {result.missing_inputs}. Logged as infeasible, so the "
            "surrogate drops it rather than treating it as a poor outcome."
        )

    observation = BoObservation(
        bo_run_id=run_id,
        experiment_id=experiment.id,
        parameters=payload.recipe,
        objective_value=result.log_value if feasible else None,
        fom_score_id=score_id,
        is_feasible=feasible,
    )
    db.add(observation)
    db.commit()
    db.refresh(observation)

    if result.uses_modeled_inputs:
        warnings.append(
            "At least one input was tiered modeled, so this score is ILLUSTRATIVE and must not "
            "be reported alongside measurement-based results (Sec. 2.3)."
        )

    return IngestExperimentResponse(
        run_id=run_id,
        experiment_id=experiment.id,
        material_id=material.id,
        properties_stored=stored,
        fom_name=fom_name,
        fom_status=result.status.value,
        fom_value=result.value,
        fom_log_value=result.log_value,
        missing_inputs=result.missing_inputs,
        fom_score_id=score_id,
        observation_id=observation.id,
        objective_value=observation.objective_value,
        warnings=warnings,
    )
