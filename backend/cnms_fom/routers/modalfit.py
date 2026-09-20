"""/modalfit — multi-technique co-refinement records from ModalFit.

The endpoints divide along one line, and it is the line FOM_PROOF cares about:

``/import``, ``/samples``, ``/compare``  read and reason about fits.  Safe, and
    the main thing the research assistant uses.

``/promote``  writes fitted values into ``property_values`` / ``descriptor_values``
    as MEASURED. Gated hard, defaults to a dry run, and requires an explicit
    material identity. This is instrument-derived data, which is why it is
    allowed at all — it is a different path from retrieval, and there is
    deliberately no equivalent endpoint for a RAG answer.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from cnms_fom.db.base import get_db
from cnms_fom.modalfit.compare import (
    COMPARABLE_PARAMETERS,
    compare_parameter,
    cross_technique_report,
    describe_fit,
    fits_for_sample,
)
from cnms_fom.modalfit.promote import PromotionRefused, promote_fit, promotion_plan
from cnms_fom.modalfit.records import FitImportError, import_directory, import_fit
from cnms_fom.schemas.modalfit import (
    CompareRequest,
    FitDatasetOut,
    FitImportRequest,
    FitImportResponse,
    FitLayerOut,
    FitRecordOut,
    PromoteRequest,
    SampleFitsResponse,
)

router = APIRouter(prefix="/modalfit", tags=["modalfit"])


def _record_out(record) -> FitRecordOut:
    return FitRecordOut(
        fit_record_id=record.id,
        sample_id=record.sample_id,
        stack_id=record.stack_id,
        techniques=record.techniques or [],
        algorithm=record.algorithm,
        chi2_total=record.chi2_total,
        chi2_by_technique=record.chi2_by_technique,
        n_free_parameters=record.n_free_parameters,
        uses_placeholder_optical_constants=record.uses_placeholder_optical_constants,
        resolution_smearing_applied=record.resolution_smearing_applied,
        roughness_applied_to_spr=record.roughness_applied_to_spr,
        fitted_at=str(record.fitted_at) if record.fitted_at else None,
        operator=record.operator,
        layers=[
            FitLayerOut(
                layer_index=layer.layer_index,
                role=layer.role,
                label=layer.label,
                material=layer.material,
                formula=layer.formula,
                thickness_ang=layer.thickness_ang,
                roughness_ang=layer.roughness_ang,
                density_g_cm3=layer.density_g_cm3,
                free_parameters=layer.free_parameters or [],
                bounds=layer.bounds,
                uncertainties=layer.uncertainties,
                parameters=layer.parameters,
            )
            for layer in record.layers
        ],
        datasets=[
            FitDatasetOut(
                technique=dataset.technique.value,
                source_filename=dataset.source_filename,
                n_points=dataset.n_points,
                x_min=dataset.x_min,
                x_max=dataset.x_max,
                x_units=dataset.x_units,
                chi2=dataset.chi2,
                weight=dataset.weight,
            )
            for dataset in record.datasets
        ],
        description=describe_fit(record),
    )


@router.post("/import", response_model=FitImportResponse)
def import_fits(payload: FitImportRequest, db: Session = Depends(get_db)) -> FitImportResponse:
    """Import a ModalFit export, or every export under a directory.

    Idempotent by content hash, so a directory sweep can be re-run after adding
    files. Read the ``warnings`` on the response: a fit that claims a technique
    with no matching slab-model block, or whose parameters finished clamped on
    their bounds, is stored but is not a measurement anyone should quote.
    """
    from cnms_fom.db.models import FitRecord

    path = Path(payload.path)
    if not path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{path} is not visible to the API. Under docker-compose, mount it and use the "
            "container path (./data is mounted at /app/data).",
        )

    shared = {
        "techniques": [t.value for t in payload.techniques] if payload.techniques else None,
        "algorithm": payload.algorithm.value if payload.algorithm else None,
        "chi2_total": payload.chi2_total,
        "chi2_by_technique": payload.chi2_by_technique,
        "technique_weights": payload.technique_weights,
        "length_units": payload.length_units,
        "operator": payload.operator,
        "notes": payload.notes,
        "resolution_smearing_applied": payload.resolution_smearing_applied,
        "roughness_applied_to_spr": payload.roughness_applied_to_spr,
    }

    try:
        if path.is_dir():
            results = import_directory(db, path, **shared)
        else:
            results = [
                import_fit(
                    db,
                    path,
                    datasets=[d.model_dump() for d in payload.datasets] if payload.datasets else None,
                    sample_id=payload.sample_id,
                    datafed_record_id=payload.datafed_record_id,
                    material_id=payload.material_id,
                    experiment_id=payload.experiment_id,
                    **shared,
                )
            ]
        db.commit()
    except FitImportError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Fit import failed: {exc}"
        ) from exc

    total = db.query(FitRecord).count()
    warnings = [w for result in results for w in result.get("warnings", [])]
    return FitImportResponse(imported=results, total_fits=total, warnings=warnings)


@router.get("/samples", response_model=list[dict])
def list_samples(
    limit: int = Query(default=25, ge=1, le=200), db: Session = Depends(get_db)
) -> list[dict]:
    """Samples with stored refinements, most recently fitted first."""
    from sqlalchemy import func

    from cnms_fom.db.models import FitRecord

    rows = (
        db.query(FitRecord.sample_id, func.count(FitRecord.id), func.max(FitRecord.fitted_at))
        .filter(FitRecord.sample_id.isnot(None))
        .group_by(FitRecord.sample_id)
        .order_by(func.max(FitRecord.fitted_at).desc().nullslast())
        .limit(limit)
        .all()
    )
    return [
        {
            "sample_id": sample_id,
            "n_fits": int(count),
            "most_recent_fit": str(latest) if latest else None,
        }
        for sample_id, count, latest in rows
    ]


@router.get("/samples/{sample_id}/fits", response_model=SampleFitsResponse)
def sample_fits(sample_id: str, db: Session = Depends(get_db)) -> SampleFitsResponse:
    """Every stored refinement for one sample, with its caveats."""
    records = fits_for_sample(db, sample_id)
    return SampleFitsResponse(
        sample_id=sample_id,
        n_fits=len(records),
        fits=[_record_out(record) for record in records],
    )


@router.get("/fits/{fit_record_id}", response_model=FitRecordOut)
def get_fit(fit_record_id: int, db: Session = Depends(get_db)) -> FitRecordOut:
    from cnms_fom.db.models import FitRecord

    record = db.get(FitRecord, fit_record_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No fit record {fit_record_id}.")
    return _record_out(record)


@router.post("/compare", response_model=dict)
def compare(payload: CompareRequest, db: Session = Depends(get_db)) -> dict:
    """Compare one parameter across every technique that determined it.

    Returns each determination separately with its caveats, plus the spread and a
    verdict. Deliberately never an average: FOM_PROOF Sec. 2.1 forbids merging
    records without a declared aggregation rule, and two techniques 40% apart on
    a thickness do not have a mean worth reporting — they have a discrepancy
    someone has to explain.
    """
    if payload.parameter not in COMPARABLE_PARAMETERS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{payload.parameter!r} is not cross-technique comparable. "
            f"Available: {', '.join(sorted(COMPARABLE_PARAMETERS))}.",
        )
    return compare_parameter(
        db, payload.sample_id, parameter=payload.parameter, layer_label=payload.layer_label
    )


@router.get("/samples/{sample_id}/disagreements", response_model=dict)
def disagreements(
    sample_id: str, layer_label: str | None = None, db: Session = Depends(get_db)
) -> dict:
    """Every cross-technique disagreement on one sample, across all parameters."""
    return cross_technique_report(db, sample_id, layer_label=layer_label)


@router.get("/fits/{fit_record_id}/promotion-plan", response_model=dict)
def plan(fit_record_id: int, db: Session = Depends(get_db)) -> dict:
    """What would be promoted from this fit, and what would be refused, and why.

    The refusal list is the actionable half: it names the parameters to free, the
    bounds to widen, and the fits to re-run.
    """
    try:
        return promotion_plan(db, fit_record_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/fits/{fit_record_id}/promote", response_model=dict)
def promote(
    fit_record_id: int, payload: PromoteRequest, db: Session = Depends(get_db)
) -> dict:
    """Promote eligible fitted values into the analysis tables as MEASURED.

    Dry run by default. These rows are instrument-derived — a forward model with
    no interpretive freedom reproduced a measured curve — which is why they may
    enter ``property_values`` at all. There is no equivalent path from a RAG
    answer, and ``rag_backend.chains.assert_not_property_ingestion`` exists to
    keep it that way.
    """
    try:
        result = promote_fit(
            db,
            fit_record_id,
            material_id=payload.material_id,
            layer_label=payload.layer_label,
            temperature_k=payload.temperature_k,
            substrate=payload.substrate,
            dry_run=payload.dry_run,
        )
        if not payload.dry_run:
            db.commit()
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except PromotionRefused as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Promotion failed: {exc}"
        ) from exc
    return result
