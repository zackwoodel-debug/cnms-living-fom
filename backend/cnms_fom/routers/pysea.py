"""HTTP surface for the pySEA integration.

The verbs mirror ``/modalfit``: import a container, look at what it holds, ask
what could be promoted and why not, promote with an explicit identity, compare
against another platform.

Status codes carry meaning here. A container that fails validation is still
stored, so import returns 200 with ``validation_status: "invalid"`` and the issue
list; the caller reads the issues rather than a bare 422. A promotion the protocol
forbids is 409, because the request was well formed and the answer is no.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from cnms_fom.db.base import get_db
from cnms_fom.pysea.compare import compare_across_platforms, cross_platform_disagreements
from cnms_fom.pysea.contract import ContractError
from cnms_fom.pysea.promote import PromotionRefused, promote, promotion_plan
from cnms_fom.pysea.records import (
    PySeaImportError,
    import_pysea_record,
    list_records,
    record_detail,
)
from cnms_fom.pysea.validate import validate_envelope
from cnms_fom.schemas.pysea import (
    PySeaCompareRequest,
    PySeaCompareResponse,
    PySeaImportRequest,
    PySeaImportResponse,
    PySeaPromoteRequest,
    PySeaRecordSummary,
)

router = APIRouter(prefix="/pysea", tags=["pysea"])


@router.post("/validate", response_model=dict)
def validate(payload: PySeaImportRequest) -> dict:
    """Check a container without storing it.

    Separate from import on purpose. An operator fixing an export wants the issue
    list, and should not have to write a row to get it.
    """
    envelope = _envelope_from(payload)
    report = validate_envelope(envelope)
    return {
        "validation_status": report.status,
        "issues": report.as_dicts(),
        "n_errors": len(report.errors),
        "n_warnings": len(report.warnings),
    }


@router.post("/import", response_model=PySeaImportResponse)
def import_record(
    payload: PySeaImportRequest, db: Session = Depends(get_db)
) -> PySeaImportResponse:
    """Import one container. Idempotent by content hash.

    An invalid container is stored with its issues rather than refused: the
    acquisition happened, and refusing would leave the fact unrecorded. What an
    invalid record cannot do is promote.
    """
    envelope = _envelope_from(payload)
    try:
        summary = import_pysea_record(
            db,
            envelope,
            source_filename=payload.source_filename or payload.path,
            unit_hints=payload.unit_hints,
        )
    except ContractError as exc:
        #  A version this build cannot read. 422: the request is well formed and
        #  the payload cannot be interpreted.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except PySeaImportError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    #  ``get_db`` closes the session and does not commit, matching the modalfit
    #  router: the endpoint that wrote decides whether the write stands.
    db.commit()
    return PySeaImportResponse(**summary)


@router.get("/records", response_model=list[PySeaRecordSummary])
def records(
    sample_id: str | None = Query(default=None),
    record_kind: str | None = Query(default=None),
    instrument_id: str | None = Query(default=None),
    datafed_record_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[PySeaRecordSummary]:
    """Stored records, filtered and bounded."""
    rows = list_records(
        db,
        sample_id=sample_id,
        record_kind=record_kind,
        instrument_id=instrument_id,
        datafed_record_id=datafed_record_id,
        limit=limit,
    )
    return [PySeaRecordSummary(**row) for row in rows]


@router.get("/records/{pysea_record_id}", response_model=dict)
def get_record(pysea_record_id: int, db: Session = Depends(get_db)) -> dict:
    """One record in full: state, calibrations, signals, scalars. No arrays."""
    try:
        return record_detail(db, pysea_record_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/records/{pysea_record_id}/promotion-plan", response_model=dict)
def plan(pysea_record_id: int, db: Session = Depends(get_db)) -> dict:
    """What would be promoted, what would be refused, and why. No writes."""
    try:
        return promotion_plan(db, pysea_record_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/records/{pysea_record_id}/promote", response_model=dict)
def promote_record(
    pysea_record_id: int,
    payload: PySeaPromoteRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Promote eligible scalars for an explicitly named material.

    409 on refusal. The request was well formed; the protocol says no, and the
    body explains which gate closed.
    """
    try:
        result = promote(
            db,
            pysea_record_id,
            material_id=payload.material_id,
            scalar_ids=payload.scalar_ids,
            dry_run=payload.dry_run,
            operator=payload.operator,
        )
        if not payload.dry_run:
            db.commit()
        return result
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except PromotionRefused as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/compare", response_model=PySeaCompareResponse)
def compare(payload: PySeaCompareRequest, db: Session = Depends(get_db)) -> PySeaCompareResponse:
    """Every determination of one quantity for one sample, across platforms.

    Never averages. Two platforms that disagree are two findings.
    """
    result = compare_across_platforms(db, payload.sample_id, quantity=payload.quantity)
    return PySeaCompareResponse(**result)


@router.get("/samples/{sample_id}/disagreements", response_model=dict)
def disagreements(sample_id: str, db: Session = Depends(get_db)) -> dict:
    """Quantities this sample has more than one determination of, and their verdicts."""
    return cross_platform_disagreements(db, sample_id)


def _envelope_from(payload: PySeaImportRequest) -> dict:
    """The envelope, from the request's inline body or its path."""
    from cnms_fom.pysea.contract import load_envelope

    if payload.envelope is None and not payload.path:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Give either an inline envelope or a path to one.",
        )
    if payload.envelope is not None:
        return payload.envelope
    try:
        return load_envelope(payload.path or "")
    except (ContractError, OSError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
