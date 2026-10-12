"""/suite: the first integrated workflow (docs/specs/suite-workflow/SPEC.md).

Reference data comes from the materials-db service; this router adds what CNMS owns: the tier/label vocabulary shown on every number, the fit diagnostics, and the
promotion preview that is already ``/modalfit/fits/{id}/promotion-plan``. Nothing here writes a measured property: promotion stays on ``/modalfit/.../promote``,
which defaults to a dry run and refuses a real write without a named operator.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from cnms_fom.config import get_settings
from cnms_fom.db.base import get_db
from cnms_fom.materials_release.provenance import EvidenceInput, map_evidence
from cnms_fom.suite.client import MaterialsDbClient, SuiteRefusal, SuiteUnavailable
from cnms_fom.suite.diagnostics import diagnose

router = APIRouter(prefix="/suite", tags=["suite"])


def get_client():
    client = MaterialsDbClient(get_settings().materials_db_url, timeout=get_settings().materials_db_timeout_s)
    try:
        yield client
    finally:
        client.close()


def _call(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except SuiteRefusal as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except SuiteUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def with_labels(evidence: dict) -> dict:
    """Add, to every dataset, how CNMS may show it. ``kind`` is the badge vocabulary of the workflow page; ``reviewed`` is separate from it."""
    for ds in evidence["datasets"]:
        states = {k: v["state"] for k, v in (ds.get("conditions") or {}).items()}
        review = ds["review"]
        mapped = map_evidence(EvidenceInput(dataset_id=ds["dataset_id"], provenance=ds["provenance"], review=review["status"], review_conflict=bool(review["conflict"]),
                                            states=states, basis=ds.get("basis", "")))
        ds["cnms"] = dict(label=mapped.label, tier=mapped.tier.value if mapped.tier else None, destination=mapped.destination, reviewed=mapped.reviewed,
                          reasons=mapped.reasons, kind=_KIND[ds["provenance"]])
    return evidence


#  The workflow page's vocabulary. "user_assumption" and "experimental_fit" never come from materials-db: the page adds them for what the user typed and what a fit returned.
_KIND = {"measured": "reference_literature", "reported": "reference_literature", "calculated": "calculated", "modeled": "synthetic_prediction",
         "fitted": "reference_literature", "unknown": "unknown"}
BADGE_KINDS = ("reference_literature", "user_assumption", "calculated", "synthetic_prediction", "experimental_fit", "unknown")


@router.get("/info")
def info(client: MaterialsDbClient = Depends(get_client)) -> dict:
    return {**_call(client.info), "badge_kinds": list(BADGE_KINDS)}


@router.get("/materials")
def materials(q: str | None = None, limit: int = 25, client: MaterialsDbClient = Depends(get_client)) -> list[dict]:
    return _call(client.materials, q, limit)


@router.get("/materials/{material_id}/evidence")
def evidence(material_id: int, client: MaterialsDbClient = Depends(get_client)) -> dict:
    return with_labels(_call(client.evidence, material_id))


@router.get("/optical/search")
def optical_search(
    wavelength_nm: float,
    n_min: float | None = None,
    n_max: float | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
    q: str | None = None,
    provenance: str | None = None,
    approved_only: bool = False,
    limit: int = 50,
    client: MaterialsDbClient = Depends(get_client),
) -> dict:
    """Tabulated n and k at one wavelength, filtered by range. Interpolated values are flagged and nothing is extrapolated."""
    result = _call(client.optical_search, wavelength_nm=wavelength_nm, n_min=n_min, n_max=n_max, k_min=k_min, k_max=k_max, q=q, provenance=provenance,
                   approved_only=approved_only, limit=limit)
    for hit in result["results"]:
        mapped = map_evidence(EvidenceInput(dataset_id=hit["dataset_id"], provenance=hit["provenance"], review=hit["review"], review_conflict=False, states={}))
        hit["cnms"] = dict(label=mapped.label, kind=_KIND[hit["provenance"]], reviewed=mapped.reviewed)
    return result


class ReadinessIn(BaseModel):
    selection: dict
    technique: str
    window: dict | None = None
    conditions: dict | None = None


@router.post("/readiness")
def readiness(payload: ReadinessIn, client: MaterialsDbClient = Depends(get_client)) -> dict:
    return _call(client.readiness, payload.selection, payload.technique, payload.window, payload.conditions)


class BundleIn(BaseModel):
    layers: list
    substrate: str = "Silicon"
    ambient: str = "air"
    techniques: list
    window: dict | None = None
    mode: str = "strict"
    assumptions: list = []
    length_unit: str = "angstrom"


@router.post("/bundle")
def bundle(payload: BundleIn, client: MaterialsDbClient = Depends(get_client)) -> Response:
    """A strict ModalFit bundle for the stack. A refusal (missing input, ambiguity, uncovered window) is a 422 with the reason; nothing is filled in."""
    import json

    data, summary = _call(client.bundle, payload.model_dump())
    assumptions = summary.get("assumptions_used") or []
    headers = {"X-Bundle-Summary": json.dumps(dict(summary, user_assumptions=list(payload.assumptions), assumptions_used=assumptions), separators=(",", ":"))}
    return Response(content=data, media_type="application/zip", headers=headers)


@router.get("/fits/{fit_record_id}/diagnostics")
def diagnostics(fit_record_id: int, db: Session = Depends(get_db)) -> dict:
    from cnms_fom.db.models import FitRecord

    record = db.get(FitRecord, fit_record_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No fit record {fit_record_id}.")
    findings = diagnose(record)
    return dict(fit_record_id=record.id, findings=findings, blocks_promotion=any(f["severity"] == "blocks_promotion" for f in findings))
