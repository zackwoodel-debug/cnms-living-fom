"""/fom — scores, correlations, mediated effects, and integrity checks.

This router is the HTTP surface of FOM_PROOF. Each endpoint maps to a section:

    POST /fom/score      Sec. 6.2   weighted geometric score
    POST /fom/correlate  Sec. 7-8   correlation block + permutation + FDR
    POST /fom/mediate    Sec. 10    M = B Gamma, the primary result
    POST /fom/integrity  Sec. 11    covariance identity, null overlap, leakage
    GET  /fom/hypotheses Sec. 4.2   the pre-registered table and its fingerprint
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload

from cnms_fom.cnms_integration.provenance import RunProvenance
from cnms_fom.config import get_settings
from cnms_fom.db.base import get_db
from cnms_fom.db.enums import CorrelationBlock, ProvenanceTier, ScoreStatus, Transform
from cnms_fom.db.models import AnalysisRun, CorrelationResult, FomDefinition, FomScore, Material
from cnms_fom.db.models import MediationResult as MediationRow
from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES, STRUCTURAL_DESCRIPTORS
from cnms_fom.fom_engine import hypotheses as hypothesis_module
from cnms_fom.fom_engine.correlations import correlation_block
from cnms_fom.fom_engine.definitions import all_draft_foms, to_definition_kwargs
from cnms_fom.fom_engine.eligibility import ContextFilter, build_analysis_table
from cnms_fom.fom_engine.integrity import (
    covariance_to_correlation,
    excess_correlation,
    leakage_check,
    null_score_correlation,
    reconstruct_log_score_covariance,
    weight_matrix,
)
from cnms_fom.fom_engine.mediation import mediated_effect_from_theory
from cnms_fom.fom_engine.scores import FomSpec, score_material
from cnms_fom.schemas.fom import (
    CorrelationRequest,
    CorrelationResponse,
    FomDefinitionIn,
    FomDefinitionOut,
    HypothesisRegistryOut,
    IntegrityRequest,
    IntegrityResponse,
    MediationRequest,
    MediationResponse,
    ScoreRequest,
    ScoreResponse,
)

router = APIRouter(prefix="/fom", tags=["fom"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_spec(db: Session, name: str, version: int | None) -> tuple[FomSpec, int | None]:
    """Find a FOM by name/version, falling back to the built-in drafts.

    The fallback keeps a fresh database usable, and costs nothing in rigour: a
    draft carries ``approved=False``, so every score it produces is labelled
    DRAFT and the publish guard rejects it.
    """
    query = db.query(FomDefinition).filter(FomDefinition.name == name)
    if version is not None:
        query = query.filter(FomDefinition.version == version)
    row = query.order_by(FomDefinition.version.desc()).first()
    if row is not None:
        return FomSpec.from_definition(row), row.id

    drafts = all_draft_foms(floor_eps=get_settings().score_floor_eps)
    if name not in drafts:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No FOM definition {name!r}. Stored: "
            f"{[r.name for r in db.query(FomDefinition.name).distinct()]}; "
            f"built-in drafts: {sorted(drafts)}.",
        )
    return drafts[name], None


def _context_from_request(payload) -> ContextFilter:
    """Build the eligibility context from a request.

    The tier filter defaults to measured + calculated. Passing ``modeled`` opts
    into the separate modeled-scenario analysis of Sec. 2.3 — permitted, but the
    results are labelled ILLUSTRATIVE and must never overwrite a
    measurement-based result.
    """
    tiers = getattr(payload, "provenance_tiers", None)
    context = ContextFilter(
        specimen_forms=tuple(payload.specimen_forms) if payload.specimen_forms else None,
        temperature_k=tuple(payload.temperature_k) if getattr(payload, "temperature_k", None) else None,
        frequency_hz=tuple(payload.frequency_hz) if getattr(payload, "frequency_hz", None) else None,
    )
    if tiers:
        context.provenance_tiers = tuple(tiers)
    return context


def _eligible_materials(db: Session, material_ids: list[int] | None = None) -> list[Material]:
    query = db.query(Material).options(
        selectinload(Material.descriptors), selectinload(Material.properties)
    )
    if material_ids:
        query = query.filter(Material.id.in_(material_ids))
    return query.order_by(Material.id).all()


# ---------------------------------------------------------------------------
# Definitions and pre-registration
# ---------------------------------------------------------------------------


@router.get("/definitions", response_model=list[FomDefinitionOut])
def list_definitions(db: Session = Depends(get_db)) -> list[FomDefinition]:
    return db.query(FomDefinition).order_by(FomDefinition.name, FomDefinition.version).all()


@router.post(
    "/definitions", response_model=FomDefinitionOut, status_code=status.HTTP_201_CREATED
)
def create_definition(payload: FomDefinitionIn, db: Session = Depends(get_db)) -> FomDefinition:
    """Create a FOM definition, or a new version of one.

    A frozen definition is never edited — Sec. 5.3 makes the bounds, transform,
    direction, floor, and weights part of the score's identity, so changing any
    of them changes what the score means. Post a new version instead.
    """
    unknown = sorted(set(payload.weights) - set(PHYSICAL_PROPERTIES))
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unknown properties {unknown}. Register them in descriptors.registry first, "
            "with units, direction, and required context.",
        )

    existing = (
        db.query(FomDefinition)
        .filter(FomDefinition.name == payload.name, FomDefinition.version == payload.version)
        .one_or_none()
    )
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{payload.name} v{payload.version} already exists"
            + (" and is frozen." if existing.frozen else ". Bump the version.")
        )

    definition = FomDefinition(
        name=payload.name,
        version=payload.version,
        application=payload.application,
        description=payload.description,
        weights=payload.weights,
        normalization={k: v.model_dump(mode="json") for k, v in payload.normalization.items()},
        floor_eps=payload.floor_eps,
        bounds_basis=payload.bounds_basis,
        eligible_set_query=payload.eligible_set_query,
        n_materials_in_bounds=payload.n_materials_in_bounds,
        approved=payload.approved,
        approved_by=payload.approved_by,
    )
    db.add(definition)
    db.commit()
    db.refresh(definition)
    return definition


@router.post("/definitions/seed-drafts", response_model=list[FomDefinitionOut])
def seed_draft_definitions(db: Session = Depends(get_db)) -> list[FomDefinition]:
    """Write the built-in draft definitions (Table 4) into the database.

    Convenience for a fresh install. They land unapproved and unfrozen, with
    DRAFT screening bounds that must be re-derived from the frozen eligible set
    before anything is reported.
    """
    created: list[FomDefinition] = []
    for spec in all_draft_foms(floor_eps=get_settings().score_floor_eps).values():
        exists = (
            db.query(FomDefinition)
            .filter(FomDefinition.name == spec.name, FomDefinition.version == spec.version)
            .one_or_none()
        )
        if exists is None:
            definition = FomDefinition(**to_definition_kwargs(spec))
            db.add(definition)
            created.append(definition)
    db.commit()
    for definition in created:
        db.refresh(definition)
    return created


@router.get("/hypotheses", response_model=HypothesisRegistryOut)
def get_hypotheses() -> HypothesisRegistryOut:
    """The pre-registered sign table (Sec. 4.2) and its fingerprint.

    The fingerprint is stamped onto every analysis run. If a predicted sign is
    edited after results are seen, runs before and after no longer agree on what
    was tested — which is the whole point of declaring signs in advance.
    """
    return HypothesisRegistryOut(
        fingerprint=hypothesis_module.registry_fingerprint(),
        hypotheses=hypothesis_module.as_records(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@router.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, db: Session = Depends(get_db)) -> ScoreResponse:
    """Eq. (29): score materials under one FOM definition.

    Materials missing a required input come back as ``not_scored`` with the list
    of what is missing. They are not dropped and not partially scored.
    """
    spec, definition_id = _resolve_spec(db, payload.fom_name, payload.fom_version)
    materials = _eligible_materials(db, payload.material_ids)
    if not materials:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No materials matched the request.")

    table = build_analysis_table(
        materials,
        property_keys=list(spec.required_properties),
        descriptor_keys=[],
        context=_context_from_request(payload),
    )

    run_id = None
    if payload.persist:
        provenance = RunProvenance(
            kind="score",
            eligible_material_ids=[m.id for m in materials],
            params={"fom_name": spec.name, "fom_version": spec.version},
        )
        run = AnalysisRun(**provenance.as_run_kwargs(), status="running")
        db.add(run)
        db.flush()
        run_id = run.id

    results = []
    for index, material in enumerate(materials):
        material_key = table.material_keys[index]
        properties = {q: table.columns[q][index] for q in spec.required_properties}
        #  Carry each value's tier through, so a modeled input forces the score
        #  to ILLUSTRATIVE instead of passing as a measurement-based result.
        provenance = {
            q: ProvenanceTier(tier)
            for q, tier in table.value_provenance.get(material_key, {}).items()
            if q in spec.required_properties
        }
        result = score_material(
            properties, spec, material_key=material_key, provenance=provenance
        )
        payload_out = result.as_dict()
        payload_out["material_id"] = material.id
        results.append(payload_out)

        if payload.persist:
            db.add(
                FomScore(
                    material_id=material.id,
                    fom_definition_id=definition_id,
                    run_id=run_id,
                    value=result.value,
                    log_value=result.log_value,
                    status=result.status,
                    missing_inputs=result.missing_inputs,
                    components=result.components,
                    uses_modeled_inputs=result.uses_modeled_inputs,
                )
            )

    if payload.persist:
        if definition_id is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Cannot persist scores against the built-in draft {spec.name!r}: it has no "
                "database row. POST /fom/definitions (or /fom/definitions/seed-drafts) first.",
            )
        db.query(AnalysisRun).filter(AnalysisRun.id == run_id).update({"status": "complete"})
        db.commit()

    warnings: list[str] = []
    if not spec.approved:
        warnings.append(
            f"FOM {spec.name!r} v{spec.version} is not approved. Sec. 6.2: weights are an "
            "application-policy choice and must be documented and approved before a ranking "
            "is reported."
        )
    out_of_bounds = sorted({q for r in results for q in r["out_of_bounds"]})
    if out_of_bounds:
        warnings.append(
            f"Properties outside the declared normalization bounds: {out_of_bounds}. Re-derive "
            "the bounds from the frozen eligible set under a new FOM version (Sec. 5.3)."
        )
    n_illustrative = sum(
        1 for r in results if r["status"] == ScoreStatus.ILLUSTRATIVE.value
    )
    if n_illustrative:
        warnings.append(
            f"{n_illustrative} score(s) use modeled inputs and are ILLUSTRATIVE. Sec. 2.3: report "
            "them separately; they must never overwrite a measurement-based result."
        )

    return ScoreResponse(
        fom_name=spec.name,
        fom_version=spec.version,
        approved=spec.approved,
        n_scored=sum(1 for r in results if r["status"] == ScoreStatus.SCORED.value),
        n_not_scored=sum(1 for r in results if r["status"] == ScoreStatus.NOT_SCORED.value),
        n_illustrative=n_illustrative,
        results=results,  # type: ignore[arg-type]
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


@router.post("/correlate", response_model=CorrelationResponse)
def correlate(payload: CorrelationRequest, db: Session = Depends(get_db)) -> CorrelationResponse:
    """Eqs. (35)-(38) with permutation p-values (Eq. 41) and BH-FDR (Eq. 44).

    Every cell carries its own complete-case n and the material identifiers that
    entered it. There is no run-level n in the response, by design (Sec. 7.3).
    """
    try:
        block = CorrelationBlock(payload.block)
    except ValueError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unknown block {payload.block!r}; use one of "
            f"{[b.value for b in CorrelationBlock]}.",
        ) from None

    descriptor_keys = payload.descriptor_keys or list(STRUCTURAL_DESCRIPTORS)
    property_keys = payload.property_keys or list(PHYSICAL_PROPERTIES)
    materials = _eligible_materials(db)
    if not materials:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No materials in the database.")

    table = build_analysis_table(
        materials,
        property_keys=property_keys,
        descriptor_keys=descriptor_keys,
        context=_context_from_request(payload),
    )

    transforms: dict[str, Transform] = {
        **{k: v.default_transform for k, v in STRUCTURAL_DESCRIPTORS.items()},
        **{k: v.default_transform for k, v in PHYSICAL_PROPERTIES.items()},
    }
    #  A log10 transform needs strictly positive values; fall back per column
    #  rather than failing the whole block on one non-positive entry.
    for key, values in table.columns.items():
        if transforms.get(key) is Transform.LOG10 and any(
            v is not None and v <= 0 for v in values
        ):
            transforms[key] = Transform.NONE

    x_columns = {k: table.columns[k] for k in descriptor_keys}
    y_columns = {k: table.columns[k] for k in property_keys}
    if block is CorrelationBlock.SS:
        y_columns = x_columns
    elif block is CorrelationBlock.PP:
        x_columns = y_columns

    cells = correlation_block(
        x_columns,
        y_columns,
        block=block,
        transforms=transforms,
        material_ids=table.material_keys,
        permutations=payload.permutations,
        seed=payload.seed if payload.seed is not None else get_settings().random_seed,
        hypotheses=hypothesis_module.hypothesis_map(),
        alpha=payload.alpha,
    )

    provenance = RunProvenance(
        kind="correlation",
        eligible_material_ids=[m.id for m in materials],
        permutation_b=payload.permutations,
        random_seed=payload.seed,
        params={"block": block.value, "alpha": payload.alpha},
    )

    run_id = None
    if payload.persist:
        run = AnalysisRun(**provenance.as_run_kwargs(), status="complete")
        db.add(run)
        db.flush()
        run_id = run.id
        db.add_all(
            CorrelationResult(
                run_id=run_id,
                block=cell.block,
                x_key=cell.x_key,
                y_key=cell.y_key,
                x_transform=cell.x_transform,
                y_transform=cell.y_transform,
                pearson_r=cell.pearson_r,
                spearman_rho=cell.spearman_rho,
                n_complete=cell.n_complete,
                p_permutation=cell.p_permutation,
                q_fdr=cell.q_fdr,
                predicted_sign=cell.predicted_sign,
                mechanism=cell.mechanism,
                material_ids=cell.material_ids,
            )
            for cell in cells
        )
        db.commit()

    return CorrelationResponse(
        cells=[cell.as_dict() for cell in cells],  # type: ignore[arg-type]
        coverage=table.coverage(),
        exclusions=[
            {"material_key": e.material_key, "property_key": e.property_key, "reason": e.reason}
            for e in table.exclusions
        ],
        run_id=run_id,
        provenance_warnings=provenance.warnings(),
    )


# ---------------------------------------------------------------------------
# Mediated effect
# ---------------------------------------------------------------------------


@router.post("/mediate", response_model=MediationResponse)
def mediate(payload: MediationRequest, db: Session = Depends(get_db)) -> MediationResponse:
    """Eq. (57): M = B Gamma, with the Eq. (65) per-channel decomposition."""
    spec, definition_id = _resolve_spec(db, payload.fom_name, payload.fom_version)

    if payload.source != "theory":
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Empirical B from regression is not wired into this endpoint yet. Use "
            "fom_engine.regression.ols + mediation.sensitivity_from_regressions directly, or "
            "pass source='theory' for the oscillator prior (Eqs. 50-53).",
        )

    reference_properties = dict(payload.reference_properties or {})
    reference_descriptors = dict(payload.reference_descriptors or {})

    if not reference_properties or not reference_descriptors:
        #  Sec. 10.1: the elasticity conversion is valid at a reference point, so
        #  use the population median of the eligible set rather than one material.
        materials = _eligible_materials(db)
        table = build_analysis_table(
            materials,
            property_keys=[*spec.required_properties, "eps_ionic"],
            descriptor_keys=payload.descriptor_keys or list(STRUCTURAL_DESCRIPTORS),
            context=_context_from_request(payload),
        )
        for key, values in table.columns.items():
            present = [v for v in values if v is not None]
            if not present:
                continue
            median = float(np.median(present))
            if key in PHYSICAL_PROPERTIES:
                reference_properties.setdefault(key, median)
            else:
                reference_descriptors.setdefault(key, median)

    try:
        result = mediated_effect_from_theory(spec, reference_properties, reference_descriptors)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {
                "error": str(exc),
                "hint": "Supply reference_properties (including eps_ionic and k) and "
                "reference_descriptors, or ingest enough data for a population median.",
            },
        ) from exc

    run_id = None
    if payload.persist:
        provenance = RunProvenance(kind="mediation", params={"fom_name": spec.name})
        run = AnalysisRun(**provenance.as_run_kwargs(), status="complete")
        db.add(run)
        db.flush()
        run_id = run.id
        db.add_all(
            MediationRow(
                run_id=run_id,
                fom_definition_id=definition_id,
                descriptor_key=descriptor,
                application=spec.application,
                mediated_effect=value,
                contributions=result.contributions[descriptor],
                dominant_property=result.dominant_property.get(descriptor),
                sensitivity_source=result.sensitivity_source,
            )
            for descriptor, value in result.mediated_effect.items()
        )
        db.commit()

    return MediationResponse(**result.as_dict(), run_id=run_id)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


@router.post("/integrity", response_model=IntegrityResponse)
def integrity(payload: IntegrityRequest, db: Session = Depends(get_db)) -> IntegrityResponse:
    """Sec. 11: is an observed FOM-FOM correlation anything more than weight overlap?

    Returns the observed correlation, the Eq. (60) null implied by the weight
    vectors alone, and their difference (Eq. 61). A large observed correlation
    with near-zero excess is fully explained by shared weights and is not
    evidence of shared physics.
    """
    specs = [_resolve_spec(db, name, None)[0] for name in payload.fom_names]
    materials = _eligible_materials(db)
    if not materials:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No materials in the database.")

    properties = sorted({q for spec in specs for q in spec.required_properties})
    table = build_analysis_table(
        materials,
        property_keys=properties,
        descriptor_keys=list(STRUCTURAL_DESCRIPTORS),
        context=_context_from_request(payload),
    )

    #  Only materials scoreable under *every* application can enter the
    #  comparison — otherwise the applications are compared on different sets.
    ln_z: dict[str, list[float]] = {q: [] for q in properties}
    ln_scores: dict[str, list[float]] = {spec.name: [] for spec in specs}
    descriptor_columns: dict[str, list[float]] = {k: [] for k in STRUCTURAL_DESCRIPTORS}

    for index in range(len(table.material_keys)):
        row = {q: table.columns[q][index] for q in properties}
        results = [
            score_material(row, spec, material_key=table.material_keys[index]) for spec in specs
        ]
        if any(r.status is ScoreStatus.NOT_SCORED for r in results):
            continue
        for spec, result in zip(specs, results, strict=True):
            ln_scores[spec.name].append(result.log_value)  # type: ignore[arg-type]
        reference = next(iter(results))
        for q in properties:
            component = next(
                (r.components[q] for r in results if q in r.components), None
            )
            ln_z[q].append(float(np.log(component["z"])) if component else float("nan"))
        for key in descriptor_columns:
            value = table.columns.get(key, [None] * len(table.material_keys))[index]
            descriptor_columns[key].append(float(value) if value is not None else float("nan"))
        del reference

    n = len(next(iter(ln_scores.values())))
    if n < 3:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Only {n} material(s) are scoreable under all of {payload.fom_names}. "
            "Sec. 15.2: a two-material comparison does not establish a population correlation.",
        )

    score_matrix = np.array([ln_scores[spec.name] for spec in specs], dtype=float)
    observed = np.corrcoef(score_matrix)

    null, names, _ = null_score_correlation(specs, properties)
    excess = excess_correlation(observed, null)

    #  Eq. (62): reconstruct R_FF from W Cov(ln z) W^T and compare.
    w, _, w_properties = weight_matrix(specs, properties)
    z_matrix = np.array([ln_z[q] for q in w_properties], dtype=float)
    reconstruction_ok = None
    max_error = None
    if np.isfinite(z_matrix).all():
        reconstructed = covariance_to_correlation(
            reconstruct_log_score_covariance(w, np.cov(z_matrix))
        )
        finite = np.isfinite(observed) & np.isfinite(reconstructed)
        max_error = float(np.max(np.abs(observed[finite] - reconstructed[finite])))
        reconstruction_ok = bool(max_error <= 1e-6)

    usable_descriptors = {
        key: np.asarray(values, dtype=float)
        for key, values in descriptor_columns.items()
        if np.isfinite(values).all() and np.std(values) > 0
    }
    leakage = leakage_check(
        {name: np.asarray(ln_scores[name], dtype=float) for name in names},
        {q: np.asarray(ln_z[q], dtype=float) for q in properties if np.isfinite(ln_z[q]).all()},
        usable_descriptors or None,
    )

    def _clean(matrix: np.ndarray) -> list[list[float | None]]:
        return [[None if not np.isfinite(v) else float(v) for v in row] for row in matrix]

    return IntegrityResponse(
        applications=names,
        observed_correlation=_clean(observed),
        null_correlation=[[float(v) for v in row] for row in null],
        excess_correlation=_clean(excess),
        reconstruction_ok=reconstruction_ok,
        max_reconstruction_error=max_error,
        leakage=leakage.as_dict(),
        notes=[
            f"Computed over {n} material(s) scoreable under every listed application.",
            "Sec. 11.1: a FOM-FOM correlation is produced jointly by property-channel "
            "correlation and weight overlap. Interpret the excess, not the observed value.",
        ],
    )
