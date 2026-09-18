#!/usr/bin/env python
"""Pre-release check against the FOM_PROOF Sec. 16 reproducibility checklist.

Run before releasing any ranking or correlation table. It reports what the
repository can verify automatically; the remaining items need a human.

    python scripts/check_protocol_compliance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import func, inspect, select  # noqa: E402

from cnms_fom.cnms_integration.provenance import RunProvenance  # noqa: E402
from cnms_fom.db.base import get_engine, session_scope  # noqa: E402
from cnms_fom.db.context import CONTEXT_FIELDS, context_digest, describe_context  # noqa: E402
from cnms_fom.db.enums import ProvenanceTier  # noqa: E402
from cnms_fom.db.models import (  # noqa: E402
    ExternalRecord,
    FomDefinition,
    FomScore,
    Material,
    PropertyValue,
    SpectralPoint,
    SpectralSeries,
)
from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES  # noqa: E402
from cnms_fom.fom_engine.eligibility import missing_context_fields  # noqa: E402

#  Indices the FOM and BO hot paths depend on.  A missing one is a performance
#  cliff rather than a wrong answer, so it is reported as a warning.
EXPECTED_INDICES: dict[str, tuple[str, ...]] = {
    "property_values": ("ix_property_material_key", "ix_property_key_tier"),
    "descriptor_values": ("ix_descriptor_material_key",),
    "bo_observations": ("ix_bo_obs_run_feasible",),
    "fom_scores": ("ix_score_definition_status",),
    "spectral_points": ("ix_spectral_point_series_x",),
}

def _database_checks(db) -> tuple[list[str], list[str]]:
    """Checks that only the database can answer.

    The in-memory checks below verify the maths. These verify that what is
    *stored* still satisfies the protocol — which is a different question, and
    the one that matters once more than one process writes to the database.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # --- derived columns still agree with their source columns --------------
    #  ORM events maintain context_digest, but bulk inserts and raw SQL bypass
    #  them. A stale digest silently splits duplicates across two fingerprints,
    #  defeating the uniqueness constraint, so it is recomputed here.
    columns = ("id", "material_id", "property_key", "context_digest", *CONTEXT_FIELDS)
    stale = 0
    for row in db.execute(
        select(*(getattr(PropertyValue, name) for name in columns))
    ).mappings():
        if row["context_digest"] != context_digest(dict(row)):
            stale += 1
            if stale <= 3:
                failures.append(
                    f"PropertyValue {row['id']} ({row['property_key']}) has a stale "
                    f"context_digest — written by a path that bypassed the ORM events. "
                    f"Context: {describe_context(dict(row))}"
                )
    if stale > 3:
        failures.append(f"... and {stale - 3} more stale context digests.")

    # --- a computational value must name its functional ---------------------
    #  The registry cannot require xc_functional unconditionally: an optically
    #  derived eps_inf has no exchange-correlation functional. So the rule is
    #  conditional on the method actually being a calculation (Sec. 16 item 8).
    computational = ("dft", "dfpt", "pbe", "hse", "scan", "lda", "gga", "gw", "b3lyp")
    for row in db.execute(
        select(PropertyValue.id, PropertyValue.property_key, PropertyValue.method).where(
            PropertyValue.value.isnot(None),
            PropertyValue.xc_functional.is_(None),
            PropertyValue.method.isnot(None),
        )
    ).all():
        method = (row.method or "").lower()
        if any(token in method for token in computational):
            failures.append(
                f"PropertyValue {row.id} ({row.property_key}) has a computational method "
                f"({row.method!r}) but no xc_functional. Sec. 16 item 8: values from different "
                "functionals are not interchangeable."
            )

    # --- the same measurement stored twice ----------------------------------
    duplicates = db.execute(
        select(
            PropertyValue.material_id,
            PropertyValue.property_key,
            PropertyValue.context_digest,
            func.count().label("n"),
        )
        .group_by(
            PropertyValue.material_id, PropertyValue.property_key, PropertyValue.context_digest
        )
        .having(func.count() > 1)
    ).all()
    for material_id, key, _digest, n in duplicates:
        failures.append(
            f"{n} property values share material {material_id} / {key} / context. "
            "Sec. 2.1: merging needs an explicit aggregation rule; the database should have "
            "rejected these, so the uniqueness constraint is probably missing."
        )

    # --- scores must not outlive the definition that explains them -----------
    orphan_scores = db.execute(
        select(func.count(FomScore.id)).where(FomScore.fom_definition_id.is_(None))
    ).scalar()
    if orphan_scores:
        failures.append(
            f"{orphan_scores} FOM score(s) have no definition. A score without its weights, "
            "bounds, and transform is not interpretable (Sec. 5.3)."
        )

    unfrozen = db.execute(
        select(func.count(FomScore.id))
        .select_from(FomScore)
        .join(FomDefinition, FomScore.fom_definition_id == FomDefinition.id)
        .where(FomDefinition.frozen.is_(False))
    ).scalar()
    if unfrozen:
        warnings.append(
            f"{unfrozen} score(s) reference an unfrozen definition. Freeze it before publishing "
            "a ranking, or the definition can change underneath the numbers (Sec. 5.3)."
        )

    # --- spectral series denormalised counts --------------------------------
    mismatched = db.execute(
        select(SpectralSeries.id, SpectralSeries.n_points, func.count(SpectralPoint.id))
        .outerjoin(SpectralPoint, SpectralPoint.series_id == SpectralSeries.id)
        .group_by(SpectralSeries.id, SpectralSeries.n_points)
        .having(SpectralSeries.n_points != func.count(SpectralPoint.id))
    ).all()
    if mismatched:
        warnings.append(
            f"{len(mismatched)} spectral series report an n_points that disagrees with their "
            "actual point count."
        )

    # --- external ingestion backlog -----------------------------------------
    quarantined = db.execute(
        select(ExternalRecord.quarantine_reason, func.count())
        .where(ExternalRecord.status == "quarantined")
        .group_by(ExternalRecord.quarantine_reason)
    ).all()
    for reason, n in quarantined:
        warnings.append(f"{n} external record(s) quarantined: {reason}")

    # --- indices --------------------------------------------------------------
    inspector = inspect(get_engine())
    present_tables = set(inspector.get_table_names())
    for table, expected in EXPECTED_INDICES.items():
        if table not in present_tables:
            continue
        present = {ix["name"] for ix in inspector.get_indexes(table)}
        for name in expected:
            if name not in present:
                warnings.append(
                    f"Index {name} missing on {table}: a hot query path will fall back to a "
                    "sequential scan. Run `cnms-fom migrate up`."
                )

    return failures, warnings


MANUAL_ITEMS = [
    "Tensor aggregation rules are stated wherever a scalar was derived (item 3).",
    "The eligible-material set was frozen before the final calculation (item 10).",
    "Composite-score correlations are presented as downstream outputs (item 15).",
    "Remaining gaps are marked [DATA GAP: explicitly unresolved] (item 17).",
]


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    provenance = RunProvenance(kind="compliance-check")
    failures.extend(provenance.warnings())

    with session_scope() as db:
        materials = db.query(Material).count()
        if materials == 0:
            warnings.append("No materials in the database; nothing to check.")

        # Item 1: polymorph and specimen form on every record.
        for material in db.query(Material).all():
            if not material.polymorph:
                failures.append(f"Material {material.id} has no polymorph (item 1).")

        # Items 4-9: every value carries the context its property requires.
        missing_context = 0
        for value in db.query(PropertyValue).filter(PropertyValue.value.isnot(None)).all():
            absent = missing_context_fields(value, value.property_key)
            if absent:
                missing_context += 1
                if missing_context <= 5:
                    failures.append(
                        f"PropertyValue {value.id} ({value.property_key}) missing {absent} "
                        "(items 4-9)."
                    )
        if missing_context > 5:
            failures.append(f"... and {missing_context - 5} more values missing required context.")

        # Item 2: measured / calculated / modeled are labelled separately.
        modeled = (
            db.query(PropertyValue)
            .filter(PropertyValue.provenance_tier == ProvenanceTier.MODELED)
            .count()
        )
        if modeled:
            warnings.append(
                f"{modeled} modeled value(s) present. Any score using them is ILLUSTRATIVE and "
                "must be reported separately from measurement-based results (item 2)."
            )

        # Item 11: transforms, bounds, floors, and weights are versioned.
        for definition in db.query(FomDefinition).all():
            if not definition.approved:
                warnings.append(
                    f"FOM {definition.name} v{definition.version} is unapproved — not reportable."
                )
            if not definition.bounds_basis or "DRAFT" in (definition.bounds_basis or ""):
                failures.append(
                    f"FOM {definition.name} v{definition.version} still carries draft bounds "
                    "(item 11). Re-derive them from the frozen eligible set."
                )
            unknown = sorted(set(definition.weights) - set(PHYSICAL_PROPERTIES))
            if unknown:
                failures.append(f"FOM {definition.name} weights unknown properties {unknown}.")

        db_failures, db_warnings = _database_checks(db)
        failures.extend(db_failures)
        warnings.extend(db_warnings)

    print("FOM_PROOF Sec. 16 — automated checks\n" + "=" * 42)
    for warning in warnings:
        print(f"  [warn] {warning}")
    for failure in failures:
        print(f"  [FAIL] {failure}")
    if not failures:
        print("  [ok]   No automated check failed.")

    print("\nStill requires a human:")
    for item in MANUAL_ITEMS:
        print(f"  [ ] {item}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
