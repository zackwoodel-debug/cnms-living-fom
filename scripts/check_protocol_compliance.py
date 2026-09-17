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

from cnms_fom.cnms_integration.provenance import RunProvenance  # noqa: E402
from cnms_fom.db.base import session_scope  # noqa: E402
from cnms_fom.db.enums import ProvenanceTier  # noqa: E402
from cnms_fom.db.models import FomDefinition, Material, PropertyValue  # noqa: E402
from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES  # noqa: E402
from cnms_fom.fom_engine.eligibility import missing_context_fields  # noqa: E402

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
