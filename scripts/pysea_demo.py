#!/usr/bin/env python3
"""The pySEA integration end to end, offline.

Five steps: validate a container, import it, ask what could be promoted and why
not, promote one scalar, then compare the result against a second determination of
the same quantity from a different platform.

No network, no pySEA install, no DataFed account, no database server. An in-memory
SQLite database is built and thrown away.

    python scripts/pysea_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from cnms_fom.db import models  # noqa: E402,F401  (registers the mappers)
from cnms_fom.db.base import Base  # noqa: E402
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm  # noqa: E402
from cnms_fom.db.models import FitRecord, Material, PropertyValue  # noqa: E402
from cnms_fom.pysea.compare import compare_across_platforms  # noqa: E402
from cnms_fom.pysea.promote import promote, promotion_plan  # noqa: E402
from cnms_fom.pysea.records import import_pysea_record  # noqa: E402
from cnms_fom.pysea.validate import validate_envelope  # noqa: E402

FIXTURES = ROOT / "backend" / "tests" / "fixtures" / "pysea"
SAMPLE_ID = "HFO2-SI-042"
QUANTITY = "eps_inf"


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def step_1_validate() -> dict:
    rule("1. Validate a container, without storing it")
    envelope = load("example_experimental.json")
    report = validate_envelope(envelope)

    print(f"  record      {envelope['record_id']}")
    print(f"  kind        {envelope['record_kind']}")
    print(f"  status      {report.status}")
    print(f"  errors      {len(report.errors)}")
    print(f"  warnings    {len(report.warnings)}")
    for issue in report.issues:
        print(f"    [{issue.severity}] {issue.code} at {issue.field_path}")
        print(f"        {issue.message}")

    print("\n  And the same check on a container with faults planted in it:")
    broken = validate_envelope(load("example_invalid.json"))
    print(f"    status {broken.status}, {len(broken.errors)} error(s)")
    for issue in broken.errors:
        print(f"      {issue.code:<34} {issue.field_path}")
        print(f"        -> {issue.remediation}")
    return envelope


def step_2_import(session, envelope: dict) -> int:
    rule("2. Import it")
    result = import_pysea_record(session, envelope, source_filename="example_experimental.json")
    session.commit()

    print(f"  stored as      pysea_record #{result['id']}")
    print(f"  sample         {result['sample_id']}")
    print(f"  instrument     {result['instrument_id']}")
    print(f"  DataFed        {result['datafed_record_id']}")
    print(f"  signals        {result['n_signals']}   scalars {result['n_scalars']}")
    print(f"  content hash   {result['content_sha256'][:16]}...")

    again = import_pysea_record(session, envelope)
    session.commit()
    print(f"\n  Re-importing the same container: created={again['created']}, "
          f"id={again['id']} (idempotent by content hash)")
    return result["id"]


def step_3_plan(session, record_id: int) -> None:
    rule("3. What could be promoted, and what could not")
    plan = promotion_plan(session, record_id)

    print(f"  instrument state is quantitative: {plan['instrument_quantitative']}")
    print(f"\n  ELIGIBLE ({len(plan['eligible'])}):")
    for item in plan["eligible"]:
        print(f"    {item['property_key']} = {item['value']} +/- {item['uncertainty']} "
              f"{item['units']}  -> tier {item['proposed_tier']}")

    print(f"\n  REFUSED ({len(plan['refused'])}):")
    for item in plan["refused"]:
        print(f"    {item['name']}")
        for reason in item["reasons"]:
            print(f"      - {reason}")

    print(f"\n  {plan['note']}")


def step_4_promote(session, record_id: int) -> None:
    rule("4. Promote one scalar, for an explicitly named material")

    material = Material(
        formula="HfO2",
        formula_reduced="HfO2",
        #  Supplied by a person. The importer never sets this: a sample label is
        #  not a composition, a polymorph and a specimen form.
        polymorph="monoclinic",
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
    )
    session.add(material)
    session.flush()
    print(f"  material #{material.id}: {material.formula_reduced}, {material.polymorph}")

    preview = promote(session, record_id, material_id=material.id)
    print(f"\n  Dry run (the default): written={preview['written']}, "
          f"rows in property_values={session.query(PropertyValue).count()}")

    result = promote(session, record_id, material_id=material.id, dry_run=False,
                     operator="demo")
    session.commit()
    print(f"  Commit:  written={result['written']}")

    for created in result["created"]:
        row = session.get(PropertyValue, created["property_value_id"])
        print(f"\n    property_value #{row.id}")
        print(f"      {row.property_key} = {row.value} +/- {row.uncertainty} {row.units}")
        print(f"      tier        {row.provenance_tier.value}")
        print(f"      source      {row.source_locator}  ({row.database_identifier})")
        print(f"      method      {row.method}")

    #  The same quantity from the simulation, to show the tier is decided by the
    #  record kind rather than by how close the numbers are.
    sim_id = import_pysea_record(session, load("example_simulation.json"))["id"]
    session.commit()
    sim = promote(session, sim_id, material_id=material.id, dry_run=False)
    session.commit()
    sim_row = session.get(PropertyValue, sim["created"][0]["property_value_id"])
    print("\n    The simulated determination of the same quantity:")
    print(f"      {sim_row.property_key} = {sim_row.value} {sim_row.units}  "
          f"tier {sim_row.provenance_tier.value}")
    print("      It agrees to within 4%, and it is still MODELED. record_kind decides.")

    _add_modalfit_determination(session, material)


def _add_modalfit_determination(session, material: Material) -> None:
    """A second platform measuring the same film, so the demo has something to compare.

    Written straight into ``property_values`` here because the demo is about the
    pySEA path; in a real run this arrives through ``modalfit.promote`` and its own
    gates.
    """
    fit = FitRecord(
        sample_id=SAMPLE_ID, stack_id="hfo2-se-stack", content_sha256="demo-modalfit-hash",
        techniques=["SE"], chi2_total=1.42, algorithm="LMA",
    )
    session.add(fit)
    session.flush()
    session.add(
        PropertyValue(
            material_id=material.id,
            property_key=QUANTITY,
            value=5.90,
            units="dimensionless",
            uncertainty=0.08,
            method=f"ModalFit SE refinement; Tauc-Lorentz oscillator; fit record #{fit.id}",
            software="ModalFit",
            provenance_tier=ProvenanceTier.MEASURED,
            source_locator=f"fit_record:{fit.id}",
        )
    )
    session.commit()


def step_5_compare(session) -> None:
    rule("5. Compare across platforms")
    result = compare_across_platforms(session, SAMPLE_ID, quantity=QUANTITY)

    print(f"  sample     {result['sample_id']}")
    print(f"  quantity   {result['quantity']}")
    print(f"  platforms  {', '.join(result['platforms'])}")
    print(f"\n  {result['n_determinations']} determination(s), none merged:\n")
    for item in result["determinations"]:
        uncertainty = f" +/- {item['uncertainty']}" if item["uncertainty"] is not None else ""
        print(f"    {item['value']}{uncertainty} {item['units'] or ''}   [{item['provenance_tier']}]")
        print(f"      {item['label']}")
        print(f"      {item['method'][:100]}")
        print()

    print(f"  VERDICT: {result['verdict']}")
    print(f"\n  {result['summary']}")
    print(f"\n  {result['note']}")


def main() -> int:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()

    try:
        envelope = step_1_validate()
        record_id = step_2_import(session, envelope)
        step_3_plan(session, record_id)
        step_4_promote(session, record_id)
        step_5_compare(session)
    finally:
        session.close()
        engine.dispose()

    rule("Done")
    print(
        "  Everything above ran offline against pysea-canonical/0.1, which is a\n"
        "  contract written from pySEA's abstracts rather than from its format.\n"
        "  docs/PYSEA_INTEGRATION.md lists what still has to be confirmed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
