"""Promotion gates: what reaches ``property_values`` and what is refused.

The rules under test come from FOM_PROOF and are the reason this package exists
rather than a direct write. A simulation is MODELED. Identity is supplied, never
inferred. A number derived under a nominal column state is not a measurement.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db import models  # noqa: F401 - registers the mappers
from cnms_fom.db.base import Base
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
from cnms_fom.db.models import Material, PropertyValue, PySeaDerivedScalar
from cnms_fom.pysea.promote import PromotionRefused, promote, promotion_plan
from cnms_fom.pysea.records import import_pysea_record
from tests.pysea_fixtures import EXPERIMENTAL, INVALID, SIMULATION, fixture_envelope


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'promote.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def material(db):
    record = Material(
        formula="HfO2",
        formula_reduced="HfO2",
        polymorph="monoclinic",
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
    )
    db.add(record)
    db.flush()
    return record


def _import(db, name=EXPERIMENTAL, mutate=None):
    envelope = fixture_envelope(name)
    if mutate:
        mutate(envelope)
    result = import_pysea_record(db, envelope)
    db.commit()
    return result["id"]


# --- the plan --------------------------------------------------------------


def test_the_plan_writes_nothing(db):
    record_id = _import(db)
    before = db.query(PropertyValue).count()
    promotion_plan(db, record_id)
    assert db.query(PropertyValue).count() == before


def test_a_complete_experimental_scalar_is_eligible_as_measured(db):
    plan = promotion_plan(db, _import(db))
    eligible = {item["property_key"]: item for item in plan["eligible"]}

    assert "eps_inf" in eligible
    assert eligible["eps_inf"]["proposed_tier"] == ProvenanceTier.MEASURED.value
    assert plan["instrument_quantitative"] is True


def test_a_scalar_without_a_registry_key_is_refused(db):
    """The LO phonon energy has no specification here: no context, no direction."""
    plan = promotion_plan(db, _import(db))
    refused = {item["name"]: item for item in plan["refused"]}
    phonon = refused["longitudinal optical phonon energy"]
    assert any("registry key" in reason for reason in phonon["reasons"])


def test_a_simulated_scalar_is_planned_as_modeled(db):
    plan = promotion_plan(db, _import(db, SIMULATION))
    eligible = {item["property_key"]: item for item in plan["eligible"]}
    assert eligible["eps_inf"]["proposed_tier"] == ProvenanceTier.MODELED.value


def test_a_nominal_state_refuses_every_scalar(db):
    def make_nominal(envelope):
        envelope["instrument_state"]["twin_reconstructed"] = False
        envelope["instrument_state"]["lens_strength_source"] = "nominal"

    plan = promotion_plan(db, _import(db, mutate=make_nominal))
    assert plan["eligible"] == []
    assert plan["instrument_quantitative"] is False
    reasons = " ".join(r for item in plan["refused"] for r in item["reasons"])
    assert "twin-reconstructed" in reasons


def test_a_missing_collection_angle_refuses(db):
    def drop_angle(envelope):
        del envelope["instrument_state"]["collection_semi_angle_mrad"]
        del envelope["calibrations"]["cal-veels-2026-Q2"]["collection_angle_mrad"]

    plan = promotion_plan(db, _import(db, mutate=drop_angle))
    reasons = " ".join(r for item in plan["refused"] for r in item["reasons"])
    assert "collection semi-angle" in reasons


def test_a_scalar_without_uncertainty_is_refused(db):
    def drop_uncertainty(envelope):
        del envelope["derived_scalars"][0]["uncertainty"]

    plan = promotion_plan(db, _import(db, mutate=drop_uncertainty))
    refused = {item["property_key"]: item for item in plan["refused"]}
    assert any("uncertainty" in reason for reason in refused["eps_inf"]["reasons"])


def test_missing_required_context_is_refused(db):
    """eps_inf requires a method; without it the value is unusable (Sec. 16)."""
    def drop_method(envelope):
        envelope["derived_scalars"][0]["context"] = {"thickness_nm": 18.4}

    plan = promotion_plan(db, _import(db, mutate=drop_method))
    refused = {item["property_key"]: item for item in plan["refused"]}
    assert any("required context" in reason for reason in refused["eps_inf"]["reasons"])


# --- promotion -------------------------------------------------------------


def test_promotion_is_dry_by_default(db, material):
    record_id = _import(db)
    result = promote(db, record_id, material_id=material.id)
    assert result["dry_run"] is True
    assert result["written"] == 0
    assert db.query(PropertyValue).count() == 0


def test_committing_writes_a_property_value_linked_back(db, material):
    record_id = _import(db)
    result = promote(db, record_id, material_id=material.id, dry_run=False)
    db.commit()

    assert result["written"] == 1
    row = db.query(PropertyValue).one()
    assert row.property_key == "eps_inf"
    assert row.value == 4.18
    assert row.uncertainty == 0.09
    assert row.provenance_tier is ProvenanceTier.MEASURED
    assert row.source_locator == f"pysea_record:{record_id}"
    assert row.database_identifier == "d/41772093"
    #  The method string records the column state the number rests on.
    assert "60.0 keV" in row.method
    assert "twin-reconstructed" in row.method

    scalar = db.query(PySeaDerivedScalar).filter_by(property_key="eps_inf").one()
    assert scalar.promotion_status == "promoted"
    assert scalar.promoted_property_value_id == row.id


def test_a_simulation_lands_modeled_however_well_it_agrees(db, material):
    record_id = _import(db, SIMULATION)
    promote(db, record_id, material_id=material.id, dry_run=False)
    db.commit()

    row = db.query(PropertyValue).one()
    assert row.provenance_tier is ProvenanceTier.MODELED
    assert "simulated with pySEA multislice" in row.method


def test_promotion_refuses_an_unknown_material(db):
    record_id = _import(db)
    with pytest.raises(PromotionRefused, match="No material"):
        promote(db, record_id, material_id=9999, dry_run=False)


def test_promotion_refuses_an_invalid_record(db, material):
    record_id = _import(db, INVALID)
    with pytest.raises(PromotionRefused, match="failed validation"):
        promote(db, record_id, material_id=material.id, dry_run=False)


def test_promotion_refuses_a_non_twin_state(db, material):
    def make_nominal(envelope):
        envelope["instrument_state"]["twin_reconstructed"] = False

    record_id = _import(db, mutate=make_nominal)
    with pytest.raises(PromotionRefused, match="quantitative"):
        promote(db, record_id, material_id=material.id, dry_run=False)


def test_promoting_an_ineligible_scalar_by_id_is_refused(db, material):
    record_id = _import(db)
    plan = promotion_plan(db, record_id)
    refused_id = plan["refused"][0]["scalar_id"]

    with pytest.raises(PromotionRefused, match="not eligible"):
        promote(db, record_id, material_id=material.id, scalar_ids=[refused_id], dry_run=False)


def test_a_refused_scalar_records_why_after_a_commit(db, material):
    record_id = _import(db)
    promote(db, record_id, material_id=material.id, dry_run=False)
    db.commit()

    phonon = (
        db.query(PySeaDerivedScalar)
        .filter_by(name="longitudinal optical phonon energy")
        .one()
    )
    assert phonon.promotion_status == "refused"
    assert phonon.refusal_reasons


def test_promotion_records_the_material_on_the_record(db, material):
    """The importer never sets it; a person supplying it does."""
    record_id = _import(db)
    from cnms_fom.db.models import PySeaRecord

    assert db.get(PySeaRecord, record_id).material_id is None
    promote(db, record_id, material_id=material.id, dry_run=False)
    db.commit()
    assert db.get(PySeaRecord, record_id).material_id == material.id


def test_a_missing_record_raises_lookup_error(db, material):
    with pytest.raises(LookupError):
        promote(db, 9999, material_id=material.id)
