"""Protocol invariants enforced by the database, not by convention.

The FOM engine already refuses to manufacture a score. These tests check the
*storage* layer refuses too — which matters as soon as anything other than the
API writes to the database: an ingestion script, a migration, a notebook, or a
colleague with psql open.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import ProvenanceTier, ScoreStatus, SpecimenForm
from cnms_fom.db.models import FomDefinition, FomScore, Material, PropertyValue


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'constraints.db'}", future=True)
    #  SQLite does not enforce foreign keys unless asked, and the protocol
    #  depends on a score never outliving its definition.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db:
        yield db
    engine.dispose()


@pytest.fixture
def material(session):
    row = Material(
        formula="HfO2",
        formula_reduced="HfO2",
        polymorph="monoclinic",
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
    )
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def definition(session):
    row = FomDefinition(
        name="logic",
        version=1,
        application="logic",
        weights={"k": 1.0},
        normalization={"k": {"lo": 0.0, "hi": 2.0, "direction": "benefit", "transform": "log10"}},
        floor_eps=1e-3,
    )
    session.add(row)
    session.commit()
    return row


def _property(material, **overrides) -> PropertyValue:
    payload = dict(
        material_id=material.id,
        property_key="k",
        value=25.0,
        provenance_tier=ProvenanceTier.MEASURED,
        temperature_k=300.0,
        frequency_hz=1e4,
        tensor_component="zz",
    )
    payload.update(overrides)
    return PropertyValue(**payload)


# ---------------------------------------------------------------------------
# Material identity — Sec. 2.1
# ---------------------------------------------------------------------------


def test_blank_polymorph_is_rejected(session):
    """A whitespace polymorph would reinstate the formula-only identity."""
    session.add(
        Material(
            formula="TiO2",
            formula_reduced="TiO2",
            polymorph="   ",
            specimen_form=SpecimenForm.CERAMIC,
        )
    )
    with pytest.raises(IntegrityError, match="ck_material_polymorph_not_blank"):
        session.commit()


def test_same_formula_different_polymorph_is_allowed(session):
    """Rutile and anatase are different materials, and both must be storable."""
    for polymorph in ("rutile", "anatase"):
        session.add(
            Material(
                formula="TiO2",
                formula_reduced="TiO2",
                polymorph=polymorph,
                specimen_form=SpecimenForm.CERAMIC,
            )
        )
    session.commit()
    assert session.query(Material).count() == 2


# ---------------------------------------------------------------------------
# Measurement context — Eq. (3)
# ---------------------------------------------------------------------------


def test_context_digest_is_assigned_automatically(session, material):
    value = _property(material)
    session.add(value)
    session.commit()
    assert value.context_digest and len(value.context_digest) == 32


def test_context_digest_follows_a_context_edit(session, material):
    value = _property(material)
    session.add(value)
    session.commit()
    before = value.context_digest

    value.temperature_k = 400.0
    session.commit()
    assert value.context_digest != before


def test_duplicate_measurement_is_rejected(session, material):
    """Sec. 2.1: the same source, same context, same quantity, twice.

    Matched on the columns rather than the constraint name: SQLite reports a
    uniqueness violation as "UNIQUE constraint failed: <columns>" and never
    names the constraint, while Postgres names it. The columns appear in both.
    """
    session.add(_property(material))
    session.commit()
    session.add(_property(material))
    with pytest.raises(IntegrityError, match="context_digest"):
        session.commit()


def test_same_property_in_a_different_context_is_allowed(session, material):
    """Two frequencies are two measurements, never an average."""
    session.add(_property(material, frequency_hz=1e4))
    session.add(_property(material, frequency_hz=1e6, value=23.0))
    session.commit()
    assert session.query(PropertyValue).count() == 2


def test_independent_sources_can_both_be_stored(session, material):
    """Two papers reporting the same quantity are evidence, not a conflict."""
    session.add(_property(material, doi="10.1/aaa"))
    session.add(_property(material, doi="10.1/bbb", value=26.0))
    session.commit()
    assert session.query(PropertyValue).count() == 2


@pytest.mark.parametrize(
    ("field", "bad_value", "constraint"),
    [
        ("temperature_k", -5.0, "ck_property_temperature_positive"),
        ("frequency_hz", -1.0, "ck_property_frequency_nonneg"),
        ("thickness_nm", 0.0, "ck_property_thickness_positive"),
        ("area_cm2", -1e-4, "ck_property_area_positive"),
        ("uncertainty", -0.1, "ck_property_uncertainty_nonneg"),
    ],
)
def test_unphysical_values_are_rejected(session, material, field, bad_value, constraint):
    session.add(_property(material, **{field: bad_value}))
    with pytest.raises(IntegrityError, match=constraint):
        session.commit()


# ---------------------------------------------------------------------------
# Score completeness — Sec. 6.2 and 2.3
# ---------------------------------------------------------------------------


def test_not_scored_may_not_carry_a_value(session, material, definition):
    """The "no manufactured score" rule, as a database constraint."""
    session.add(
        FomScore(
            material_id=material.id,
            fom_definition_id=definition.id,
            value=0.5,
            status=ScoreStatus.NOT_SCORED,
        )
    )
    with pytest.raises(IntegrityError, match="ck_score_not_scored_has_no_value"):
        session.commit()


def test_scored_must_carry_a_value(session, material, definition):
    session.add(
        FomScore(
            material_id=material.id,
            fom_definition_id=definition.id,
            value=None,
            status=ScoreStatus.SCORED,
        )
    )
    with pytest.raises(IntegrityError, match="ck_score_scored_has_value"):
        session.commit()


def test_modeled_input_forces_illustrative(session, material, definition):
    """Sec. 2.3: a modeled input cannot be laundered into a measurement-based score."""
    session.add(
        FomScore(
            material_id=material.id,
            fom_definition_id=definition.id,
            value=0.4,
            status=ScoreStatus.SCORED,
            uses_modeled_inputs=True,
        )
    )
    with pytest.raises(IntegrityError, match="ck_score_modeled_is_illustrative"):
        session.commit()


# ---------------------------------------------------------------------------
# FOM definitions — Sec. 5.3 and 6.2
# ---------------------------------------------------------------------------


def test_approval_needs_a_named_approver(session):
    session.add(
        FomDefinition(
            name="power",
            version=1,
            application="power",
            weights={"k": 1.0},
            normalization={"k": {"lo": 0.0, "hi": 2.0}},
            approved=True,
            approved_by=None,
        )
    )
    with pytest.raises(IntegrityError, match="ck_fom_approved_has_approver"):
        session.commit()


def test_floor_eps_must_be_a_small_positive_number(session):
    session.add(
        FomDefinition(
            name="rf",
            version=1,
            application="rf",
            weights={"k": 1.0},
            normalization={"k": {"lo": 0.0, "hi": 2.0}},
            floor_eps=0.0,
        )
    )
    with pytest.raises(IntegrityError, match="ck_fom_floor_eps_in_unit_interval"):
        session.commit()


def test_frozen_definition_cannot_be_edited(session, definition):
    """Sec. 5.3: editing frozen weights silently redefines every published score."""
    definition.frozen = True
    session.commit()

    definition.weights = {"k": 0.5, "Eg": 0.5}
    with pytest.raises(PermissionError, match="frozen"):
        session.commit()
    session.rollback()


def test_frozen_definition_cannot_be_deleted(session, definition):
    definition.frozen = True
    session.commit()

    session.delete(definition)
    with pytest.raises(PermissionError, match="frozen"):
        session.commit()
    session.rollback()


def test_freezing_itself_is_allowed(session, definition):
    """The transition into frozen has to be possible, or nothing could freeze."""
    definition.frozen = True
    session.commit()
    assert definition.frozen


def test_approval_metadata_stays_editable_on_a_frozen_definition(session, definition):
    """Recording who approved it does not change what the score means."""
    definition.frozen = True
    session.commit()

    definition.approved = True
    definition.approved_by = "Z. Woodel"
    definition.approved_at = datetime.now(timezone.utc)
    session.commit()
    assert definition.approved_by == "Z. Woodel"
