"""External-database ingestion, against a synthetic source built in the test.

Deliberately not pointed at the real ``materials_oxide_test.db``: that file is
20 MB and lives on one laptop, so a test depending on it would pass locally and
silently skip in CI. The fixture below reproduces its schema and its awkward
parts — a material with no polymorph anywhere, a packed ``dataset_label``, and
an ``xray_sld`` column that carries the real part in one row and the imaginary
part in another — which is what the ingester actually has to survive.
"""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import SpecimenForm
from cnms_fom.db.models import (  # noqa: F401
    DescriptorValue,
    ExternalRecord,
    Material,
    PropertyValue,
    SpectralPoint,
    SpectralSeries,
)
from cnms_fom.ingest import materials_db

SOURCE_SCHEMA = """
CREATE TABLE materials (
    material_id INTEGER PRIMARY KEY, name TEXT, formula TEXT, inchikey TEXT
);
CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY, doi TEXT, title TEXT, authors TEXT,
    year INTEGER, url TEXT, uncertainty REAL
);
CREATE TABLE physical_properties (
    record_id INTEGER PRIMARY KEY, material_id INTEGER, density_g_cm3 REAL,
    xray_sld REAL, neutron_sld REAL, dielectric_constant REAL,
    temperature_c REAL, frequency_hz REAL, dataset_label TEXT, source_id INTEGER
);
CREATE TABLE optical_dispersion (
    record_id INTEGER PRIMARY KEY, material_id INTEGER, wavelength_nm REAL,
    n REAL, k REAL, temperature_c REAL, dataset_label TEXT, source_id INTEGER
);
"""


@pytest.fixture
def source_db(tmp_path):
    """A miniature stand-in for the CNMS oxide export."""
    path = tmp_path / "source.db"
    connection = sqlite3.connect(path)
    connection.executescript(SOURCE_SCHEMA)

    connection.execute(
        "INSERT INTO materials VALUES (1, 'Aluminium oxide / sapphire', 'Al2O3', 'X')"
    )
    #  No phase recoverable anywhere for this one — it must quarantine.
    connection.execute("INSERT INTO materials VALUES (2, 'Hafnium dioxide', 'HfO2', 'Y')")
    connection.execute(
        "INSERT INTO sources VALUES (1, '10.1364/JOSA.62.001405', 'Malitson', 'I. Malitson', "
        "1972, 'https://example.org', 0.001)"
    )

    #  Same column, two different physical quantities, told apart only by label.
    connection.execute(
        "INSERT INTO physical_properties VALUES "
        "(1, 1, 3.95, 32.5, NULL, NULL, NULL, NULL, "
        "'corundum/sapphire | xray_sld_real | periodictable_CuKalpha', 1)"
    )
    connection.execute(
        "INSERT INTO physical_properties VALUES "
        "(2, 1, NULL, 0.41, NULL, NULL, NULL, NULL, "
        "'corundum/sapphire | xray_sld_imag | periodictable_CuKalpha', 1)"
    )
    #  HfO2: data present, phase absent.
    connection.execute(
        "INSERT INTO physical_properties VALUES (3, 2, 9.68, NULL, NULL, NULL, NULL, NULL, "
        "'density_MP_DFT', 1)"
    )

    #  n only, no k: the common case in a literature compilation, and the one
    #  where transparency cannot be verified.
    for index, (wavelength, n, k) in enumerate(
        [(500.0, 1.774, None), (1000.0, 1.755, None), (1500.0, 1.746, None), (1800.0, 1.738, None)]
    ):
        connection.execute(
            "INSERT INTO optical_dispersion VALUES (?, 1, ?, ?, ?, 19.85, ?, 1)",
            (index + 1, wavelength, n, k, "corundum/sapphire | Malitson1972 | o-ray"),
        )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'target.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db:
        yield db
    engine.dispose()


def test_survey_writes_nothing_and_reports_coverage(source_db, session):
    report = materials_db.survey(source_db)
    assert report.mode == "survey"
    assert report.source_counts["optical_dispersion"] == 4
    assert session.query(Material).count() == 0
    assert any("No FOM input property" in note for note in report.notes)


def test_survey_counts_rows_whose_phase_cannot_be_recovered(source_db):
    report = materials_db.survey(source_db)
    assert report.coverage["optical_rows_with_recoverable_phase"] == 4
    assert report.coverage["optical_rows_without_phase"] == 0


def test_staging_is_idempotent(source_db, session):
    first = materials_db.stage(session, source_db)
    session.flush()
    second = materials_db.stage(session, source_db)
    assert first.staged == 9  # 2 materials + 3 physical + 4 optical
    assert second.staged == 0


def test_promotion_creates_materials_with_a_recovered_polymorph(source_db, session):
    materials_db.stage(session, source_db)
    report = materials_db.promote(
        session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL, operator="tester"
    )
    assert report.materials_created == 1

    material = session.query(Material).one()
    assert material.formula_reduced == "Al2O3"
    #  Recovered from inside the dataset_label, not invented.
    assert material.polymorph == "corundum/sapphire"
    assert material.specimen_form is SpecimenForm.BULK_SINGLE_CRYSTAL
    #  The asserted specimen form is on the record.
    assert "tester" in (material.notes or "")


def test_material_without_a_phase_is_quarantined_not_guessed(source_db, session):
    materials_db.stage(session, source_db)
    report = materials_db.promote(
        session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL
    )
    assert report.quarantined >= 1
    assert any("no polymorph" in reason for reason in report.quarantine_reasons)

    quarantined = (
        session.query(ExternalRecord).filter(ExternalRecord.status == "quarantined").all()
    )
    assert any("polymorph" in (record.missing_fields or []) for record in quarantined)
    assert not session.query(Material).filter(Material.formula_reduced == "HfO2").count()


def test_phase_map_lets_a_person_supply_what_the_source_omits(source_db, session):
    materials_db.stage(session, source_db)
    materials_db.promote(
        session,
        source_db,
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
        phase_map={"Hafnium dioxide": "monoclinic"},
    )
    hafnia = session.query(Material).filter(Material.formula_reduced == "HfO2").one()
    assert hafnia.polymorph == "monoclinic"


def test_real_and_imaginary_sld_are_kept_apart(source_db, session):
    """One column, two quantities. Conflating them breaks any reflectivity fit."""
    materials_db.stage(session, source_db)
    materials_db.promote(session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL)

    by_key = {
        row.property_key: row.value
        for row in session.query(PropertyValue).filter(
            PropertyValue.property_key.like("sld_xray%")
        )
    }
    assert by_key["sld_xray"] == pytest.approx(32.5)
    assert by_key["sld_xray_imag"] == pytest.approx(0.41)


def test_density_becomes_a_structural_descriptor(source_db, session):
    materials_db.stage(session, source_db)
    materials_db.promote(session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL)
    descriptor = session.query(DescriptorValue).filter(DescriptorValue.descriptor_key == "rho").one()
    assert descriptor.value == pytest.approx(3.95)


def test_optical_rows_become_a_series_with_its_axis(source_db, session):
    materials_db.stage(session, source_db)
    materials_db.promote(session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL)

    #  Only n was supplied, so only an n series exists — a missing k curve is a
    #  data gap, not something to fill in.
    assert session.query(SpectralSeries).filter(SpectralSeries.quantity == "k").count() == 0
    series = session.query(SpectralSeries).filter(SpectralSeries.quantity == "n").one()
    assert series.axis == "o-ray"
    #  Sec. 3.2: the axis survives as a tensor component, so a scalar derived
    #  later still knows which direction it came from.
    assert series.tensor_component == "xx"
    assert series.n_points == 4
    assert session.query(SpectralPoint).filter(SpectralPoint.series_id == series.id).count() == 4


def test_eps_inf_derivation_requires_transparency(source_db, session):
    """n^2 - k^2 is only eps_inf where the material does not absorb."""
    materials_db.stage(session, source_db)
    materials_db.promote(session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL)

    #  No k series in this fixture, so transparency cannot be verified and the
    #  derivation must decline rather than assume.
    report = materials_db.derive_eps_inf(session, window_nm=(1000.0, 2000.0))
    assert report.properties_created == 0
    assert any("transparency" in reason for reason in report.quarantine_reasons)


def test_eps_inf_is_derived_when_a_k_curve_proves_transparency(source_db, session):
    materials_db.stage(session, source_db)
    materials_db.promote(session, source_db, specimen_form=SpecimenForm.BULK_SINGLE_CRYSTAL)

    series = session.query(SpectralSeries).filter(SpectralSeries.quantity == "n").one()
    k_series = SpectralSeries(
        material_id=series.material_id,
        quantity="k",
        independent_variable="wavelength_nm",
        axis=series.axis,
        provenance_tier=series.provenance_tier,
        n_points=4,
    )
    session.add(k_series)
    session.flush()
    session.add_all(
        SpectralPoint(series_id=k_series.id, x_value=x, y_value=0.0)
        for x in (500.0, 1000.0, 1500.0, 1800.0)
    )
    session.flush()

    report = materials_db.derive_eps_inf(session, window_nm=(1000.0, 2000.0))
    assert report.properties_created == 1

    value = (
        session.query(PropertyValue).filter(PropertyValue.property_key == "eps_inf").one()
    )
    #  Sapphire: n ~ 1.746 in the near IR, so eps_inf ~ 3.05.
    assert value.value == pytest.approx(3.05, abs=0.1)
    #  Derived from a measurement under a stated approximation, so CALCULATED.
    assert value.provenance_tier.value == "calculated"
    assert "n^2 - k^2" in value.method
