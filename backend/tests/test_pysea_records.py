"""Importing a container: idempotency, no bulk arrays, invalid records still stored."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db import models  # noqa: F401 - registers the mappers
from cnms_fom.db.base import Base
from cnms_fom.db.models import PySeaDerivedScalar, PySeaRecord, PySeaSignalRow
from cnms_fom.pysea.contract import ContractError
from cnms_fom.pysea.records import import_pysea_record, list_records, record_detail
from tests.pysea_fixtures import EXPERIMENTAL, INVALID, SIMULATION, fixture_envelope


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'pysea.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


def test_an_experimental_container_imports_with_its_signals_and_scalars(db):
    result = import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()

    assert result["created"] is True
    assert result["validation_status"] == "valid"
    assert result["n_signals"] == 2
    assert result["n_scalars"] == 2

    record = db.get(PySeaRecord, result["id"])
    assert record.record_kind == "experimental"
    assert record.sample_id == "HFO2-SI-042"
    assert record.datafed_record_id == "d/41772093"
    assert record.instrument_state["twin_reconstructed"] is True


def test_reimporting_the_same_container_returns_the_existing_row(db):
    """Idempotent by content hash: one acquisition, one row."""
    first = import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()
    second = import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()

    assert second["created"] is False
    assert second["id"] == first["id"]
    assert db.query(PySeaRecord).count() == 1
    assert db.query(PySeaSignalRow).count() == 2


def test_a_reordered_export_is_the_same_acquisition(db):
    envelope = fixture_envelope(EXPERIMENTAL)
    reordered = dict(reversed(list(envelope.items())))

    import_pysea_record(db, envelope)
    db.commit()
    second = import_pysea_record(db, reordered)
    db.commit()

    assert second["created"] is False
    assert db.query(PySeaRecord).count() == 1


def test_bulk_arrays_are_stripped_before_storage(db):
    """A 4D-STEM scan is gigabytes. Postgres stores the locator, not the detector."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["signals"][0]["data"] = [[1.0] * 16] * 16

    result = import_pysea_record(db, envelope)
    db.commit()

    record = db.get(PySeaRecord, result["id"])
    blob = str(record.raw_metadata)
    assert "data" not in record.raw_metadata["signals"][0]
    assert len(blob) < 20000
    #  The reference survives.
    signal = db.query(PySeaSignalRow).filter_by(signal_id="veels-q-resolved").one()
    assert signal.data_ref["record_id"] == "d/41772093"
    assert signal.shape == [64, 64, 1024]


def test_an_invalid_container_is_stored_with_its_issues(db):
    """The acquisition happened. Refusing would leave that unrecorded."""
    result = import_pysea_record(db, fixture_envelope(INVALID))
    db.commit()

    assert result["created"] is True
    assert result["validation_status"] == "invalid"
    assert len(result["issues"]) > 0

    record = db.get(PySeaRecord, result["id"])
    assert record.validation_status == "invalid"
    assert any(issue["severity"] == "error" for issue in record.validation_issues)


def test_the_importer_never_sets_material_id(db):
    """Identity is composition + polymorph + specimen form, supplied at promotion."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["sample"]["material_id"] = 42

    result = import_pysea_record(db, envelope)
    db.commit()

    assert db.get(PySeaRecord, result["id"]).material_id is None


def test_a_simulation_keeps_its_provenance_block(db):
    result = import_pysea_record(db, fixture_envelope(SIMULATION))
    db.commit()

    record = db.get(PySeaRecord, result["id"])
    assert record.record_kind == "simulation"
    assert record.simulation["code"] == "pySEA multislice"
    assert record.simulation["interatomic_potential"].startswith("MACE")


def test_an_unsupported_contract_version_is_refused(db):
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["contract_version"] = "pysea-canonical/1.0"
    with pytest.raises(ContractError):
        import_pysea_record(db, envelope)


def test_unmapped_fields_survive_into_the_stored_metadata(db):
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["aberration_coefficients"] = {"C30_um": 1.2}

    result = import_pysea_record(db, envelope)
    db.commit()

    record = db.get(PySeaRecord, result["id"])
    unmapped = record.raw_metadata["extensions"]["_unmapped"]
    assert unmapped["aberration_coefficients"] == {"C30_um": 1.2}


def test_a_scalar_without_a_registry_key_keeps_a_null_key(db):
    """The LO phonon energy has no registry key; inventing one is not an option."""
    result = import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()

    scalars = db.query(PySeaDerivedScalar).filter_by(pysea_record_id=result["id"]).all()
    by_name = {s.name: s for s in scalars}
    phonon = by_name["longitudinal optical phonon energy"]
    assert phonon.property_key is None
    assert phonon.value == 0.0731


def test_listing_filters_by_sample_and_kind(db):
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    import_pysea_record(db, fixture_envelope(SIMULATION))
    db.commit()

    assert len(list_records(db, sample_id="HFO2-SI-042")) == 2
    assert len(list_records(db, record_kind="simulation")) == 1
    assert len(list_records(db, sample_id="nobody")) == 0


def test_record_detail_returns_metadata_and_no_arrays(db):
    result = import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()

    detail = record_detail(db, result["id"])
    assert detail["contract_version"] == "pysea-canonical/0.1"
    assert len(detail["signals"]) == 2
    assert len(detail["derived_scalars"]) == 2
    assert all("data" not in signal for signal in detail["signals"])


def test_record_detail_raises_for_an_unknown_id(db):
    with pytest.raises(LookupError):
        record_detail(db, 9999)
