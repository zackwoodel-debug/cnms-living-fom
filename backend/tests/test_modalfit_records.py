"""Importing ModalFit refinements, and what the importer refuses to infer.

The rule under test throughout: a fit is a measurement record, and the importer
does not manufacture any part of it. A technique list is not reconstructed from
which slab-model blocks are populated, a missing chi-squared does not become
zero, and a clamped or fixed parameter is flagged rather than quietly accepted.
"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.models import FitDataset, FitLayer, FitRecord
from cnms_fom.modalfit.records import FitImportError, import_directory, import_fit

EXPORT = {
    "stack_id": "20260901_hfo2_v3",
    "sample_id": "HFO2-PILOT-07",
    "fit": {
        "techniques": ["XRR"],
        "algorithm": "L-BFGS-B",
        "chi2": 1.84,
        "chi2_by_technique": {"XRR": 1.84},
        "tech_settings": {"XRR": {"energy_keV": 8.04, "wavelength_A": 1.5406}},
    },
    "stack": [
        {"role": "ambient", "label": "air", "optical": {"n": 1.0, "k": 0.0}},
        {
            "role": "layer",
            "label": "hfo2_film",
            "material": "HfO2",
            "structural": {
                "thickness": {"value": 103.4, "min": 50.0, "max": 200.0, "vary": True},
                "roughness": {"value": 4.2, "min": 0.0, "max": 20.0, "vary": True},
            },
            "xray": {
                "sld_real": {"value": 64.6, "min": 55.0, "max": 75.0, "vary": True},
                "sld_imag": {"value": 1.2, "min": 0.0, "max": 5.0},
            },
            "molecular": {"formula": "HfO2", "density": {"value": 9.1, "min": 8.0, "max": 10.0, "vary": True}},
        },
        {"role": "substrate", "label": "silicon", "material": "Si", "xray": {"sld_real": 20.07}},
    ],
}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'fits.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


def test_import_stores_layers_datasets_and_free_parameters(db):
    result = import_fit(db, copy.deepcopy(EXPORT))
    db.commit()

    assert result["techniques"] == ["XRR"]
    assert result["n_free_parameters"] == 4
    assert result["chi2_total"] == pytest.approx(1.84)

    record = db.get(FitRecord, result["fit_record_id"])
    assert record.sample_id == "HFO2-PILOT-07"
    assert [layer.role for layer in record.layers] == ["ambient", "layer", "substrate"]

    film = record.layers[1]
    assert film.thickness_ang == pytest.approx(103.4)
    assert film.density_g_cm3 == pytest.approx(9.1)
    #  sld_imag has a value and no vary flag: an input, not a result.
    assert "sld_imag" not in film.free_parameters
    assert set(film.free_parameters) == {"thickness", "roughness", "sld_real", "density"}
    assert film.parameters["xray"]["sld_real"] == pytest.approx(64.6)

    dataset = db.query(FitDataset).one()
    assert dataset.technique.value == "XRR"
    assert dataset.chi2 == pytest.approx(1.84)
    #  Units recorded, not assumed.
    assert dataset.x_units == "1/angstrom"


def test_import_is_idempotent_by_content_hash(db):
    first = import_fit(db, copy.deepcopy(EXPORT))
    db.commit()
    second = import_fit(db, copy.deepcopy(EXPORT))
    db.commit()

    assert second["skipped"] is True
    assert second["fit_record_id"] == first["fit_record_id"]
    assert db.query(FitRecord).count() == 1


def test_reordered_keys_are_the_same_fit(db):
    """The hash is over canonical JSON: re-exporting is not a new measurement."""
    import_fit(db, copy.deepcopy(EXPORT))
    db.commit()

    reordered = {key: copy.deepcopy(EXPORT[key]) for key in reversed(list(EXPORT))}
    assert import_fit(db, reordered)["skipped"] is True
    assert db.query(FitRecord).count() == 1


def test_a_fit_with_no_technique_is_refused(db):
    """A refinement with no technique is not a measurement of anything."""
    payload = copy.deepcopy(EXPORT)
    payload.pop("fit")
    with pytest.raises(FitImportError, match="No techniques recorded"):
        import_fit(db, payload)


def test_technique_list_is_not_inferred_from_populated_blocks(db):
    """The stack has an xray block; that alone must not imply XRR was fitted."""
    payload = copy.deepcopy(EXPORT)
    payload.pop("fit")
    with pytest.raises(FitImportError, match="not inferred"):
        import_fit(db, payload)


def test_unknown_technique_is_refused(db):
    payload = copy.deepcopy(EXPORT)
    payload["fit"]["techniques"] = ["XRD"]
    with pytest.raises(FitImportError, match="Unknown technique"):
        import_fit(db, payload)


def test_explicit_arguments_override_the_export(db):
    result = import_fit(
        db, copy.deepcopy(EXPORT), techniques=["XRR", "SE"], algorithm="DREAM (emcee)"
    )
    db.commit()
    record = db.get(FitRecord, result["fit_record_id"])
    assert record.techniques == ["XRR", "SE"]
    assert record.algorithm == "DREAM (emcee)"
    assert {d.technique.value for d in record.datasets} == {"XRR", "SE"}


def test_warns_when_a_claimed_technique_has_no_matching_block(db):
    payload = copy.deepcopy(EXPORT)
    payload["fit"]["techniques"] = ["XRR", "NR"]
    result = import_fit(db, payload)
    db.commit()
    assert any("NR is recorded as refined" in w and "neutron" in w for w in result["warnings"])


def test_warns_about_a_clamped_parameter(db):
    payload = copy.deepcopy(EXPORT)
    payload["stack"][1]["structural"]["thickness"]["value"] = 200.0  # == max
    result = import_fit(db, payload)
    db.commit()
    assert any("finished on a fit bound" in w for w in result["warnings"])


def test_warns_when_no_chi_squared_was_recorded(db):
    payload = copy.deepcopy(EXPORT)
    payload["fit"].pop("chi2")
    payload["fit"].pop("chi2_by_technique")
    result = import_fit(db, payload)
    db.commit()
    #  Absent, never zero — zero would read as a perfect fit.
    assert db.get(FitRecord, result["fit_record_id"]).chi2_total is None
    assert any("No chi-squared recorded" in w for w in result["warnings"])


def test_flags_placeholder_optical_constants(db):
    """ModalFit's README says the bundled Si/Au/Cr/Ti n,k are not real data."""
    payload = copy.deepcopy(EXPORT)
    payload["stack"][2]["optical"] = {"n": 3.88, "k": 0.018}
    result = import_fit(db, payload)
    db.commit()
    record = db.get(FitRecord, result["fit_record_id"])
    assert record.uses_placeholder_optical_constants is True
    assert any("placeholder n/k" in w for w in result["warnings"])


def test_known_limitation_flags_default_to_what_modalfit_actually_does(db):
    result = import_fit(db, copy.deepcopy(EXPORT))
    db.commit()
    record = db.get(FitRecord, result["fit_record_id"])
    #  refnx models built with dq=0, SPR path ignores roughness.
    assert record.resolution_smearing_applied is False
    assert record.roughness_applied_to_spr is False


def test_nanometre_import_scales_stored_thickness(db):
    result = import_fit(db, copy.deepcopy(EXPORT), length_units="nm")
    db.commit()
    film = (
        db.query(FitLayer)
        .filter(FitLayer.fit_record_id == result["fit_record_id"], FitLayer.role == "layer")
        .one()
    )
    assert film.thickness_ang == pytest.approx(1034.0)


def test_directory_import_reports_bad_files_instead_of_dropping_them(db, tmp_path):
    import json

    (tmp_path / "good.json").write_text(json.dumps(EXPORT))
    (tmp_path / "bad.json").write_text(json.dumps({"sample_id": "no-stack"}))

    results = import_directory(db, tmp_path)
    db.commit()

    assert len(results) == 2
    assert any(r.get("error") for r in results)
    assert any(r.get("fit_record_id") for r in results)
    assert db.query(FitRecord).count() == 1
