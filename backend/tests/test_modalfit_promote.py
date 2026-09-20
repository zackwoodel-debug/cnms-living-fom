"""The gates on promoting a fitted value into the analysis tables.

Promotion is the only path from a ModalFit refinement into ``property_values``,
and every test here is a refusal it must keep making. The refusals are the
feature: a fixed parameter, a clamped one, a technique that never constrained the
quantity, a fit with no goodness-of-fit, and a material identity nobody supplied
are all ways to produce a number that looks exactly like a measurement and is not.
"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
from cnms_fom.db.models import DescriptorValue, Material, PropertyValue
from cnms_fom.modalfit.promote import PromotionRefused, promote_fit, promotion_plan
from cnms_fom.modalfit.records import import_fit

EXPORT = {
    "stack_id": "20260901_hfo2_v3",
    "sample_id": "HFO2-PILOT-07",
    "fit": {
        "techniques": ["XRR"],
        "algorithm": "L-BFGS-B",
        "chi2": 1.84,
        "tech_settings": {"XRR": {"energy_keV": 8.04, "wavelength_A": 1.5406}},
    },
    "stack": [
        {"role": "ambient", "label": "air"},
        {
            "role": "layer",
            "label": "hfo2_film",
            "material": "HfO2",
            "structural": {
                "thickness": {"value": 103.4, "min": 50.0, "max": 200.0, "vary": True},
                "roughness": {"value": 4.2, "min": 0.0, "max": 20.0, "vary": True},
            },
            "xray": {
                "sld_real": {"value": 40.1, "min": 30.0, "max": 50.0, "vary": True},
                "sld_imag": {"value": 1.2, "min": 0.0, "max": 5.0},
            },
            "molecular": {"formula": "HfO2", "density": {"value": 9.1, "min": 8.0, "max": 10.0, "vary": True}},
        },
        {"role": "substrate", "label": "silicon", "material": "Si", "xray": {"sld_real": 20.07}},
    ],
}


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


def _import(db, payload=None):
    result = import_fit(db, copy.deepcopy(payload or EXPORT))
    db.commit()
    return result["fit_record_id"]


def test_plan_lists_what_is_eligible_and_why_the_rest_is_not(db):
    plan = promotion_plan(db, _import(db))

    eligible = {(item["registry_key"], item["value"]) for item in plan["eligible"]}
    assert ("sld_xray", 40.1) in eligible
    assert ("rho", 9.1) in eligible

    #  sld_imag had no vary flag.
    refused = {item["registry_key"]: item["reasons"] for item in plan["refused"]}
    assert any("held fixed" in reason for reason in refused["sld_xray_imag"])
    assert plan["requires_material_id"] is True


def test_thickness_is_not_promoted_as_a_property(db):
    """Thickness is measurement context (Table 1), not a property of a material."""
    plan = promotion_plan(db, _import(db))
    assert "thickness" not in {item["parameter"] for item in plan["eligible"]}
    assert "roughness" not in {item["parameter"] for item in plan["eligible"]}


def test_substrate_layers_are_never_promoted(db):
    """The wafer's SLD must not be filed against the film's material."""
    plan = promotion_plan(db, _import(db))
    assert all(item["layer_label"] == "hfo2_film" for item in plan["eligible"] + plan["refused"])


def test_promotion_writes_measured_rows_with_context(db, material):
    fit_id = _import(db)
    result = promote_fit(db, fit_id, material_id=material.id, temperature_k=300.0, dry_run=False)
    db.commit()

    assert result["written"] == 2

    sld = db.query(PropertyValue).filter(PropertyValue.property_key == "sld_xray").one()
    assert sld.value == pytest.approx(40.1)
    assert sld.provenance_tier is ProvenanceTier.MEASURED
    assert sld.software == "ModalFit"
    #  Thickness rides along as context, in nm, converted from the stored Å.
    assert sld.thickness_nm == pytest.approx(10.34)
    assert sld.substrate == "Si"
    assert "Nevot-Croce" in sld.interface
    assert sld.source_locator == f"fit_record:{fit_id}"
    #  The registry requires `method`, and says why: X-ray SLD is energy-dependent.
    assert "8.04 keV" in sld.method
    assert "L-BFGS-B" in sld.method

    rho = db.query(DescriptorValue).filter(DescriptorValue.descriptor_key == "rho").one()
    assert rho.value == pytest.approx(9.1)
    assert rho.provenance_tier is ProvenanceTier.MEASURED


def test_dry_run_writes_nothing(db, material):
    result = promote_fit(db, _import(db), material_id=material.id, dry_run=True)
    assert result["written"] == 0
    assert db.query(PropertyValue).count() == 0
    assert db.query(DescriptorValue).count() == 0


def test_missing_material_is_refused_not_invented(db):
    """Sec. 2.1: the importer does not get to guess a polymorph."""
    with pytest.raises(PromotionRefused, match="polymorph"):
        promote_fit(db, _import(db), material_id=999, dry_run=False)


def test_a_clamped_parameter_is_refused(db, material):
    payload = copy.deepcopy(EXPORT)
    payload["stack"][1]["xray"]["sld_real"]["value"] = 50.0  # == max
    fit_id = _import(db, payload)

    result = promote_fit(db, fit_id, material_id=material.id, dry_run=True)
    reasons = {item["registry_key"]: item["reasons"] for item in result["refused"]}
    assert any("clamped" in reason for reason in reasons["sld_xray"])
    assert "sld_xray" not in {item["registry_key"] for item in result["eligible"]}


def test_a_technique_that_did_not_constrain_the_value_is_refused(db, material):
    """An SLD carried through a QCM-only fit was never constrained by the data."""
    payload = copy.deepcopy(EXPORT)
    payload["fit"]["techniques"] = ["QCM"]
    payload["stack"][1]["viscoelastic"] = {"density": {"value": 9.1, "vary": True}}
    fit_id = _import(db, payload)

    result = promote_fit(db, fit_id, material_id=material.id, dry_run=True)
    reasons = {item["registry_key"]: item["reasons"] for item in result["refused"]}
    assert any("nothing in the data constrained" in reason for reason in reasons["sld_xray"])
    #  QCM does reach density acoustically, so rho survives.
    assert "rho" in {item["registry_key"] for item in result["eligible"]}


def test_a_fit_with_no_chi_squared_promotes_nothing(db, material):
    payload = copy.deepcopy(EXPORT)
    payload["fit"].pop("chi2")
    fit_id = _import(db, payload)

    result = promote_fit(db, fit_id, material_id=material.id, dry_run=True)
    assert result["eligible"] == []
    assert all(
        any("no chi-squared" in reason for reason in item["reasons"])
        for item in result["refused"]
    )


def test_ambiguous_film_layer_is_refused_rather_than_chosen(db, material):
    payload = copy.deepcopy(EXPORT)
    payload["stack"].insert(
        2,
        {
            "role": "layer",
            "label": "interlayer",
            "structural": {"thickness": {"value": 12.0, "vary": True}},
            "xray": {"sld_real": {"value": 20.0, "vary": True}},
        },
    )
    fit_id = _import(db, payload)

    with pytest.raises(PromotionRefused, match="refusing to guess|Refusing to guess"):
        promote_fit(db, fit_id, material_id=material.id, dry_run=False)

    #  Naming the layer resolves it.
    result = promote_fit(
        db, fit_id, material_id=material.id, layer_label="interlayer", dry_run=False
    )
    db.commit()
    assert result["written"] >= 1
    assert db.query(PropertyValue).filter(PropertyValue.property_key == "sld_xray").one().value == 20.0


def test_promotion_is_not_the_retrieval_path(db, material):
    """The RAG guard still refuses, and promotion does not route through it."""
    from cnms_fom.rag_backend.chains import assert_not_property_ingestion

    result = promote_fit(db, _import(db), material_id=material.id, dry_run=False)
    db.commit()
    assert result["provenance_tier"] == "measured"

    with pytest.raises(PermissionError, match="Sec. 2.3"):
        assert_not_property_ingestion("property_values")


def test_unknown_fit_id_raises_lookup_error(db, material):
    with pytest.raises(LookupError):
        promote_fit(db, 4242, material_id=material.id, dry_run=True)
