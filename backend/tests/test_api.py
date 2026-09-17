"""API smoke tests against a temporary SQLite database.

Covers the endpoints that must work without Postgres, pgvector, Ollama, or the
heavy science extras — which is most of the FOM surface.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base, get_db
from cnms_fom.db.models import Material, PropertyValue  # noqa: F401 - registers mappers
from cnms_fom.main import app


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def test_root_and_health(client):
    assert client.get("/").json()["name"] == "CNMS Living FOM"
    assert client.get("/health").json()["status"] == "ok"


def test_descriptor_dictionary_is_complete(client):
    """Sec. 13.1: the dictionary ships with the analysis."""
    body = client.get("/materials/dictionary").json()
    keys = {d["key"] for d in body["structural_descriptors"]}
    assert {"V_fu", "rho", "CN", "d_M_O", "sigma_d", "delta_chi",
            "Z_RMS_star", "omega_TO_min", "S_osc", "A_eps"} == keys

    for descriptor in body["structural_descriptors"]:
        assert descriptor["units"] and descriptor["formula"] and descriptor["interpretation"]
        assert descriptor["missing_policy"].startswith("exclude")


def test_hypotheses_are_preregistered_with_a_fingerprint(client):
    body = client.get("/fom/hypotheses").json()
    assert len(body["fingerprint"]) == 64
    signs = {(h["x_key"], h["y_key"]): h["expected_sign"] for h in body["hypotheses"]}
    assert signs[("Z_RMS_star", "eps_ionic")] == "+"
    assert signs[("omega_TO_min", "eps_ionic")] == "-"
    assert signs[("k", "Eg")] == "test"


def test_material_requires_a_polymorph(client):
    """Sec. 2.1: a chemical formula is not a material identifier."""
    response = client.post(
        "/materials", json={"formula": "TiO2", "specimen_form": "ceramic"}
    )
    assert response.status_code == 422


def test_material_create_and_duplicate_conflict(client):
    payload = {"formula": "HfO2", "polymorph": "monoclinic", "specimen_form": "crystalline_film"}
    created = client.post("/materials", json=payload)
    assert created.status_code == 201
    assert client.post("/materials", json=payload).status_code == 409


def test_property_rejected_without_required_context(client):
    """Sec. 16 item 6: a breakdown field needs its measurement context."""
    material_id = client.post(
        "/materials",
        json={"formula": "HfO2", "polymorph": "monoclinic", "specimen_form": "crystalline_film"},
    ).json()["id"]

    response = client.post(
        f"/materials/{material_id}/properties",
        json={"property_key": "Ebd", "value": 4.0, "provenance_tier": "measured"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "thickness_nm" in detail["error"]


def test_property_accepted_with_full_context(client):
    material_id = client.post(
        "/materials",
        json={"formula": "HfO2", "polymorph": "monoclinic", "specimen_form": "crystalline_film"},
    ).json()["id"]

    response = client.post(
        f"/materials/{material_id}/properties",
        json={
            "property_key": "Ebd",
            "value": 4.0,
            "provenance_tier": "measured",
            "thickness_nm": 10.0,
            "electrode": "TiN",
            "temperature_k": 300.0,
            "area_cm2": 1e-4,
            "failure_criterion": "1 mA/cm^2",
        },
    )
    assert response.status_code == 201
    assert response.json()["value"] == 4.0


def _seed_scoreable_material(client) -> int:
    material_id = client.post(
        "/materials",
        json={"formula": "HfO2", "polymorph": "monoclinic", "specimen_form": "crystalline_film"},
    ).json()["id"]

    properties = [
        {"property_key": "k", "value": 25.0, "temperature_k": 300.0, "frequency_hz": 1e4,
         "tensor_component": "zz"},
        {"property_key": "Eg", "value": 5.7, "method": "HSE06", "xc_functional": "HSE06"},
        {"property_key": "dEc", "value": 1.5, "substrate": "Si(001)",
         "interface": "HfO2/SiO2/Si", "method": "XPS"},
        {"property_key": "Ebd", "value": 4.0, "thickness_nm": 10.0, "electrode": "TiN",
         "temperature_k": 300.0, "area_cm2": 1e-4, "failure_criterion": "1 mA/cm^2"},
    ]
    for payload in properties:
        response = client.post(
            f"/materials/{material_id}/properties",
            json={**payload, "provenance_tier": "measured"},
        )
        assert response.status_code == 201, response.text
    return material_id


def test_score_endpoint_flags_draft_weights(client):
    _seed_scoreable_material(client)
    body = client.post("/fom/score", json={"fom_name": "logic"}).json()

    assert body["n_scored"] == 1
    assert body["approved"] is False
    assert any("not approved" in w for w in body["warnings"])
    assert 0.0 < body["results"][0]["value"] <= 1.0


def test_score_reports_not_scored_for_incomplete_material(client):
    """Sec. 6.2: report 'not scored' rather than a manufactured score."""
    material_id = client.post(
        "/materials",
        json={"formula": "TiO2", "polymorph": "rutile", "specimen_form": "ceramic"},
    ).json()["id"]
    client.post(
        f"/materials/{material_id}/properties",
        json={"property_key": "k", "value": 100.0, "temperature_k": 300.0,
              "frequency_hz": 1e4, "tensor_component": "iso", "provenance_tier": "measured"},
    )

    body = client.post("/fom/score", json={"fom_name": "logic"}).json()
    result = next(r for r in body["results"] if r["material_id"] == material_id)
    assert result["status"] == "not_scored"
    assert set(result["missing_inputs"]) == {"Eg", "dEc", "Ebd"}
    assert result["value"] is None


def test_seed_drafts_then_persist_scores(client):
    _seed_scoreable_material(client)
    assert len(client.post("/fom/definitions/seed-drafts").json()) == 3

    body = client.post("/fom/score", json={"fom_name": "logic", "persist": True}).json()
    assert body["n_scored"] == 1


def test_mediate_returns_the_dominant_channel(client):
    body = client.post(
        "/fom/mediate",
        json={
            "fom_name": "logic",
            "reference_properties": {"k": 25.0, "Eg": 5.7, "dEc": 1.5, "Ebd": 4.0,
                                     "eps_ionic": 20.5},
            "reference_descriptors": {"Z_RMS_star": 4.5, "omega_TO_min": 140.0,
                                      "V_fu": 34.0, "mu_eff": 12.0},
        },
    ).json()

    assert body["mediated_effect"]["Z_RMS_star"] > 0
    assert body["mediated_effect"]["omega_TO_min"] < 0
    assert body["dominant_property"]["Z_RMS_star"] == "k"
    assert any("No theoretical channel" in n for n in body["notes"])


def test_bo_instruments_are_labelled_placeholder(client):
    instruments = client.get("/bo/instruments").json()
    assert instruments
    assert all(i["source"] == "placeholder" for i in instruments)


def test_bo_run_rejects_an_impossible_envelope(client):
    response = client.post(
        "/bo/run",
        json={
            "name": "too hot",
            "instrument_id": "CNMS-ALD-01",  # max 400 C
            "search_space": [
                {"name": "substrate_temp_c", "kind": "continuous", "lower": 600.0,
                 "upper": 900.0, "units": "degC"}
            ],
        },
    )
    assert response.status_code == 422
    assert "does not overlap" in response.json()["detail"]


def test_bo_observation_must_lie_in_the_search_space(client):
    run_id = client.post(
        "/bo/run",
        json={
            "name": "pld campaign",
            "search_space": [
                {"name": "substrate_temp_c", "kind": "continuous", "lower": 400.0, "upper": 850.0}
            ],
        },
    ).json()["id"]

    response = client.post(
        f"/bo/run/{run_id}/observe",
        json={"parameters": {"substrate_temp_c": 2000.0}, "objective_value": -1.0},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["violations"][0]["parameter"] == "substrate_temp_c"
