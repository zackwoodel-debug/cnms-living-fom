"""The /pysea HTTP surface: status codes carry meaning.

An invalid container returns 200 with its issues, because the acquisition happened
and the caller needs the list to fix the export. A promotion the protocol forbids
returns 409, because the request was well formed and the answer is no.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db import models  # noqa: F401 - registers the mappers
from cnms_fom.db.base import Base, get_db
from cnms_fom.db.enums import SpecimenForm
from cnms_fom.db.models import Material
from cnms_fom.main import app
from tests.pysea_fixtures import EXPERIMENTAL, INVALID, SIMULATION, fixture_envelope


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'pysea_api.db'}", future=True)
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


def _material(client) -> int:
    db = next(app.dependency_overrides[get_db]())
    record = Material(
        formula="HfO2", formula_reduced="HfO2", polymorph="monoclinic",
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
    )
    db.add(record)
    db.commit()
    return record.id


def _import(client, name=EXPERIMENTAL) -> dict:
    response = client.post("/pysea/import", json={"envelope": fixture_envelope(name)})
    assert response.status_code == 200, response.text
    return response.json()


def test_validate_checks_without_storing(client):
    response = client.post("/pysea/validate", json={"envelope": fixture_envelope(INVALID)})
    assert response.status_code == 200
    body = response.json()
    assert body["validation_status"] == "invalid"
    assert body["n_errors"] > 0
    #  Nothing was written.
    assert client.get("/pysea/records").json() == []


def test_import_returns_the_record_summary(client):
    body = _import(client)
    assert body["validation_status"] == "valid"
    assert body["record_kind"] == "experimental"
    assert body["n_signals"] == 2
    assert body["created"] is True


def test_importing_twice_is_idempotent(client):
    first = _import(client)
    second = _import(client)
    assert second["created"] is False
    assert second["id"] == first["id"]
    assert len(client.get("/pysea/records").json()) == 1


def test_an_invalid_container_is_stored_and_returns_its_issues(client):
    response = client.post("/pysea/import", json={"envelope": fixture_envelope(INVALID)})
    assert response.status_code == 200
    body = response.json()
    assert body["validation_status"] == "invalid"
    assert any(issue["severity"] == "error" for issue in body["issues"])


def test_an_unsupported_contract_version_is_422(client):
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["contract_version"] = "pysea-canonical/1.0"
    response = client.post("/pysea/import", json={"envelope": envelope})
    assert response.status_code == 422


def test_import_without_an_envelope_or_path_is_422(client):
    assert client.post("/pysea/import", json={}).status_code == 422


def test_records_filter_by_sample_and_kind(client):
    _import(client, EXPERIMENTAL)
    _import(client, SIMULATION)

    assert len(client.get("/pysea/records", params={"sample_id": "HFO2-SI-042"}).json()) == 2
    assert len(client.get("/pysea/records", params={"record_kind": "simulation"}).json()) == 1
    assert client.get("/pysea/records", params={"sample_id": "nope"}).json() == []


def test_a_missing_record_is_404(client):
    assert client.get("/pysea/records/9999").status_code == 404
    assert client.get("/pysea/records/9999/promotion-plan").status_code == 404


def test_the_promotion_plan_separates_eligible_from_refused(client):
    record = _import(client)
    body = client.get(f"/pysea/records/{record['id']}/promotion-plan").json()

    assert body["instrument_quantitative"] is True
    assert [item["property_key"] for item in body["eligible"]] == ["eps_inf"]
    assert body["refused"]
    assert body["requires_material_id"] is True


def test_promotion_is_dry_by_default_over_http(client):
    record = _import(client)
    material_id = _material(client)

    body = client.post(
        f"/pysea/records/{record['id']}/promote", json={"material_id": material_id}
    ).json()
    assert body["dry_run"] is True
    assert body["written"] == 0


def test_committing_over_http_writes_one_value(client):
    record = _import(client)
    material_id = _material(client)

    response = client.post(
        f"/pysea/records/{record['id']}/promote",
        json={"material_id": material_id, "dry_run": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["written"] == 1
    assert body["created"][0]["tier"] == "measured"


def test_a_refused_promotion_is_409(client):
    """The request was well formed; the protocol says no."""
    record = _import(client, INVALID)
    material_id = _material(client)

    response = client.post(
        f"/pysea/records/{record['id']}/promote",
        json={"material_id": material_id, "dry_run": False},
    )
    assert response.status_code == 409
    assert "validation" in response.json()["detail"]


def test_promoting_to_an_unknown_material_is_409(client):
    record = _import(client)
    response = client.post(
        f"/pysea/records/{record['id']}/promote",
        json={"material_id": 9999, "dry_run": False},
    )
    assert response.status_code == 409


def test_compare_returns_a_verdict(client):
    _import(client)
    response = client.post(
        "/pysea/compare", json={"sample_id": "HFO2-SI-042", "quantity": "eps_inf"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "single_determination"
    assert body["n_determinations"] == 1
    assert "average" in body["note"]


def test_compare_rejects_a_blank_sample(client):
    response = client.post("/pysea/compare", json={"sample_id": "  ", "quantity": "eps_inf"})
    assert response.status_code == 422


def test_the_experiment_and_its_simulation_compare_without_merging(client):
    _import(client, EXPERIMENTAL)
    _import(client, SIMULATION)

    body = client.post(
        "/pysea/compare", json={"sample_id": "HFO2-SI-042", "quantity": "eps_inf"}
    ).json()
    assert body["n_determinations"] == 2
    tiers = sorted(d["provenance_tier"] for d in body["determinations"])
    assert tiers == ["measured", "modeled"]


def test_the_disagreements_overview_lists_quantities(client):
    _import(client)
    body = client.get("/pysea/samples/HFO2-SI-042/disagreements").json()
    assert body["sample_id"] == "HFO2-SI-042"
    assert "eps_inf" in body["unchecked"]


def test_pysea_appears_in_the_openapi_tags(client):
    schema = client.get("/openapi.json").json()
    tags = {tag["name"] for tag in schema.get("tags", [])}
    assert "pysea" in tags
