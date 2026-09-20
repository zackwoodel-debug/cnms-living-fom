"""API tests for /modalfit and the new /rag surface, on SQLite with no Ollama.

What is checked here is the contract at the edge: that a refusal comes back as a
status code a client can act on rather than a 500, and that the endpoints that
must work without a model server do.
"""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base, get_db
from cnms_fom.db.models import Material  # noqa: F401 - registers mappers
from cnms_fom.main import app

SAMPLE = "HFO2-PILOT-07"

EXPORT = {
    "stack_id": "20260901_hfo2_v3",
    "sample_id": SAMPLE,
    "fit": {
        "techniques": ["XRR"],
        "algorithm": "L-BFGS-B",
        "chi2": 1.84,
        "tech_settings": {"XRR": {"energy_keV": 8.04}},
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
            "xray": {"sld_real": {"value": 64.6, "min": 55.0, "max": 75.0, "vary": True}},
            "molecular": {"formula": "HfO2", "density": {"value": 9.1, "min": 8.0, "max": 10.0, "vary": True}},
        },
        {"role": "substrate", "label": "silicon", "material": "Si"},
    ],
}


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.db'}", future=True)
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


@pytest.fixture
def export_file(tmp_path):
    path = tmp_path / "hfo2_fitted.json"
    path.write_text(json.dumps(EXPORT))
    return path


def test_import_returns_the_record_and_its_warnings(client, export_file):
    response = client.post("/modalfit/import", json={"path": str(export_file)})
    assert response.status_code == 200
    body = response.json()
    assert body["total_fits"] == 1
    assert body["imported"][0]["techniques"] == ["XRR"]


def test_import_of_a_missing_path_is_404(client):
    response = client.post("/modalfit/import", json={"path": "/nope/missing.json"})
    assert response.status_code == 404
    assert "not visible to the API" in response.json()["detail"]


def test_import_without_a_technique_is_422_not_500(client, tmp_path):
    payload = copy.deepcopy(EXPORT)
    payload.pop("fit")
    path = tmp_path / "no_fit_meta.json"
    path.write_text(json.dumps(payload))

    response = client.post("/modalfit/import", json={"path": str(path)})
    assert response.status_code == 422
    assert "No techniques recorded" in response.json()["detail"]


def test_an_unknown_length_unit_is_rejected_at_the_schema(client, export_file):
    response = client.post(
        "/modalfit/import", json={"path": str(export_file), "length_units": "cubits"}
    )
    assert response.status_code == 422


def test_sample_listing_and_fit_detail(client, export_file):
    client.post("/modalfit/import", json={"path": str(export_file)})

    samples = client.get("/modalfit/samples").json()
    assert samples[0]["sample_id"] == SAMPLE
    assert samples[0]["n_fits"] == 1

    fits = client.get(f"/modalfit/samples/{SAMPLE}/fits").json()
    assert fits["n_fits"] == 1
    fit = fits["fits"][0]
    assert fit["techniques"] == ["XRR"]
    assert fit["resolution_smearing_applied"] is False
    film = next(layer for layer in fit["layers"] if layer["role"] == "layer")
    assert film["thickness_ang"] == pytest.approx(103.4)
    assert "thickness" in film["free_parameters"]
    assert "dq=0" in fit["description"]


def test_unknown_fit_is_404(client):
    assert client.get("/modalfit/fits/999").status_code == 404


def test_compare_rejects_an_uncomparable_parameter(client, export_file):
    client.post("/modalfit/import", json={"path": str(export_file)})
    response = client.post(
        "/modalfit/compare", json={"sample_id": SAMPLE, "parameter": "bandgap"}
    )
    assert response.status_code == 422
    assert "not cross-technique comparable" in response.json()["detail"]


def test_compare_reports_a_disagreement_without_averaging(client, export_file, tmp_path):
    client.post("/modalfit/import", json={"path": str(export_file)})

    se = copy.deepcopy(EXPORT)
    se["stack_id"] = "se_v1"
    se["fit"] = {"techniques": ["SE"], "algorithm": "Nelder-Mead", "chi2": 3.0}
    se["stack"][1]["structural"]["thickness"]["value"] = 152.0
    se["stack"][1]["optical"] = {"n": {"value": 2.05, "vary": True}}
    se_path = tmp_path / "se_fitted.json"
    se_path.write_text(json.dumps(se))
    client.post("/modalfit/import", json={"path": str(se_path)})

    body = client.post(
        "/modalfit/compare", json={"sample_id": SAMPLE, "parameter": "thickness"}
    ).json()
    assert body["verdict"] == "disagreement"
    assert sorted(d["value"] for d in body["determinations"]) == [103.4, 152.0]
    assert "midpoint" in body and "mean" not in body

    rollup = client.get(f"/modalfit/samples/{SAMPLE}/disagreements").json()
    assert "thickness" in rollup["disagreements"]


def test_promotion_plan_lists_refusals(client, export_file):
    client.post("/modalfit/import", json={"path": str(export_file)})
    plan = client.get("/modalfit/fits/1/promotion-plan").json()
    assert {item["registry_key"] for item in plan["eligible"]} == {"sld_xray", "rho"}
    assert plan["requires_material_id"] is True


def test_promotion_without_a_material_is_409_not_500(client, export_file):
    client.post("/modalfit/import", json={"path": str(export_file)})
    response = client.post(
        "/modalfit/fits/1/promote", json={"material_id": 999, "dry_run": False}
    )
    assert response.status_code == 409
    assert "polymorph" in response.json()["detail"]


def test_promotion_writes_measured_values_for_a_real_material(client, export_file):
    client.post("/modalfit/import", json={"path": str(export_file)})
    material = client.post(
        "/materials",
        json={
            "formula": "HfO2",
            "polymorph": "monoclinic",
            "specimen_form": "crystalline_film",
        },
    ).json()

    #  Dry run first: the refusal list is the actionable half.
    dry = client.post(
        "/modalfit/fits/1/promote", json={"material_id": material["id"]}
    ).json()
    assert dry["written"] == 0

    written = client.post(
        "/modalfit/fits/1/promote",
        json={"material_id": material["id"], "dry_run": False, "temperature_k": 300.0},
    ).json()
    assert written["written"] == 2
    assert written["provenance_tier"] == "measured"

    detail = client.get(f"/materials/{material['id']}").json()
    sld = next(p for p in detail["properties"] if p["property_key"] == "sld_xray")
    assert sld["value"] == pytest.approx(64.6)
    assert sld["provenance_tier"] == "measured"
    assert sld["thickness_nm"] == pytest.approx(10.34)
    assert "8.04 keV" in sld["method"]

    #  Density lands as a structural descriptor, not a property.
    assert any(d["descriptor_key"] == "rho" for d in detail["descriptors"])


def test_assistant_config_is_reachable_without_a_model_server(client):
    body = client.get("/rag/assistant").json()
    assert body["provider"] == "ollama"
    assert "compare_fit_techniques" in body["tools"]


def test_search_degrades_to_lexical_and_reports_the_backend(client):
    """No Ollama here, so the dense leg is unavailable and lexical carries it."""
    from cnms_fom.db.enums import SynthesisTechnique
    from cnms_fom.db.models import Document, DocumentChunk

    db = next(app.dependency_overrides[get_db]())
    document = Document(
        title="XRR practice",
        filename="xrr.pdf",
        content_sha256="hash-xrr",
        technique=SynthesisTechnique.CNMS_USER_DOC,
        n_pages=1,
    )
    db.add(document)
    db.flush()
    db.add(
        DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            page=3,
            text="Interfacial width uses a Nevot-Croce roughness factor, not a graded layer.",
        )
    )
    db.commit()

    body = client.post(
        "/rag/search", json={"query": "Nevot-Croce roughness factor", "diagnostics": True}
    ).json()
    assert body["n_hits"] >= 1
    assert body["hits"][0]["found_by"] == ["lexical"]
    assert body["lexical_backend"] == "python_term_overlap"
    assert body["diagnostics"]["dense_error"]


def test_a_search_that_cannot_run_is_503_not_an_empty_result(client):
    """Claiming the corpus lacks something it was never searched for is worse than failing."""
    response = client.post("/rag/search", json={"query": "gallium arsenide"})
    assert response.status_code in (501, 503)


def test_an_unknown_provider_is_rejected_at_the_request(client):
    response = client.post("/rag/chat", json={"question": "anything", "provider": "gpt"})
    assert response.status_code == 422
    assert "Unknown RAG_LLM_PROVIDER" in response.json()["detail"]


def test_sessions_listing_is_empty_and_an_unknown_transcript_is_404(client):
    assert client.get("/rag/sessions").json() == []
    assert client.get("/rag/sessions/nope").status_code == 404
    assert client.delete("/rag/sessions/nope").status_code == 404
