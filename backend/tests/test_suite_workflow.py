"""The first integrated suite workflow, end to end (docs/specs/suite-workflow/SPEC.md): reference data -> evidence -> readiness -> strict bundle -> (ModalFit) -> fit import ->
diagnostics -> promotion preview -> confirmed promotion by a named operator.

The materials-db side is REPLAYED from responses recorded from its real service over a labelled fixture database (scripts/record_suite_fixtures.py), so this runs with no
network, no materials-db install and no ModalFit. Steps 5-6 (simulate and refine) happen in ModalFit; the fit file it returns is the recorded EXPORT below.
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base, get_db
from cnms_fom.db.models import DescriptorValue, Material, PropertyValue  # noqa: F401 - registers mappers
from cnms_fom.main import app
from cnms_fom.routers import suite as suite_router
from cnms_fom.suite.client import MaterialsDbClient
from test_api_modalfit import EXPORT

REC = json.loads((Path(__file__).parent / "fixtures" / "suite" / "recorded.json").read_text())
GOLD = REC["_ids"]["gold"]


def replay(request: httpx.Request) -> httpx.Response:
    path, method = request.url.path, request.method
    if (method, path) == ("GET", "/suite/info"):
        return httpx.Response(200, json=REC["info"])
    if (method, path) == ("GET", "/suite/materials"):
        return httpx.Response(200, json=REC["materials_gold"])
    if (method, path) == ("GET", f"/suite/materials/{GOLD}/evidence"):
        return httpx.Response(200, json=REC["evidence_gold"])
    if (method, path) == ("POST", "/suite/readiness"):
        body = json.loads(request.content)
        if body["technique"] == "XRR":
            return httpx.Response(200, json=REC["readiness_gold_xrr"])
        return httpx.Response(422, json={"detail": REC["readiness_silica_se_refused"]["detail"]})
    if (method, path) == ("POST", "/suite/bundle"):
        body = json.loads(request.content)
        if body["techniques"] == ["SE"]:
            return httpx.Response(REC["bundle_refused"]["status"], json={"detail": REC["bundle_refused"]["detail"]})
        ok = REC["bundle_ok"]
        return httpx.Response(200, content=base64.b64decode(ok["zip_b64"]), headers={"content-type": "application/zip", "x-bundle-summary": json.dumps(ok["summary"])})
    return httpx.Response(404, json={"detail": f"unrecorded {method} {path}"})


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'suite.db'}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    def override_client():
        c = MaterialsDbClient("http://materials-db.test", transport=httpx.MockTransport(replay))
        yield c
        c.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[suite_router.get_client] = override_client
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def test_the_whole_slice_from_reference_data_to_a_confirmed_promotion(client, tmp_path):
    # 1. browse + evidence: every dataset says what kind of number it is and whether anyone reviewed it
    assert client.get("/suite/info").json()["badge_kinds"] == ["reference_literature", "user_assumption", "calculated", "synthetic_prediction", "experimental_fit", "unknown"]
    found = client.get("/suite/materials", params={"q": "Gold"}).json()
    assert found[0]["name"] == "Gold"
    ev = client.get(f"/suite/materials/{found[0]['material_id']}/evidence").json()
    optical = next(d for d in ev["datasets"] if d["family"] == "optical")
    assert optical["provenance"] == "measured" and optical["review"]["status"] == "unreviewed"
    assert optical["cnms"] == {"label": "literature_measured", "tier": None, "destination": "reference", "reviewed": False, "reasons": [], "kind": "reference_literature"}

    # 2/4. readiness: an answer with reasons, or a refusal that says what to supply; nothing defaulted
    ready = client.post("/suite/readiness", json={"selection": {"material_id": GOLD}, "technique": "XRR"}).json()
    assert ready["status"] == "ready"
    refused = client.post("/suite/readiness", json={"selection": {"material_id": GOLD}, "technique": "SE"})
    assert refused.status_code == 422 and "window" in refused.json()["detail"]

    # 3/5. stack -> strict bundle for ModalFit; user assumptions are carried alongside, never blended into reference data
    body = {"layers": [{"material": "Gold", "thickness": 50.0, "roughness": 0.3}], "techniques": ["XRR"], "assumptions": []}
    bundle = client.post("/suite/bundle", json=body)
    assert bundle.status_code == 200 and bundle.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(bundle.content)).namelist()
    assert any(n.endswith("manifest.json") for n in names)
    summary = json.loads(bundle.headers["x-bundle-summary"])
    assert summary["mode"] == "strict" and summary["user_assumptions"] == [] and summary["n_layers"] == 1
    no = client.post("/suite/bundle", json={"layers": [{"material": "Fused silica", "thickness": 50.0}], "techniques": ["SE"], "window": {"wavelength_nm": [400, 700]}})
    assert no.status_code == 422 and "none was chosen" in no.json()["detail"]

    # 6/8. ModalFit's fit comes back as a file; import it
    fit_file = tmp_path / "fit.json"
    fit_file.write_text(json.dumps(EXPORT))
    imported = client.post("/modalfit/import", json={"path": str(fit_file), "operator": "zackwoodel-debug"})
    assert imported.status_code == 200

    # 7. diagnostics: what the record does and does not support
    diag = client.get("/suite/fits/1/diagnostics").json()
    codes = {f["code"] for f in diag["findings"]}
    assert "single_technique" in codes and "no_uncertainty" in codes and diag["blocks_promotion"] is False

    # 9. promotion preview writes nothing; a real write needs a confirming operator; the tier is MEASURED only after both
    material = client.post("/materials", json={"formula": "HfO2", "polymorph": "monoclinic", "specimen_form": "crystalline_film"}).json()
    preview = client.post("/modalfit/fits/1/promote", json={"material_id": material["id"]}).json()
    assert preview["written"] == 0 and preview["eligible"]
    assert _count_pv(client) == 0
    refusal = client.post("/modalfit/fits/1/promote", json={"material_id": material["id"], "dry_run": False})
    assert refusal.status_code == 409 and "operator" in refusal.json()["detail"]
    done = client.post("/modalfit/fits/1/promote", json={"material_id": material["id"], "dry_run": False, "operator": "zackwoodel-debug", "temperature_k": 300.0}).json()
    assert _count_pv(client) == done["written"] > 0 and done["operator"] == "zackwoodel-debug" and done["provenance_tier"] == "measured"


def _count_pv(client):
    gen = app.dependency_overrides[get_db]()
    db = next(gen)
    try:
        return db.query(PropertyValue).count() + db.query(DescriptorValue).count()
    finally:
        gen.close()


def test_an_unreachable_service_is_503_with_instructions_not_an_empty_answer(client):
    def dead(request):
        raise httpx.ConnectError("refused")

    app.dependency_overrides[suite_router.get_client] = lambda: MaterialsDbClient("http://127.0.0.1:9", transport=httpx.MockTransport(dead))
    r = client.get("/suite/materials", params={"q": "Gold"})
    assert r.status_code == 503 and "uvicorn" in r.json()["detail"] and "SSH tunnel" in r.json()["detail"]


def test_reference_values_never_reach_the_measured_store_through_the_suite_router(client):
    client.get(f"/suite/materials/{GOLD}/evidence")
    client.post("/suite/readiness", json={"selection": {"material_id": GOLD}, "technique": "XRR"})
    client.post("/suite/bundle", json={"layers": [{"material": "Gold", "thickness": 50.0}], "techniques": ["XRR"]})
    from cnms_fom.routers import suite as s
    import inspect
    assert "PropertyValue" not in inspect.getsource(s) and "promote_fit" not in inspect.getsource(s)


def test_a_fit_with_a_blocking_diagnostic_says_so(client, tmp_path):
    bad = json.loads(json.dumps(EXPORT))
    bad["fit"].pop("chi2")
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad))
    client.post("/modalfit/import", json={"path": str(f)})
    diag = client.get("/suite/fits/1/diagnostics").json()
    assert diag["blocks_promotion"] is True and any(x["code"] == "chi2_not_recorded" for x in diag["findings"])
    assert client.get("/suite/fits/999/diagnostics").status_code == 404


def test_wavelength_search_is_proxied_filtered_and_labelled(client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"wavelength_nm": 633.0, "results": [{"material_id": 1, "material": "Gold", "dataset_id": 7, "label": "J", "provenance": "measured",
                                                                               "review": "approved", "n": 0.2, "k": 3.1, "interpolated": True, "between_nm": [600, 650]}],
                                         "truncated": False, "datasets_spanning_wavelength": 1, "skipped_without_k": 0, "note": "n"})

    app.dependency_overrides[suite_router.get_client] = lambda: MaterialsDbClient("http://x", transport=httpx.MockTransport(handler))
    r = client.get("/suite/optical/search", params={"wavelength_nm": 633, "k_min": 1.0, "approved_only": True})
    assert r.status_code == 200 and seen["k_min"] == "1.0" and seen["approved_only"] == "true" and "n_min" not in seen
    hit = r.json()["results"][0]
    assert hit["cnms"] == {"label": "literature_measured (reviewed)", "kind": "reference_literature", "reviewed": True} and hit["interpolated"] is True
