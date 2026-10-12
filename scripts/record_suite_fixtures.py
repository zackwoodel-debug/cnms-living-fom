#!/usr/bin/env python3
"""Record the materials-db /suite answers the smoke test replays (backend/tests/fixtures/suite/).

    python scripts/record_suite_fixtures.py --materials-db ~/code/materials-db

Builds the small labelled fixture database that materials-db's own tests use, runs materials-db's real service code over it and saves the responses. Re-run it when
the /suite contract changes; the diff then shows exactly what changed.
"""
import argparse
import base64
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "backend" / "tests" / "fixtures" / "suite"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials-db", required=True)
    root = Path(ap.parse_args(argv).materials_db).expanduser()
    for sub in ("src", "scripts", "tests"):
        sys.path.insert(0, str(root / sub))
    from fastapi.testclient import TestClient
    from materials_db.access import http
    from materials_db.access.suite import SuiteService
    from materials_db.provenance import schema
    from test_modalfit_hardening import build_db

    with tempfile.TemporaryDirectory() as td:
        path = build_db(Path(td), "fixture.sqlite")
        conn = sqlite3.connect(path)
        schema.apply_migration(conn, now="2026-10-11T12:00:00Z")
        schema.backfill(conn, Path(td) / "reg.csv", classify_optical=lambda raw: ("measured", "dataset_kind:measured_page"), first_release="fixture", importer_version="fixture",
                        now="2026-10-11T12:00:00Z")
        conn.commit()
        conn.close()
        svc = SuiteService(sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False), db_path=path, version="fixture")
        http.get_suite = lambda: svc
        client = TestClient(http.app)
        gold = client.get("/suite/materials", params={"q": "Gold"}).json()[0]["material_id"]
        silica = client.get("/suite/materials", params={"q": "Fused silica"}).json()[0]["material_id"]
        recs = {
            "info": client.get("/suite/info").json(),
            "materials_gold": client.get("/suite/materials", params={"q": "Gold", "limit": 25}).json(),
            "evidence_gold": client.get(f"/suite/materials/{gold}/evidence").json(),
            "readiness_gold_xrr": client.post("/suite/readiness", json=dict(selection=dict(material_id=gold), technique="XRR", window=None, conditions=None)).json(),
            "readiness_silica_se_refused": client.post("/suite/readiness", json=dict(selection=dict(material_id=silica), technique="SE", window=None, conditions=None)).json(),
        }
        body = dict(layers=[dict(material="Gold", thickness=50.0, roughness=0.3)], substrate="Silicon", ambient="air", techniques=["XRR"], window=None, mode="strict",
                    assumptions=[], length_unit="angstrom")
        ok = client.post("/suite/bundle", json=body)
        recs["bundle_ok"] = dict(status=ok.status_code, summary=json.loads(ok.headers["x-bundle-summary"]), zip_b64=base64.b64encode(ok.content).decode())
        refused = client.post("/suite/bundle", json=dict(body, layers=[dict(material="Fused silica", thickness=50.0)], techniques=["SE"], window=dict(wavelength_nm=[400, 700])))
        recs["bundle_refused"] = dict(status=refused.status_code, detail=refused.json()["detail"])
        recs["_ids"] = dict(gold=gold, silica=silica)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recorded.json").write_text(json.dumps(recs, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT / "recorded.json")


if __name__ == "__main__":
    main()
