# Integrated suite workflow

One path from a reference material to a promoted fit. Spec: `docs/specs/suite-workflow/SPEC.md`. UI: the **Workflow** tab of `frontend/`.

| Owner | What it holds |
|---|---|
| materials-db | versioned reference datasets, their provenance and review state, readiness rules, strict ModalFit bundles |
| ModalFit | simulation (step 5) and refinement against your data (step 6) |
| CNMS (this repo) | stacks, fit records, diagnostics, promotion, scoring |

## Where things run
Everything runs on your machine except, optionally, the database:

```
 browser ── frontend (5173) ── CNMS API (8001) ──HTTP──► materials-db service (8000) ── release .sqlite
                                                           ▲ local, or on a remote host through an SSH tunnel
 ModalFit (its own web app) ◄── bundle zip you download
```

* Run the CNMS API on 8001 (the materials-db service is on 8000) and start the frontend with `VITE_API_BASE_URL=http://localhost:8001 npm run dev`; its proxy otherwise targets 8000, i.e. materials-db itself, which has no `cnms` labels and breaks the page.
* CNMS reads the reference data **only over HTTP** from the materials-db service (`MATERIALS_DB_URL`, default `http://127.0.0.1:8000`); it never opens or writes the release file.
* To keep the full database on a remote host: run the service there, bind it to localhost, and reach it with `ssh -L 8000:localhost:8000 <host>`. Then nothing else changes.
  The service has **no authentication**: do not expose it on a network interface; use the tunnel.
* Quark and ModalFit can read the same service (`/suite/*`, `/materials/*`, `/sql`).

## Setup
1. materials-db service (in the materials-db repo, with a release built or downloaded, e.g. v0.24.0):
   ```
   MATERIALS_DB_RELEASE=release/materials-db-v0.24.0 PYTHONPATH=src uvicorn materials_db.access.http:app --host 127.0.0.1 --port 8000
   ```
   To serve several databases together (for example the curated release and the JARVIS-DFT calculated shard), name them:
   ```
   MATERIALS_DB_RELEASES="curated=release/materials-db-v0.26.0,jarvis=release/materials-db-jarvis-2025.09.24/materials-db-jarvis-dft-3d-2025.09.24.sqlite" \
     PYTHONPATH=src uvicorn materials_db.access.http:app --host 127.0.0.1 --port 8000
   ```
   Material and dataset ids are only unique inside one database, so every answer names its `release` and every call that takes an id takes the release too (the first one is the default).
   A search without a release looks in all of them; a ModalFit bundle is built from one database (the exporter reads one), so a stack cannot mix layers from two.
   The calculated shards (the JARVIS-DFT shard and the Alexandria shards, built by materials-db's `scripts/build_jarvis_shard.py` and `scripts/build_bulk_shard.py`) can be added to the same list, for example
   `alex-pbe-3d=<path to materials-db-alexandria-pbe-3d-2024.10.1.sqlite>`. They hold density, scattering length densities, lattices and DFT energies and gaps, all `calculated`; no optical constants.
2. CNMS API (this repo): `make install`, then run the backend as usual; set `MATERIALS_DB_URL` in `.env` if the service is not at `http://127.0.0.1:8000`.
3. Frontend: `cd frontend && npm install && npm run dev`, open the **Workflow** tab.
4. ModalFit: run it from its own checkout (pinned commit in the materials-db docs); it opens the bundle zip this page downloads.

## Steps in the page
1. Search a material; **Inspect** shows every dataset with its kind (reference/literature, calculated, synthetic prediction), its basis, source, range, stated conditions and review state.
2. Pick a technique, window and optional temperature; **Check readiness** answers ready / not ready / ambiguous with a reason for each. Nothing is defaulted.
3. Add layers (your thickness and roughness are labelled as your entries) and **Build strict ModalFit bundle**. A refusal shows the exporter's reason and the way forward.
4. In ModalFit: simulate, load data, refine, export the fit file. Back here: **Import**; the diagnostics list what the fit does not support (held-fixed parameters, bounds, missing chi², placeholder optics).
5. **Preview** shows what would be promoted and what is refused. **Confirm** needs the material and your name and writes MEASURED rows tagged with that name.

## Rules the page enforces
* A literature value is never shown as a measurement; "approved by a person" is a separate badge from the kind of number.
* Promotion never happens without a preview and a named operator (also enforced in the API: `POST /modalfit/fits/{id}/promote` returns 409 without `operator` when `dry_run` is false).
* The suite router has no code path into the measured-property store (a test checks that the module does not reference it).

## Tests
`make smoke-suite` runs the whole slice (steps 1-5 minus ModalFit's own work) against recorded materials-db answers: `backend/tests/fixtures/suite/recorded.json`, regenerated with
`python scripts/record_suite_fixtures.py --materials-db ~/code/materials-db` whenever the `/suite` contract changes.
Not covered by CI: a live run against a real service and real ModalFit (do it by hand once per release).
