# Suite integration audit: cnms-living-fom, materials-db, modalfit

Audit date: 2026-10-11. Audit only. No application code was edited, no database was modified, no dependency was
installed, no dataset was downloaded, and no existing change was discarded.

## 0. How to read this document

Every statement is tagged by how it is known:

- **Observed**: read directly in code or data, or produced by a read-only command. A file/function reference is given.
- **Documented**: a repository's own prose says so; the claim was *not* checked against code, or was checked and is
  flagged as such.
- **Inferred**: follows from observed facts but was not executed.
- **Proposed**: a recommendation, not a fact.

**What was and was not run.** Run: read-only `git` (`--no-optional-locks`), `sqlite3` with `mode=ro`, `grep`, `sed`, `ls`,
`shasum`. **Not run: any test suite, any ModalFit server, refnx, refellips, or the launcher.** So "the tests pass" is not
claimed anywhere below. Where a behavior depends on running ModalFit's numerics (for example refellips wavelength units),
it is listed under Unknowns.

**Not traced to code.** Workflow 1 (source → ingestion) was traced through `scripts/build_release.py::validate`, the
README, `docs/INVARIANTS.md`, `docs/versioning.md` and the release `MANIFEST.json`. The per-family loaders
(`scripts/build_*_csv.py`, `src/materials_db/pipeline/`) were not read.

---

## 1. Repositories as found

| Repo | Path | Branch / HEAD | Working tree |
|---|---|---|---|
| cnms-living-fom | `/Users/zackwoodel/code/cnms-living-fom` | `feat/materials-atlas`, `252d741` | 29 changed paths in `git status --short` (modified atlas/assistant work, plus untracked `.claude/`, `CLAUDE.md`, `graphify-out/`, `docs/suite-integration-audit.md`). Not pushed (per project notes). |
| materials-db | `/Users/zackwoodel/code/materials-db` | `main` = `origin/main`, `1d0e765` (`git describe`: `v0.20.2-16-g1d0e765`) | clean |
| modalfit | `/Users/zackwoodel/modalfit` | **detached HEAD** `4eef754`; `main` and `origin/main` are the same commit | clean |

Observed: a second, **stale** materials-db copy exists at `~/Desktop/nat deb /materials-db` (HEAD `dcc931f`, release work
only up to v0.20.0 in `release/`). A second CNMS copy exists at `~/Desktop/CNMS LIVING FOM`. Neither was audited beyond
`git log`. Nothing here should be read from those copies.

Observed: `~/code/materials-db/.modalfit_path` contains `/Users/zackwoodel/modalfit`, and that clone's `HEAD`
(`4eef754295be…`) equals `MODALFIT_COMMIT_SHA` in `src/materials_db/export/modalfit_contract.py:44`. The pin matches today.

Observed: the ModalFit clone has **no `tests/` directory** and no test files. Its `requirements.txt` pins nothing below
`>=`. Its README lists a `nk_data/` directory ("Bundled placeholder optical constants", README line 241) that is **not in
the clone** (`ls` shows `static substrates templates` and the `.py` files only).

---

## 2. Authoritative database, schema and release per repository

| Repo | Authoritative thing | Evidence |
|---|---|---|
| materials-db | The **release package** `release/materials-db-vX.Y.Z/` built by `scripts/build_release.py`. Current local release: **0.22.0**, 2129 materials, 2605 optical datasets, `MANIFEST.json` `git_commit` `a624aaaa…`, built 2026-10-01. | `docs/versioning.md` (four independent version axes); `MANIFEST.json`; `CHANGELOG.md` |
| materials-db (secondary) | `data/materials_oxide_test.db`, 22 MB, **135 materials / 182 optical datasets**, tracked in git. | `sqlite3 mode=ro` counts |
| cnms-living-fom | Its **own** SQLAlchemy schema (`backend/cnms_fom/db/models.py`) and the materials-db release read by `cnms_fom/materials_release/release.py`. Configured to `~/code/materials-db/release/materials-db-v0.22.0` (`backend/cnms_fom/config.py:124`, `materials-atlas/build.py:46`). | code |
| modalfit | **None.** No database. State is an in-process `Session` keyed by cookie (`session_store.py`); the only persistent artifact is an exported model JSON. | `server.py:98-141`, `:797-817` |

Observed, important: **materials-db has two databases that disagree about scope, and the ModalFit exporter reads the
smaller one.** `src/materials_db/export/modalfit.py:44` sets `DEFAULT_DB = data/materials_oxide_test.db` (135 materials).
CNMS reads the release SQLite (2129 materials). For the 135 shared rows, `material_id` and `name` agree (135 of 135 match in
a read-only join). **Not checked:** whether optical/physical *values* agree for those 135, or whether the sidecar CSV
for a material would differ if exported from the release instead. The two files have different SHA-256 values, as
expected for different contents.

Observed: CNMS's own docs name the release it was built against (v0.22.0). Nothing in ModalFit's export, or in the
materials-db ModalFit exporter's JSON output as read here, was checked for a release-version stamp. **Unknown** whether an
exported slab model records which release it came from. (The exporter emits a `materials_db` block per layer
(`library_export.py::build_library_entry` merges `layer["materials_db"]`), contents not enumerated.)

---

## 3. Workflow traces

### 3.1 Scientific source → materials-db ingestion → published dataset

Observed:
- Release build: `scripts/build_release.py::build_db` → `validate(conn)` (line 200) → `package` (552). `validate` is
  FAIL-closed: integrity check, FK check, descriptor-row count equals material count, `n` positive and finite, wavelength
  positive and finite, negative `k` only for reviewed `(material, dataset_label)` pairs (`NEGATIVE_K_ALLOWED`), duplicate
  optical rows only if the source file itself repeats the wavelength, duplicate physical rows, then
  `invariants.evaluate(conn, "release", …)`.
- `docs/INVARIANTS.md` lists FAIL invariants: every optical/physical row has a source and the source exists; density has
  a known state; n positive; wavelength positive; ids and normalized names unique. `sources_have_a_locator` is WARN only.
- `build_release.py:543 check_changelog`: a versioned build needs a `## [X.Y.Z]` entry in `CHANGELOG.md`.
- The release carries `MANIFEST.json` (counts, per-family counts, `git_commit`, `git_worktree_clean`,
  `refractiveindex_info_commit`, `materials_project_database_version`, software versions), `SHA256SUMS`,
  `material_registry.csv`, `family_tables/`, `csv/`, `data_dictionary.json`.
- CNMS verifies a release before trusting it: `materials_release/release.py` checks the SQLite SHA-256 against
  `SHA256SUMS`, every manifest table/dataset/family count, and that the registry covers exactly the `materials` table.

Documented only: refractiveindex.info YAML wavelengths are converted µm → nm on ingest (README "Data flow"). The loader
that does it was not read.

Observed gap: the module docstring of `export/modalfit.py` points to `docs/for_aiden.md` for "the gaps found along the
way". **`docs/for_aiden.md` does not exist** in the repository (`ls` fails).

### 3.2 Material selection → ModalFit model export → optical asset loading

Observed path:
1. Selection: `launcher/catalog.py` (`list_materials`, `selectable` flag) and `scripts/modalfit_launcher.py`.
2. Export: `export/modalfit.py::export_layer` / `export_stack` build a layer dict with `molecular`, `xray`, `neutron`,
   a `scattering` alias, `structural`, and either a scalar `optical` or, for tabulated data,
   `{"model": "Tabulated n,k", "params": {"file": <name>}}` (`export/modalfit.py:531`) plus a sidecar CSV written with
   header `wavelength_nm,n,k` (`export/modalfit.py:411-416`). `export_stack` writes the CSV path **relative to the JSON**.
3. Library export for ModalFit's builder: `launcher/library_export.py::build_library_entry` rewrites the file path to
   **absolute** (`library_export.py:139-141`) so a library entry resolves from anywhere.
4. Loading in ModalFit:
   - Desktop tool `model_predictor.py` sets `_json_dir` when loading a model from disk (`model_predictor.py:549-552`).
   - Web app: `server.py:98-112 model_upload` → `_apply_model_payload` → `physics.normalize_stack`. **`_json_dir` is not
     set for an uploaded model.** The only place `_json_dir` is assigned in `physics.py` is for built-in substrate-library
     entries (`physics.py:1368`). The upload route accepts exactly one file (`request.files.get("file")`, one JSON).

Observed duplication: `modalfit_export.py` at the repository root (1619 lines) and
`src/materials_db/export/modalfit.py` (718 lines) both write ModalFit models and both emit the tabulated-CSV form
(`modalfit_export.py:185, 1101` vs `export/modalfit.py:531`). A line diff between them is 1203 lines. The root file's
last commit touching it is `8e99b43 Add standalone ModalFit exporters`; the package file's latest are v0.21.0 work
(`dc7d599`) and a `k_status` change. **Not checked:** whether anything imports `modalfit_export`, or which of the two
modules `tests/test_modalfit_export.py` exercises (an attempted import search failed on a shell glob error and its result
was not used).

### 3.3 ModalFit prediction → experimental refinement → fit export

Observed (`~/modalfit/server.py`, 25 routes; listed with `grep @app.route`):
- Prediction: `/api/predict` (`:497`) returns model curves and `chi2` per technique (`:551-703`).
- Refinement: `/api/fit/start` (`:713`) runs `physics.FitEngine` in a thread, then writes best values into `sess.params` and
  `sess.stack` via `P._apply_vary_params` (`:752-757`).
- Export: `/api/model/export` (`:797-817`) writes `{**sess.model_meta, "stack": _stack_for_display(sess.stack)}`.
  `sess.model_meta` is built in `_apply_model_payload` as exactly `stack_id`, `sample_id`, `filename`, `n_layers`.

Consequence, Observed: **a fit exported from ModalFit's own UI contains no technique list, no chi-squared, no weights,
no algorithm, no timestamp, no operator, and no parameter uncertainties.** A grep for
`uncertain|stderr|std_err|covariance|bootstrap` in `physics.py` and `server.py` returns nothing. Chi-squared exists only
transiently in `/api/predict` responses and `FitEngine._best_chi2` (`physics.py:1036, 1184`).

### 3.4 Fit export → CNMS import → eligibility/promotion → FOM analysis

Observed:
- Import: `modalfit/slab_model.py::parse_slab_model` / `read_fit_metadata`; `modalfit/records.py` writes
  `fit_records`, `fit_layers`, `fit_datasets`; routes in `routers/modalfit.py`; CLI import in `cli.py:146`.
- `read_fit_metadata` (`slab_model.py:473`) looks for fit data under `_FIT_CONTAINERS = ("fit", "fit_result",
  "refinement", "fit_meta", "meta")` and for keys `techniques`, `chi2`, `weights`, … **ModalFit's export writes none of
  those containers or keys (3.3).** The CNMS tests (`backend/tests/test_modalfit_slab_model.py:204-220`) exercise a
  hand-built payload containing `"fit": {"techniques": […], "chi2": 1.84}`. That shape is CNMS's, not ModalFit's.
- Consequence, Inferred from the two observed sides: importing a ModalFit-produced export yields an empty technique list,
  which `docs/MODALFIT_INTEGRATION.md` §3 says is **refused outright**, unless the caller supplies `techniques` by hand
  (the doc's "or from the caller"). And `modalfit/promote.py` refuses any fit with no chi-squared (module docstring), so
  **promotion cannot succeed on a real ModalFit export without hand-entered chi² and techniques.** I did not execute
  an import to confirm the refusal text.
- Promotion table: `promote.py` `PROMOTABLE` maps `(xray, sld_real)`, `(xray, sld_imag)`, `(neutron, …)` and
  `(molecular|viscoelastic, density)` to registry keys with units `1e-6 A^-2` and `g/cm^3`, each requiring the
  technique(s) listed. Thickness and roughness are deliberately not promotable.
- Eligibility then applies context matching on specimen form, provenance tier (default MEASURED and CALCULATED),
  temperature, etc. (`fom_engine/eligibility.py`). A `material_id` must be supplied by a person for promotion.

Observed risk in ModalFit's forward model, relevant to what "fitted SLD" means: `physics._make_xrr_slab`
(`physics.py:336-377`) uses `MaterialSLD(formula, density)` **whenever** `molecular.formula` and a density are present and
`use_material_sld` is true (default `True` at every call site, `server.py:571,580,602,611`); the stack's direct
`xray.sld_real` is used only in the `elif`. materials-db's exporter always writes `molecular` with formula and density.
So for a materials-db-exported layer, the forward model's SLD is derived from formula+density. **Inferred, not executed:**
a promoted `xray.sld_real` from such a layer may not be what the fit actually used, and the number that was refined would
be the density. Whether `extract_params` exposes `sld_real` as a free parameter on those layers was not read.

### 3.5 Process recipe → simulated properties → synthetic measurement

Observed (CNMS only; neither materials-db nor ModalFit is involved):
- `pilot/workflow.py::evaluate_recipe` (line 160): `export_stack_artifacts` (394) → `pilot/stack.py` builds a stack
  (SLD from the CNMS DB when present, else computed from composition and density, with `sld_source` recorded) →
  `pilot/simulate.py::simulate_xrr` (own Parratt implementation; `refnx` "used instead when available and
  `prefer_refnx` is set") → `pilot/properties.py::simulate_properties` (derived physics plus **`HEURISTIC`** relations
  with invented coefficients) → stored as `MODELED`, FOM comes back `ILLUSTRATIVE`.
- This is a **third** reflectivity implementation: CNMS `pilot/simulate.py`, materials-db
  `simulation/xrr.py` + `calculators/xrr_engine.py`, and ModalFit's refnx calls in `physics.py`.
- "Synthetic measurement" in the sense of generating noisy data from a simulated curve was **not found** in the pilot path
  as read; `scripts/make_synthetic_corpus.py` exists but concerns a text corpus (not read). materials-db has
  `scripts/generate_ml_spectra.py` and `gen_pinn_training_data.py` (not read). **Unknown** whether any of them is the
  intended "synthetic measurement" step.

---

## 4. Units at every boundary

| Quantity | materials-db | Export / sidecar | ModalFit | CNMS |
|---|---|---|---|---|
| Wavelength | `optical_dispersion.wavelength_nm` (nm); invariant requires positive finite | sidecar header `wavelength_nm,n,k` | `nk_tabulated` guesses µm vs nm from the column median (`median < 50` ⇒ µm ⇒ ×1000, `physics.nk_tabulated`); `_make_refellips_ri` passes `data[:,0]` **unconverted** and names it `wl_um` (`physics._make_refellips_ri`, tabulated branch), while its non-tabulated branch passes `wl_nm/1000` | no wavelength handled in `modalfit/` beyond dataset window with units |
| SLD | `xray_sld`, `neutron_sld` in 1e-6 Å⁻² | `xray.sld_real`, 1e-6 Å⁻² | 1e-6 Å⁻² per `modalfit_schema_notes.md`; GUI labels disagree (README: LayerEditor "x1e-6", SubstrateEditor "Å⁻²", per `modalfit_contract.py` docstring) | `PROMOTABLE` units `"1e-6 A^-2"`; `pilot/stack.py` `sld_real # 1e-6 A^-2`; `simulate.py` `SLD_SCALE = 1e-6` |
| Density | g/cm³ | `molecular.density_g_cm3` | `molecular.density` read as g/cm³; **viscoelastic density read as kg/m³** and ÷1000 (`physics._make_xrr_slab`) | promotion key `rho`, `g/cm^3`; the same `rho` key is also fed from `viscoelastic.density` (kg/m³ per ModalFit) — **no conversion visible in `PROMOTABLE`** |
| Length (thickness, roughness) | not stored | Å | Å (physics backends) but bundled `substrates/library.json` is **nm** | stored as Å (`thickness_ang`); `LENGTH_UNITS` table with default `angstrom`; a file's own `length_units` overrides |

Observed: the CNMS promotion table maps both `("molecular","density")` and `("viscoelastic","density")` to descriptor
`rho` in `g/cm^3`. ModalFit stores viscoelastic density in kg/m³ (`physics._make_xrr_slab`: `float(rho_ve) / 1000.0`). Whether
CNMS divides by 1000 before writing was **not read** (`promote.py` lines 270-320). If it does not, a QCM-sourced `rho` is
1000× too large. This is a concrete item to prove or refute (Plan step 5).

---

## 5. Identity

| Concept | materials-db | ModalFit | CNMS |
|---|---|---|---|
| Material | `materials.material_id` permanent, `material_registry.csv` stable key, e.g. `batch2_31:As2S3@amorphous` | none; a layer is `label` + `molecular.formula` | `Material` row; FOM_PROOF Eq. (3): composition + polymorph + specimen form |
| Specimen | none (`dataset_label` carries phase/source/axis) | `sample_id`, `stack_id` | `SpecimenForm` enum; `ContextFilter.specimen_forms` |
| Dataset | `dataset_label`, a token string; v0.21.0 moved "amorphous" out of the polymorph slot into `structure:amorphous`; `label_aliases` keeps old labels working | none | `fit_datasets` per technique; imports keyed by content hash |
| Layer | n/a (a material, not a layer) | `label`, `role`, `material_type`; no stable layer id | `fit_layers` (role, label, index); promotion picks by `layer_label` or index |

Observed: a ModalFit layer name is **a free label**, and materials-db's exporter writes `"name"` from
`_MODALFIT_SUBSTRATE_NAME_BY_FORMULA.get(formula, name)` (`library_export.py`), so a layer's name may be a substrate
alias rather than the DB name. Neither the exported JSON as read here nor the CNMS importer carries `material_id`
(CNMS: "promotion requires a `material_id` someone supplied", doc §3). So the round trip
materials-db material → ModalFit layer → CNMS material is **not machine-linked**; it is re-established by a person.
**Unknown** whether the `materials_db` block in an exported layer includes `material_id`/stable key. Checking this is
Plan step 2.

The release changed the layer name for As2S3 (`As2S3_amorphous` → `As2S3`) and for SiO2 etc. (v0.21.0/v0.22.0
`CHANGELOG`). Any previously imported CNMS fit using the old layer names would carry the old strings. Not checked.

---

## 6. Provenance, uncertainty, conditions, review state

| | materials-db | ModalFit export | CNMS |
|---|---|---|---|
| Provenance | `sources` table; FAIL invariant on every optical/physical row; `density_state` verified vs bulk approximation | `stack_id`, `sample_id`; the schema notes claim a `provenance` block exists and is tolerated, but the real export does not write one (3.3) | `ProvenanceTier` MEASURED/CALCULATED/MODELED; `DescriptorValue`/`PropertyValue` carry method strings; `content hash` on import |
| Uncertainty | none for optical `n,k`; consensus properties table (`consensus_properties`, 3971 rows) not examined | **none** (3.3) | columns exist (`fit_layers.uncertainties`, `promote.py:283,311`, `compare.py:284`); **nothing upstream supplies them from a ModalFit export** |
| Conditions | `process_condition` grammar (`structure:amorphous`, temperature, etc.) in `dataset_label`; temperature on some rows | none | `ContextFilter.temperature_k`, direction, method |
| Review state | release invariants WARN/FAIL; `dataset_validation` (417 rows, not examined) | none | `approved=False` on FOM weights; brief review (`generate_brief`, reviewed/rejected); `PromotionRefused` |

---

## 7. Where missing values become defaults or assumptions

Observed:
- ModalFit: missing `thickness` → `0.0` (`_layer_thickness_A`); missing viscoelastic density → `1200 kg/m³`
  (`physics.py`, per `modalfit_schema_notes.md`; the default was not re-read); `nk_tabulated` outside its range **clamps to
  the edge value** and only warns (`physics.nk_tabulated`); missing `role` is treated as `"layer"`; unknown optical model
  falls through to `n = params.n or entry.n or 1.0`, `k = … or 0.0` (`physics.py:232-235`).
- ModalFit's built-in substrate library: constant, non-dispersive "placeholder" n/k (the library's own `_comment`).
- materials-db exporter: raises `ExportError` rather than guessing (module docstring and `build_library_entry`
  docstring); density is classified verified vs bulk approximation and the label carries
  `*** BULK_APPROXIMATION density ***`.
- CNMS importer: absent chi² stays absent, technique list is never inferred from populated blocks, polymorph is never
  invented (doc §3).
- CNMS pilot: all properties are `MODELED` heuristics with coefficients in `PilotConstants` marked `TODO(CNMS)`.
- The four project-level placeholders (FOM weights, normalization bounds, instrument envelopes, ModalFit optical constants)
  remain, per `docs/` and `TODO(FOM_PROOF)/TODO(CNMS)` markers. Not re-counted here.

### Placeholder-optics detector does not match ModalFit's schema
Observed: `modalfit/records.py::detect_placeholder_optics` (line 118) skips any layer whose `extra["optical"]` dict has
`nk_file` **or `model`**. ModalFit's real per-layer shape is `optical: {"model": …, "params": {…}}`
(`physics.get_nk`, `:216-235`). Every layer written by the real builder therefore has `optical.model` set and is
skipped by the detector; the key `nk_file` does not exist in ModalFit (`params.file` does). The detector flags only the
flat form (`"n"`/`"k"` at layer top level, as in `substrates/library.json`). The test at
`backend/tests/test_modalfit_records.py:171` was not read in full to confirm which shape it uses. **Inferred:** for a real
export of a Si/Au/Cr/Ti layer with a scalar model, the placeholder warning would not fire. Prove with Plan step 4.

---

## 8. Do tabulated n,k assets work in ModalFit's web application?

**Observed: not by the route materials-db's `export_stack` uses.**

- `server.py` `/api/model/upload` takes one JSON file. Of the 25 routes listed in 3.3, `request.files.get("file")`
  appears only at `server.py:101` (model upload) and `:1006` (QR decode); `/api/data/upload` (`:319`) was not read, and it
  is for experimental data, not n,k tables. No route was found that accepts a companion n,k CSV.
- `_resolve_nk_path(fpath, json_dir)` (`physics.py:110-119`) tries `fpath` as given (relative to the **server process's
  working directory**, or absolute), then `json_dir/fpath`; `json_dir` is empty for uploaded models.
- Therefore a relative sidecar name resolves only if a file of that name exists in the Flask process's CWD; an absolute
  path resolves only if it exists **on the machine running the server**. `library_export.py` writes absolute paths, which
  work when the browser and server are the same machine and the files have not moved; `export_stack` writes relative
  paths, which do not.
- materials-db's own `modalfit_bridge.py` docstring and `modalfit_contract.py` (`optical_nk_path_resolution`) state the
  same conclusion; this audit re-derived it independently from `server.py` and `physics.py`.

**Not executed:** an actual web-app load. So "does not work" is a reading of the code path, not a run.

**Separate Unknown (needs execution):** two code paths in `physics.py` treat column 0 of a tabulated CSV differently.
`nk_tabulated` (used by `get_nk` for XRR/SPR-style evaluation) auto-converts µm → nm when the column median is below 50.
`_make_refellips_ri` (used for SE) hands `data[:,0]` to `refellips.RI` unchanged, under a variable named `wl_um`. The
exporter writes nm. If `refellips.RI` expects µm, the SE path would be wrong by 1000× for every sidecar. Whether it
expects nm or µm was not verified (refellips was not installed or read).

---

## 9. Which tests prove numerical compatibility, and which prove only parsing

| Test | What it proves | Class |
|---|---|---|
| materials-db `tests/test_simulation_xrr_parratt.py` | Its Parratt matches **hard-coded golden values from a one-off refnx run** (`:41` "Golden values below were computed directly against refnx"); refnx is not a test dependency (`:16`) | numerical, against refnx **indirectly** |
| materials-db `tests/test_modalfit_export.py`, `test_launcher_modalfit_bridge.py`, `test_export_all_materials_modalfit.py` | exporter output shape, bridge; **not read** | unclear |
| materials-db `modalfit_contract.py` `CONFIRMED_BEHAVIORS` | notes some behaviors were "empirical" (refnx `MaterialSLD` for TiO2 reproduced DB SLD 34.5178 / 1.742017) | one-time experiment recorded in prose |
| CNMS `backend/tests/test_pilot.py` | Parratt properties: critical q for silicon, total external reflection, high-q decay, roughness damping, thickness encoded; SLD vs literature with tolerance | numerical, against **analytic/literature** values; **no comparison to refnx** (the only `refnx` mention in `backend/tests/` is in `test_modalfit_records.py`) |
| CNMS `test_modalfit_slab_model.py`, `test_modalfit_records.py`, `test_modalfit_compare.py`, `test_modalfit_promote.py`, `test_api_modalfit.py` | parse a **hand-built** payload, including a `fit` container ModalFit never writes | **parsing**, not compatibility with a real ModalFit export |
| ModalFit | none exist | none |

**Not found anywhere:** a test that takes an export produced by ModalFit itself (or by materials-db's exporter run
through ModalFit's `physics.py`) and imports it into CNMS; a test that CNMS's Parratt agrees with refnx on the same
stack; a test that materials-db's sidecar CSV is read by `nk_tabulated` or `_make_refellips_ri`.

---

## 10. Documented claims not demonstrated by an executable test

1. "ModalFit documents an `examples/` directory" / "`nk_data/` bundled placeholder constants": neither exists in the
   clone (README line 241; `modalfit_contract.py` docstring about `examples/`).
2. `docs/MODALFIT_INTEGRATION.md` §1: "the HTTP layer was not part of the source we had". **Stale:** the clone at
   `~/modalfit` contains `server.py` with 25 routes. The integration remains file-based; the stated reason no longer
   holds as written, and the doc says nothing of the export lacking fit metadata (3.3).
3. Same doc §2: "A file declaring its own `length_units` overrides" — ModalFit's real export never declares one.
4. Same doc §4: "`resolution_smearing_applied` — ModalFit builds its refnx models with `dq=0.0`". Not checked in
   `physics.py` here.
5. `docs/modalfit_schema_notes.md`: "the `provenance` dict is a safe, existing place to put citation/mp_id metadata".
   It is tolerated on input, but ModalFit's export does not round-trip it: export is `{**model_meta, "stack": …}` with
   `model_meta` limited to four keys, so metadata placed in `provenance` is **dropped on re-export**. Inferred from
   `_apply_model_payload`; not executed.
6. `docs/modalfit_schema_notes.md` / `export/modalfit.py` cite `docs/for_aiden.md`, which does not exist.
7. `README` of materials-db: "34 checks" in `verify_all.py` — not run.
8. CNMS `docs/DB_PROTOCOL.md` §7 survey numbers were written against an older base DB (133 densities, 266 SLDs,
   124,547 optical rows, 2026-09-17); the release now holds 1,753,649 optical rows and 3,023 physical rows. Not rechecked.

---

## 11. Duplicated implementations

| Capability | Copies |
|---|---|
| ModalFit slab-model export from materials-db | `modalfit_export.py` (root, 1619 L) and `src/materials_db/export/modalfit.py` (718 L); plus `scripts/export_all_materials_modalfit.py`, `export_all_oxides_modalfit.py`, `launcher/library_export.py` |
| Launch ModalFit | `scripts/modalfit_launcher.py`, `scripts/modalfit_builder_launcher.py`, `scripts/launch_modalfit.command`, `launcher/modalfit_bridge.py` |
| ModalFit schema reading | materials-db exporter (writes it); CNMS `modalfit/slab_model.py` (reads it, with its own unit and "two shapes" logic); ModalFit `physics.py`, `slab_model_builder.py` **and** `model_predictor.py` (three separate SLD readers, documented in `modalfit_schema_notes.md`) |
| Release reading | materials-db `access/` (`ReleaseDB`, MCP, HTTP) and CNMS `materials_release/` (own checksum/manifest/registry verification) |
| Parratt reflectivity | CNMS `pilot/simulate.py`; materials-db `simulation/xrr.py`, `calculators/xrr_engine.py`, `simulate_xrr.py`; ModalFit via refnx |
| X-ray SLD from composition | CNMS `pilot/stack.py::xray_sld_from_composition`; materials-db `calculators/sld_calculator.py`; refnx `MaterialSLD` inside ModalFit |
| Density/SLD/optical tables | materials-db release; CNMS `property_values`; CNMS `materials-atlas/` vault |

---

## 12. Unknowns and blockers

1. Whether ModalFit's `refellips.RI` takes wavelengths in nm or µm (decides whether SE with a materials-db sidecar is
   off by 1000×). Needs ModalFit's dependencies installed, which this audit was not allowed to do.
2. Whether `extract_params` frees `xray.sld_real` for layers that also carry `molecular` formula+density (3.4).
3. Whether CNMS promotion converts `viscoelastic.density` (kg/m³) to g/cm³ before writing `rho` (4).
4. Whether the sidecar/library values for the 135 shared materials equal the release's values.
5. Whether an exported layer's `materials_db` block carries `material_id` and release version.
6. What "synthetic measurement" is meant to be, and whether any existing script implements it (3.5).
7. ModalFit's license and any upstream contact: no `LICENSE` file was listed in the clone (`ls` shows none); ownership
   of ModalFit is external (`github.com/agauer/modalfit`). Any plan item that edits ModalFit needs upstream agreement.
8. Blocker for any numerical-compatibility test: refnx/refellips/PyMoosh must be installed somewhere. materials-db's
   tests deliberately avoid it; CNMS does not declare it. A decision on where that dependency lives is needed.
9. CNMS has 29 uncommitted paths on `feat/materials-atlas`; plan steps that touch CNMS should start from a commit or a
   branch off it.

---

## 13. Proposed ownership map (Proposed, not observed)

| Concern | Proposed owner | Reason |
|---|---|---|
| Reference data, units of record, identity (`material_id`, stable key, `dataset_label` grammar), release versioning | **materials-db** | already the source of truth with FAIL-closed release validation |
| Reading a release | **one** shared reader, kept in materials-db's `access/` and consumed by CNMS, or CNMS's verified reader kept and materials-db's dropped as a consumer | two verified readers drift; pick one |
| Exporting a material to ModalFit (slab-model JSON + sidecar) | **materials-db**, a single module | one exporter, versioned against the ModalFit pin |
| Slab-model schema and what an export contains | **ModalFit** (external) | CNMS and materials-db should treat it as a contract, tested against a real export |
| Fit metadata (techniques, chi², weights, uncertainties) | **ModalFit** must emit it, **or** a thin adapter owned by CNMS must collect it from the session; today nobody does | 3.3 |
| Fit storage, comparison, promotion, eligibility, FOM | **cnms-living-fom** | already implemented with explicit refusals |
| Simulation for the pilot | **cnms-living-fom**, validated against refnx, or call the shared engine | three implementations today |

---

## 14. Ordered implementation plan with acceptance tests (Proposed)

Each step names the check that would show it is done. None has been started.

0. **Freeze a baseline.** Commit or branch the 29 pending CNMS paths; record the three HEAD SHAs in this file.
   *Accept:* `git status` clean in all three repos; SHAs recorded.
1. **Capture a real ModalFit export.** In a throwaway venv (outside all three repos), run ModalFit at `4eef754`, load a
   materials-db-exported model, fit it against synthetic data, and save the `/api/model/export` file as a fixture.
   *Accept:* fixture committed with its generation script; its keys listed; confirms 3.3 (no `fit`, no chi²).
2. **Identity round trip.** Make the exported layer carry `material_id`, stable key and release version, and prove they
   survive ModalFit's re-export (5, 10.5).
   *Accept:* test loads fixture → ModalFit upload/export → CNMS import and asserts the three fields are equal at the end,
   or documents which are dropped.
3. **Decide where fit metadata comes from** (ModalFit emits it, or an adapter).
   *Accept:* importing the step-1 fixture produces a non-empty technique list with the source of each value recorded; no
   field is invented.
4. **Fix and test the placeholder detector** against the real `optical: {model, params}` shape.
   *Accept:* a failing test first (a scalar Si/Au layer in the step-1 fixture is **not** flagged today), then flagged.
5. **Density unit.** Add a test with a viscoelastic density of 2330 kg/m³ promoted to `rho`.
   *Accept:* stored `rho` is 2.33 g/cm³ (or the code is shown to already do this).
6. **Which SLD drove the fit.** Using the fixture, decide whether `xray.sld_real` or formula+density determined the
   curve; make promotion refuse or relabel accordingly.
   *Accept:* test shows a layer with `molecular` present does not promote `sld_real` as MEASURED unless it was free.
7. **Tabulated n,k.** Choose one mechanism (absolute path, embedded arrays, or a documented companion upload) and test
   it against ModalFit's web app.
   *Accept:* a script starts the Flask app, uploads the model, calls `/api/predict` for SE, and the returned `n,k` equal
   the DB values at three wavelengths to 1e-9; the nm/µm question (12.1) is answered by the same run.
8. **Single exporter.** Remove or reduce `modalfit_export.py` to a shim over `export/modalfit.py`; confirm the base-DB vs
   release question (2, 12.4).
   *Accept:* for the 135 shared materials, exported layers from both databases are byte-identical, or the differences are
   listed and explained.
9. **Parratt cross-check.** Add a test comparing CNMS `pilot/simulate.py` with refnx on the HfO2/SiO2/Si stack used in
   `test_pilot.py`, gated on refnx being importable.
   *Accept:* `log10(R)` agrees to a stated tolerance over the q range used in `materials-db`'s golden test.
10. **Define "synthetic measurement."** Specify noise model, resolution (`dq`), and who owns it; implement once.
    *Accept:* a recipe → simulated curve → noisy data → ModalFit fit → CNMS import run recovers the input thickness
    within the fit's reported uncertainty, with the seed fixed.
11. **Doc corrections.** Fix 10.2, 10.5, 10.6, 10.8; add a "last verified" footer tied to the ModalFit pin.
    *Accept:* the cited files and keys exist.

Stop point: this document ends the requested audit. Nothing above has been implemented.

---

## Update 2026-10-11 (after the export hardening, Prompt 2): what is now verified

These items were Unknowns or inferences above. Each was run, in a throwaway virtual environment outside both repositories (flask 3.1.3,
refellips 0.0.6, PyMoosh; `--system-site-packages`), against the ModalFit clone at the pinned commit `4eef754`, which stayed unmodified.

| Earlier statement | Now |
|---|---|
| 12.1 "Whether `refellips.RI` takes nm or um is unknown" | **Observed.** refellips documents the tabulated wavelength column as **micrometres**. ModalFit's SE path (`physics._make_refellips_ri`) passes the raw column, so a nanometre file returns the first row's n at every wavelength (n = 1.5 at 550, 633 and 750 nm for a 1.50-1.80 table). The same table in micrometres gives 1.55, 1.633, 1.75 on **both** ModalFit paths. The exporter now writes micrometres. This is an upstream ModalFit defect, not edited. |
| 8 "tabulated n,k in the web app: reading of the code path" | **Observed with the real Flask app.** `/api/model/upload` accepts a model whose asset path is absolute (HTTP 200). ModalFit keeps each user's model in a **cookie-keyed session**: a script's upload is invisible to a browser and to a later request without the same cookie. A person should load the resolved model with the web app's own button; scripts need one session. |
| 3.3 "export drops the top-level `provenance`" | **Observed.** `/api/model/export?fmt=json` returns `stack_id`, `sample_id`, `filename`, `n_layers`, `stack`; layer-level keys (`layer_id`, `materials_db.*` with material and dataset ids) survive. |
| 11 "duplicated implementations: two exporters" | **There are three:** `src/materials_db/export/modalfit.py` (maintained), `modalfit_export.py` (single file) and the package `modalfit_db_export/`. The other two are referenced nowhere, are not packaged, and keep the pre-hardening behaviour; both are now marked DEPRECATED with a runtime warning. |
| (new) repeated wavelengths in a dataset | ModalFit stays finite on a repeated wavelength (a step at that point); the exporter records `repeated_wavelengths` and keeps the rows as published. |
| 12.2 / 4 "which SLD drove the fit" and the viscoelastic density unit | still **open** (not addressed by the export work). |
