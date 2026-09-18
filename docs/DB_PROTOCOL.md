# Database protocol

How the relational schema encodes FOM_PROOF, what the BO loop needs from it, and
how external data gets in without breaking either.

Companion to `FOM_PROTOCOL.md` (equations → code) and `ARCHITECTURE.md` (module
boundaries). This one is equations → **tables**.

---

## 1. The three-layer schema

```
Material ──┬── StructureRecord ──── DescriptorValue      S  (structure)
           ├── PropertyValue                             P  (property)
           ├── SpectralSeries ──── SpectralPoint         P  (as a curve)
           └── FomScore ──── FomDefinition               F  (function)

AnalysisRun ──┬── CorrelationResult      Sec. 7-8
              ├── SensitivityEstimate    Sec. 10.1   (B)
              ├── MediationResult        Sec. 10.3   (M = BΓ)
              ├── IntegrityCheck         Sec. 11
              └── AnalysisExclusion      Sec. 2.3    (why a row was dropped)
```

### Identity is not the formula

FOM_PROOF Eq. (3) makes the unit of analysis

```
(composition, polymorph, specimen form, temperature, direction, method)
```

The first three are `Material`; the rest live on each `PropertyValue`. That split
is a storage normalisation, not a departure: a separate `Material` row per
temperature would duplicate composition and polymorph across thousands of rows.
An eligible analysis row is a `Material` joined to a *context-matched* set of
values, and that filter is implemented once in `fom_engine/eligibility.py`.

| Constraint | What it prevents |
|---|---|
| `uq_material_identity` on (formula_reduced, polymorph, specimen_form) | Rutile and anatase collapsing into one "TiO₂" |
| `ck_material_polymorph_not_blank` | A blank polymorph reinstating formula-only identity |

### Context identity: `context_digest`

Twenty-odd nullable columns cannot be a composite key, so `PropertyValue` carries
a 32-character fingerprint of its context columns (`db/context.py`), maintained
by an ORM event and enforced by `uq_property_value_context` on
(material_id, property_key, context_digest).

What is in the digest, and why:

* **In:** tensor component, direction, reduction rule, temperature, frequency,
  field amplitude, thickness, electrode, substrate, interface, area, failure
  criterion, processing route, method, software, XC functional, pseudopotential,
  provenance tier, and **source identity** (DOI, database identifier, locator).
* **Out:** `value` and `uncertainty`.

Source identity is *in* on purpose. Two papers reporting the same quantity under
the same conditions are independent measurements; Sec. 2.1 forbids merging them
without a declared aggregation rule, but it does not forbid storing both.
Keeping both preserves the evidence and lets `build_analysis_table` surface the
ambiguity at analysis time rather than the database silently choosing one.

`value` is *out* so that a second reading of an identical context is rejected as
a duplicate. If two such readings disagree, that is a data-quality problem to
resolve, not two rows to average.

Floats are canonicalised to 12 significant figures (300.0 and 300.00000000000006
are one temperature) and empty strings normalise to NULL (a blank electrode
field and a missing one mean the same thing).

---

## 2. Protocol invariants as constraints

Rules the FOM engine already enforces are now enforced in storage too, because
the API is not the only writer: ingestion scripts, migrations, notebooks and
`psql` all reach the same tables.

| FOM_PROOF | Constraint | Effect |
|---|---|---|
| Sec. 6.2 score completeness | `ck_score_not_scored_has_no_value` | A `not_scored` row cannot carry a number |
| Sec. 6.2 | `ck_score_scored_has_value` | A `scored` row cannot be empty |
| Sec. 2.3 modeled quarantine | `ck_score_modeled_is_illustrative` | A modeled input cannot be laundered into a measurement-based score |
| Sec. 2.1 | `uq_property_value_context` | The same measurement cannot be stored twice |
| Sec. 5.3 versioning | `events.py` frozen guard | A frozen `FomDefinition` cannot be edited or deleted |
| Sec. 6.2 policy ownership | `ck_fom_approved_has_approver` | An approval with nobody attached is not an approval |
| Eq. (26) | `ck_fom_floor_eps_in_unit_interval` | The floor stays a small positive number |
| Eq. (41) | `ck_corr_p_range` | `p_permutation` is in (0, 1] — a zero p-value means a bug |
| Sec. 7 | `ck_corr_r_range`, `ck_corr_rho_range` | Correlations stay in [-1, 1] |
| Sec. 16 | `ck_property_*_positive` | Negative temperatures, zero thicknesses and the like |

The frozen-definition rule needs an event rather than a CHECK: "this row was
writable yesterday and is not today" has no SQL expression. `_FROZEN_EDITABLE`
lists the columns that stay mutable (approval metadata), because recording who
approved a definition does not change what its scores mean.

**Scope.** ORM events fire on flushes. `Session.bulk_*` and raw Core `insert()`
bypass them, so `scripts/check_protocol_compliance.py` recomputes every digest
and reports mismatches — a bypass shows up as a reported failure rather than as
silent corruption.

---

## 3. Enum storage

Columns store the member **value** (`'measured'`), not the Python **name**
(`'MEASURED'`). SQLAlchemy defaults to the name, which meant the database said
`MODELED` while the API, the JSON and the docs all said `modeled` — and
`WHERE provenance_tier = 'measured'` typed by hand returned zero rows.

`models.enum_column()` sets `values_callable` to fix that, and
`native_enum=False` so both Postgres and SQLite get `VARCHAR + CHECK`. The
constraint is equally strong, the schema is identical across backends, and
adding a member later is an ordinary constraint change rather than `ALTER TYPE`
— which matters because `SpecimenForm` and `SynthesisTechnique` will grow.

Revision `0002` rewrites existing data before installing the new CHECK, and does
so dialect-aware: Postgres native ENUM columns are relaxed to `VARCHAR` first
(`USING col::text`), since a native ENUM refuses any label outside its type.

---

## 4. Indices, and the queries that need them

| Index | Query it serves |
|---|---|
| `ix_property_material_key` (material_id, property_key) | `build_analysis_table` — the dominant FOM read |
| `ix_property_key_tier` (property_key, provenance_tier) | Eligibility filtering by tier |
| `ix_descriptor_material_key` | Same, for the S layer |
| `ix_bo_obs_run_feasible` (bo_run_id, is_feasible) | Every `suggest()` call reads exactly this slice |
| `ix_score_definition_status` | "How many materials scored under this FOM?" |
| `ix_spectral_point_series_x` (series_id, x_value) | Wavelength-window queries for the ε∞ derivation |
| `ix_chunk_embedding_hnsw` | RAG similarity search (Postgres + pgvector only) |

The HNSW index is created with raw DDL in revision `0002` and excluded from
autogenerate (`include_object` in `migrations/env.py`), because the operator
class has no ORM representation and autogenerate would propose dropping it on
every run. HNSW rather than IVFFlat: no training pass, so it can be built on an
empty table and stay correct as the corpus grows.

### N+1 patterns removed

* `GET /bo/runs` counted observations and suggestions by eager-loading both
  collections and calling `len()`. It now uses correlated `COUNT` subqueries — a
  long-running campaign holds thousands of observations, and materialising them
  to produce one integer is the difference between a listing that stays fast and
  one that degrades as the science progresses.
* `POST /bo/runs/{id}/suggest` walked `run.observations`; it now queries
  `BoObservation` directly with the unevaluated rows filtered in SQL, so the
  `(bo_run_id, is_feasible)` index does the work.
* `POST /materials/{id}/descriptors/compute` ran one existence query per
  descriptor; it now loads the material's descriptors once and keys them by
  `(descriptor_key, method)` — the unique constraint's natural key.

**Known scaling point, not yet addressed:** the no-pgvector RAG fallback loads
every chunk to rank in Python. Fine into the low tens of thousands; enable
pgvector beyond that.

---

## 5. Persisted analysis intermediates

Three tables exist because computing a result and returning it over HTTP is not
the same as being able to re-read it. Sec. 16 item 11 requires transformations,
bounds and weights to be versioned; a number you cannot recover is not auditable.

| Table | Holds | Why it is not enough to return it |
|---|---|---|
| `sensitivity_estimates` | Each B<sub>jq</sub> with its source, elasticity, standard error, **VIF**, and reference point | B is the empirical half of the mediated effect and the half a reviewer will question. Sec. 9.2 makes VIF part of whether a coefficient is interpretable at all |
| `integrity_checks` | Observed / null / excess correlation matrices, Eq. (62) reconstruction error, Eq. (63) leakage report | A stored *failure* is the evidence that a ranking was not released. Re-running until it passes without recording the failures defeats the check |
| `analysis_exclusions` | Per (material, property): why it was dropped | "Excluded" and "never existed" are different findings, and only one is fixed by going back to the literature |

`sensitivity_estimates.source` is constrained to `regression` or `theory`. An
empirical B is evidence about this material population; a theoretical B is a
physical prior. They support different claims and must never be silently mixed.

---

## 6. Spectra

`SpectralSeries` + `SpectralPoint` store curves: n(λ), k(λ), ε(ω), IR spectra.

This closes a real gap. ω<sub>TO,min</sub> and S<sub>osc</sub> (Table 2) are
*derived from* an IR spectrum, and external databases publish dielectric
information as dispersion far more often than as scalars. With nowhere to put a
curve, every such dataset arrives pre-reduced by someone else under an
aggregation rule nobody recorded — exactly what Sec. 3.2 prohibits.

Split into series + points so context is stored once per curve rather than once
per point: the CNMS oxide import is ~205 series and ~114,000 points. The series
carries the axis (`o-ray`, `alpha`, …) and its `tensor_component`, so a scalar
derived later still knows which direction it came from.

---

## 7. External ingestion

### Two-phase, always

```
external SQLite ──▶ external_records ──▶ materials / property_values / spectra
                    (staged, verbatim)      (promoted, only if eligible)
                            │
                            └────────────▶ quarantined + reason
```

Everything lands in `external_records` with its raw payload. Only rows carrying
a phase, a specimen form, and the context their property requires are promoted.
The rest stay quarantined with a reason, which turns "this database is not usable
yet" into a queryable list of exactly what is missing.

A single-phase importer facing a missing polymorph has two options: invent it or
drop the row silently. Sec. 2.3 forbids the first and auditability forbids the
second.

### What `materials_oxide_test.db` actually contains

Surveyed with `python scripts/ingest_materials_db.py survey <db>`:

| | |
|---|---|
| Materials | 135 |
| Optical dispersion rows | 124,547 (all with n, 73,957 with k) |
| Physical property rows | 665 — **133 density, 266 x-ray SLD, 266 neutron SLD** |
| Dielectric constant rows | **0** (column present, never populated) |
| Bandgap / band offset / breakdown / loss / thermal | **absent entirely** |

**No material imported from this source can be FOM-scored today.** A logic score
needs k, Eg, ΔEc and Ebd; none are present. Imported materials correctly report
`not_scored`. That is the finding, not a failure of the importer.

Three further structural problems, and how each is handled:

1. **No polymorph column.** Identity is name + formula — the formula-only
   identity Sec. 2.1 rejects. One row is literally `"Titanium dioxide (rutile /
   anatase)"`.
2. **No specimen form.** Required for comparability and simply not recorded.
3. **Phase and optical axis buried in `dataset_label`**, e.g.
   `"corundum/sapphire | Malitson1972 | o-ray"`.

`ingest/labels.py` recovers (3) by classifying tokens against a **vocabulary**
rather than by position — the field count varies from one to three and the same
slot means different things in different rows. Unrecognised tokens go to
`unclassified` rather than being guessed at. Of 124,547 optical rows, 72,977
yield a phase and 51,570 do not; the latter quarantine.

(1) and (2) are supplied by a person, explicitly and on the record:
`--specimen-form` is required for `promote` and is written into every material's
`notes` along with the operator's name; `--phase-map` accepts a JSON map for
records whose label carries no phase. A documented human assertion is not the
same as an importer guessing.

### Mapping

| External | CNMS | Notes |
|---|---|---|
| `materials.formula` | `Material.formula` | Polymorph from label or phase map; specimen form asserted |
| `physical_properties.density_g_cm3` | `DescriptorValue['rho']` | |
| `physical_properties.xray_sld` | `PropertyValue['sld_xray']` | New registry entry, `fom_eligible=False` |
| `physical_properties.neutron_sld` | `PropertyValue['sld_neutron']` | Isotope-dependent; record the assumption in `method` |
| `physical_properties.dielectric_constant` | `PropertyValue['k']` | Mapped, but zero rows exist |
| `optical_dispersion.n`, `.k` | `SpectralSeries` + `SpectralPoint` | Axis → `tensor_component` |
| `sources.*` | `doi`, `source_url`, `uncertainty` | Denormalised onto the staged payload so the citation survives |

SLD and the optical quantities are marked `fom_eligible=False`: worth storing
and correlating, but not "better when larger" for any device, so they may not
appear in a composite score until someone defines one that uses them.

### The one real bridge to the FOM engine

`derive-eps-inf` estimates ε<sub>∞</sub> from optical dispersion, which feeds
Eq. (11) and hence `mediation.propagate_ionic_elasticities_to_k`.

The exact relation is ε<sub>real</sub> = n² − k², and identifying that with
ε<sub>∞</sub> requires the material to be transparent and dispersion-free in the
window — below the electronic edge, above the phonon resonances.

`max_k` (default 0.1) enforces that and is **not** optional. Without it the
function returns "ε∞ = 36" for chromium: a metal has large k and a negative
ε<sub>real</sub> in the infrared, dominated by free carriers, and n² there is not
a permittivity at all. A material with no k data is skipped rather than assumed
transparent.

On the oxide database over 1000–2000 nm this yields 10 values and skips 105.
Spot-checked against literature: diamond n = 2.367 (lit. ≈ 2.38), ZnS 2.30
(≈ 2.27), Ta₂O₅ 2.045 (≈ 2.05), with Fe₂O₃ and CdS keeping their xx/zz
anisotropy as separate values. Every metal is excluded.

Results are tiered `CALCULATED`, never `MEASURED`: derived from a measurement
under a stated approximation, and Sec. 2.3 keeps those apart. The window is
recorded in `method` on every value.

---

## 8. Migrations

```bash
cnms-fom init-db                    # fresh database → head
cnms-fom init-db --stamp-baseline   # database created before Alembic existed
cnms-fom migrate up | down | current | history
```

| Revision | Contains |
|---|---|
| `0001` | Baseline — the schema as first created by `create_all` |
| `0002` | Enum storage rewrite, context digest, new tables, constraints, indices, HNSW |

`0001` exists so a pre-Alembic database has something to be stamped at. A fresh
database just runs both.

Three things autogenerate got wrong, all corrected by hand in `0002` and worth
knowing about before drafting a revision `0003`:

1. **CheckConstraints are invisible to autogenerate.** Every one on a
   pre-existing table is hand-written in `CHECK_CONSTRAINTS`.
2. **`add_column(..., nullable=False)` with no backfill** fails on any table
   with rows. `context_digest` is added nullable, backfilled, then tightened.
3. **Enum type changes do not rewrite data.** `_rewrite_enum_values()` does,
   before the new CHECK is installed.

Two SQLite-specific points:

* `render_as_batch=True` in `env.py`: SQLite cannot `ALTER` a column or drop a
  constraint in place, so alembic recreates the table. Batch mode is what lets
  one script serve both backends.
* Batch blocks are grouped **one per table, not one per constraint**. Every
  block copies the whole table, so 28 separate blocks meant 28 copies — and each
  copy is a chance for a constraint the reflector did not carry over to be
  quietly lost.
* SQLite exposes an inline `UNIQUE` only as an anonymous auto-index, so
  `drop_constraint("uq_property_value_context")` raises "No such constraint".
  `_drop_context_unique_constraint()` skips it there; the CHECK-constraint
  rebuild removes it instead, and `test_downgrade_removes_the_hardening_but_keeps_the_rows`
  asserts it is genuinely gone rather than leaving that assumed.

`test_migrated_schema_matches_the_orm_metadata` compares `upgrade head` against
`metadata.create_all`. A divergence means a model change landed without a
migration, which otherwise surfaces much later as a constraint that exists in
tests but not in production.

---

## 9. Checking a live database

```bash
python scripts/check_protocol_compliance.py
```

Beyond the in-memory maths checks, it now verifies what is *stored*:

* every `context_digest` still matches its context columns (catches ORM bypass);
* no duplicate (material, property, context) rows;
* no `FomScore` orphaned from its definition, and a warning for any referencing
  an unfrozen one;
* any value whose `method` mentions DFT/PBE/HSE/GW but has no `xc_functional`
  — Sec. 16 item 8, and the reason `eps_inf` cannot require `xc_functional`
  unconditionally: an optically derived value has no functional;
* `spectral_series.n_points` agreeing with the actual point count;
* the quarantine backlog, grouped by reason;
* the hot-path indices actually existing.

---

## 10. Open decisions

Deliberately not guessed at. Each is a `TODO(FOM_PROOF)` or `TODO(CNMS)` in the
source.

| Decision | Where | Blocked on |
|---|---|---|
| Production FOM weights | `fom_engine/definitions.py` | Application-policy decision with a named approver (Sec. 6.2). Uniform placeholders make the gap visible |
| Normalization bounds | `DRAFT_BOUNDS` | Re-derivation from the *frozen* eligible set (Sec. 5.3). Current values are literature screening ranges |
| Real instrument envelopes | `cnms_integration/instruments.py` | CNMS instrument registry. `assert_real_registry` blocks outward-facing actions meanwhile |
| Proposal / user-agreement binding | `cnms_integration/experiments.py` | `CNMS_PROPOSAL_API_URL`. Data-release and embargo terms govern what may appear in a published ranking |
| Specimen form for imported data | `--specimen-form` | Per-import human assertion. A `specimen_form` column in the external DB would remove the need |
| Aggregation rule for genuine duplicates | `eligibility.build_analysis_table` | Currently refuses and records an exclusion. A declared rule (Sec. 2.1) would let it choose |
| Empirical B through the API | `POST /fom/mediate` | Which confounders and which descriptors survive collinearity — decisions Sec. 9 requires a human to make |
| Partitioning `spectral_points` | — | Not needed at ~10⁵ rows; revisit past ~10⁷ |
