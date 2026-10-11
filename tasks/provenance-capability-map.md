# Capability map: scientific provenance contract (Prompt 1)

Status: **proposed, awaiting approval.** No module spec, plan, branch or code exists yet.
Inputs: `docs/suite-integration-audit.md`, `~/code/materials-db` at `1d0e765`, release `0.22.0` schema read read-only.

## Why a map first

The request bundles capabilities that can be built and verified separately (vocabularies, new evidence tables, source
validation, review, quality findings, readiness, CNMS mapping). One spec over all of them would be too large to review.

## What already exists and will be extended, not replaced (observed)

| Existing | Where | Used as |
|---|---|---|
| `sources` (doi, title, year, technique, url, `uncertainty` scalar, notes) | release schema | citation; has no locator, checksum or access state |
| `optical_dispersion`, `physical_properties`, `mechanical_properties`, `rheology` rows with `dataset_label`, `raw_record_table`, `raw_record_id`, `source_id`, `temperature_c` | release schema | the property records the contract attaches to; **there is no dataset id**, a dataset is `(material_id, dataset_label)` |
| `material_registry.csv`, `materials.material_id`, stable keys like `batch2_31:As2S3@amorphous` | release | stable material identity |
| `process_condition` grammar and `DENSITY_STATE` (verified / bulk approximation) | `pipeline/process_condition.py` | conditions: structure, deposition, thickness, temperature, anneal |
| `label_aliases` | `core/label_aliases.py` | label renames; the new dataset id must not depend on a label |
| `k_status` (measured / partially_measured / not_measured) | `export/modalfit.py` | seed for "do not infer absorption" |
| FAIL/WARN invariants | `core/invariants.py`, `docs/INVARIANTS.md` | home for new structural checks |
| Additive migrations as `migrations/00N_*.sql` using `IF NOT EXISTS` and triggers | `migrations/001_constraints_indexes.sql` | migration style |
| Release build with validation, `MANIFEST.json`, `SHA256SUMS`, `CHANGELOG` gate | `scripts/build_release.py` | how new tables ship |
| CNMS `ProvenanceTier` = measured / calculated / modeled / unavailable; `method`, `source_url`, `source_locator` columns | `cnms_fom/db/enums.py`, `models.py` | target of the mapping |

## Modules

| Module id | Responsibility | Depends on |
|---|---|---|
| `prov-vocab` | Closed vocabularies and the rules between them: provenance (measured, fitted, calculated, modeled, reported, unknown), review (unreviewed, approved, rejected, needs_clarification), finding class (structural, physical, applicability, heuristic), "not reported" states for uncertainty and other optional fields | none |
| `prov-schema` | Additive migration: stable `dataset_id` registry, evidence record per dataset/property (conditions, validity domain, units, uncertainty state), locator and raw-source checksum fields, transformation history; backfill that asserts nothing new | prov-vocab |
| `prov-source-checks` | Two separate checks stored separately: source-metadata validation (DOI/URL resolves, bibliographic match) and numeric/source verification (value compared to the source, by a person or a proposed-then-confirmed extraction); inaccessible source stays `unverified`; extraction proposals carry proposer identity | prov-schema |
| `prov-review` | Review decisions (decision, reviewer, rationale, timestamp), append-only; conflicting determinations coexist; an LLM or script can propose, never approve | prov-schema |
| `prov-findings` | Quality findings table with the four classes, linked to records, never overwriting values; existing FAIL/WARN invariants emit findings where they map | prov-schema |
| `prov-readiness` | A function, not stored data: given technique + measurement window, report ready / not ready with reasons from the record's stored conditions; no extrapolation, no inferred temperature/phase/k=0 | prov-schema, prov-findings |
| `prov-cnms-map` | Mapping from the contract to CNMS: which states may become `ProvenanceTier` values, which never do, and a dry-run preview; literature records never enter the measured store implicitly; "verified" never becomes a fixed fit bound | prov-vocab, prov-schema, prov-review |

Build order: `prov-vocab` → `prov-schema` → (`prov-review`, `prov-source-checks`, `prov-findings` in any order) →
`prov-readiness` → `prov-cnms-map`.

## Assumptions I am making (correct me now)

1. **Extend the release schema with new tables keyed to existing ids; do not alter existing tables.** SQLite cannot add
   CHECKs to existing tables without a rebuild, and existing consumers (ModalFit export, CNMS reader, atlas) read them.
2. **A dataset id is assigned once and stored in a permanent registry file** (like `material_registry.csv`), not derived
   from `dataset_label`, because labels were renamed in v0.21.0 and v0.22.0.
3. **Backfill is conservative.** Every existing dataset gets review = `unreviewed`. Provenance gets `unknown`, except
   where the dataset itself states it (for example `MP_DFPT` rows are `calculated`). Raw-source checksum, locator and
   uncertainty get an explicit `not_recorded` / `not_reported` state, never a guessed value. Nothing is `approved`.
4. **Raw-source checksums can only be recorded where the raw file is on disk** (`raw_record_table` names a file; the
   refractiveindex.info clone is pinned in the manifest). Elsewhere the checksum is `not_recorded`.
5. **Readiness here is data-level** (conditions and coverage). Which inputs each forward model needs is Prompt 2.
6. **The CNMS side is a mapping module plus tests**, not a CNMS schema change, unless the mapping proves it needs one.
7. **Work is on a new branch in `~/code/materials-db`**, with tests using small fixture databases. A full release
   rebuild would be written to a temp directory and would not touch `release/materials-db-v0.22.0`.

## Open questions (need your decision)

1. **Release version.** Should this ship as a new dataset release (0.23.0: new tables, manifest, data dictionary,
   CHANGELOG entry), or stop at migration + tests + a documented build step, leaving the release build to you?
2. **Backfill for `reported`.** Most rows come from refractiveindex.info pages whose own text says measured or
   calculated, but the database does not record which. Is `unknown` for all of these the right conservative default,
   or do you want a reviewed per-family mapping to `reported`?
3. **Who are reviewers?** Reviewer identity needs a stable form (name string, ORCID, GitHub handle). Which?
4. **CNMS: does `fitted` get its own tier?** CNMS has no `fitted` or `reported` tier today. Proposed: `fitted` maps to
   `MEASURED` only through the existing explicit promotion; `reported` maps to nothing (reference store only). OK?
5. **Is a `needs_clarification` review blocking** for readiness and promotion, like `rejected`, or only a warning?

## Stop point

Per the spec-driven workflow, I stop here. After you approve the map (and answer the questions), I write one spec per
module in build order, then the plan and tasks, and only then implement.
