# T8.1: how CNMS can carry the provenance contract (read-only notes, 2026-10-11)

Read: `backend/cnms_fom/db/models.py` (`PropertyValue`, `DescriptorValue`), `backend/cnms_fom/ingest/materials_db.py`,
`backend/cnms_fom/db/enums.py`, `backend/cnms_fom/materials_release/kinds.py`.

## What CNMS stores today (observed)

- `ProvenanceTier`: `measured`, `calculated`, `modeled`, `unavailable`. No `fitted`, `reported`, or review state.
- `PropertyValue` columns that can carry contract facts without a schema change:
  `provenance_tier`, `method`, `uncertainty` (NULL = not stated), `doi`, `source_url`, `database_identifier`
  (importer writes `<release file>:<table>`), `source_locator` (String 128, "table/page"; the importer currently puts
  the materials-db `record_id` there), `ingested_at`, `experiment_id` (NULL for anything not produced by a CNMS experiment).
- So an imported literature value is already distinguishable from a CNMS measurement without a new column:
  `experiment_id IS NULL` and `database_identifier IS NOT NULL`. The mapper's `literature_*` label can live in that
  distinction plus `method`; **no CNMS schema change is needed for the mapper.**

## What the importer does today (observed)

`ingest/materials_db.py` `DEFAULT_TIER = ProvenanceTier.MEASURED` (comment: "MEASURED with an explicit source DOI is the honest
reading: the measurement happened, it just was not ours"). Optical series: `MODELED if fit else MEASURED`
(~lines 848 and 968). Physical rows: `MEASURED` if the label's kind says measured, else `CALCULATED` (~lines 759, 768).
Eligibility's default `ContextFilter.provenance_tiers` is `(MEASURED, CALCULATED)`, so imported literature values count as
measured in analysis.

## The conflict (needs a decision, not made here)

The contract says a literature claim must not silently become a CNMS measurement record. The importer deliberately does the
opposite. The mapper therefore returns `tier=None` for literature `measured`/`fitted`/`reported` and
`compare_with_importer` reports where the importer's tier differs. **The importer is not changed by this work.** Changing
it (plan task T11) alters what analyses treat as measured, so it needs an explicit go-ahead.

## Where "verified" must not leak

`modalfit/promote.py` and `modalfit/compare.py` take fit bounds and uncertainties from the fit itself
(`layer.uncertainties`, free/fixed flags). Nothing there reads a review status. The mapper has no bounds field and the
modalfit package does not import it (asserted by a test).
