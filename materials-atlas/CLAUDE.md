# materials-atlas: CLAUDE.md

A living, searchable reference atlas of the materials-db release, inside CNMS Living FOM. It has two faces built from one
generator:

1. **An Obsidian vault** structured like sea-atlas (typed frontmatter, stable dot-path ids, small headed sections, typed links,
   generated reverse index and indexes), readable by a person.
2. **A read-only retrieval layer for Quark's local model** (Ollama, `llama3.1:8b`): an index plus assistant tools, so the model
   can find a material, read the right section, and quote exact values with their provenance and evidence grade.

This file is the project's rules and full plan. It applies to everything under `materials-atlas/` and to the CNMS code this
project adds. CNMS's own rules (README "What makes it defensible", docs/FOM_PROTOCOL.md, docs/RESEARCH_ASSISTANT.md) also apply,
and where they are stricter they win. Read this file, those two docs and `docs/ARCHITECTURE.md` before changing anything.

---

## 1. Inputs, outputs, and what this is not

- **Input (read-only):** a materials-db release, currently **v0.22.0**:
  - `~/code/materials-db/release/materials-db-v0.22.0/` (the `.sqlite`, `family_tables/`, `MANIFEST.json`,
    `material_registry.csv`, `DATA_DICTIONARY.md`, `README.md`, `DATA_LICENSE.md`);
  - two helpers imported read-only from `~/code/materials-db/scripts/`: `dataset_kind.py` (measured vs model fit, reads the pinned
    refractiveindex.info checkout) and `generate_ml_release_set.py` `primary_pages()` (which dataset is primary).
  - The release location is configurable (`MATERIALS_ATLAS_RELEASE`); the generator never writes there.
- **Outputs (generated, never hand-edited):**
  - `materials-atlas/vault/`: the Obsidian vault;
  - `materials-atlas/index/atlas_index.sqlite`: chunk table + SQLite FTS5 index + typed-link table (section 7);
  - `materials-atlas/index/atlas_index.json`: the same metadata as JSON (ids, titles, paths, links, grades) for tools and tests.
- **Code:** `materials-atlas/atlas/` (generator), `materials-atlas/tests/`, `materials-atlas/build.py`; the CNMS-side reader and
  tools in `backend/cnms_fom/atlas/` and `backend/cnms_fom/rag_backend/tools.py` (section 8).
- **Not this project:** changing materials-db data; writing to `~/sea-atlas`; writing to CNMS analysis tables
  (`property_values`, materials, fits, FOM scores); turning atlas notes into knowledge cards; publishing or pushing anything.

Release facts the generator must reproduce (from v0.22.0; re-read them, the release wins over this list):

| item | count |
|---|---|
| materials | 2,129 (1,672 are manufacturer catalog glasses; the release README's prose says 1,675, but its table, the registry and the MANIFEST say 1,672: three SCHOTT keys are `absent_from_this_release`) |
| optical datasets (distinct material_id + dataset_label) | 2,605 |
| optical rows | 1,753,649 |
| physical_properties rows | 3,023 |
| sources | 417 (300 with a DOI; 64 with no authors) |
| consensus_properties | 3,971 (single_source 3,908, excellent 57, warning 5, suspicious 1) |
| dataset_validation pairs | 417 over 71 materials (excellent 347, warning 57, suspicious 13) |
| material_synonyms | 307 names for 203 materials |
| family tables | 20 (+ 16 `*_gaps.csv`) |

### 1.1 What the build found in v0.22.0 (2026-10-01), and what changed here because of it

The release wins over this file; these are the places it did, and the design decisions taken
while building it.

- **Release inconsistencies** (reported in every build's `_build.json`, the Methods page and the
  notes, never corrected): Gadolinium(III) fluoride's family table says `density_source
  literature` but the database has no density row; Copper gallium sulfide's `Boyd1971-120C |
  e-ray` names 120 degC in its tag but stores no temperature, so the release's consensus counted
  it as ambient and its stored `n_633nm | e-ray` mixes 20 and 120 degC; `Boyd1971-20C | e-ray`
  also stores no temperature.
- **n at 633 nm is per phase/variant and axis**, not per material (116 materials have more than
  one entry, e.g. TiO2's o-ray and e-ray). Frontmatter `n_633nm` is a list of entries, each with
  its own basis, grade and reason; there is no material-level `agreement` or
  `independent_groups` field, because a single one would have to pick an entry.
- **Grade T** (new): no dataset covers 633 nm at ambient temperature, datasets cover it only at
  other stated temperatures (31 entries: ices, cryogenic Ar/Kr/Xe, gases at 0 degC, AlGaAs at 600
  degC). Without it these would be "none" (false) or C (wrong temperature).
- **Contested voters** (new rule, applied before the grade table): a voter whose own tag names a
  temperature outside 15-30 degC while its rows store none is excluded and the entry graded on
  the rest. Only CuGaS2 e-ray is affected (B -> C).
- **Density kind `calculated_from_lattice`** (new): `density_experimental lattice params` (17
  rows) is computed from measured lattice parameters, neither literature nor DFT.
- **The suspicious consensus row is `k_633nm` of As2Se3**, not an n row: no n entry is graded S
  in v0.22.0.
- **TiO2 is one material** (id 42, key `oxides_50:TiO2@rutile`, name "Titanium dioxide (rutile /
  anatase)") whose data are rutile only; the formula-ambiguity examples are C (diamond, graphite)
  and C3H7OH (1- and 2-propanol).
- **`num_sources` counts datasets, not papers** (water: 8 voting datasets from 5 sources). The
  atlas shows both.
- **Independence**: organisations are also recognised from a source's title or URL when its
  authors field has no person (e.g. SCHOTT's B 270 and BOROFLOAT datasheets); compiled handbooks
  (Palik, Weber, Handbook of Mineralogy) are unknown, not organisations.
- **Where things live**: release reading and classification are in
  `backend/cnms_fom/materials_release/` (shared with the CNMS importer, so both read the release
  one way); the generator is `atlas/{facts,render,pages,links,index,vault,yamlfm,check}.py` plus
  `evidence.py`, `authors.py`, `templates.py`, `ids.py`. Views are generated Markdown tables, not
  Obsidian Bases.
- **Quark did not use the backend**: it called Ollama directly from the browser, so no tool could
  reach it. It now sends every question to `POST /rag/chat`.
- **Writes under ~/code/materials-db**: an early `git status` in the release checkouts refreshed
  git's own index stat cache (no tracked file or data changed). Every git call now uses
  `--no-optional-locks`, and the build fails if any file under materials-db or sea-atlas changes.

---

## 2. Non-negotiable rules

1. **Never invent, interpolate, impute, extrapolate, round away or zero-fill a value.** Unknown k is not zero; a missing density
   is "not in this release" plus its recorded reason. (Matches CNMS: "a missing value stays missing".)
2. **Every number carries provenance:** table, `dataset_label`, unit, release version, material id, and a link to its source.
   Frontmatter and the index hold the exact stored value; a note body may show a rounded display value and must say so.
3. **A formula is not a material identity** (CNMS Sec. 2.1). Notes and tools always carry the material name, id and the
   dataset's phase/structure/axis label (e.g. `structure:amorphous | Malitson1965`, `wurtzite | o-ray`). A lookup by formula that
   matches several materials or datasets returns all of them, labelled; it never picks one silently.
4. **Calculated is not measured.** Materials Project DFT densities and dielectric constants, all SLDs, all descriptors and
   model-fit optical datasets are labelled calculated/model in notes, index and tool output (CNMS: a simulation is MODELED).
5. **No free prose.** Sentences come only from fixed templates filled with release facts (section 5). No LLM-written summaries.
   Quoted text (flags, comments, source notes) is verbatim and marked as a quote.
6. **Grades are rubric outputs.** Each is computed by a published, tested rule (section 6) and shown with its reason; if a rule
   cannot decide, the answer is "unknown", never the favourable option.
7. **Reference data, not platform data.** Atlas values are literature/reference values from materials-db. There is no code path
   from the atlas into `property_values`, FOM scoring or the optimizer. Tools that return atlas values say
   `origin: "materials-db v0.22.0 (reference)"`.
8. **Not knowledge cards.** Atlas notes are not written as cards and do not enter the card review flow; cards are the model's
   syntheses and are non-citable until reviewed, atlas notes are generated reference records. A card may *link to* an atlas id.
9. **The release is the source of truth.** Fix data in materials-db, cut a release, regenerate. Never patch generated output.
10. **Deterministic and loud.** Same release in, byte-identical vault and index out; any count mismatch, broken link, filename
    collision, missing source or helper-version mismatch stops the build.
11. **Read-only inputs and boundaries.** Open the release sqlite `mode=ro`; never write under `~/code/materials-db` or
    `~/sea-atlas`; never modify unrelated CNMS files (there may be the user's uncommitted work, e.g. `quark/README.md`).
12. **Licence:** the data is CC BY 4.0; `Home.md`, every note footer and every tool response carry the attribution pointer.
13. **Honest reporting:** never say "all tests pass", "verified" or "reproducible" without running the check and stating its
    scope. No commit, push or release without the user's go-ahead.

---

## 3. Vault structure (sea-atlas style)

sea-atlas's patterns, adapted: stable dot-path ids, typed frontmatter, small `##` sections that each make sense alone, typed
links declared on one side with the reverse index generated, and machine indexes beside the human pages.

```
materials-atlas/
  CLAUDE.md  PROMPT.md  README.md  build.py
  atlas/            generator (release.py, helpers.py, ids.py, names.py, evidence.py, authors.py, templates.py,
                    render_*.py, links.py, index.py, check.py)
  tests/
  vault/                                   GENERATED: open this folder in Obsidian
    Home.md
    Methods/          how to read a note; every rubric; independence; measured vs model fit; what the atlas cannot tell you
    Materials/<family>/<Name>.md           one note per material (catalog glasses under Glass catalogs/<Maker>/)
    Sources/<Surname> <Year> - <short title>.md
    Families/<family>.md
    Properties/<property>.md               n at 633 nm, k status, density, X-ray SLD, neutron SLD, dielectric constant
    Views/                                 Obsidian Bases views (or generated Markdown tables if Bases format is uncertain)
    .obsidian/graph.json                   default graph filter hides Glass catalogs/
  index/                                   GENERATED
    atlas_index.sqlite                     chunks, FTS5, links, values (section 7)
    atlas_index.json
    _build.json                            release version, db sha256, MANIFEST commit, RI commit, generator commit, counts
```

### 3.1 Ids (stable, dot-path, like sea-atlas)
- material: `material.<family>.<slug>`, slug from the materials-db stable key (`oxides_50:SiO2@amorphous` ->
  `material.oxides_50.sio2-amorphous`); the materials-db `material_id` (38) and `key` are also in frontmatter.
- source: `source.<source_id>`; family: `family.<family>`; property: `property.<name>`.
- dataset (section-level, inside a material note): `<material id>#dataset.<slugified dataset_label>`.
- Ids never change for the same materials-db key; a test pins a sample of them across rebuilds.

### 3.2 Typed links (frontmatter, one side declared, reverse generated)
`member_of: family.*`, `cites: [source.*]`, `has_property: [property.*]`, `agrees_with` / `disagrees_with` (from
`dataset_validation`, carrying the class and mean relative error), `same_author_group_as` (rubric 6.4). The generator writes the
reverse lists (`cited_by`, `members`, `materials_with`) so each record states only its own side; Obsidian wikilinks mirror the
same edges for the graph view.

---

## 4. Note designs

### 4.1 Material note
Frontmatter (exact values; absent values are omitted and listed under `missing:` with their reason):

```yaml
id: material.oxides_50.sio2-amorphous
type: material
material_id: 38
key: oxides_50:SiO2@amorphous
name: Silicon dioxide (fused silica)
formula: SiO2
family: oxides_50
aliases: [Silica, Silicon dioxide, fused silica]   # material_synonyms (+ original name if the filename was sanitised)
cas: 60676-86-0
pubchem_cid: 24261
density_g_cm3: <exact>
density_label: density_literature
density_kind: literature            # literature | calculated_dft | bulk_approximation (rubric 6.2)
n_633nm: <exact>
n_633nm_basis: consensus | primary_dataset_point | not_covered
n_633nm_grade: A                    # rubric 6.1
k_status: measured | partially_measured | not_measured
optical_datasets: 3
measured_datasets: 3
model_fit_datasets: 0
wavelength_min_nm: 210
wavelength_max_nm: 6700
agreement: excellent | warning | suspicious | single_source | none
independent_groups: 2 | 1 | unknown
member_of: family.oxides_50
cites: [source.12, ...]
missing: {}
release: 0.22.0
db_sha256: <release sqlite sha256>
tags: [material, family/oxides_50, grade/A]
```
(Values shown are illustrative except id, name, CAS, CID, synonyms and wavelength range, which are v0.22.0's.)

Body: each `##` section is self-contained (it repeats the material name and id in its first line) because the index chunks by
section and a local model may read one section alone.

1. `## Summary`: template 5.1 and 5.2 sentences.
2. `## Evidence at a glance`: value, grade, basis, sources, agreement, independence, each with its reason.
3. `## Identity`: formula, key, ids, CAS, PubChem, InChIKey, synonyms.
4. `## Optical datasets`: one row per dataset: label, [[source]], range (nm), points, temperature, measured or model fit (reason
   quoted), k status, primary marker.
5. `## Physical properties`: quantity, value, unit, label, kind (measured / literature / calculated), [[source]].
6. `## Agreement between sources`: `dataset_validation` and `consensus_properties` rows.
7. `## Flags and gaps`: family-table `flags` and `*_gaps.csv` rows, verbatim quotes.
8. `## What this note cannot tell you`: template 5.4, one line per limit.
9. `## Get the data`: SQL and Python (materials_db.access) snippets for this material's rows.
10. Footer: release, db sha256 (short), ids, licence link, "generated, do not edit".

### 4.2 Source, family, property, Home, Methods
- **Source:** citation fields, DOI/URL, technique, `notes` quoted, author group (6.4), `cited_by` (generated).
- **Family:** counts, the selection rule quoted from the release README, gaps rows, `members` (catalog glasses grouped by maker).
- **Property:** definition, unit and convention quoted from DATA_DICTIONARY.md; how it is graded (link to Methods); grade
  distribution; `materials_with` (generated); the matching view.
- **Home:** what this is, release and checksum, counts, licence and citation, how to rebuild, how Quark uses it.
- **Methods:** `How to read a note`, `Evidence grades`, `Independence`, `Measured vs model fit`, `What the atlas cannot tell you`.

---

## 5. Sentence templates (the only prose)

One sentence each, release facts only, an explicit "unknown" branch, a unit test per branch (`atlas/templates.py`).

- **5.1 Summary:** "{name} ({formula}, id {material_id}) has {n} optical dataset(s) covering {min}-{max} nm{, {m} of them model
  fits} and {density phrase}." Density phrase: "a {kind} density of {value} g/cm3 ({label})" | "no density in this release
  ({reason})".
- **5.2 Agreement:** "{k} measured sources give n at 633 nm and agree within {spread}% ({class})." | "Only one source gives n at
  633 nm, so its agreement cannot be checked." | "No dataset covers 633 nm."
- **5.3 Independence:** "They come from {g} distinct author groups." | "All agreeing sources share an author, so the agreement is
  within one research group." | "Independence cannot be determined: {reason}."
- **5.4 Limits:** e.g. "k is not measured in any dataset; this does not mean k = 0." / "The density is a DFT calculation, not a
  measurement." / "The structure of this sample is not stated by its source."

---

## 6. Rubrics (defensibility)

Pure functions in `atlas/evidence.py` and `atlas/authors.py`, documented in `vault/Methods/`, unit-tested on built cases and on
named real materials. A rubric never upgrades on missing information.

### 6.1 Grade of n at 633 nm
| grade | rule |
|---|---|
| A | consensus from >= 2 measured ambient datasets, class `excellent`, and >= 2 independent author groups |
| B | consensus from >= 2 measured ambient datasets, class `excellent`, independence 1 or unknown; or class `warning` |
| C | a single measured ambient dataset covers 633 nm (consensus `single_source`) |
| D | no measured dataset covers 633 nm at ambient temperature; a model fit does |
| T | no dataset covers 633 nm at ambient temperature; datasets cover it only at other stated temperatures |
| S | consensus class `suspicious` (shows the consensus, the stored rows either side of 633 nm, and the validation rows) |
| none | no dataset covers 633 nm |

Ambient is the release's window: 15-30 degC, or no temperature stated. Applied first: a voter whose
tag names a non-ambient temperature while its rows store none is excluded (see 1.1).

### 6.2 Density kind
`literature` (with its citation), `calculated_dft` (MP_DFT: "calculated, not measured"), `bulk_approximation`
(`bulk_elemental_approximation`: a bulk value standing in for a film), `calculated_from_lattice` (`density_experimental
lattice params`), `absent` (with the gap-list reason, or a pointer to the quoted family-table flags). From the stored label only.

### 6.3 Measured vs model fit
materials-db `dataset_kind.model_fit_reason(raw_record_table)`, reason quoted. The build fails if the helper's refractiveindex.info
checkout commit differs from the MANIFEST `refractiveindex_info_commit` (`c5c2f188...` for v0.22.0).

### 6.4 Author-group independence (from sea-atlas, made conservative)
- Parse `authors` into (surname, first initial): handle "J. Walling, O. Peterson" and "Vasylechko, L.; Kodama, N."; strip
  "et al."; normalise case, accents, hyphens.
- Organisations (Materials Project Consortium, NCBI, the periodictable package, glass makers) are their own group, labelled
  "organisation".
- Same group if any (surname, initial) pair is shared; groups are connected components.
- `unknown` if any compared source has no parseable authors (64 sources have none) or only surnames without initials collide.
- Errs toward fewer groups: it can understate independence, never overstate it. sea-atlas's surname-only matching (which merges
  every "Wang") is the known failure this avoids; the limit is stated in Methods.

### 6.5 k status
`measured` / `partially_measured` / `not_measured`, per dataset and material, counted from stored rows (same definition as the
materials-db ModalFit exporter).

---

## 7. The index (what the local model searches)

`index/atlas_index.sqlite`, built from the same in-memory records as the vault (never by re-parsing Markdown). As built:
`meta`, `notes`, `chunks`, `chunks_fts`, `values` (with `reason`, `citable`, `citable_reason`, `origin`), `datasets`, `links`,
`aliases`, `sources`, `gaps` (candidates the release did not load, with its reasons); `meta.index_schema` is checked by the
CNMS reader. The FTS tokenizer is unicode61 with diacritics removed; labels such as `o-ray` and `structure:amorphous` are matched
as quoted phrases, and exact identifiers go through `aliases`. Material chunk headers name the quantities the material holds in
plain words ("refractive index n"), and a query that names a material puts that material's summary first.

- `chunks(chunk_id, note_id, section, title, path, text, material_id, family, formula, grade, kind)`: one row per `##` section;
  `text` is the section's Markdown with the material name, formula, id and aliases prepended so a chunk stands alone.
- `chunks_fts`: SQLite FTS5 over `title, text, aliases, formula` with a tokenizer that keeps chemical formulas and labels intact
  (`SiO2`, `Ge2Sb2Te5`, `o-ray`, `structure:amorphous`); tested on those tokens.
- `values(material_id, quantity, value, unit, dataset_label, kind, grade, source_id, note_id, section)`: exact numbers for
  direct lookup, so the model never has to read a number out of prose.
- `links(from_id, relation, to_id, note)`: the typed edges of 3.2, both directions.
- `aliases(alias, note_id)`: synonyms, original names, formulas, CAS numbers -> note.

Lexical first, because the vocabulary is rare exact tokens (formulas, labels, author names), which CNMS's own retrieval notes say
embeddings handle badly. Dense retrieval is an optional later phase: embed chunks with CNMS's existing Ollama embedding code and
fuse with the same rank fusion `rag_backend/hybrid.py` uses, reporting `found_by`.

---

## 8. Quark / assistant integration (CNMS side)

A read-only package `backend/cnms_fom/atlas/` that opens `atlas_index.sqlite` `mode=ro` (path from config, default
`materials-atlas/index/atlas_index.sqlite`) and new tools registered in `rag_backend/tools.py`, following that file's existing
patterns (specs in `tool_specs()`, dispatch in `run_tool`, lazy imports, typed results):

| Tool | Returns |
|---|---|
| `atlas_search(query, family?, limit=8)` | ranked chunks: note id, section, title, snippet, material name/id/formula, grade, `found_by` |
| `atlas_read(note_id, section?)` | one note or one section, with frontmatter and typed links both ways |
| `atlas_lookup(material, quantity)` | exact values from `values`: every matching material and dataset, each with label, unit, kind, grade, source citation; ambiguity is returned, not resolved |
| `atlas_compare(materials[], quantity)` | the same for several materials side by side |
| `atlas_coverage(material?)` | what exists and what is missing (and why) for a material or the whole atlas |

Every result carries `origin: "materials-db v<version> (reference)"`, the release checksum, and `citable: true` only for values
that have a resolved source (DOI or full citation); a value without one is returned with `citable: false` and the reason.

If the index is missing or stale (its `_build.json` release does not match config), the tools fail with a clear error; a search
that could not run is an error, not "no data" (CNMS RESEARCH_ASSISTANT Sec. 2 rule).

As built: `atlas_lookup` also returns `summary` lines (one ready-to-quote line per value: name and id, scope, value, grade or
kind with its meaning, source) because llama3.1:8b quotes them far more faithfully than it assembles fields, and it resolves a
name inside a phrase ("liquid water") only when every other word is a state or form qualifier ("indium nitride" is not indium).
Quark reaches the tools through `POST /rag/chat` (`quark/src/lib/cnms.ts`).

The assistant's system prompt gains one short paragraph (where the existing prompt lists tools): use `atlas_lookup` for numbers,
`atlas_search`/`atlas_read` for context, always quote the material id and dataset label, say "reference value" and never write an
atlas value into platform tables. Keep it short: `llama3.1:8b` follows short, concrete instructions best.

Tests (backend/tests): each tool on a fixture index built from 10 real materials; ambiguity (a formula with several materials);
missing index; stale index; no write path (static check that the atlas package imports no session writer).

---

## 9. Checks (`atlas/check.py`, end of every build; any failure stops the build)

1. One note per material (2,129), source (417), family (20); counts equal the MANIFEST.
2. Every frontmatter and `values` number equals the database value exactly (re-read and compare).
3. No 0 unless the database stores 0; no absent value appears as a number.
4. Every dataset listed exactly once (2,605 across notes and in `values`).
5. Every wikilink and every typed link resolves; reverse lists equal the inverted forward lists.
6. Every chunk stands alone (contains its material name and id); FTS finds every material by name, each alias, formula and CAS.
7. Every grade has a reason; distributions are written to `_build.json` and the property notes.
8. Rebuild into a temp folder: byte-identical vault and index.
9. Nothing written outside `materials-atlas/vault`, `materials-atlas/index` (and, in Phase 4, the CNMS files that phase names).

---

## 10. Retrieval evals (does the local model actually find things?)

`materials-atlas/evals/questions.yaml`: about 30 questions with expected note ids and values, run against `atlas_search` /
`atlas_lookup` (deterministic, no model) and, separately, end-to-end through the assistant with `llama3.1:8b`. Cover: by name,
synonym ("fused silica", "galena"), formula with several materials (TiO2 rutile vs anatase), CAS, property questions ("density of
As2S3 and where it comes from"), limits ("is k measured for X?" -> "not measured", never 0), a model-fit-only material (CdTe), a
single-source value, the suspicious consensus row (k of As2Se3), a catalog glass by maker code, and a material not in the atlas
(must say so). In v0.22.0, TiO2 is one material with rutile data only; the several-materials formulas are C and C3H7OH.

As built (2026-10-02): 106 questions — 40 hand-written (including 10 adversarial) in `questions.yaml`, 66 generated from the
release by plain SQL in `questions_generated.yaml` (`make_questions.py`, fixed seed, byte-identical per release). `--gate`
holds deterministic results to 100% and end-to-end results to a per-model baseline in `baselines.json`; CI runs the
deterministic gate on the fixture index. Because llama3.1:8b restates tool output unreliably, an answer that rests only on
atlas lookups is composed: the values are the lookup's `summary` lines quoted verbatim, the model writes a checked framing
(`rag_backend/compose.py`, `rag_backend/answer_checks.py`), and a framing that asserts anything no tool returned is withheld.
End-to-end results therefore report the system (what the reader sees) and the model (framings withheld, claims flagged)
separately.
Report retrieval hit@1/hit@5 and, for end-to-end, whether the answer quotes the right id, label and value. Never tune the data to
pass an eval; tune the index, tools or prompt, and say which.

---

## 11. Plan, phase by phase

Each phase ends with its checks run and a short report (what was built, what was checked and the result, anything unexpected).
Stop at **ask**.

**Phase 0: setup.** Create `materials-atlas/` structure, minimal dependencies (pandas, pyyaml; stdlib sqlite3 with FTS5 - check
it is compiled in), `atlas/release.py` opening v0.22.0 read-only and asserting the section 1 counts. Decide with the user whether
`vault/` and `index/` are committed or git-ignored (default: ignored, rebuilt by one command). **ask** about the iCloud question
(section 12) before generating thousands of files.

**Phase 1: rubrics, authors, templates (tested first).** Every rubric branch, both author formats, organisations, missing authors
-> unknown, named real cases (single-source -> C, CdTe -> D, the suspicious row -> S). Report the grade distribution and
independence summary. **ask:** approve Methods text and distributions.

**Phase 2: five sample notes + sample index.** SiO2 (fused silica), As2S3, GaAs (temperature series), SCHOTT N-BK7HT (catalog glass),
Water; their chunks in a sample index; run applicable checks; open in Obsidian. **ask:** review readability and wording (change
templates, not data).

**Phase 3: full vault and index.** Everything in sections 3-7, all section 9 checks, the deterministic part of section 10.
Report counts, distributions, build time, sizes, eval hit rates.

**Phase 4: CNMS tools.** `backend/cnms_fom/atlas/`, the five tools, the system-prompt paragraph, backend tests, a `cnms-fom atlas
build` CLI entry (wrapping build.py) and a `/health/ready` line for the index. Run the backend test suite. Then the end-to-end evals
with `llama3.1:8b` in Quark. **ask** before committing; never push without asking.

**Phase 5: hand-off.** README (rebuild in one command, what Quark can ask, limits), and the new-release procedure.

**Later, each its own ask:** dense retrieval (Ollama embeddings + rank fusion); n,k-vs-wavelength SVG plots (stored points only,
model fits dashed); a `card -> atlas` link type so reviewed cards can cite atlas ids; feeding the independence check back into
materials-db as a release feature (that is a materials-db change with its own declared-diff process).

---

## 12. Environment notes

- **iCloud.** This repo is on the iCloud-synced Desktop. materials-db had iCloud roll back a file and create conflict copies
  there, and the Desktop became unwritable for Claude Code's sandbox. Generating ~2,600 notes plus a SQLite index here can trigger
  the same. Options to put to the user in Phase 0: move the repo to `~/code/cnms-living-fom` (recommended, as was done for
  materials-db), or keep the repo here but write `vault/` and `index/` to a non-synced path set by config.
- **Sandbox.** Writes under the Desktop may need the sandbox lifted for that command; say so when it happens.
- **Local model.** Quark talks to Ollama at `localhost:11434` (`llama3.1:8b`). Tools must work without Ollama (lexical only) and
  must report when the dense leg is unavailable.

## 13. Working conventions

- Python 3.11+, pandas, pyyaml, stdlib sqlite3. Pure functions for facts and rubrics; rendering separate.
- Follow CNMS module boundaries (docs/ARCHITECTURE.md): heavy imports lazy; `fom_engine` untouched.
- pytest for `materials-atlas/tests` and `backend/tests`; state the command and result when reporting.
- Short docstrings that say why; match the surrounding code's style.
- When a release fact contradicts this file, the release wins: update this file in the same change and say so.
