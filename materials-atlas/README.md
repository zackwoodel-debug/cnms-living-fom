# materials-atlas

A generated, searchable reference atlas of the materials-db release (`~/code/materials-db`)
(v0.22.0: 2,129 materials, 2,605 optical datasets, 417 sources), with two faces built from one
set of records:

- **`vault/`** — an Obsidian vault: one note per material, source, family and property, plus
  Methods pages that publish every rubric. Open the folder in Obsidian.
- **`index/atlas_index.sqlite`** — a read-only SQLite FTS5 index with exact values, typed links
  and aliases, which the CNMS assistant (and so Quark) reads through five tools.

Every number is the stored release value, printed exactly, with where it came from, whether it
is measured, a model fit or a calculation, an evidence grade by a published rule, and what the
data cannot tell you. These are **reference values**: nothing in the atlas flows into CNMS's
`property_values`, FOM scores or optimizer. `CLAUDE.md` is the full specification and rules.

## Build it (one command)

```bash
cnms-fom atlas build              # or: python materials-atlas/build.py
cnms-fom atlas status             # is the index built, current, and what does it hold?
```

The build verifies the release against its `SHA256SUMS` and `MANIFEST.json` counts, checks that
the materials-db helper it imports (`dataset_kind.py`) and the refractiveindex.info checkout are
the ones that built the release, renders into a temporary folder, runs every check (one note per
material/source/family; every number equal to the database; every dataset once; every link
resolving both ways; every chunk standing alone; every name, alias, formula and CAS findable),
rebuilds a second time from scratch and compares the two byte for byte, confirms nothing was
written under `~/code/materials-db` or `~/sea-atlas`, and only then replaces `vault/` and
`index/`. About 36 s. `vault/` and `index/` are git-ignored: rebuild, never edit.

Options: `--sample` (the five Phase 2 notes), `--release PATH`, `--out DIR`, `--materials 38,42,…`
with `--index-only` (how `backend/tests/fixtures/atlas` is made).

## What Quark can ask

Quark sends every question to the CNMS assistant (`POST /rag/chat`), which runs llama3.1:8b
locally and must answer from tools. The atlas tools:

| Tool | Returns |
|---|---|
| `atlas_lookup(material, quantity)` | exact stored values for every matching material and entry, each with label, unit, kind, grade and its meaning, sources, `citable`; ready-to-quote `summary` lines |
| `atlas_compare(materials, quantity)` | the same side by side |
| `atlas_search(query)` | ranked note sections for context |
| `atlas_read(note_id, section?)` | one note or section, with typed links |
| `atlas_coverage(material?)` | what exists, what is missing and why |

Questions it answers well: "What is n of fused silica at 633 nm, and how good is the evidence?",
"Density of As2S3 and where it comes from?", "Is k measured for X?" (never answered as 0),
"What is the density of C?" (both diamond and graphite, by id), "Compare the densities of diamond
and graphite", "What is n of indium nitride?" (not in the release, with the release's reason).

A formula is never treated as an identity: a formula that names several materials returns all of
them, labelled. A material not in the release is reported as absent; the assistant's evidence
guard withholds an answer that rests on nothing.

## Evals

```bash
python materials-atlas/evals/run_evals.py          # deterministic: the tools, no model
python materials-atlas/evals/run_evals.py --e2e    # also through the assistant with llama3.1:8b
```

`evals/questions.yaml` holds 30 questions whose expected values were read from the release
database directly. Results go to `evals/results/` (git-ignored).

## Tests

```bash
pytest -q materials-atlas/tests                    # rubrics, authors, templates; named real materials
pytest -q backend/tests/test_atlas.py              # the five tools on a 12-material fixture index
```

## Limits

- It holds what the release holds. No band gap, band offset, breakdown field, loss tangent or
  thermal conductivity is in materials-db, so none is in the atlas.
- It never interpolates. n at 633 nm comes from the release's consensus row or, where only a model
  fit covers 633 nm, from the family table's stored `n_633`; elsewhere the note lists the stored
  rows either side of 633 nm.
- A grade measures agreement between published datasets and the independence of their authors; it
  says nothing about sample quality, specimen form or stoichiometry, and nothing about any CNMS
  sample.
- Retrieval is lexical (SQLite FTS5). There is no dense leg yet; the tools say so.

## When a new materials-db release is out

1. Point the configuration at it: `MATERIALS_DB_RELEASE_DIR` and `MATERIALS_ATLAS_RELEASE`
   (`.env`), and `--release` for a one-off build.
2. Read its `MANIFEST.json`, `README.md` and `CHANGELOG.md`; update the release-facts table in
   `CLAUDE.md` section 1 from the MANIFEST (the release wins over the table, in the same change).
3. `cnms-fom atlas build`. It stops if the materials-db checkout is not the release's (commit,
   clean refractiveindex.info checkout at the MANIFEST's commit, helper files unchanged since the
   release commit), if any count disagrees with the MANIFEST, if a label falls outside the grammar,
   or if the consensus re-run does not reproduce a stored row.
4. Run both test suites and both eval modes; compare `index/_build.json` with the previous one
   (counts, grade distribution, independence, release notices) and report what changed.
5. Rebuild the backend fixture (`build.py --materials … --out backend/tests/fixtures/atlas
   --index-only`) if the fixture's materials changed, and update the expected values in
   `evals/questions.yaml` from the new release database, never from the atlas.
