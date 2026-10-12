# Kickoff prompt

Open Claude Code in the CNMS Living FOM repo, `cd` into `materials-atlas/` (so this folder's CLAUDE.md loads), and paste the
block below. For later phases paste the matching "continue" line.

```
Build materials-atlas: a living reference atlas of the materials-db v0.22.0 release inside CNMS Living FOM, with two faces from
one generator: an Obsidian vault structured like sea-atlas (stable dot-path ids, typed frontmatter, self-contained ## sections,
typed links with a generated reverse index) and a read-only index plus assistant tools so Quark's local model (Ollama,
llama3.1:8b) can search it and quote exact values with provenance and evidence grades.

Follow materials-atlas/CLAUDE.md exactly. First read it fully, then the repo's README.md, docs/ARCHITECTURE.md,
docs/RESEARCH_ASSISTANT.md and docs/FOM_PROTOCOL.md, then the release's README.md, DATA_DICTIONARY.md and MANIFEST.json in
~/code/materials-db/release/materials-db-v0.22.0/, and skim ~/code/materials-db/scripts/dataset_kind.py and primary_pages() in
~/code/materials-db/scripts/generate_ml_release_set.py.

Every note and tool result must be readable by a scientist who has never seen the database, and defensible: each number shows
where it came from, whether it is measured, literature or calculated, how good the evidence is (by a published, tested rubric),
and what the data cannot tell you. Nothing is invented, interpolated, rounded away or zero-filled; a formula is never treated as a
material identity; atlas values are reference data and never flow into CNMS's property_values, FOM scores or optimizer.

Do Phase 0 and Phase 1 of CLAUDE.md section 11 now:
- Phase 0: ask me the iCloud question in section 12 and the commit-or-ignore question for vault/ and index/ before creating
  bulk output; set up materials-atlas/ with minimal dependencies; confirm sqlite3 has FTS5; open the release read-only and
  assert the section 1 counts against the MANIFEST and tables; check the refractiveindex.info commit used by dataset_kind
  equals the MANIFEST's, or stop.
- Phase 1: implement and unit-test evidence.py, authors.py and templates.py, covering every branch, both author-list formats,
  organisational authors, and missing authors (which must give "unknown", never "independent").

Then stop and report: the test command and result, the n-at-633-nm grade distribution over all 2,129 materials, the independence
summary for consensus values (multi-group / one group / unknown), anything in the release or the CNMS code that contradicted
CLAUDE.md, and the draft text of vault/Methods/Evidence grades.md. Do not render notes, do not touch backend/ yet, do not modify
any file outside materials-atlas/, do not write under ~/code/materials-db or ~/sea-atlas, and do not commit or push without asking.
```

## Continue lines

- **Phase 2:** `Rubrics approved. Do Phase 2 of materials-atlas/CLAUDE.md: render the five sample notes and their sample index,
  run the applicable checks, open the sample vault in Obsidian, and stop for my review.`
- **Phase 3:** `Samples approved (apply my notes first). Do Phase 3: generate the full vault and index, run every check in
  section 9 and the deterministic evals in section 10, and report counts, distributions, build time, sizes and hit rates.`
- **Phase 4:** `Do Phase 4: add backend/cnms_fom/atlas/ and the five atlas tools following rag_backend/tools.py's patterns, the
  short system-prompt paragraph, the CLI entry and the health line; run the backend tests, then the end-to-end evals with
  llama3.1:8b in Quark, and ask me before committing.`
- **Phase 5:** `Do Phase 5: write the README and the new-release procedure, run all tests, and summarise what Quark can now answer.`
- **New release:** `materials-db v<X.Y.Z> is out. Point the atlas at it, update CLAUDE.md section 1 from its MANIFEST, rebuild,
  run all checks and evals, and report what changed (notes added/removed, values changed, grade changes, eval changes).`
