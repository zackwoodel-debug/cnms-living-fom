"""The index the local model searches: ``atlas_index.sqlite`` and its JSON twin.

Built from the same in-memory notes and records as the vault, never by
re-parsing Markdown:

    meta          release version, checksum, generator, counts
    notes         one row per note, with its frontmatter as JSON
    chunks        one row per '##' section; text carries the subject so it stands alone
    chunks_fts    FTS5 over title, text, aliases, formula
    "values"      exact numbers for direct lookup, with label, unit, kind, grade, source
    datasets      every optical dataset with its kind, range, temperature and k status
    links         typed edges, both directions
    aliases       name, synonym, original name, formula, CAS, key, maker code -> note
    sources       citation fields and whether each is citable
    gaps          candidates the release did not load, with its reasons

Lexical, because the vocabulary is rare exact tokens (formulas, labels, author
names) that embeddings handle badly. The FTS5 tokenizer is unicode61 with
diacritics removed: formulas such as SiO2 and Ge2Sb2Te5 are single tokens, and
labels such as o-ray or structure:amorphous are matched as phrases, so 'o-ray'
does not match 'e-ray'. Exact identifiers go through the ``aliases`` table.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .facts import Records
from .links import Link
from .render import Note

#  Bumped whenever a table or column the reader uses changes. The CNMS reader refuses
#  an index whose schema it does not know, rather than failing on a missing column.
INDEX_SCHEMA = 1

SCHEMA = """
PRAGMA page_size = 4096;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE notes (
    note_id TEXT PRIMARY KEY, type TEXT NOT NULL, title TEXT NOT NULL, path TEXT NOT NULL,
    material_id INTEGER, family TEXT, formula TEXT, grade TEXT, frontmatter TEXT NOT NULL
);
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY, note_id TEXT NOT NULL, section TEXT NOT NULL, title TEXT NOT NULL,
    path TEXT NOT NULL, text TEXT NOT NULL, material_id INTEGER, family TEXT, formula TEXT,
    grade TEXT, kind TEXT, aliases TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    title, text, aliases, formula,
    content='chunks', content_rowid='chunk_id',
    tokenize="unicode61 remove_diacritics 2"
);
CREATE TABLE "values" (
    value_id INTEGER PRIMARY KEY, material_id INTEGER NOT NULL, note_id TEXT NOT NULL, section TEXT NOT NULL,
    quantity TEXT NOT NULL, value REAL, text_value TEXT, unit TEXT, dataset_label TEXT, scope TEXT,
    kind TEXT NOT NULL, basis TEXT, grade TEXT, reason TEXT, classification TEXT,
    voting_datasets INTEGER, voting_sources TEXT, independent_groups TEXT,
    source_id INTEGER, record_id INTEGER, origin TEXT NOT NULL,
    temperature_c REAL, frequency_hz REAL, radiation_wavelength_nm REAL,
    citable INTEGER NOT NULL, citable_reason TEXT NOT NULL
);
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY, material_id INTEGER NOT NULL, note_id TEXT NOT NULL, dataset_label TEXT NOT NULL,
    phase TEXT, tag TEXT NOT NULL, axis TEXT, kind TEXT NOT NULL, kind_reason TEXT NOT NULL,
    source_id INTEGER NOT NULL, temperature_c REAL, n_points INTEGER NOT NULL,
    wavelength_min_nm REAL NOT NULL, wavelength_max_nm REAL NOT NULL, k_status TEXT NOT NULL,
    is_primary INTEGER NOT NULL, covers_633_nm INTEGER NOT NULL, raw_record_table TEXT NOT NULL,
    temperature_note TEXT
);
CREATE TABLE links (from_id TEXT NOT NULL, relation TEXT NOT NULL, to_id TEXT NOT NULL, note TEXT NOT NULL);
CREATE TABLE aliases (alias TEXT NOT NULL, alias_norm TEXT NOT NULL, note_id TEXT NOT NULL, kind TEXT NOT NULL);
CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY, note_id TEXT NOT NULL, handle TEXT NOT NULL, citation TEXT NOT NULL,
    doi TEXT, url TEXT, year INTEGER, title TEXT, authors TEXT, technique TEXT, author_kind TEXT NOT NULL,
    citable INTEGER NOT NULL, citable_reason TEXT NOT NULL
);
CREATE TABLE gaps (family TEXT NOT NULL, name TEXT, key TEXT, gap_kind TEXT, reason TEXT, evidence TEXT, name_norm TEXT);
CREATE INDEX ix_values_material ON "values"(material_id, quantity);
CREATE INDEX ix_aliases_norm ON aliases(alias_norm);
CREATE INDEX ix_links_from ON links(from_id);
CREATE INDEX ix_links_to ON links(to_id);
CREATE INDEX ix_chunks_note ON chunks(note_id);
CREATE INDEX ix_datasets_material ON datasets(material_id);
CREATE INDEX ix_gaps_name ON gaps(name_norm);
"""


def norm(text: str) -> str:
    """Case-folded, whitespace-collapsed: how aliases and lookups compare."""
    return " ".join(str(text).casefold().split())


def chunk_text(note: Note, section) -> str:
    return f"{note.chunk_header}\n\n## {section.heading}\n{section.body}"


def _citation(records: Records, source_id: int | None) -> tuple[bool, str]:
    if source_id is None:
        return False, "no single source"
    s = records.sources[source_id]
    return s.citable, s.citable_reason


def value_rows(records: Records) -> list[tuple]:
    rows: list[tuple] = []
    origin = f"materials-db v{records.version} (reference)"
    for m in sorted(records.materials.values(), key=lambda m: m.material_id):
        for e in [*m.n633, *m.k633]:
            if e.quantity == "k_633nm" and not e.classification:
                continue
            single = e.voter_sources[0] if len(e.voter_sources) == 1 else None
            if e.voter_sources:
                cites = [records.sources[sid] for sid in e.voter_sources]
                citable = all(s.citable for s in cites)
                reason = (
                    "every voting source resolves"
                    if citable
                    else "; ".join(f"source {s.source_id}: {s.citable_reason}" for s in cites if not s.citable)
                )
            elif e.basis == "primary_dataset_point":
                citable, reason = False, "family-table value; its dataset's source is listed in the note"
            else:
                citable, reason = False, "no stored value"
            rows.append(
                (
                    m.material_id, m.note_id, "Evidence at a glance", e.quantity, e.value,
                    None, None, e.property_name, e.scope,
                    "consensus" if e.basis == "consensus" else e.basis,
                    e.basis, e.grade.grade if e.grade else None, e.grade.reason if e.grade else None,
                    e.classification, e.num_sources,
                    json.dumps(e.voter_sources), json.dumps(_groups(e)),
                    single, None, (e.value_origin or "none") + f"; {origin}",
                    None, None, None, int(citable), reason,
                )
            )
        for p in m.physical:
            citable, reason = _citation(records, p.source_id)
            rows.append(
                (
                    m.material_id, m.note_id, "Physical properties", p.quantity, p.value,
                    None, p.unit, p.label, p.phase, p.kind.kind, p.kind.provenance, None, p.kind.reason,
                    None, None, None, None, p.source_id, p.record_id,
                    f"physical_properties record {p.record_id}; {origin}",
                    p.temperature_c, p.frequency_hz, p.wavelength_nm, int(citable), reason,
                )
            )
        rows.append(
            (
                m.material_id, m.note_id, "Evidence at a glance", "k_status", None, m.k_status, None, None,
                "all datasets", "counted_from_stored_rows", None, None,
                "k on every stored row, on some, or on none; a missing k is never 0", None, None, None, None,
                None, None, f"optical_dispersion.k; {origin}", None, None, None, 0, "a status, not a value",
            )
        )
        if m.density is None:
            rows.append(
                (
                    m.material_id, m.note_id, "Evidence at a glance", "density", None, "absent", None, None, None,
                    "absent", None, None, m.density_evidence.reason, None, None, None, None, None, None,
                    f"no density row; {origin}", None, None, None, 0, "no value",
                )
            )
    return rows


def _groups(e) -> object:
    if e.groups is None:
        return None
    return e.groups.n_groups if e.groups.status == "determined" else "unknown"


def write_index(path: Path, records: Records, notes: list[Note], links: list[Link], meta: dict) -> dict:
    """Write the sqlite index at ``path`` (replacing it). Returns counts."""
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        for key, value in sorted(meta.items()):
            connection.execute("INSERT INTO meta VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))

        chunk_id = 0
        for note in sorted(notes, key=lambda n: n.note_id):
            connection.execute(
                "INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    note.note_id, note.type, note.title, note.path, note.material_id, note.family,
                    note.formula, note.grade, json.dumps(note.frontmatter, sort_keys=False, ensure_ascii=False),
                ),
            )
            aliases = " ; ".join(note.aliases)
            for section in note.sections:
                chunk_id += 1
                connection.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk_id, note.note_id, section.heading, note.title, note.path,
                        chunk_text(note, section), note.material_id, note.family, note.formula,
                        note.grade, note.kind, aliases,
                    ),
                )
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")

        connection.executemany(
            'INSERT INTO "values" (material_id, note_id, section, quantity, value, text_value, unit, '
            "dataset_label, scope, kind, basis, grade, reason, classification, voting_datasets, "
            "voting_sources, independent_groups, source_id, record_id, origin, temperature_c, frequency_hz, "
            "radiation_wavelength_nm, citable, citable_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            value_rows(records),
        )

        for m in sorted(records.materials.values(), key=lambda m: m.material_id):
            for d in m.datasets:
                connection.execute(
                    "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        d.dataset_id, m.material_id, m.note_id, d.label, d.phase, d.tag, d.axis, d.kind.kind,
                        d.kind.reason, d.source_id, d.temperature_c, d.n_rows, d.wavelength_min_nm,
                        d.wavelength_max_nm, d.k_status, int(d.primary), int(d.covers_633_n),
                        d.raw_record_table, d.temperature_note,
                    ),
                )
            alias_rows = [(m.name, "name"), (m.key, "key"), (str(m.material_id), "material_id")]
            alias_rows += [(a, "synonym") for a in m.synonyms]
            alias_rows += [(a, "original_name") for a in m.original_names]
            if m.formula:
                alias_rows.append((m.formula, "formula"))
            if m.cas:
                alias_rows.append((m.cas, "cas"))
            if m.inchikey:
                alias_rows.append((m.inchikey, "inchikey"))
            if m.family == "glass_catalogs":
                alias_rows.append((m.name.split(" ", 1)[1], "maker_code"))
            seen: set[tuple[str, str]] = set()
            for alias, kind in alias_rows:
                if (norm(alias), kind) in seen:
                    continue
                seen.add((norm(alias), kind))
                connection.execute("INSERT INTO aliases VALUES (?,?,?,?)", (alias, norm(alias), m.note_id, kind))

        connection.executemany("INSERT INTO links VALUES (?,?,?,?)", links)

        for s in sorted(records.sources.values(), key=lambda s: s.source_id):
            connection.execute(
                "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    s.source_id, s.note_id, s.handle, s.citation(), s.doi, s.url, s.year, s.title, s.authors,
                    s.technique, s.parse.kind, int(s.citable), s.citable_reason,
                ),
            )
        for family, row in records.gap_candidates:
            connection.execute(
                "INSERT INTO gaps VALUES (?,?,?,?,?,?,?)",
                (
                    family, row.get("name"), row.get("key") or row.get("selection_key"), row.get("gap_kind"),
                    row.get("reason"), row.get("evidence"), norm(row.get("name") or ""),
                ),
            )
        connection.commit()
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in ("notes", "chunks", "values", "datasets", "links", "aliases", "sources", "gaps")
        }
        connection.execute("VACUUM")
    finally:
        connection.close()
    return counts


def index_json(records: Records, notes: list[Note], links: list[Link], meta: dict) -> dict:
    """Note metadata as JSON: ids, titles, paths, grades, links out and in."""
    out_links: dict[str, list] = {}
    in_links: dict[str, list] = {}
    for frm, relation, to, _ in links:
        out_links.setdefault(frm, []).append([relation, to])
        in_links.setdefault(to, []).append([relation, frm])
    return {
        "meta": meta,
        "notes": [
            {
                "id": n.note_id,
                "type": n.type,
                "title": n.title,
                "path": n.path,
                "material_id": n.material_id,
                "family": n.family,
                "formula": n.formula,
                "grades": n.grade.split(",") if n.grade else [],
                "aliases": n.aliases,
                "links_out": out_links.get(n.note_id, []),
                "links_in": in_links.get(n.note_id, []),
            }
            for n in sorted(notes, key=lambda n: n.note_id)
        ],
    }
