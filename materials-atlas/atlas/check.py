"""The build's checks (materials-atlas/CLAUDE.md section 9). Any failure stops the build.

They read what was *written* — the Markdown files and the sqlite index — and
compare it with the release database, so a rendering or serialisation bug shows
up here even when the in-memory records were right.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .facts import Records
from .index import norm
from .links import REVERSE, Link

#  [[target]], [[target#heading]], [[target|alias]], and in a table [[target\\|alias]].
WIKILINK = re.compile(r"\[\[([^\]|#\\]+)(?:#[^\]|\\]*)?(?:\\?\|[^\]]*)?\]\]")


class CheckFailed(RuntimeError):
    pass


def _frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        raise CheckFailed("note without frontmatter")
    end = text.index("\n---\n", 4)
    return yaml.safe_load(text[4:end])


def run_checks(
    records: Records,
    vault: Path,
    index_path: Path,
    links: list[Link],
    release_connection: sqlite3.Connection,
    *,
    sample: bool = False,
) -> dict:
    """Run every check; return a report. Raises CheckFailed listing every failure."""
    failures: list[str] = []
    report: dict = {}
    files = {p.relative_to(vault).as_posix(): p for p in vault.rglob("*.md")}
    texts = {rel: p.read_text(encoding="utf-8") for rel, p in files.items()}
    fms = {rel: _frontmatter(text) for rel, text in texts.items()}
    by_id = {fm["id"]: (rel, fm) for rel, fm in fms.items()}
    index = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)

    # 1. one note per material, source, family; counts equal the MANIFEST
    types = Counter(fm["type"] for fm in fms.values())
    report["notes_by_type"] = dict(sorted(types.items()))
    if not sample:
        tables = records.manifest["counts"]["tables"]
        expected = {"material": tables["materials"], "source": tables["sources"], "family": len(records.manifest["counts"]["per_family"])}
        for kind, n in expected.items():
            if types.get(kind) != n:
                failures.append(f"1: {types.get(kind)} {kind} notes, MANIFEST says {n}")
    material_ids = [fm["material_id"] for fm in fms.values() if fm["type"] == "material"]
    if len(material_ids) != len(set(material_ids)):
        failures.append("1: a material has two notes")

    # 2/3. every frontmatter and values number equals the database value exactly
    db = release_connection
    physical = {r[0]: r for r in db.execute(
        "SELECT record_id, material_id, density_g_cm3, xray_sld, neutron_sld, dielectric_constant FROM physical_properties")}
    consensus = {(r[0], r[1]): r[2] for r in db.execute("SELECT material_id, property_name, consensus_value FROM consensus_properties")}
    compared = 0
    for rel, fm in fms.items():
        if fm["type"] != "material":
            continue
        mid = fm["material_id"]
        m = records.materials[mid]
        if "density_g_cm3" in fm:
            row = physical[m.density.record_id]
            compared += 1
            if row[2] != fm["density_g_cm3"] or not isinstance(fm["density_g_cm3"], float):
                failures.append(f"2: {rel} density {fm['density_g_cm3']!r} != database {row[2]!r}")
            if "density" in fm.get("missing", {}):
                failures.append(f"3: {rel} has a density and lists it as missing")
        elif m.density is not None:
            failures.append(f"3: {rel} omits a stored density")
        for entry in fm.get("n_633nm", []) + fm.get("k_633nm", []):
            if "value" not in entry:
                continue
            compared += 1
            if entry["basis"] == "consensus":
                stored = consensus.get((mid, entry["property_name"]))
                if stored != entry["value"]:
                    failures.append(f"2: {rel} {entry['property_name']} {entry['value']!r} != database {stored!r}")
            elif entry["basis"] == "primary_dataset_point":
                family_value = float(next(r for f, r in m.family_rows if f == m.family)["n_633"])
                if family_value != entry["value"]:
                    failures.append(f"2: {rel} family-table n_633 {entry['value']!r} != {family_value!r}")
            else:
                failures.append(f"3: {rel} shows a value with basis {entry['basis']}")
    for value_id, record_id, quantity, value in index.execute(
        'SELECT value_id, record_id, quantity, value FROM "values" WHERE record_id IS NOT NULL'
    ):
        row = physical[record_id]
        stored = {"density": row[2], "dielectric_static_total": row[5], "dielectric_electronic": row[5]}.get(
            quantity, row[3] if quantity.startswith("xray") else row[4]
        )
        compared += 1
        if stored != value:
            failures.append(f"2: index value {value_id} ({quantity}) {value!r} != database {stored!r}")
    for value_id, mid, name, value, basis in index.execute(
        'SELECT value_id, material_id, dataset_label, value, basis FROM "values" WHERE quantity IN (\'n_633nm\', \'k_633nm\') AND value IS NOT NULL'
    ):
        compared += 1
        if basis == "consensus" and consensus.get((mid, name)) != value:
            failures.append(f"2: index consensus {value_id} {value!r} != database")
    zeros = index.execute('SELECT COUNT(*) FROM "values" WHERE value = 0').fetchone()[0]
    stored_zeros = db.execute(
        "SELECT (SELECT COUNT(*) FROM physical_properties WHERE density_g_cm3 = 0 OR xray_sld = 0 OR neutron_sld = 0 OR dielectric_constant = 0)"
        " + (SELECT COUNT(*) FROM consensus_properties WHERE consensus_value = 0)"
    ).fetchone()[0]
    if zeros > stored_zeros and not sample:
        failures.append(f"3: the index holds {zeros} zero values; the database stores {stored_zeros}")
    absent_as_number = index.execute('SELECT COUNT(*) FROM "values" WHERE kind = \'absent\' AND value IS NOT NULL').fetchone()[0]
    if absent_as_number:
        failures.append(f"3: {absent_as_number} absent values carry a number")
    report["numbers_compared"] = compared

    # 4. every dataset listed exactly once
    note_datasets = Counter(d["id"] for fm in fms.values() if fm["type"] == "material" for d in fm["datasets"])
    duplicated = [k for k, v in note_datasets.items() if v > 1]
    index_datasets = index.execute("SELECT COUNT(*), COUNT(DISTINCT dataset_id) FROM datasets").fetchone()
    expected_datasets = records.n_datasets if not sample else sum(len(records.materials[m].datasets) for m in material_ids)
    if duplicated or sum(note_datasets.values()) != expected_datasets or index_datasets != (expected_datasets, expected_datasets):
        failures.append(
            f"4: datasets in notes {sum(note_datasets.values())} (duplicated {duplicated[:3]}), "
            f"in index {index_datasets}, expected {expected_datasets}"
        )
    if not sample and expected_datasets != records.manifest["counts"]["datasets"]:
        failures.append(f"4: {expected_datasets} datasets, MANIFEST says {records.manifest['counts']['datasets']}")
    report["datasets"] = expected_datasets

    # 5. every wikilink and typed link resolves; reverse lists equal inverted forward lists
    targets = {rel[:-3] for rel in files}
    broken = Counter()
    for rel, text in texts.items():
        for target in WIKILINK.findall(text):
            if target not in targets:
                broken[f"{rel} -> {target}"] += 1
    if broken:
        failures.append(f"5: {len(broken)} broken wikilinks, e.g. {sorted(broken)[:3]}")
    known_ids = set(by_id) | {
        d["id"] for fm in fms.values() if fm["type"] == "material" for d in fm["datasets"]
    }
    for frm, relation, to, _ in links:
        if relation not in REVERSE and relation not in REVERSE.values():
            failures.append(f"5: unknown relation {relation}")
        for end in (frm, to):
            if end not in known_ids:
                failures.append(f"5: typed link end {end} ({frm} {relation} {to}) has no note")
                break
    forward = defaultdict(set)
    for frm, relation, to, _ in links:
        if relation == "cites":
            forward[to].add(frm)
    for s in records.sources.values():
        fm = by_id.get(s.note_id, (None, None))[1]
        if fm is not None and set(fm["cited_by"]) != forward.get(s.note_id, set()):
            failures.append(f"5: {s.note_id} cited_by differs from the inverted cites links")
    for family in records.families.values():
        fm = by_id.get(family.note_id, (None, None))[1]
        members = {frm for frm, relation, to, _ in links if relation == "member_of" and to == family.note_id}
        if fm is not None and set(fm["members"]) != members:
            failures.append(f"5: {family.note_id} members differ from the inverted member_of links")
    link_set = set((f, r, t) for f, r, t, _ in links)
    missing_reverse = [(f, r, t) for f, r, t in link_set if r in REVERSE and (t, REVERSE[r], f) not in link_set]
    if missing_reverse:
        failures.append(f"5: {len(missing_reverse)} links without their reverse, e.g. {missing_reverse[:2]}")
    report["links"] = len(links)

    # 6. every chunk stands alone; FTS finds every material by name, alias, formula and CAS
    lonely = index.execute(
        "SELECT c.chunk_id, c.note_id FROM chunks c JOIN notes n USING (note_id) WHERE n.type = 'material' "
        "AND (instr(c.text, n.title) = 0 OR instr(c.text, 'id ' || n.material_id) = 0)"
    ).fetchall()
    if lonely:
        failures.append(f"6: {len(lonely)} material chunks lack their name or id, e.g. {lonely[:3]}")
    misses = []
    searched = 0
    for fm in fms.values():
        if fm["type"] != "material":
            continue
        terms = [fm["name"], *fm.get("aliases", [])]
        terms += [fm[k] for k in ("formula", "cas") if k in fm]
        for term in terms:
            searched += 1
            phrase = '"' + str(term).replace('"', '""') + '"'
            found = index.execute(
                "SELECT 1 FROM chunks_fts JOIN chunks ON chunks.chunk_id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? AND chunks.note_id = ? LIMIT 1",
                (phrase, fm["id"]),
            ).fetchone()
            exact = index.execute("SELECT 1 FROM aliases WHERE alias_norm = ? AND note_id = ?", (norm(term), fm["id"])).fetchone()
            if not found or not exact:
                misses.append((fm["id"], term, bool(found), bool(exact)))
    if misses:
        failures.append(f"6: {len(misses)} name/alias/formula/CAS searches miss their note, e.g. {misses[:3]}")
    report["fts_searches"] = searched

    # 7. every grade has a reason; distributions are published
    for rel, fm in fms.items():
        for entry in fm.get("n_633nm", []):
            if not entry.get("grade") or not entry.get("grade_reason"):
                failures.append(f"7: {rel} has an n_633nm entry without a grade and reason")
    index.close()
    if failures:
        raise CheckFailed("atlas checks failed:\n  " + "\n  ".join(failures[:40]) + (f"\n  ... {len(failures) - 40} more" if len(failures) > 40 else ""))
    report["result"] = "all checks passed"
    return report


def compare_trees(a: Path, b: Path) -> list[str]:
    """Check 8: the relative paths whose bytes differ between two builds."""
    def listing(root: Path) -> dict[str, bytes]:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    la, lb = listing(a), listing(b)
    return sorted(k for k in set(la) | set(lb) if la.get(k) != lb.get(k))


def files_newer_than(root: Path, stamp: float) -> list[str]:
    """Check 9: files under ``root`` modified after ``stamp`` (a read-only input must have none)."""
    out = []
    if not root.exists():
        return out
    for path in root.rglob("*"):
        try:
            if path.stat().st_mtime > stamp:
                out.append(str(path))
        except OSError:
            continue
    return sorted(out)[:20]


def frontmatter_of(path: Path) -> dict:
    return _frontmatter(path.read_text(encoding="utf-8"))


def dumps(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
