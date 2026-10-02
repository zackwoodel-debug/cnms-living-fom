"""What the assistant's atlas tools return. Pure reads over an open ``AtlasIndex``.

Five shapes, one per tool:

    search(query, family?, limit)   ranked sections (chunks) with a snippet: context, not numbers
    read(note_id, section?)         one note or one section, with its typed links both ways
    lookup(material, quantity)      exact stored values for every matching material and dataset
    compare(materials, quantity)    lookup for several materials side by side
    coverage(material?)             what exists, and what is missing and why

Three rules shape them:

* **Numbers come from ``values``, never from text.** A model reading "1.457" out of a
  paragraph can misread it; a row cannot be misread.
* **Ambiguity is returned, not resolved.** A formula that names several materials, or a
  material with several phases or axes, comes back as several labelled matches. Picking
  one is the formula-as-identity error FOM_PROOF Sec. 2.1 forbids.
* **Every result carries its origin** — ``materials-db v<x> (reference)``, the release
  checksum, and per value ``citable`` with the reason — so a value cannot leave the tool
  without saying what it is.
"""

from __future__ import annotations

import json
import re

from .reader import AtlasIndex

ATTRIBUTION = (
    "Data: materials-db, CC BY 4.0. Cite materials-db and each value's own source; the atlas note "
    "'Home' lists the upstream citations."
)
MAX_HITS = 12
MAX_MATCHES = 12

#  quantity asked for -> the ``values.quantity`` rows that answer it.
QUANTITIES: dict[str, tuple[str, ...]] = {
    "n_633nm": ("n_633nm",),
    "k_633nm": ("k_633nm",),
    "k_status": ("k_status",),
    "density": ("density",),
    "xray_sld": ("xray_sld_real", "xray_sld_imag"),
    "neutron_sld": ("neutron_sld_real", "neutron_sld_imag"),
    "sld": ("xray_sld_real", "xray_sld_imag", "neutron_sld_real", "neutron_sld_imag"),
    "dielectric": ("dielectric_static_total", "dielectric_electronic"),
    "all": (
        "n_633nm", "k_633nm", "k_status", "density", "xray_sld_real", "xray_sld_imag",
        "neutron_sld_real", "neutron_sld_imag", "dielectric_static_total", "dielectric_electronic",
    ),
}
_SYNONYMS = {
    "n": "n_633nm", "n633": "n_633nm", "n at 633 nm": "n_633nm", "refractive index": "n_633nm",
    "index of refraction": "n_633nm", "n_633": "n_633nm",
    "k": "k_633nm", "k633": "k_633nm", "k at 633 nm": "k_633nm", "extinction coefficient": "k_633nm",
    "k_633": "k_633nm", "is k measured": "k_status", "k measured": "k_status", "kstatus": "k_status",
    "mass density": "density", "density_g_cm3": "density", "rho": "density",
    "x-ray sld": "xray_sld", "xray sld": "xray_sld", "sld_xray": "xray_sld",
    "neutron sld": "neutron_sld", "sld_neutron": "neutron_sld", "scattering length density": "sld",
    "dielectric constant": "dielectric", "permittivity": "dielectric", "epsilon": "dielectric",
    "dielectric_constant": "dielectric", "eps_inf": "dielectric", "everything": "all",
}


#  What a grade or a consensus class means, in the words an answer should use. From the
#  published rubric (materials-atlas Methods/Evidence grades) and the release's classes.
GRADE_MEANING = {
    "A": "two or more independent author groups agree within 2%",
    "B": "agreement only within one author group, or only within 10%",
    "C": "one measured dataset, not cross-checked",
    "D": "model fit only, no measured dataset at 633 nm",
    "T": "measured at 633 nm only at another temperature",
    "S": "the measured datasets disagree by 10% or more",
    "none": "no dataset covers 633 nm",
}
CLASS_MEANING = {
    "single_source": "one dataset; agreement cannot be checked",
    "excellent": "the voting datasets agree within 2% of their median",
    "warning": "the voting datasets agree only within 10% of their median",
    "suspicious": "the voting datasets DISAGREE: spread of 10% of the median or more",
}


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _envelope(atlas: AtlasIndex, **payload) -> dict:
    return {"origin": atlas.origin, "release_sha256": atlas.sha256, "attribution": ATTRIBUTION, **payload}


#  quantity -> the frontmatter ``missing`` keys that explain its absence.
MISSING_KEYS: dict[str, tuple[str, ...]] = {
    "n_633nm": ("n_633nm",),
    "density": ("density",),
    "xray_sld": ("sld",),
    "neutron_sld": ("sld",),
    "sld": ("sld",),
    "dielectric": ("dielectric_constant",),
}


def normalise_quantity(quantity: str | None) -> str | None:
    """'refractive index' -> 'n_633nm', 'x-ray SLD' -> 'xray_sld'; None when unknown."""
    if quantity is None or not str(quantity).strip():
        return "all"
    raw = _norm(quantity)
    for candidate in (raw, raw.replace(" ", "_"), raw.replace("-", " "), raw.replace("_", " ")):
        if candidate in QUANTITIES:
            return candidate
        if candidate in _SYNONYMS:
            return _SYNONYMS[candidate]
    return None


def fts_query(text: str) -> str:
    """Each word as a quoted FTS5 phrase, so ':', '-', 'OR' and quotes are data, not syntax."""
    words = re.findall(r"[^\s\"]+", text)
    return " ".join('"' + w.replace('"', '""') + '"' for w in words)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search(atlas: AtlasIndex, query: str, *, family: str | None = None, limit: int = 8) -> dict:
    """Ranked sections for a question. Lexical: BM25 over title, text, aliases and formula."""
    limit = max(1, min(int(limit), MAX_HITS))
    words = re.findall(r"[^\s\"]+", query or "")
    if not words:
        return _envelope(atlas, query=query, hits=[], error="empty query")
    exact = [row["note_id"] for row in atlas.query("SELECT DISTINCT note_id FROM aliases WHERE alias_norm = ?", (_norm(query),))]
    named = [n for n in _named_in(atlas, query) if n not in exact]

    def run(match: str) -> list:
        sql = (
            "SELECT c.chunk_id, c.note_id, c.section, c.title, c.material_id, c.formula, c.grade, c.family, c.path, "
            "snippet(chunks_fts, 1, '[', ']', ' … ', 24) AS snippet, bm25(chunks_fts, 5.0, 1.0, 4.0, 4.0) AS score "
            "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid WHERE chunks_fts MATCH ?"
        )
        params: list = [match]
        if family:
            sql += " AND c.family = ?"
            params.append(family)
        sql += " ORDER BY score, c.chunk_id LIMIT ?"
        params.append(limit * 4)
        return atlas.query(sql, tuple(params))

    rows = run(" AND ".join(fts_query(w) for w in words))
    mode = "all words"
    if len(rows) < limit and len(words) > 1:
        seen = {r["chunk_id"] for r in rows}
        rows = list(rows) + [r for r in run(" OR ".join(fts_query(w) for w in words)) if r["chunk_id"] not in seen]
        mode = "all words, then any word"
    #  A material the query names exactly (the whole query, or a name, synonym, formula
    #  or CAS number inside it) has its summary put first, fetched if FTS ranked it low.
    rows = list(rows)
    present = {(r["note_id"], r["section"]) for r in rows}
    for note_id in [*exact, *named]:
        if (note_id, "Summary") not in present:
            rows += atlas.query(
                "SELECT chunk_id, note_id, section, title, material_id, formula, grade, family, path, "
                "substr(text, 1, 240) AS snippet, 0.0 AS score FROM chunks WHERE note_id = ? AND section = 'Summary'",
                (note_id,),
            )

    def priority(r) -> tuple:
        top = r["section"] in ("Summary", "Evidence at a glance")
        if r["note_id"] in exact and top:
            return (0, exact.index(r["note_id"]))
        if r["note_id"] in named and top:
            return (1, named.index(r["note_id"]))
        return (2, 0)

    rows = sorted(rows, key=priority)
    hits = []
    for r in rows[:limit]:
        found_by = "lexical"
        if r["note_id"] in exact:
            found_by += ", exact alias"
        elif r["note_id"] in named:
            found_by += ", named in the query"
        hits.append(
            {
                "note_id": r["note_id"],
                "section": r["section"],
                "title": r["title"],
                "material_id": r["material_id"],
                "formula": r["formula"],
                "grades": r["grade"].split(",") if r["grade"] else [],
                "family": r["family"],
                "snippet": r["snippet"],
                "found_by": found_by,
            }
        )
    payload = _envelope(atlas, query=query, family=family, match_mode=mode, hits=hits)
    payload["note"] = (
        "Sections for context. For any number use atlas_lookup, which returns the stored value with its "
        "label, unit, grade and source; do not read numbers out of a snippet."
        if hits
        else "No atlas section matches. That is a statement about the atlas, not about the literature."
    )
    return payload


def _named_in(atlas: AtlasIndex, query: str) -> list[str]:
    """Material notes whose name, synonym, formula, CAS or maker code appears as words in the query.

    Longest phrases first. A bare material id is never matched inside a query ("633 nm"
    is not material 633), and neither is a one-character alias ("C").
    """
    words = [w.strip(",.;:?!()\"'") for w in query.split()]
    found: list[str] = []
    for size in range(min(6, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start : start + size]).strip()
            if len(phrase) < 2:
                continue
            for row in atlas.query(
                "SELECT DISTINCT note_id FROM aliases WHERE alias_norm = ? AND kind != 'material_id' ORDER BY note_id",
                (_norm(phrase),),
            ):
                if row["note_id"] not in found:
                    found.append(row["note_id"])
    return found


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def read(atlas: AtlasIndex, note_id: str, *, section: str | None = None) -> dict:
    note = atlas.query("SELECT * FROM notes WHERE note_id = ?", (note_id,))
    if not note:
        candidates = resolve(atlas, note_id)
        return _envelope(
            atlas,
            note_id=note_id,
            error=f"No atlas note has the id {note_id!r}.",
            did_you_mean=[c["note_id"] for c in candidates[:5]],
        )
    row = note[0]
    chunks = atlas.query("SELECT section, text FROM chunks WHERE note_id = ? ORDER BY chunk_id", (note_id,))
    sections = [{"section": c["section"], "text": c["text"]} for c in chunks]
    if section:
        wanted = [s for s in sections if _norm(s["section"]) == _norm(section)]
        if not wanted:
            return _envelope(
                atlas, note_id=note_id, error=f"No section {section!r}.", sections_available=[s["section"] for s in sections]
            )
        sections = wanted
    out_links = atlas.query("SELECT relation, to_id, note FROM links WHERE from_id = ? ORDER BY relation, to_id", (note_id,))
    return _envelope(
        atlas,
        note_id=note_id,
        type=row["type"],
        title=row["title"],
        path=row["path"],
        frontmatter=json.loads(row["frontmatter"]),
        sections=sections,
        links=[{"relation": r["relation"], "to": r["to_id"], "note": r["note"]} for r in out_links[:60]],
    )


# ---------------------------------------------------------------------------
# lookup / compare
# ---------------------------------------------------------------------------


def resolve(atlas: AtlasIndex, material: str) -> list[dict]:
    """Every material note the text names: exact name, synonym, formula, CAS, key, id or maker code."""
    text = str(material).strip()
    rows = atlas.query(
        "SELECT a.note_id, a.kind, a.alias, n.title, n.material_id FROM aliases a JOIN notes n USING (note_id) "
        "WHERE a.alias_norm = ? ORDER BY n.material_id",
        (_norm(text),),
    )
    out: dict[str, dict] = {}
    for r in rows:
        entry = out.setdefault(
            r["note_id"], {"note_id": r["note_id"], "material": r["title"], "material_id": r["material_id"], "matched_on": []}
        )
        entry["matched_on"].append(r["kind"])
    return list(out.values())


def _candidates(atlas: AtlasIndex, material: str) -> list[dict]:
    """Near misses for an unresolved name: titles and aliases sharing its words."""
    #  Prefix terms, so a truncated or partly misspelt name ("silicon nitrid") still finds candidates.
    match = " AND ".join(fts_query(w) + "*" for w in re.findall(r"[^\s\"]+", material))
    if not match:
        return []
    rows = atlas.query(
        "SELECT DISTINCT c.note_id, c.title, c.material_id FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? AND c.material_id IS NOT NULL AND c.section = 'Identity' "
        "ORDER BY bm25(chunks_fts, 5.0, 1.0, 4.0, 4.0) LIMIT 6",
        ("{title aliases formula} : (" + match + ")",),
    )
    return [{"note_id": r["note_id"], "material": r["title"], "material_id": r["material_id"]} for r in rows]


def _value_row(atlas: AtlasIndex, r) -> dict:
    out = {
        "quantity": r["quantity"],
        "value": r["value"],
        "unit": r["unit"] or ("dimensionless" if r["value"] is not None else None),
        "label": r["dataset_label"],
        "scope": r["scope"],
        "kind": r["kind"],
        "basis": r["basis"],
        "origin": r["origin"],
        "citable": bool(r["citable"]),
        "citable_reason": r["citable_reason"],
    }
    if r["text_value"] is not None:
        out["status"] = r["text_value"]
        if r["value"] is None:
            out.pop("value")
    for key in ("grade", "classification", "voting_datasets"):
        if r[key] is not None:
            out[key] = r[key]
    if r["reason"] is not None:
        #  Why a value has its grade (n at 633 nm) or its kind (everything else).
        out["grade_reason" if r["grade"] is not None else "kind_reason"] = r["reason"]
    if r["grade"] is not None:
        out["grade_meaning"] = GRADE_MEANING.get(r["grade"], "")
    if r["classification"] is not None:
        out["classification_meaning"] = CLASS_MEANING.get(r["classification"], "")
    if r["independent_groups"] is not None:
        out["independent_groups"] = json.loads(r["independent_groups"])
    for key in ("temperature_c", "frequency_hz", "radiation_wavelength_nm"):
        if r[key] is not None:
            out[key] = r[key]
    source_ids = json.loads(r["voting_sources"]) if r["voting_sources"] else ([r["source_id"]] if r["source_id"] else [])
    if source_ids:
        placeholders = ",".join("?" for _ in source_ids)
        out["sources"] = [
            {"source_id": s["source_id"], "citation": s["citation"], "doi": s["doi"], "citable": bool(s["citable"])}
            for s in atlas.query(f"SELECT * FROM sources WHERE source_id IN ({placeholders}) ORDER BY source_id", tuple(source_ids))
        ]
    return out


def _quote(match: dict, value: dict, version: str) -> str:
    """One ready-to-quote line for a value: who, what, how much, how good, from where."""
    who = f"{match['material']} (id {match['material_id']})"
    what = value["label"] or value["quantity"]
    if "status" in value and value.get("value") is None:
        extra = "; a missing k is unknown, not 0" if value["quantity"] == "k_status" else ""
        return f"{who}: {value['quantity']} = {value['status']}{extra} (materials-db v{version})"
    if value.get("value") is None:
        return f"{who}: {what}: no stored value ({value.get('grade_reason') or value.get('kind_reason') or value.get('basis')})"
    unit = f" {value['unit']}" if value.get("unit") and value["unit"] != "dimensionless" else ""
    parts = [f"{who}: {what} = {value['value']}{unit}"]
    if value.get("grade"):
        parts.append(f"grade {value['grade']} ({value.get('grade_meaning')})")
        if "excluding" in (value.get("grade_reason") or ""):
            parts.append(value["grade_reason"])
    elif value.get("kind"):
        parts.append(f"{value['kind']} ({value.get('kind_reason') or value.get('basis')})")
    if value.get("classification") and value["classification"] != "single_source":
        parts.append(f"{value['classification']}: {value.get('classification_meaning')}")
    sources = value.get("sources") or []
    if sources:
        parts.append("source: " + "; ".join(src["citation"] for src in sources[:3]) + (" ..." if len(sources) > 3 else ""))
    parts.append(f"reference value, materials-db v{version}" + ("" if value.get("citable") else ", not citable"))
    return " — ".join(parts)


#  Words that may surround a material's name without changing which material it is:
#  states, forms and conditions. "liquid water" is water; "indium nitride" is not indium.
QUALIFIERS = frozenset(
    "liquid solid gas gaseous vapour vapor film films thin thick bulk crystal crystals crystalline single "
    "polycrystalline amorphous glass glassy fused powder ceramic layer coating sample samples pure the a an of "
    "at in room ambient temperature nm um µm optical".split()
)


def _named_matches(atlas: AtlasIndex, text: str) -> list[dict]:
    """Materials named inside the text ('liquid water'), when every other word only qualifies them.

    The longest names win and their sub-phrases are not matched again. If any word left
    over is not a state, form or condition (``QUALIFIERS``) or a number, nothing is
    returned: "indium nitride" contains "indium" and is not indium.
    """
    words = [w.strip(",.;:?!()\"'") for w in text.split()]
    covered: set[int] = set()
    found: dict[str, dict] = {}
    for size in range(min(6, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            span = set(range(start, start + size))
            if span & covered:
                continue
            phrase = " ".join(words[start : start + size]).strip()
            if len(phrase) < 2:
                continue
            rows = atlas.query(
                "SELECT a.note_id, a.kind, n.title, n.material_id FROM aliases a JOIN notes n USING (note_id) "
                "WHERE a.alias_norm = ? AND a.kind != 'material_id' ORDER BY n.material_id",
                (_norm(phrase),),
            )
            if rows:
                covered |= span
            for r in rows:
                found.setdefault(
                    r["note_id"],
                    {"note_id": r["note_id"], "material": r["title"], "material_id": r["material_id"],
                     "matched_on": [f"{r['kind']} named inside {text!r}: {phrase!r}"]},
                )
    leftover = [w for i, w in enumerate(words) if i not in covered and w]
    if any(w.casefold() not in QUALIFIERS and not w.replace(".", "", 1).isdigit() for w in leftover):
        return []
    return list(found.values())


def lookup(atlas: AtlasIndex, material: str, quantity: str | None = "all") -> dict:
    """Exact stored values for every material ``material`` names. Ambiguity is returned, not resolved."""
    q = normalise_quantity(quantity)
    if q is None:
        return _envelope(
            atlas,
            material=material,
            quantity=quantity,
            error=f"Unknown quantity {quantity!r}.",
            valid_quantities=sorted(QUANTITIES),
        )
    matches = resolve(atlas, material)
    resolved_inside = False
    gaps = [] if matches else atlas.query(
        "SELECT family, name, key, gap_kind, reason FROM gaps WHERE name_norm = ? OR key = ?", (_norm(material), material)
    )
    if not matches and not gaps:
        matches = _named_matches(atlas, material)
        resolved_inside = bool(matches)
    if not matches:
        payload = _envelope(atlas, material=material, quantity=q, matches=[])
        if gaps:
            payload["absent_from_release"] = [dict(g) for g in gaps]
            payload["note"] = (
                f"{material!r} is not a material in materials-db v{atlas.version}. The release considered it and "
                "recorded why it is not included (quoted in absent_from_release). Say so; do not supply a value."
            )
        else:
            payload["candidates"] = _candidates(atlas, material)
            payload["note"] = (
                f"No material in materials-db v{atlas.version} is named {material!r} (searched names, synonyms, "
                "formulas, CAS numbers, keys and maker codes). If a candidate is the material meant, look it up by "
                "its id; otherwise the atlas has no value, and none may be supplied from memory."
            )
        return payload

    out = []
    for m in matches[:MAX_MATCHES]:
        note = atlas.query("SELECT frontmatter FROM notes WHERE note_id = ?", (m["note_id"],))[0]
        fm = json.loads(note["frontmatter"])
        placeholders = ",".join("?" for _ in QUANTITIES[q])
        rows = atlas.query(
            f'SELECT * FROM "values" WHERE note_id = ? AND quantity IN ({placeholders}) ORDER BY value_id',
            (m["note_id"], *QUANTITIES[q]),
        )
        entry = {
            **m,
            "key": fm.get("key"),
            "formula": fm.get("formula"),
            "family": fm.get("family"),
            "values": [_value_row(atlas, r) for r in rows],
        }
        missing = {k: v for k, v in fm.get("missing", {}).items() if q == "all" or k in MISSING_KEYS.get(q, ())}
        if missing:
            entry["missing"] = missing
        if q in ("n_633nm", "k_633nm", "k_status", "all"):
            entry["datasets"] = [
                {
                    "label": d["dataset_label"],
                    "kind": d["kind"],
                    "k_status": d["k_status"],
                    "range_nm": [d["wavelength_min_nm"], d["wavelength_max_nm"]],
                    "temperature_c": d["temperature_c"],
                    "covers_633_nm": bool(d["covers_633_nm"]),
                }
                for d in atlas.query(
                    "SELECT * FROM datasets WHERE note_id = ? ORDER BY dataset_label LIMIT 40", (m["note_id"],)
                )
            ]
        out.append(entry)
    summary = []
    for entry in out:
        summary += [_quote(entry, v, atlas.version) for v in entry["values"]]
        for key, reason in entry.get("missing", {}).items():
            summary.append(f"{entry['material']} (id {entry['material_id']}): no {key} in materials-db v{atlas.version} — {reason}")
    payload = _envelope(atlas, material=material, quantity=q)
    notes = []
    if len(matches) > 1:
        notes.append(
            f"{len(matches)} materials match {material!r}. A formula is not a material identity: report each "
            "with its name and id, and do not present any one of them as the answer."
        )
    if resolved_inside:
        notes.append(f"No material is called {material!r}; it was matched by a name it contains. Say which material this is.")
    if any(len([v for v in e["values"] if v["quantity"] == "n_633nm"]) > 1 for e in out):
        notes.append("Several n-at-633-nm entries (phases, variants or axes): report each with its scope.")
    if notes:
        payload["note"] = " ".join(notes)
    payload["summary"] = summary
    payload["matches"] = out
    payload["ambiguous"] = len(matches) > 1
    if len(matches) > MAX_MATCHES:
        payload["truncated"] = f"{len(matches) - MAX_MATCHES} more matches not shown"
    return payload


def compare(atlas: AtlasIndex, materials: list[str], quantity: str | None = "all") -> dict:
    results = [lookup(atlas, name, quantity) for name in list(materials)[:8]]
    payload = _envelope(atlas, quantity=normalise_quantity(quantity), materials=results)
    payload["note"] = (
        "Side by side, each with its own grade, kind and source. Values of different kinds (measured, model fit, "
        "calculated) or different phases, axes or temperatures are not the same quantity; never average them."
    )
    return payload


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def coverage(atlas: AtlasIndex, material: str | None = None) -> dict:
    if not material:
        meta = atlas.meta
        families = atlas.query(
            "SELECT family, COUNT(*) AS n FROM notes WHERE type = 'material' GROUP BY family ORDER BY family"
        )
        build = {
            "materials": atlas.query("SELECT COUNT(*) FROM notes WHERE type = 'material'")[0][0],
            "datasets": atlas.query("SELECT COUNT(*) FROM datasets")[0][0],
            "model_fit_datasets": atlas.query("SELECT COUNT(*) FROM datasets WHERE kind = 'model_fit'")[0][0],
            "sources": atlas.query("SELECT COUNT(*) FROM sources")[0][0],
            "values": atlas.query('SELECT COUNT(*) FROM "values" WHERE value IS NOT NULL')[0][0],
        }
        grades = atlas.query(
            "SELECT grade, COUNT(*) AS n FROM \"values\" WHERE quantity = 'n_633nm' GROUP BY grade ORDER BY grade"
        )
        kinds = atlas.query(
            "SELECT kind, COUNT(*) AS n FROM \"values\" WHERE quantity = 'density' GROUP BY kind ORDER BY kind"
        )
        return _envelope(
            atlas,
            coverage={
                "counts": build,
                "families": {r["family"]: r["n"] for r in families},
                "n_633nm_grades": {r["grade"]: r["n"] for r in grades},
                "density_kinds": {r["kind"]: r["n"] for r in kinds},
                "quantities": sorted(QUANTITIES),
                "not_in_the_atlas": [
                    "band gap, band offset, breakdown field, loss tangent, thermal conductivity (no FOM input "
                    "property is in materials-db)",
                    "mechanical properties and rheology (tables reserved, empty in every release)",
                    "values at wavelengths other than 633 nm as scalars (the stored rows are available)",
                    "specimen form (bulk, film, thickness) of any sample",
                ],
                "materials_db_release_commit": meta.get("materials_db_release_commit"),
            },
        )
    matches = resolve(atlas, material)
    if not matches:
        return lookup(atlas, material, "all")
    out = []
    for m in matches[:MAX_MATCHES]:
        fm = json.loads(atlas.query("SELECT frontmatter FROM notes WHERE note_id = ?", (m["note_id"],))[0]["frontmatter"])
        quantities = sorted(
            {r["quantity"] for r in atlas.query('SELECT quantity FROM "values" WHERE note_id = ? AND value IS NOT NULL', (m["note_id"],))}
        )
        out.append(
            {
                **m,
                "optical_datasets": fm.get("optical_datasets"),
                "measured_datasets": fm.get("measured_datasets"),
                "model_fit_datasets": fm.get("model_fit_datasets"),
                "wavelength_range_nm": [fm.get("wavelength_min_nm"), fm.get("wavelength_max_nm")],
                "k_status": fm.get("k_status"),
                "density_kind": fm.get("density_kind"),
                "n_633nm_grades": fm.get("n_633nm_grades"),
                "quantities_with_values": quantities,
                "missing": fm.get("missing", {}),
            }
        )
    return _envelope(atlas, material=material, coverage=out, ambiguous=len(matches) > 1)
