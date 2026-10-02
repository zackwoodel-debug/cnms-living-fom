"""Typed links: each declared once, on one side; the reverse is generated.

Relations (materials-atlas/CLAUDE.md 3.2):

    material  member_of            family     reverse: members
    material  cites                source     reverse: cited_by
    material  has_property         property   reverse: materials_with
    dataset   agrees_with          dataset    (dataset_validation excellent; symmetric)
    dataset   partly_agrees_with   dataset    (warning; symmetric)
    dataset   disagrees_with       dataset    (suspicious; symmetric)
    source    same_author_group_as source     (rubric 6.4, sources that share an author; symmetric)
"""

from __future__ import annotations

from .facts import Records, has_properties

REVERSE: dict[str, str] = {
    "member_of": "members",
    "cites": "cited_by",
    "has_property": "materials_with",
    "agrees_with": "agrees_with",
    "partly_agrees_with": "partly_agrees_with",
    "disagrees_with": "disagrees_with",
    "same_author_group_as": "same_author_group_as",
}

Link = tuple[str, str, str, str]  # (from_id, relation, to_id, note)


def declared_links(records: Records) -> list[Link]:
    out: list[Link] = []
    for m in sorted(records.materials.values(), key=lambda m: m.material_id):
        out.append((m.note_id, "member_of", f"family.{m.family}", ""))
        for sid in m.cites:
            what = [d.label for d in m.datasets if d.source_id == sid] + [
                p.label for p in m.physical if p.source_id == sid
            ]
            out.append((m.note_id, "cites", f"source.{sid}", "; ".join(what)))
        for name in has_properties(m):
            out.append((m.note_id, "has_property", f"property.{name}", ""))
        ids = {d.label: d.dataset_id for d in m.datasets}
        for v in m.validations:
            note = f"{v.property_name}: {v.classification}, mean relative error {v.mean_relative_error!r}"
            out.append((ids[v.dataset_a], v.relation, ids[v.dataset_b], note))

    seen: set[tuple[int, int]] = set()
    for m in sorted(records.materials.values(), key=lambda m: m.material_id):
        for e in [*m.n633, *m.k633]:
            if e.groups is None or e.groups.status != "determined":
                continue
            for component in e.groups.components:
                for i, a in enumerate(component):
                    for b in component[i + 1 :]:
                        if (a, b) in seen:
                            continue
                        shared = _shared(records, a, b)
                        if shared:
                            seen.add((a, b))
                            out.append((f"source.{a}", "same_author_group_as", f"source.{b}", shared))
    return sorted(out)


def _shared(records: Records, a: int, b: int) -> str:
    pa, pb = records.sources[a].parse, records.sources[b].parse
    if a == b:
        return ""
    if pa.kind == "organisation" and pb.kind == "organisation" and pa.organisation == pb.organisation:
        return f"same organisation: {pa.organisation}"
    common = sorted(
        {(p.surname, p.initial) for p in pa.people if p.initial} & {(p.surname, p.initial) for p in pb.people if p.initial}
    )
    return ("shared author: " + ", ".join(f"{s}, {i}." for s, i in common)) if common else ""


def with_reverse(links: list[Link]) -> list[Link]:
    """Forward links plus each one's reverse, sorted, without duplicates."""
    out = set(links)
    for frm, relation, to, note in links:
        out.add((to, REVERSE[relation], frm, note))
    return sorted(out)
