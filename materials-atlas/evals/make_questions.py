#!/usr/bin/env python
"""Generate eval questions from the materials-db release database itself.

    python materials-atlas/evals/make_questions.py   # writes evals/questions_generated.yaml

Every expectation here comes from plain SQL on the release (opened read-only), never
from the atlas code, so an eval cannot pass by agreeing with itself: which materials a
query names (exact name, synonym, formula, CAS number, stable key or catalog maker
code), their stored n at 633 nm and density, a density's kind as its label states it,
and k status counted from stored rows. Sampling uses a fixed seed over sorted ids, so
the file only changes when the release does. Re-run it for a new release and commit the
result with the release bump.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_RELEASE = Path("~/code/materials-db/release/materials-db-v0.22.0").expanduser()
SEED = 20261001


def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def density_kind(label: str) -> str:
    if "MP_DFT" in label:
        return "calculated_dft"
    if "bulk_elemental_approximation" in label:
        return "bulk_approximation"
    if "experimental lattice params" in label:
        return "calculated_from_lattice"
    if "literature" in label:
        return "literature"
    raise ValueError(label)


def build(release: Path) -> list[dict]:
    version = release.name.split("-v", 1)[1]
    db = sqlite3.connect(f"file:{release / f'materials-db-v{version}.sqlite'}?mode=ro", uri=True)
    q = lambda sql, *a: db.execute(sql, a).fetchall()  # noqa: E731
    registry = {int(r["material_id"]): r for r in csv.DictReader((release / "material_registry.csv").open())}
    family = {mid: r["key"].split(":", 1)[0] for mid, r in registry.items()}
    names = {mid: name for mid, name in q("SELECT material_id, name FROM materials")}

    #  Every way the atlas lets a query name a material, built from the release alone.
    index: dict[str, set[int]] = defaultdict(set)
    for mid, name, formula, cas in q("SELECT material_id, name, formula, cas_number FROM materials"):
        for alias in (name, formula, cas, registry[mid]["key"]):
            if alias:
                index[norm(alias)].add(mid)
        if family[mid] == "glass_catalogs":
            index[norm(name.split(" ", 1)[1])].add(mid)
    for mid, synonym in q("SELECT material_id, synonym FROM material_synonyms"):
        index[norm(synonym)].add(mid)

    rng = random.Random(SEED)
    out: list[dict] = []

    def pick(items, n):
        items = sorted(items)
        return rng.sample(items, min(n, len(items)))

    single_n = {
        mid: (name, value, cls)
        for mid, name, value, cls in q(
            "SELECT material_id, property_name, consensus_value, classification FROM consensus_properties "
            "WHERE property_name LIKE 'n_633nm%' AND material_id IN (SELECT material_id FROM consensus_properties "
            "WHERE property_name LIKE 'n_633nm%' GROUP BY material_id HAVING COUNT(*) = 1)"
        )
    }
    by_name = [mid for mid in single_n if index[norm(names[mid])] == {mid} and family[mid] != "glass_catalogs"]
    for mid in pick(by_name, 12):
        label, value, cls = single_n[mid]
        out.append({
            "id": f"gen-name-{mid}", "category": "name", "kind": "lookup",
            "question": f"What is the refractive index of {names[mid]} at 633 nm, and how good is the evidence?",
            "material": names[mid], "quantity": "n_633nm", "material_ids": [mid],
            "values": [{"quantity": "n_633nm", "value": value, "classification": cls}],
            "e2e": {"contains": [f"id {mid}", repr(value)[:5]], "absent": []},
        })

    synonyms = [(s, mid) for mid, s in q("SELECT material_id, synonym FROM material_synonyms")
                if index[norm(s)] == {mid} and norm(s) != norm(names[mid])]
    for synonym, mid in pick(synonyms, 8):
        out.append({
            "id": f"gen-synonym-{mid}", "category": "synonym", "kind": "lookup",
            "question": f"What does the atlas hold for {synonym}?",
            "material": synonym, "quantity": "all", "material_ids": [mid],
            "e2e": {"contains": [f"id {mid}"], "absent": []},
        })

    cas = [(c, mid) for mid, c in q("SELECT material_id, cas_number FROM materials WHERE cas_number IS NOT NULL")
           if index[norm(c)] == {mid}]
    for number, mid in pick(cas, 6):
        out.append({
            "id": f"gen-cas-{mid}", "category": "cas", "kind": "lookup",
            "question": f"Which material has CAS number {number}?",
            "material": number, "quantity": "all", "material_ids": [mid],
            "e2e": {"contains": [names[mid]], "absent": []},
        })

    formulas = defaultdict(set)
    for mid, formula in q("SELECT material_id, formula FROM materials WHERE formula IS NOT NULL"):
        formulas[formula].add(mid)
    unique = [f for f, ids in formulas.items() if len(ids) == 1 and index[norm(f)] == ids]
    for formula in pick(unique, 8):
        [mid] = formulas[formula]
        out.append({
            "id": f"gen-formula-{mid}", "category": "formula", "kind": "lookup",
            "question": f"What is the density of {formula}?",
            "material": formula, "quantity": "density", "material_ids": [mid],
            "e2e": {"contains": [f"id {mid}"], "absent": []},
        })
    shared = [f for f, ids in formulas.items() if len(ids) > 1]
    for formula in pick(shared, 5):
        ids = sorted(index[norm(formula)])
        out.append({
            "id": f"gen-formula-shared-{norm(formula)}", "category": "formula_ambiguous", "kind": "lookup",
            "question": f"What is the refractive index of {formula} at 633 nm?",
            "material": formula, "quantity": "n_633nm", "material_ids": ids, "ambiguous": True,
            "e2e": {"contains": [f"id {mid}" for mid in ids], "absent": []},
        })

    codes = [(names[mid].split(" ", 1)[1], mid) for mid in names if family[mid] == "glass_catalogs"]
    codes = [(c, mid) for c, mid in codes if index[norm(c)] == {mid}]
    densities = {mid: (value, label) for mid, value, label in q(
        "SELECT material_id, density_g_cm3, dataset_label FROM physical_properties WHERE density_g_cm3 IS NOT NULL")}
    for code, mid in pick([c for c in codes if c[1] in densities], 6):
        value, label = densities[mid]
        out.append({
            "id": f"gen-maker-{mid}", "category": "maker_code", "kind": "lookup",
            "question": f"What is the density of the catalog glass {code}?",
            "material": code, "quantity": "density", "material_ids": [mid],
            "values": [{"quantity": "density", "value": value, "kind": density_kind(label)}],
            "e2e": {"contains": [names[mid].split(" ")[0], repr(value)[:4]], "absent": []},
        })

    kinds = defaultdict(list)
    for mid, (_value, label) in densities.items():
        if index[norm(names[mid])] == {mid}:
            kinds[density_kind(label)].append(mid)
    for kind, ids in sorted(kinds.items()):
        for mid in pick(ids, 3 if kind == "literature" else 2):
            value, _ = densities[mid]
            out.append({
                "id": f"gen-density-{mid}", "category": f"density_{kind}", "kind": "lookup",
                "question": f"What is the density of {names[mid]}, and is it measured?",
                "material": names[mid], "quantity": "density", "material_ids": [mid],
                "values": [{"quantity": "density", "value": value, "kind": kind}],
                "e2e": {"contains": [f"id {mid}", repr(value)[:4]], "absent": []},
            })
    no_density = [mid for mid in names if mid not in densities and index[norm(names[mid])] == {mid}]
    for mid in pick(no_density, 2):
        out.append({
            "id": f"gen-no-density-{mid}", "category": "density_absent", "kind": "lookup",
            "question": f"What is the density of {names[mid]}?",
            "material": names[mid], "quantity": "density", "material_ids": [mid],
            "e2e": {"contains": [f"id {mid}"], "absent": []},
        })

    rows = q("SELECT material_id, dataset_label, COUNT(*), COUNT(k) FROM optical_dispersion GROUP BY 1, 2")
    per_material = defaultdict(list)
    for mid, _label, n, with_k in rows:
        per_material[mid].append("measured" if with_k == n else "not_measured" if with_k == 0 else "partially_measured")
    status = {mid: ("measured" if all(s == "measured" for s in ss) else
                    "not_measured" if all(s == "not_measured" for s in ss) else "partially_measured")
              for mid, ss in per_material.items()}
    for wanted in ("measured", "partially_measured", "not_measured"):
        for mid in pick([m for m, s in status.items() if s == wanted and index[norm(names[m])] == {m}], 2):
            out.append({
                "id": f"gen-k-{mid}", "category": f"k_{wanted}", "kind": "lookup",
                "question": f"Is k measured for {names[mid]}?",
                "material": names[mid], "quantity": "k_status", "material_ids": [mid],
                "values": [{"quantity": "k_status", "status": wanted}],
                "e2e": {"contains": [f"id {mid}"], "absent": ["k = 0", "k is 0", "k is zero"]},
            })

    gaps = []
    for path in sorted((release / "family_tables").glob("*_gaps.csv")):
        for row in csv.DictReader(path.open()):
            name = row.get("name") or ""
            if name and not index[norm(name)] and row.get("reason"):
                gaps.append(name)
    for name in pick(sorted(set(gaps)), 4):
        out.append({
            "id": f"gen-absent-{norm(name).replace(' ', '-')[:40]}", "category": "absent", "kind": "absent",
            "question": f"What is the refractive index of {name}?",
            "material": name, "quantity": "n_633nm",
            "e2e": {"contains": [], "absent": []},
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    questions = build(args.release)
    header = (
        "# GENERATED by make_questions.py from the materials-db release (plain SQL, seed "
        f"{SEED}); do not edit.\n# Regenerate for a new release and commit with the release bump.\n"
    )
    body = yaml.safe_dump({"release": args.release.name.split("-v", 1)[1], "questions": questions},
                          sort_keys=False, allow_unicode=True, width=120)
    (HERE / "questions_generated.yaml").write_text(header + body, encoding="utf-8")
    print(f"{len(questions)} questions written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
