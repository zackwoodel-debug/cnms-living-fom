#!/usr/bin/env python
"""Build the materials-atlas vault and index from a materials-db release.

    python materials-atlas/build.py                 # full vault + index, every check
    python materials-atlas/build.py --sample        # the five sample notes and their index
    python materials-atlas/build.py --release PATH  # another release folder (or its .sqlite)

Outputs (generated, never hand-edited): ``vault/`` (open it in Obsidian),
``index/atlas_index.sqlite``, ``index/atlas_index.json`` and ``index/_build.json``,
under ``--out`` (default: this folder). The release is opened read-only and
nothing is written outside the output folder; the build checks that too.

The build renders into a temporary folder, runs the checks of CLAUDE.md
section 9 there, rebuilds a second time from scratch and compares the two byte
for byte, and only then replaces the output. Any failure leaves the previous
output untouched and exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for path in (HERE, REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from atlas import check, pages, render  # noqa: E402
from atlas.facts import Records, build_records, subset  # noqa: E402
from atlas.index import INDEX_SCHEMA, index_json, write_index  # noqa: E402
from atlas.links import declared_links, with_reverse  # noqa: E402
from atlas.vault import render_files, write_tree  # noqa: E402

from cnms_fom.materials_release import Release  # noqa: E402

DEFAULT_RELEASE = Path(
    os.environ.get("MATERIALS_ATLAS_RELEASE", "~/code/materials-db/release/materials-db-v0.22.0")
).expanduser()

#  Phase 2's five sample materials, by stable key (ids never change for a key).
SAMPLE_KEYS = (
    "oxides_50:SiO2@amorphous",
    "batch2_31:As2S3@amorphous",
    "semiconductors:GaAs",
    "glass_catalogs:SCHOTT-optical/N-BK7HT",
    "liquids:H2O",
)


def generator_fingerprint() -> dict:
    """What generated this output: a hash of the generator's own source files."""
    files = sorted(
        [*(HERE / "atlas").glob("*.py"), HERE / "build.py", *(REPO / "backend/cnms_fom/materials_release").glob("*.py")]
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return {"generator_sha256": digest.hexdigest(), "generator_files": [p.relative_to(REPO).as_posix() for p in files]}


def sample_records(records: Records) -> Records:
    keep = {m.material_id for m in records.materials.values() if m.key in SAMPLE_KEYS}
    if len(keep) != len(SAMPLE_KEYS):
        raise SystemExit(f"sample keys not all found: {SAMPLE_KEYS}")
    return subset(records, keep)


def all_notes(records: Records, sample: bool) -> list[render.Note]:
    materials = [records.materials[i] for i in sorted(records.materials)]
    #  Every source in a full build (PubChem is credited for identifiers, not cited by a row);
    #  the sources the kept materials cite in a sample.
    source_ids = sorted({sid for m in materials for sid in m.cites}) if sample else sorted(records.sources)
    families = sorted(records.families)
    notes = [render.material_note(records, m) for m in materials]
    notes += [render.source_note(records, records.sources[sid]) for sid in source_ids]
    notes += [render.family_note(records, f) for f in families]
    notes += [render.property_note(records, p) for p in records.properties]
    notes += [pages.home(records), *pages.methods(records), *pages.views(records)]
    return notes


def build_once(
    release: Release, out: Path, *, sample: bool, materials: tuple[int, ...] = ()
) -> tuple[Records, list[render.Note], list, dict]:
    records = build_records(release)
    if sample:
        records = sample_records(records)
    elif materials:
        missing = set(materials) - set(records.materials)
        if missing:
            raise SystemExit(f"no material with id {sorted(missing)}")
        records = subset(records, set(materials))
    notes = all_notes(records, sample or bool(materials))
    ids = {n.note_id for n in notes} | {
        d["id"] for n in notes if n.type == "material" for d in n.frontmatter["datasets"]
    }
    links = [link for link in with_reverse(declared_links(records)) if link[0] in ids and link[2] in ids]
    meta = {
        "release_version": records.version,
        "db_sha256": records.sha256,
        "materials_db_release_commit": records.manifest.get("git_commit"),
        "refractiveindex_info_commit": records.manifest.get("refractiveindex_info_commit"),
        "materials_project_database_version": records.manifest.get("materials_project_database_version"),
        "helper_provenance": records.helper_provenance,
        "index_schema": INDEX_SCHEMA,
        "sample": sample,
        #  A test fixture: a real subset, accepted by the reader, never shipped as the atlas.
        "subset": f"materials {list(materials)}" if materials else None,
        **generator_fingerprint(),
    }
    vault = out / "vault"
    index_dir = out / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    tree = write_tree(vault, render_files(notes))
    counts = write_index(index_dir / "atlas_index.sqlite", records, notes, links, meta)
    (index_dir / "atlas_index.json").write_text(
        json.dumps(index_json(records, notes, links, meta), indent=1, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build = {
        **meta,
        "counts": {
            "materials": sum(n.type == "material" for n in notes),
            "sources": sum(n.type == "source" for n in notes),
            "families": sum(n.type == "family" for n in notes),
            "properties": sum(n.type == "property" for n in notes),
            "notes": len(notes),
            "datasets": counts["datasets"],
            "chunks": counts["chunks"],
            "values": counts["values"],
            "links": counts["links"],
            "aliases": counts["aliases"],
            "gap_candidates": counts["gaps"],
        },
        "n_633nm_grades": pages.grade_distribution(records),
        "independence": pages.independence_summary(records),
        "density_kinds": records.properties["density"].distribution,
        "k_status": records.properties["k_status"].distribution,
        "release_notices": records.notices,
    }
    (index_dir / "_build.json").write_text(json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records, notes, links, {"tree": tree, "build": build}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--sample", action="store_true", help="only the five sample materials")
    parser.add_argument("--skip-determinism", action="store_true", help="skip the second build (check 8)")
    parser.add_argument("--materials", help="comma-separated material ids: a subset build (test fixtures)")
    parser.add_argument("--index-only", action="store_true", help="write the index but not the vault")
    args = parser.parse_args(argv)

    started = time.time()
    stamp = started - 1.0
    release = Release(args.release)
    out = args.out.expanduser().resolve()
    materials = tuple(sorted(int(x) for x in args.materials.split(","))) if args.materials else ()
    subset_build = args.sample or bool(materials)
    with tempfile.TemporaryDirectory(prefix="materials-atlas-") as tmp:
        first = Path(tmp) / "first"
        records, notes, links, info = build_once(release, first, sample=args.sample, materials=materials)
        connection = release.connect()
        report = check.run_checks(
            records, first / "vault", first / "index" / "atlas_index.sqlite", links, connection, sample=subset_build
        )
        report["determinism"] = "skipped"
        if not args.skip_determinism:
            second = Path(tmp) / "second"
            build_once(release, second, sample=args.sample, materials=materials)
            differ = check.compare_trees(first, second)
            if differ:
                raise check.CheckFailed(f"8: two builds differ in {len(differ)} files, e.g. {differ[:5]}")
            report["determinism"] = "two independent builds are byte-identical"

        written = {"vault": "not written (--index-only)"}
        if not args.index_only:
            written["vault"] = write_tree(
                out / "vault",
                {p.relative_to(first / "vault").as_posix(): p.read_bytes() for p in (first / "vault").rglob("*") if p.is_file()},
            )
        index_out = out / "index"
        index_out.mkdir(parents=True, exist_ok=True)
        for name in ("atlas_index.sqlite", "atlas_index.json", "_build.json"):
            (index_out / name).write_bytes((first / "index" / name).read_bytes())

    touched = []
    for root in (Path("~/code/materials-db").expanduser(), Path("~/sea-atlas").expanduser()):
        touched += check.files_newer_than(root, stamp)
    if touched:
        raise check.CheckFailed(f"9: files under read-only inputs changed during the build: {touched[:5]}")
    report["read_only_inputs_untouched"] = True
    report["seconds"] = round(time.time() - started, 1)
    summary = {
        "output": str(out),
        "sample": args.sample,
        "counts": info["build"]["counts"],
        "vault_sync": written["vault"],
        "checks": report,
        "n_633nm_grades": info["build"]["n_633nm_grades"]["entries"],
        "release_notices": records.notices,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
