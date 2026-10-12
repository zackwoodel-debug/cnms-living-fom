"""Home, Methods and Views: the pages that explain the records rather than hold them.

Methods text is written from the rubric constants in ``evidence`` and
``authors`` and from distributions computed over the records, so the published
rule and the rule the code applies cannot drift apart. Views are generated
Markdown tables (Obsidian Bases' file format is not stable enough to generate
and test against; a table is).
"""

from __future__ import annotations

from collections import Counter

from .authors import ORGANISATIONS
from .evidence import (
    AMBIENT_C,
    DENSITY_KIND_NAMES,
    GRADE_RULES,
    GRADES,
    TAG_TEMPERATURE_TOLERANCE_C,
    fmt_number,
)
from .facts import Records
from .render import LICENCE_LINE, Note, Section, cell, code, footer, link, num, quote, table


def _note(records, note_id, title, path, intro, sections, kind) -> Note:
    return Note(
        note_id=note_id,
        type=kind,
        title=title,
        path=path,
        frontmatter={"id": note_id, "type": kind, "release": records.version, "db_sha256": records.sha256, "tags": [kind]},
        intro=intro,
        sections=sections,
        chunk_header=f"{title} | {note_id}",
        kind=kind,
        footer=footer(records, note_id),
    )


# ---------------------------------------------------------------------------
# Statistics used by Methods and the build report
# ---------------------------------------------------------------------------


def grade_distribution(records: Records) -> dict:
    entries = Counter(e.grade.grade for m in records.materials.values() for e in m.n633)
    by_material = Counter(
        "+".join(sorted({e.grade.grade for e in m.n633}, key=GRADES.index)) for m in records.materials.values()
    )
    return {
        "entries": {g: entries.get(g, 0) for g in GRADES},
        "n_entries": sum(entries.values()),
        "materials_by_grade_set": dict(sorted(by_material.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def independence_summary(records: Records) -> dict:
    out: dict = {}
    for quantity in ("n_633nm", "k_633nm"):
        counts: Counter = Counter()
        for m in records.materials.values():
            for e in m.n633 if quantity == "n_633nm" else m.k633:
                if e.classification in ("excellent", "warning", "suspicious"):
                    if e.groups is None or e.groups.status != "determined":
                        counts["unknown"] += 1
                    elif (e.groups.n_groups or 0) >= 2:
                        counts["multi-group"] += 1
                    else:
                        counts["one group"] += 1
        out[quantity] = {k: counts.get(k, 0) for k in ("multi-group", "one group", "unknown")}
    return out


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


def home(records: Records) -> Note:
    m = records.manifest
    counts = m.get("counts", {}).get("tables", {})
    rows = [
        ["materials", str(len(records.materials))],
        ["optical datasets", str(records.n_datasets)],
        ["optical rows", str(counts.get("optical_dispersion"))],
        ["physical_properties rows", str(counts.get("physical_properties"))],
        ["sources", str(len(records.sources))],
        ["consensus_properties rows", str(counts.get("consensus_properties"))],
        ["dataset_validation pairs", str(counts.get("dataset_validation"))],
        ["families", str(len(records.families))],
    ]
    licence = records_licence_quote(records)
    sections = [
        Section(
            "What this is",
            "\n".join(
                [
                    f"materials-atlas: a generated reference atlas of materials-db v{records.version}. One note per "
                    "material, source, family and property, each value exactly as the release stores it, with where "
                    "it came from, whether it is measured, a model fit or a calculation, how good the evidence is by "
                    "a published rubric, and what the data cannot tell you.",
                    "",
                    "These are **reference values**, not CNMS measurements. Nothing here flows into CNMS's "
                    "property_values, FOM scores or optimizer.",
                ]
            ),
        ),
        Section(
            "Release",
            "\n".join(
                [
                    f"materials-db v{records.version}, built {m.get('built_utc')} from commit {code(str(m.get('git_commit')))}.",
                    "",
                    table(
                        ["Fact", "Value"],
                        [
                            ["sqlite sha256", code(records.sha256)],
                            ["refractiveindex.info commit", code(str(m.get("refractiveindex_info_commit")))],
                            ["Materials Project database", cell(m.get("materials_project_database_version"))],
                            ["licence", cell(m.get("license"))],
                        ],
                    ),
                    "",
                    table(["Count", "Value"], rows),
                ]
            ),
        ),
        Section("Licence and citation", licence),
        Section(
            "How to read it",
            "\n".join(
                [
                    "Start with [[Methods/How to read a note]], then [[Methods/Evidence grades]].",
                    "",
                    "- [[Methods/Evidence grades]] · [[Methods/Independence]] · [[Methods/Measured vs model fit]] · "
                    "[[Methods/What the atlas cannot tell you]]",
                    "- Families: " + ", ".join(link(f.path, f.name) for f in records.families.values()),
                    "- Properties: " + ", ".join(link(p.path, p.title) for p in records.properties.values()),
                    "- Views: [[Views/n at 633 nm by grade]], [[Views/Densities by kind]], [[Views/Model-fit datasets]], "
                    "[[Views/Disagreeing sources]], [[Views/Release notices]]",
                ]
            ),
        ),
        Section(
            "How to rebuild",
            "\n".join(
                [
                    "The vault and the index are generated; never edit them. Rebuild with one command from the "
                    "repository root:",
                    "",
                    "```bash",
                    "cnms-fom atlas build          # or: python materials-atlas/build.py",
                    "```",
                    "",
                    "The build verifies the release's checksums and counts, the refractiveindex.info commit and the "
                    "helper files, re-runs every check, and stops on any mismatch.",
                ]
            ),
        ),
        Section(
            "How Quark uses it",
            "Quark's assistant (local llama3.1:8b through the CNMS backend) has read-only atlas tools: "
            "atlas_search and atlas_read for context, atlas_lookup and atlas_compare for exact values, atlas_coverage "
            "for what exists and what is missing. Every tool result carries the release version and checksum, and "
            "marks each value citable only when its source resolves.",
        ),
    ]
    return _note(records, "home", "materials-atlas", "Home.md", f"materials-db v{records.version} reference atlas.", sections, "home")


def records_licence_quote(records: Records) -> str:
    return "\n".join(
        [
            f"The data is materials-db v{records.version}, licensed {records.manifest.get('license')}. From the "
            "release's DATA_LICENSE.md, quoted:",
            "",
            quote(
                "You may share and adapt it for any purpose, including commercially, provided you give attribution: "
                "cite this dataset **and** the upstream sources below."
            ),
            "",
            "Upstream sources (DATA_LICENSE.md): refractiveindex.info (M. N. Polyanskiy, Sci. Data 11, 94 (2024), "
            "doi:10.1038/s41597-023-02898-2, plus each dataset's original paper), Materials Project (A. Jain et al., "
            "APL Mater. 1, 011002 (2013), doi:10.1063/1.4812323), PubChem (S. Kim et al., Nucleic Acids Res. 51, "
            "D1373-D1380 (2023), doi:10.1093/nar/gkac956), periodictable (P. A. Kienzle), pymatgen (S. P. Ong et al., "
            "Comput. Mater. Sci. 68, 314-319 (2013), doi:10.1016/j.commatsci.2012.10.028), RDKit.",
            "",
            "Each note lists the sources of its own values; cite those too.",
        ]
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def methods(records: Records) -> list[Note]:
    dist = grade_distribution(records)
    ind = independence_summary(records)
    return [
        _note(records, "methods.how-to-read-a-note", "How to read a note", "Methods/How to read a note.md",
              "How a material note is laid out, and what each field means.", _how_to_read(records), "methods"),
        _note(records, "methods.evidence-grades", "Evidence grades", "Methods/Evidence grades.md",
              "The published rules behind every grade in the atlas.", _evidence_grades(records, dist, ind), "methods"),
        _note(records, "methods.independence", "Independence", "Methods/Independence.md",
              "How the atlas decides whether agreeing sources are independent.", _independence(records, ind), "methods"),
        _note(records, "methods.measured-vs-model-fit", "Measured vs model fit", "Methods/Measured vs model fit.md",
              "Which optical datasets are measurements and which are model fits, and how that is decided.", _model_fit(records), "methods"),
        _note(records, "methods.what-the-atlas-cannot-tell-you", "What the atlas cannot tell you", "Methods/What the atlas cannot tell you.md",
              "Limits of the atlas as a whole.", _cannot(records), "methods"),
    ]


def _how_to_read(records: Records) -> list[Section]:
    return [
        Section(
            "Layout",
            "Every material note has the same sections, and each section's first line names the material and its id, "
            "so any one section can be read alone: Summary, Evidence at a glance, Identity, Optical datasets, Physical "
            "properties, Agreement between sources, Flags and gaps, What this note cannot tell you, Get the data.",
        ),
        Section(
            "Values",
            "Every number is the stored release value, printed exactly (no rounding, no unit conversion). The atlas "
            "never interpolates: n at 633 nm comes from the release's consensus_properties row or, where only a "
            "model fit covers 633 nm, from the family table's stored n_633; otherwise the note lists the stored rows "
            "either side of 633 nm and says there is no stored value. An absent value is absent and listed under "
            "`missing` with its reason; it is never 0.",
        ),
        Section(
            "Frontmatter",
            "\n".join(
                [
                    "The frontmatter holds the same values for tools and Obsidian views. The fields that need reading "
                    "carefully:",
                    "",
                    "- `n_633nm`: one entry per phase or variant and axis (a uniaxial crystal has an o-ray and an "
                    "e-ray entry; there is no single 'n of the material'). Each entry has its `basis` (consensus, "
                    "primary_dataset_point, model_fit_only, other_temperature_only, not_covered), its `grade` and "
                    "`grade_reason`, and for a consensus the release's class, voting datasets and distinct sources.",
                    "- `voting_datasets` counts datasets, as the release does; `voting_sources` counts distinct "
                    "sources. One paper's temperature series can be several voting datasets.",
                    "- `density_kind`: " + "; ".join(f"`{k}` = {v}" for k, v in DENSITY_KIND_NAMES.items()) + ".",
                    "- `k_status`: measured / partially_measured / not_measured, counted from stored rows.",
                    "- `cites`, `member_of`, `has_property`: typed links; the reverse lists (`cited_by`, `members`, "
                    "`materials_with`) are generated on the other side.",
                ]
            ),
        ),
        Section(
            "Formula is not identity",
            "A formula is not a material identity (CNMS FOM_PROOF Sec. 2.1). A note is one materials-db material, "
            "with its stable key and id; datasets carry their phase or variant and axis. Ask for a formula and the "
            "tools return every material and dataset that matches, labelled, and never pick one.",
        ),
    ]


def _evidence_grades(records: Records, dist: dict, ind: dict) -> list[Section]:
    rows = [[f"**{g}**", GRADE_RULES[g], str(dist["entries"][g])] for g in GRADES]
    by_set = [[cell(k or "(no entry)"), str(v)] for k, v in dist["materials_by_grade_set"].items()]
    return [
        Section(
            "Grade of n at 633 nm",
            "\n".join(
                [
                    "One grade per n-at-633-nm entry (one material, phase or variant, and axis), from the release's "
                    "own consensus and the independence rubric. A grade never rises on missing information: where a "
                    "rule cannot decide, the lower grade or 'unknown' applies.",
                    "",
                    table(["Grade", "Rule", f"Entries in v{records.version}"], rows),
                    "",
                    f"Applied first: a voter whose own source tag names a temperature outside {fmt_number(AMBIENT_C[0])}-"
                    f"{fmt_number(AMBIENT_C[1])} degC while its rows store none is excluded (the release counted it "
                    "as ambient only because the temperature is missing), and the entry is graded on the voters that "
                    f"remain. Tag and stored temperatures within {fmt_number(TAG_TEMPERATURE_TOLERANCE_C)} degC are "
                    "the same temperature (273 K is stored as -0.15 degC).",
                ]
            ),
        ),
        Section(
            "Inputs, all from the release",
            "\n".join(
                [
                    "- **Consensus** (consensus_properties): the median of the MEASURED datasets of one phase and "
                    f"axis that cover 633 nm at ambient temperature ({fmt_number(AMBIENT_C[0])}-{fmt_number(AMBIENT_C[1])} "
                    "degC, or no temperature stated); model fits never vote. Classes on the spread (max-min)/median: "
                    "excellent < 2%, warning < 10%, suspicious >= 10%, single_source for one.",
                    "- **Voters**: the release stores the median and the count, not the voters. The atlas re-runs the "
                    "release's rule to name them and checks that it reproduces every stored row exactly (value and "
                    "count); the build stops otherwise. The re-run's interpolated values are compared and discarded, "
                    "never shown.",
                    "- **Independence**: [[Methods/Independence]].",
                    "- **Measured vs model fit**: [[Methods/Measured vs model fit]].",
                ]
            ),
        ),
        Section(
            "Distribution",
            "\n".join(
                [
                    f"Over all {dist['n_entries']} n-at-633-nm entries of {len(records.materials)} materials, by grade: "
                    + ", ".join(f"{g} {dist['entries'][g]}" for g in GRADES)
                    + ". By material (the set of grades of its entries):",
                    "",
                    table(["Grades", "Materials"], by_set),
                    "",
                    "Independence of the consensus values that have more than one voter: n at 633 nm "
                    + ", ".join(f"{k} {v}" for k, v in ind["n_633nm"].items())
                    + "; k at 633 nm "
                    + ", ".join(f"{k} {v}" for k, v in ind["k_633nm"].items())
                    + ".",
                ]
            ),
        ),
        Section(
            "Other rubrics",
            "\n".join(
                [
                    "- **Density kind** (from the stored label only): "
                    + "; ".join(f"`{k}`: {v}" for k, v in DENSITY_KIND_NAMES.items())
                    + ".",
                    "- **k status**: `measured` (k on every stored row of every dataset), `not_measured` (on none), "
                    "`partially_measured` (otherwise). A NULL k means the source gives none; it is never 0.",
                    "- **SLDs and dielectric constants** are always calculated (periodictable; Materials Project DFPT).",
                    "- **Citable**: a value is citable when its source has a DOI, or a year with authors (or an "
                    "organisation) and a title or URL. Otherwise it is returned with citable false and the reason.",
                ]
            ),
        ),
    ]


def _independence(records: Records, ind: dict) -> list[Section]:
    orgs = sorted({name for _, name in ORGANISATIONS})
    return [
        Section(
            "Rule",
            "\n".join(
                [
                    "Two sources are in the same author group if they share any author, identified by surname and "
                    "first initial (accents, case, hyphens and spaces removed; 'Jr.' and 'et al.' dropped). Groups "
                    "are the connected components of that relation: if A shares an author with B and B with C, all "
                    "three are one group.",
                    "",
                    "- Organisations are their own group, recognised from a fixed list: " + ", ".join(orgs) + ".",
                    "- A compiled handbook (Palik, Weber, the Handbook of Mineralogy) is not an organisation: whose "
                    "measurement it relays is unknown, so its authors are unknown.",
                    "- The answer is **unknown** when any compared source has no readable authors, or when two groups "
                    "would be joined only by a surname with no initial on one side.",
                ]
            ),
        ),
        Section(
            "What it can and cannot do",
            "The rule errs toward fewer groups: two different people who share a surname and first initial are "
            "merged, which can understate independence; it never overstates it. Matching on surname alone, which "
            "merges every Wang, is the failure it avoids. It cannot see shared samples, shared instruments, or one "
            "group re-publishing its own data under new co-authors; those can make 'independent' groups less "
            "independent than the names suggest.",
        ),
        Section(
            "Summary",
            "Consensus values with more than one voter: n at 633 nm "
            + ", ".join(f"{k} {v}" for k, v in ind["n_633nm"].items())
            + "; k at 633 nm "
            + ", ".join(f"{k} {v}" for k, v in ind["k_633nm"].items())
            + ". Only multi-group excellent consensus earns grade A.",
        ),
    ]


def _model_fit(records: Records) -> list[Section]:
    fits = sorted(
        ((m, d) for m in records.materials.values() for d in m.datasets if d.kind.kind == "model_fit"),
        key=lambda md: (md[0].name, md[1].label),
    )
    rows = [[link(m.path, m.name), code(d.label).replace("|", "\\|"), cell(d.kind.reason)] for m, d in fits]
    prov = records.helper_provenance
    return [
        Section(
            "Rule",
            "\n".join(
                [
                    "A dataset is a model fit of the dielectric function only when its source says so: materials-db's "
                    "scripts/dataset_kind.py, quoted:",
                    "",
                    quote(
                        "A page counts as a model fit only when the source says so explicitly: its REFERENCES carry "
                        "RI.info's \"Calculation script\" link (the tabulated values were generated from a published "
                        "model), its COMMENTS say the data is a fit to a (simplified) model, or was generated from a "
                        "model, or the referenced paper's title makes a model its subject. Sellmeier/Cauchy-type formula "
                        "pages are MEASURED: they are the authors' representation of their measured n."
                    ),
                    "",
                    "The helper is imported read-only from the materials-db checkout, and only after the build has "
                    "checked that the checkout is the one that built the release:",
                    "",
                    table(
                        ["Check", "Value"],
                        [
                            ["release commit", code(str(prov.get("materials_db_release_commit")))],
                            ["refractiveindex.info commit (equals the MANIFEST's, clean)", code(str(prov.get("refractiveindex_info_commit")))],
                            ["helper files unchanged since the release commit", cell(", ".join(prov.get("helper_files_unchanged_since_release_commit", [])))],
                        ],
                    ),
                ]
            ),
        ),
        Section(
            "Every model fit",
            "\n".join([f"The {len(fits)} model-fit datasets in v{records.version}, with the evidence quoted:", "", table(["Material", "Dataset", "Evidence"], rows)]),
        ),
    ]


def _cannot(records: Records) -> list[Section]:
    return [
        Section(
            "Limits",
            "\n".join(
                [
                    "- It holds what the release holds and nothing else: a material or a property that is not in "
                    "materials-db is not in the atlas, and the tools say so rather than answer from elsewhere.",
                    "- It does not interpolate. A value at a wavelength other than 633 nm, or at 633 nm where no "
                    "scalar is stored, comes from the stored rows (Get the data) or the release's own interpolation "
                    "(materials_db.access nk_at), never from the atlas.",
                    "- k is unknown where a source gives none; that is not k = 0.",
                    "- Calculated values (DFT and DFPT, all SLDs, all descriptors) are not measurements. Model-fit "
                    "datasets are not measured data.",
                    "- A grade measures agreement between published datasets and the independence of their authors. "
                    "It does not measure sample quality, specimen form (bulk, film, thickness), stoichiometry, or "
                    "whether any value applies to a CNMS sample.",
                    "- Specimen form is not recorded in the release. Bringing a value into a CNMS analysis needs a "
                    "person to assert it (ingest_materials_db.py --specimen-form).",
                    "- These are reference data. They never enter CNMS's property_values, FOM scores or optimizer.",
                ]
            ),
        ),
        Section(
            "Release notices",
            "\n".join(
                ["Inconsistencies inside the release that the build found, reported and did not correct:", ""]
                + [f"- {cell(n)}" for n in records.notices]
                if records.notices
                else ["The build found no inconsistency inside the release."]
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def views(records: Records) -> list[Note]:
    mats = sorted(records.materials.values(), key=lambda m: (m.name, m.material_id))
    out: list[Note] = []

    by_grade: dict[str, list] = {g: [] for g in GRADES}
    for m in mats:
        for e in m.n633:
            by_grade[e.grade.grade].append((m, e))
    sections = []
    for g in GRADES:
        rows = [[link(m.path, m.name), cell(e.scope), num(e.value) if e.value is not None else "—", cell(e.basis.replace("_", " "))] for m, e in by_grade[g]]
        body = f"Grade {g} ({GRADE_RULES[g]}): {len(rows)} entries."
        if rows:
            body += "\n\n" + table(["Material", "Scope", "n at 633 nm", "Basis"], rows)
        sections.append(Section(f"Grade {g}", body))
    out.append(_note(records, "view.n-633-by-grade", "n at 633 nm by grade", "Views/n at 633 nm by grade.md",
                     "Every n-at-633-nm entry, grouped by grade.", sections, "view"))

    kinds: dict[str, list] = {}
    for m in mats:
        kinds.setdefault(m.density_evidence.kind, []).append(m)
    sections = []
    for kind in list(DENSITY_KIND_NAMES):
        group = kinds.get(kind, [])
        rows = [[link(m.path, m.name), num(m.density.value) if m.density else "—", code(m.density.label) if m.density else cell(m.density_evidence.reason)] for m in group]
        body = f"Density kind {kind} ({DENSITY_KIND_NAMES[kind]}): {len(rows)} materials."
        if rows:
            body += "\n\n" + table(["Material", "Density (g/cm3)", "Label or reason"], rows)
        sections.append(Section(f"Density: {kind}", body))
    out.append(_note(records, "view.densities-by-kind", "Densities by kind", "Views/Densities by kind.md",
                     "Every material by the kind of its density.", sections, "view"))

    fits = [(m, d) for m in mats for d in m.datasets if d.kind.kind == "model_fit"]
    out.append(_note(records, "view.model-fit-datasets", "Model-fit datasets", "Views/Model-fit datasets.md",
                     "Optical datasets that are model fits, not measured data.",
                     [Section("Model fits", f"{len(fits)} model-fit datasets.\n\n" + table(["Material", "Dataset", "Evidence"], [[link(m.path, m.name), code(d.label).replace("|", "\\|"), cell(d.kind.reason)] for m, d in fits]))], "view"))

    pairs = [(m, v) for m in mats for v in m.validations if v.classification != "excellent"]
    out.append(_note(records, "view.disagreeing-sources", "Disagreeing sources", "Views/Disagreeing sources.md",
                     "Dataset pairs the release classes warning or suspicious.",
                     [Section("Pairs", f"{len(pairs)} pairs classed warning or suspicious.\n\n" + table(
                         ["Material", "Property", "Dataset A", "Dataset B", "Class", "Mean rel. error"],
                         [[link(m.path, m.name), v.property_name, code(v.dataset_a).replace("|", "\\|"), code(v.dataset_b).replace("|", "\\|"), v.classification, num(v.mean_relative_error)] for m, v in pairs]))], "view"))

    out.append(_note(records, "view.release-notices", "Release notices", "Views/Release notices.md",
                     "Inconsistencies inside the release found by the build.",
                     [Section("Notices", "\n".join(f"- {cell(n)}" for n in records.notices) or "None.")], "view"))

    for p in records.properties.values():
        rows = [[link(records.materials[i].path, records.materials[i].name), cell(records.materials[i].formula)] for i in p.materials_with]
        out.append(_note(records, f"view.{p.name}-materials", f"{p.title} - materials", f"Views/{p.title} - materials.md",
                         f"Materials with a stored {p.title}.", [Section("Materials", f"{len(rows)} materials.\n\n" + table(["Material", "Formula"], rows))], "view"))
    return out


LICENCE = LICENCE_LINE
