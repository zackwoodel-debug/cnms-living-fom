"""Release -> in-memory records. The one place facts are assembled.

The vault and the index are both rendered from the ``Records`` this module
returns, never from each other, so a number in a note and the same number in the
index are one Python float read once from the release.

What is computed here, and what is not
--------------------------------------
Every value a note or the index shows is a stored release value: a database
column, a family-table cell, or a MANIFEST field. The atlas computes no number
it shows.

It does re-run one release computation: the n and k at 633 nm consensus
(materials-db ``scripts/release_validation.py``), to learn *which datasets
voted* — the release stores the median and the count, not the voters, and
independence (rubric 6.4) is a property of the voters' authors. The re-run must
reproduce every stored ``consensus_properties`` row exactly (value and count),
and every row it produces must be stored; otherwise the build stops. Its
interpolated values are compared and discarded, never published.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

import numpy as np

from cnms_fom.materials_release import (
    DatasetKindHelper,
    PhysicalKind,
    Release,
    ReleaseError,
    parse_optical_label,
    parse_physical_label,
    physical_kind,
)

from .authors import AuthorParse, GroupResult, author_groups, parse_authors
from .evidence import (
    REF_WL_NM,
    TAG_TEMPERATURE_TOLERANCE_C,
    DatasetKind,
    DensityEvidence,
    Grade,
    dataset_kind,
    density_evidence,
    grade_n633,
    is_ambient,
    k_status,
    material_k_status,
    tag_temperature_c,
    validation_relation,
)
from .evidence import fmt_number as _fmt
from .ids import (
    check_unique,
    dataset_id_for,
    family_id_for,
    material_id_for,
    property_id_for,
    safe_filename,
    short_title,
    source_id_for,
)

#  material_kind values (descriptor_json) for which a crystal phase is meaningful.
SOLID_KINDS = frozenset(
    {
        "inorganic compound",
        "element",
        "solid solution",
        "doped crystal or conducting oxide",
        "hybrid organic-inorganic perovskite",
    }
)

#  Physical quantity -> (display name, unit as the data dictionary states it).
QUANTITY_UNITS: dict[str, str | None] = {
    "density": "g/cm^3",
    "xray_sld_real": "1e-6 / Angstrom^2",
    "xray_sld_imag": "1e-6 / Angstrom^2",
    "neutron_sld_real": "1e-6 / Angstrom^2",
    "neutron_sld_imag": "1e-6 / Angstrom^2",
    "dielectric_static_total": None,
    "dielectric_electronic": None,
}


class BuildError(RuntimeError):
    """The release and the atlas disagree about a fact. The build stops."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class SourceRecord:
    source_id: int
    doi: str | None
    title: str | None
    authors: str | None
    journal: str | None
    year: int | None
    technique: str | None
    url: str | None
    notes: str | None
    parse: AuthorParse
    citable: bool
    citable_reason: str
    note_id: str = ""
    path: str = ""
    handle: str = ""
    cited_by: list[str] = field(default_factory=list)

    def citation(self) -> str:
        """The stored citation fields joined in a fixed order, verbatim."""
        parts = [p for p in (self.authors, self.title, self.journal) if p and p.strip()]
        text = ". ".join(part.strip().rstrip(".") for part in parts)
        if self.year is not None:
            text += f" ({self.year})"
        if self.doi:
            text += f". doi:{self.doi}"
        elif self.url:
            text += f". {self.url}"
        return text.strip(". ") or f"source {self.source_id}"


@dataclass
class DatasetRecord:
    material_id: int
    label: str
    phase: str | None
    tag: str
    axis: str | None
    dataset_id: str
    temperature_c: float | None
    source_id: int
    raw_record_table: str
    n_rows: int
    n_with_n: int
    n_with_k: int
    wavelength_min_nm: float
    wavelength_max_nm: float
    kind: DatasetKind
    k_status: str
    primary: bool
    covers_633_n: bool
    covers_633_k: bool
    #  The temperature the source tag's series suffix names, and any conflict with
    #  the stored temperature (a release defect, reported, never corrected here).
    tag_temperature_c: float | None = None
    temperature_note: str | None = None
    #  Stored rows at the nearest stored wavelengths at or either side of 633 nm:
    #  (wavelength_nm, n, k), exactly as stored. Empty when 633 nm is not covered.
    bracket_633: tuple[tuple[float, float | None, float | None], ...] = ()


@dataclass
class ConsensusEntry:
    """One n or k at 633 nm entry: one material, phase or variant, and axis."""

    quantity: str  # "n_633nm" | "k_633nm"
    property_name: str  # the release's name, e.g. "n_633nm | rutile | e-ray"
    phase: str | None
    axis: str | None
    datasets: list[str]
    voters: list[str]
    voter_sources: list[int]
    model_fits: list[str]
    other_temperatures: list[tuple[str, float | None]]
    value: float | None = None
    basis: str = "not_covered"
    value_origin: str | None = None
    std_dev: float | None = None
    num_sources: int | None = None
    confidence_score: float | None = None
    classification: str | None = None
    groups: GroupResult | None = None
    grade: Grade | None = None
    #  Voters whose label names a non-ambient temperature while their rows store none.
    contested: list[tuple[str, float]] = field(default_factory=list)

    @property
    def scope(self) -> str:
        """'rutile | e-ray', or 'no phase or axis stated'."""
        parts = [p for p in (self.phase, self.axis) if p]
        return " | ".join(parts) if parts else "no phase or axis stated"


@dataclass
class PhysicalRecord:
    record_id: int
    quantity: str
    value: float
    unit: str | None
    label: str
    phase: str | None
    origin: str | None
    method: str | None
    kind: PhysicalKind
    temperature_c: float | None
    frequency_hz: float | None
    wavelength_nm: float | None
    energy_ev: float | None
    source_id: int


@dataclass
class ValidationRecord:
    validation_id: int
    property_name: str
    dataset_a: str
    dataset_b: str
    pearson_r: float | None
    rmse: float | None
    mean_relative_error: float | None
    classification: str
    notes: str | None
    relation: str


@dataclass
class MaterialRecord:
    material_id: int
    key: str
    name: str
    family: str
    note_id: str
    path: str
    formula: str | None
    smiles: str | None
    inchikey: str | None
    molecular_weight: float | None
    cas: str | None
    pubchem_cid: int | None
    synonyms: list[str]
    original_names: list[str]
    material_kind: str | None
    material_class: str | None
    datasets: list[DatasetRecord]
    physical: list[PhysicalRecord]
    density: PhysicalRecord | None
    density_evidence: DensityEvidence
    n633: list[ConsensusEntry]
    k633: list[ConsensusEntry]
    validations: list[ValidationRecord]
    family_rows: list[tuple[str, dict[str, str]]]
    gaps: list[tuple[str, dict[str, str]]]
    k_status: str
    dielectric_withheld: str | None
    phase_unstated: bool
    cites: list[int]
    notices: list[str] = field(default_factory=list)

    @property
    def aliases(self) -> list[str]:
        out: list[str] = []
        for name in [*self.synonyms, *self.original_names]:
            if name not in out and name != self.name:
                out.append(name)
        return out

    @property
    def wavelength_range(self) -> tuple[float | None, float | None]:
        if not self.datasets:
            return None, None
        return (
            min(d.wavelength_min_nm for d in self.datasets),
            max(d.wavelength_max_nm for d in self.datasets),
        )

    @property
    def n_model_fits(self) -> int:
        return sum(d.kind.kind == "model_fit" for d in self.datasets)


@dataclass
class FamilyRecord:
    name: str
    note_id: str
    path: str
    members: list[int]
    table_rows: int
    counted_elsewhere: list[tuple[str, str]]  # (material name, family it is counted under)
    gaps_not_loaded: list[dict[str, str]]
    manifest_counts: dict


@dataclass
class PropertyRecord:
    name: str
    note_id: str
    path: str
    title: str
    definition: list[tuple[str, str]]  # (where, quoted text)
    unit: str | None
    distribution: dict[str, int]
    materials_with: list[int]


@dataclass
class Records:
    version: str
    sha256: str
    manifest: dict
    helper_provenance: dict
    data_dictionary: dict
    materials: dict[int, MaterialRecord]
    sources: dict[int, SourceRecord]
    families: dict[str, FamilyRecord]
    properties: dict[str, PropertyRecord]
    gap_candidates: list[tuple[str, dict[str, str]]]
    notices: list[str]

    @property
    def n_datasets(self) -> int:
        return sum(len(m.datasets) for m in self.materials.values())


# ---------------------------------------------------------------------------
# The release's consensus rule, re-run to learn the voters
# ---------------------------------------------------------------------------


def _series(rows, column: int) -> tuple[np.ndarray, np.ndarray]:
    """release_validation._series: repeated wavelengths averaged, missing values dropped."""
    acc: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        value = row[column]
        if value is not None:
            acc[row[0]].append(value)
    wavelengths = np.array(sorted(acc))
    return wavelengths, np.array([np.mean(acc[w]) for w in wavelengths])


def _value_at(rows, column: int, wl: float = REF_WL_NM) -> float | None:
    """release_validation.value_at. Used to identify voters and verify; never published."""
    w, v = _series(rows, column)
    return float(np.interp(wl, w, v)) if len(w) >= 2 and w[0] <= wl <= w[-1] else None


def _bracket(rows, column: int, wl: float = REF_WL_NM):
    """Stored rows at the nearest stored wavelengths at or either side of ``wl``."""
    with_value = [row for row in rows if row[column] is not None]
    below = [row[0] for row in with_value if row[0] <= wl]
    above = [row[0] for row in with_value if row[0] >= wl]
    if not below or not above:
        return ()
    targets = {max(below), min(above)}
    return tuple(
        (row[0], row[1], row[2]) for row in sorted(with_value, key=lambda r: (r[0], r[3])) if row[0] in targets
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _citable(row: dict, parse: AuthorParse) -> tuple[bool, str]:
    if row["doi"]:
        return True, f"resolved by DOI {row['doi']}"
    has_who = bool((row["authors"] or "").strip()) or parse.kind == "organisation"
    if row["year"] is not None and has_who and ((row["title"] or "").strip() or row["url"]):
        return True, "full citation (authors, title or URL, and year) without a DOI"
    missing = [
        name
        for name, ok in (
            ("DOI", False),
            ("year", row["year"] is not None),
            ("authors", has_who),
            ("title or URL", bool((row["title"] or "").strip() or row["url"])),
        )
        if not ok
    ]
    return False, "no " + ", no ".join(missing)


def build_records(release: Release, *, helper: DatasetKindHelper | None = None) -> Records:
    """Read the whole release into records. Raises BuildError/ReleaseError on any disagreement."""
    sha = release.verify_checksums()
    connection = release.connect()
    release.assert_manifest_counts(connection)
    helper = helper or DatasetKindHelper(release)
    primary_paths = helper.primary_pages()
    registry = release.registry()
    manifest = release.manifest
    dictionary = json.loads(release.file_text("data_dictionary.json"))
    notices: list[str] = []

    def rows(sql: str, *params):
        return [dict(r) for r in connection.execute(sql, params)]

    # --- sources ---------------------------------------------------------
    sources: dict[int, SourceRecord] = {}
    for row in rows("SELECT * FROM sources ORDER BY source_id"):
        parse = parse_authors(row["authors"], title=row["title"], url=row["url"])
        citable, why = _citable(row, parse)
        sources[row["source_id"]] = SourceRecord(
            source_id=row["source_id"],
            doi=row["doi"],
            title=row["title"],
            authors=row["authors"],
            journal=row["journal"],
            year=row["year"],
            technique=row["technique"],
            url=row["url"],
            notes=row["notes"],
            parse=parse,
            citable=citable,
            citable_reason=why,
        )

    # --- materials, synonyms, descriptors ---------------------------------
    base = {r["material_id"]: r for r in rows("SELECT * FROM materials ORDER BY material_id")}
    synonyms: dict[int, list[str]] = defaultdict(list)
    for r in rows("SELECT material_id, synonym FROM material_synonyms ORDER BY material_id, synonym"):
        synonyms[r["material_id"]].append(r["synonym"])
    descriptors = {
        r["material_id"]: json.loads(r["descriptor_json"] or "{}")
        for r in rows("SELECT material_id, descriptor_json FROM chemical_descriptors")
    }
    renamed = {
        entry["key"]: entry["registered_name"]
        for entry in manifest.get("material_ids", {}).get("name_changes", [])
    }
    dielectric_withheld: dict[str, str] = {}
    dielectric = manifest.get("dielectric", {})
    for entry in dielectric.get("not_stored_reference_only", []):
        dielectric_withheld[entry["material"]] = (
            f"its Materials Project entry {entry['mp_id']} is only a {entry['applies_to']}"
        )
    for entry in dielectric.get("not_stored_narrow_gap", []):
        dielectric_withheld[entry["material"]] = (
            f"its Materials Project band gap ({entry['mp_band_gap_ev']} eV) is below the release's "
            f"{dielectric.get('min_gap_ev')} eV cut-off for DFPT values"
        )

    # --- optical rows -> datasets ----------------------------------------------
    grouped: dict[tuple[int, str], list[tuple]] = defaultdict(list)
    meta: dict[tuple[int, str], dict] = {}
    for record_id, mid, label, wl, n, k, t, sid, table in connection.execute(
        "SELECT record_id, material_id, dataset_label, wavelength_nm, n, k, temperature_c, source_id, "
        "raw_record_table FROM optical_dispersion ORDER BY material_id, dataset_label, wavelength_nm, record_id"
    ):
        grouped[(mid, label)].append((wl, n, k, record_id))
        info = meta.setdefault((mid, label), {"temps": set(), "sources": set(), "tables": set()})
        info["temps"].add(t)
        info["sources"].add(sid)
        info["tables"].add(table)

    note_ids = {mid: material_id_for(entry.key) for mid, entry in registry.items()}
    datasets_by_material: dict[int, list[DatasetRecord]] = defaultdict(list)
    for (mid, label), drows in grouped.items():
        info = meta[(mid, label)]
        for what in ("temps", "sources", "tables"):
            if len(info[what]) != 1:
                raise BuildError(f"dataset {mid} {label!r} has {len(info[what])} {what}; a dataset has one")
        parsed = parse_optical_label(label)
        table = next(iter(info["tables"]))
        name = registry[mid].name
        primary_path = primary_paths.get(name, (None, None))[1]
        wavelengths = [r[0] for r in drows]
        n_with_n = sum(r[1] is not None for r in drows)
        n_with_k = sum(r[2] is not None for r in drows)
        covers_n = _value_at(drows, 1) is not None
        covers_k = _value_at(drows, 2) is not None
        stored_t = next(iter(info["temps"]))
        tag_t = tag_temperature_c(parsed.tag)
        temperature_note = None
        if tag_t is not None and stored_t is None:
            temperature_note = f"the label names {_fmt(tag_t)} degC but its rows store no temperature"
        elif tag_t is not None and abs(stored_t - tag_t) > TAG_TEMPERATURE_TOLERANCE_C:
            temperature_note = f"the label names {_fmt(tag_t)} degC but its rows store {_fmt(stored_t)} degC"
        datasets_by_material[mid].append(
            DatasetRecord(
                material_id=mid,
                label=label,
                phase=parsed.phase,
                tag=parsed.tag,
                axis=parsed.axis,
                dataset_id=dataset_id_for(note_ids[mid], label),
                temperature_c=stored_t,
                source_id=next(iter(info["sources"])),
                raw_record_table=table,
                n_rows=len(drows),
                n_with_n=n_with_n,
                n_with_k=n_with_k,
                wavelength_min_nm=min(wavelengths),
                wavelength_max_nm=max(wavelengths),
                kind=dataset_kind(helper.model_fit_reason(table)),
                k_status=k_status(len(drows), n_with_k),
                primary=(table == primary_path),
                covers_633_n=covers_n,
                covers_633_k=covers_k,
                tag_temperature_c=tag_t,
                temperature_note=temperature_note,
                bracket_633=_bracket(drows, 1) if covers_n else (),
            )
        )

    # --- consensus: re-run to learn voters, verify against the stored rows -----------
    stored_consensus = {
        (r["material_id"], r["property_name"]): r for r in rows("SELECT * FROM consensus_properties")
    }
    produced: set[tuple[int, str]] = set()
    consensus_by_material: dict[int, dict[str, list[ConsensusEntry]]] = defaultdict(lambda: {"n": [], "k": []})
    for mid in sorted(datasets_by_material):
        groups: dict[tuple, list[DatasetRecord]] = defaultdict(list)
        for d in datasets_by_material[mid]:
            groups[(d.phase, d.axis)].append(d)
        for (phase, axis), members in sorted(groups.items(), key=lambda x: (x[0][0] or "", x[0][1] or "")):
            members = sorted(members, key=lambda d: d.label)
            for column, quantity, letter in ((1, "n_633nm", "n"), (2, "k_633nm", "k")):
                covers = (lambda d: d.covers_633_n) if letter == "n" else (lambda d: d.covers_633_k)
                voters = [d for d in members if d.kind.kind == "measured" and is_ambient(d.temperature_c) and covers(d)]
                name = " | ".join(x for x in (quantity, phase, axis) if x)
                entry = ConsensusEntry(
                    quantity=quantity,
                    property_name=name,
                    phase=phase,
                    axis=axis,
                    datasets=[d.label for d in members],
                    voters=[d.label for d in voters],
                    voter_sources=sorted({d.source_id for d in voters}),
                    model_fits=[d.label for d in members if d.kind.kind == "model_fit" and is_ambient(d.temperature_c) and covers(d)],
                    other_temperatures=[(d.label, d.temperature_c) for d in members if not is_ambient(d.temperature_c) and covers(d)],
                )
                stored = stored_consensus.get((mid, name))
                if voters:
                    votes = [_value_at(grouped[(mid, d.label)], column) for d in voters]
                    median = float(np.median(votes))
                    if stored is None or stored["num_sources"] != len(votes) or stored["consensus_value"] != median:
                        raise BuildError(
                            f"consensus re-run does not reproduce {mid} {name!r}: stored "
                            f"{None if stored is None else (stored['consensus_value'], stored['num_sources'])}, "
                            f"re-run ({median}, {len(votes)})"
                        )
                    produced.add((mid, name))
                    entry.value = stored["consensus_value"]
                    entry.basis = "consensus"
                    entry.value_origin = "consensus_properties"
                    entry.std_dev = stored["std_dev"]
                    entry.num_sources = stored["num_sources"]
                    entry.confidence_score = stored["confidence_score"]
                    entry.classification = stored["classification"]
                    if len(entry.voter_sources) >= 1 and len(voters) >= 2:
                        entry.groups = author_groups({sid: sources[sid].parse for sid in entry.voter_sources})
                    entry.contested = [
                        (d.label, d.tag_temperature_c)
                        for d in voters
                        if d.temperature_c is None
                        and d.tag_temperature_c is not None
                        and not is_ambient(d.tag_temperature_c)
                    ]
                elif stored is not None:
                    raise BuildError(f"stored consensus {mid} {name!r} has no voters in the re-run")
                if letter == "n" and not voters and not entry.model_fits and not entry.other_temperatures and not any(covers(d) for d in members):
                    entry.basis = "not_covered"
                consensus_by_material[mid][letter].append(entry)
    unproduced = set(stored_consensus) - produced
    if unproduced:
        raise BuildError(f"{len(unproduced)} stored consensus rows were not reproduced: {sorted(unproduced)[:3]}")

    # --- physical values ------------------------------------------------------------
    physical_by_material: dict[int, list[PhysicalRecord]] = defaultdict(list)
    for r in rows("SELECT * FROM physical_properties ORDER BY material_id, record_id"):
        label = parse_physical_label(r["dataset_label"])
        column = {
            "density": "density_g_cm3",
            "xray_sld_real": "xray_sld",
            "xray_sld_imag": "xray_sld",
            "neutron_sld_real": "neutron_sld",
            "neutron_sld_imag": "neutron_sld",
            "dielectric_static_total": "dielectric_constant",
            "dielectric_electronic": "dielectric_constant",
        }[label.quantity]
        populated = [c for c in ("density_g_cm3", "xray_sld", "neutron_sld", "dielectric_constant") if r[c] is not None]
        if populated != [column]:
            raise BuildError(f"physical row {r['record_id']} label {label.raw!r} fills {populated}, expected {column}")
        physical_by_material[r["material_id"]].append(
            PhysicalRecord(
                record_id=r["record_id"],
                quantity=label.quantity,
                value=r[column],
                unit=QUANTITY_UNITS[label.quantity],
                label=label.raw,
                phase=label.phase,
                origin=label.origin,
                method=label.method,
                kind=physical_kind(label),
                temperature_c=r["temperature_c"],
                frequency_hz=r["frequency_hz"],
                wavelength_nm=r["wavelength_nm"],
                energy_ev=r["energy_ev"],
                source_id=r["source_id"],
            )
        )

    # --- family tables and gap lists ------------------------------------------------------
    family_rows: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    table_sizes: dict[str, int] = {}
    for family in release.families():
        table = release.family_table(family)
        table_sizes[family] = len(table)
        for row in table:
            family_rows[row["name"]].append((family, row))
    names_to_id = {entry.name: mid for mid, entry in registry.items()}
    gaps_by_material: dict[int, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    gap_candidates: list[tuple[str, dict[str, str]]] = []
    for family, gap_rows in release.gap_tables().items():
        for row in gap_rows:
            mid = names_to_id.get(row.get("name", ""))
            if mid is not None:
                gaps_by_material[mid].append((family, row))
            else:
                gap_candidates.append((family, row))

    # --- validation pairs -----------------------------------------------------------------
    validations: dict[int, list[ValidationRecord]] = defaultdict(list)
    for r in rows("SELECT * FROM dataset_validation ORDER BY validation_id"):
        labels = {d.label for d in datasets_by_material[r["material_id"]]}
        if r["dataset_a"] not in labels or r["dataset_b"] not in labels:
            raise BuildError(f"validation {r['validation_id']} names a dataset the material does not have")
        validations[r["material_id"]].append(
            ValidationRecord(
                validation_id=r["validation_id"],
                property_name=r["property_name"],
                dataset_a=r["dataset_a"],
                dataset_b=r["dataset_b"],
                pearson_r=r["pearson_r"],
                rmse=r["rmse"],
                mean_relative_error=r["mean_relative_error"],
                classification=r["classification"],
                notes=r["notes"],
                relation=validation_relation(r["classification"]),
            )
        )

    # --- materials ----------------------------------------------------------------------------
    materials: dict[int, MaterialRecord] = {}
    for mid, entry in sorted(registry.items()):
        row = base[mid]
        name = entry.name
        datasets = sorted(datasets_by_material.get(mid, []), key=lambda d: d.label)
        physical = physical_by_material.get(mid, [])
        densities = [p for p in physical if p.quantity == "density"]
        if len(densities) > 1:
            raise BuildError(f"material {mid} has {len(densities)} density rows; the atlas expects at most one")
        density = densities[0] if densities else None
        gaps = gaps_by_material.get(mid, [])
        if density is None:
            reasons = [g["reason"] for _, g in gaps if g.get("gap_kind") == "density" and g.get("reason")]
            if not reasons and any(r.get("flags") for _, r in family_rows.get(name, [])):
                reasons = ["no gap-list entry; the family-table flags are quoted under Flags and gaps"]
            evidence = density_evidence(None, present=False, absent_reason="; ".join(reasons) or None)
        else:
            evidence = density_evidence(density.origin, present=True)
        rows_here = family_rows.get(name, [])
        if not rows_here:
            raise BuildError(f"material {mid} {name!r} is in no family table")
        own_family_rows = [(f, r) for f, r in rows_here if f == entry.family]
        if not own_family_rows:
            raise BuildError(f"material {mid} {name!r} is not in its own family table {entry.family}")
        descriptor = descriptors.get(mid, {})
        n633 = consensus_by_material[mid]["n"]
        k633 = consensus_by_material[mid]["k"]
        notices_here: list[str] = []

        #  n at 633 nm entries without a consensus: a stored scalar exists only where the
        #  family table names the primary dataset's n_633.
        primary = [d for d in datasets if d.primary]
        for e in n633:
            if e.basis == "consensus":
                e.grade = grade_n633(
                    consensus_class=e.classification,
                    n_voting=e.num_sources or 0,
                    n_voting_sources=len(e.voter_sources),
                    groups=e.groups,
                    contested=e.contested,
                )
                continue
            e.grade = grade_n633(
                consensus_class=None,
                ambient_model_fits=e.model_fits,
                other_temperatures=e.other_temperatures,
            )
            if e.grade.grade == "D":
                in_group = [d for d in primary if d.label in e.model_fits]
                own_row = own_family_rows[0][1]
                if in_group and own_row.get("n_633"):
                    e.value = float(own_row["n_633"])
                    e.basis = "primary_dataset_point"
                    e.value_origin = f"family_tables/{entry.family}.csv n_633 (primary dataset {in_group[0].label})"
                else:
                    e.basis = "model_fit_only"
            elif e.grade.grade == "T":
                e.basis = "other_temperature_only"
            else:
                e.basis = "not_covered"

        density_source_col = own_family_rows[0][1].get("density_source", "")
        if density is None and density_source_col == "literature":
            notices_here.append(
                f"family_tables/{entry.family}.csv gives density_source 'literature' for this material, "
                "but the database has no density row for it"
            )

        for d in datasets:
            if d.temperature_note:
                notices_here.append(f"dataset {d.label!r}: {d.temperature_note}")
        for e in [*n633, *k633]:
            for label, t in e.contested:
                notices_here.append(
                    f"{e.property_name}: the release counted {label!r} as ambient, but its label names "
                    f"{_fmt(t)} degC; the stored consensus mixes temperatures"
                )

        cites = sorted({d.source_id for d in datasets} | {p.source_id for p in physical})
        materials[mid] = MaterialRecord(
            material_id=mid,
            key=entry.key,
            name=name,
            family=entry.family,
            note_id=note_ids[mid],
            path="",
            formula=row["formula"],
            smiles=row["smiles"],
            inchikey=row["inchikey"],
            molecular_weight=row["molecular_weight"],
            cas=row["cas_number"],
            pubchem_cid=row["pubchem_cid"],
            synonyms=synonyms.get(mid, []),
            original_names=[renamed[entry.key]] if entry.key in renamed else [],
            material_kind=descriptor.get("material_kind"),
            material_class=descriptor.get("material_class"),
            datasets=datasets,
            physical=physical,
            density=density,
            density_evidence=evidence,
            n633=n633,
            k633=k633,
            validations=validations.get(mid, []),
            family_rows=rows_here,
            gaps=gaps,
            k_status=material_k_status([d.k_status for d in datasets]),
            dielectric_withheld=dielectric_withheld.get(name),
            phase_unstated=(
                descriptor.get("material_kind") in SOLID_KINDS
                and entry.polymorph is None
                and not any(d.phase for d in datasets)
            ),
            cites=cites,
            notices=notices_here,
        )
        for sid in cites:
            sources[sid].cited_by.append(note_ids[mid])
        notices.extend(f"{name} (id {mid}): {text}" for text in notices_here)

    missing_withheld = sorted(set(dielectric_withheld) - {m.name for m in materials.values()})
    if missing_withheld:
        raise BuildError(f"MANIFEST dielectric lists names no material has: {missing_withheld}")

    _assign_paths(materials, sources)
    families = _families(materials, registry, table_sizes, family_rows, gap_candidates, manifest)
    properties = _properties(materials, dictionary)

    return Records(
        version=release.version,
        sha256=sha,
        manifest=manifest,
        helper_provenance=helper.provenance,
        data_dictionary=dictionary,
        materials=materials,
        sources=sources,
        families=families,
        properties=properties,
        gap_candidates=gap_candidates,
        notices=notices,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _assign_paths(materials: dict[int, MaterialRecord], sources: dict[int, SourceRecord]) -> None:
    for m in materials.values():
        filename = safe_filename(m.name)
        if filename != m.name and m.name not in m.original_names:
            m.original_names.append(m.name)
        if m.family == "glass_catalogs":
            maker = safe_filename(m.name.split()[0])
            m.path = str(PurePosixPath("Materials", "Glass catalogs", maker, f"{filename}.md"))
        else:
            m.path = str(PurePosixPath("Materials", m.family, f"{filename}.md"))
    check_unique([(str(m.material_id), m.note_id) for m in materials.values()], "material id")
    check_unique([(str(m.material_id), m.path) for m in materials.values()], "material path")

    for s in sources.values():
        s.note_id = source_id_for(s.source_id)
        first = None
        if s.parse.kind == "people" and s.parse.people:
            first = _display_surname(s.authors or "", s.parse.people[0].surname)
        elif s.parse.kind == "organisation":
            first = s.parse.organisation
        handle = " ".join(x for x in (first or f"Source {s.source_id}", str(s.year) if s.year else "") if x)
        s.handle = handle
        title = short_title(s.title) or short_title(s.authors if not first else None)
        stem = safe_filename(f"{handle} - {title}") if title else safe_filename(handle)
        s.path = str(PurePosixPath("Sources", f"{stem}.md"))
    #  A collision gets the source id appended: deterministic, and stable for the sources involved.
    by_path: dict[str, list[SourceRecord]] = defaultdict(list)
    for s in sources.values():
        by_path[s.path.casefold()].append(s)
    for group in by_path.values():
        if len(group) > 1:
            for s in group:
                s.path = s.path[: -len(".md")] + f" (source {s.source_id}).md"
    check_unique([(str(s.source_id), s.path) for s in sources.values()], "source path")


def _display_surname(authors: str, folded: str) -> str:
    """The first author's surname as written ('Schröter'), found by its folded form."""
    for word in authors.replace(",", " ").replace(";", " ").split():
        bare = word.strip(".")
        plain = "".join(ch for ch in unicodedata.normalize("NFKD", bare) if not unicodedata.combining(ch)).lower()
        if plain.replace("-", "").replace("'", "").replace("’", "") == folded:
            return safe_filename(bare)
    return folded.capitalize()


# ---------------------------------------------------------------------------
# Families and properties
# ---------------------------------------------------------------------------


def _families(materials, registry, table_sizes, family_rows, gap_candidates, manifest) -> dict[str, FamilyRecord]:
    out: dict[str, FamilyRecord] = {}
    by_family: dict[str, list[int]] = defaultdict(list)
    family_of_name = {m.name: m.family for m in materials.values()}
    for m in materials.values():
        by_family[m.family].append(m.material_id)
    for family in sorted(table_sizes):
        #  Rows of this table whose material the release counts under another family
        #  ("a material in two family tables is counted once, under the first").
        elsewhere = sorted(
            (name, family_of_name[name])
            for name, rows_ in family_rows.items()
            if any(f == family for f, _ in rows_) and family_of_name.get(name) not in (family, None)
        )
        out[family] = FamilyRecord(
            name=family,
            note_id=family_id_for(family),
            path=str(PurePosixPath("Families", f"{family}.md")),
            members=sorted(by_family.get(family, []), key=lambda i: (materials[i].name, i)),
            table_rows=table_sizes[family],
            counted_elsewhere=elsewhere,
            gaps_not_loaded=[row for f, row in gap_candidates if f == family],
            manifest_counts=manifest.get("counts", {}).get("per_family", {}).get(family, {}),
        )
    return out


def has_properties(m: MaterialRecord) -> list[str]:
    """Property notes this material has a stored value for (k status: every material has one).

    The single definition behind a material's ``has_property`` list and each
    property's ``materials_with`` list, so the typed link and its reverse agree.
    """
    out = []
    if any(e.value is not None for e in m.n633):
        out.append("n_633nm")
    if any(e.classification for e in m.k633):
        out.append("k_633nm")
    out.append("k_status")
    if m.density is not None:
        out.append("density")
    for name in ("xray_sld", "neutron_sld", "dielectric"):
        if any(p.quantity.startswith(name) for p in m.physical):
            out.append(name)
    return out


PROPERTY_SPECS: tuple[tuple[str, str], ...] = (
    ("n at 633 nm", "n_633nm"),
    ("k at 633 nm", "k_633nm"),
    ("k status", "k_status"),
    ("density", "density"),
    ("X-ray SLD", "xray_sld"),
    ("neutron SLD", "neutron_sld"),
    ("dielectric constant", "dielectric"),
)


def _column(dictionary: dict, table: str, column: str) -> dict:
    for col in dictionary["tables"][table]["columns"]:
        if col["name"] == column:
            return col
    raise ReleaseError(f"data_dictionary.json has no {table}.{column}")


def _properties(materials: dict[int, MaterialRecord], dictionary: dict) -> dict[str, PropertyRecord]:
    out: dict[str, PropertyRecord] = {}
    for title, name in PROPERTY_SPECS:
        definition: list[tuple[str, str]] = []
        unit: str | None = None
        distribution: dict[str, int] = defaultdict(int)
        with_it: list[int] = []
        if name in ("n_633nm", "k_633nm"):
            definition.append(("consensus_properties", dictionary["tables"]["consensus_properties"]["description"]))
            for col in ("property_name", "consensus_value", "classification"):
                definition.append((f"consensus_properties.{col}", _column(dictionary, "consensus_properties", col)["description"]))
            for m in materials.values():
                entries = m.n633 if name == "n_633nm" else m.k633
                for e in entries:
                    key = (e.grade.grade if e.grade else e.classification) if name == "n_633nm" else (e.classification or "no consensus")
                    if name == "k_633nm" and e.classification is None:
                        continue
                    distribution[key] += 1
                if any(e.value is not None for e in entries):
                    with_it.append(m.material_id)
        elif name == "k_status":
            col = _column(dictionary, "optical_dispersion", "k")
            definition.append(("optical_dispersion.k", col["description"]))
            for m in materials.values():
                distribution[m.k_status] += 1
                if m.k_status != "not_measured":
                    with_it.append(m.material_id)
        elif name == "density":
            col = _column(dictionary, "physical_properties", "density_g_cm3")
            definition.append(("physical_properties.density_g_cm3", col["description"]))
            definition.append(("label grammar", dictionary["label_grammar"]["physical_properties.dataset_label"]))
            unit = col["unit"]
            for m in materials.values():
                distribution[m.density_evidence.kind] += 1
                if m.density is not None:
                    with_it.append(m.material_id)
        else:
            column = {"xray_sld": "xray_sld", "neutron_sld": "neutron_sld", "dielectric": "dielectric_constant"}[name]
            col = _column(dictionary, "physical_properties", column)
            definition.append((f"physical_properties.{column}", col["description"]))
            unit = col["unit"]
            for m in materials.values():
                quantities = {p.quantity for p in m.physical if p.quantity.startswith(name)}
                if quantities:
                    with_it.append(m.material_id)
                    for q in sorted(quantities):
                        distribution[q] += 1
        with_it = sorted(m.material_id for m in materials.values() if name in has_properties(m))
        out[name] = PropertyRecord(
            name=name,
            note_id=property_id_for(name),
            path=str(PurePosixPath("Properties", f"{title}.md")),
            title=title,
            definition=definition,
            unit=unit,
            distribution=dict(sorted(distribution.items())),
            materials_with=sorted(with_it),
        )
    return out


def subset(records: Records, keep: set[int]) -> Records:
    """The same records restricted to some materials, with every reverse list restricted too.

    Used for the Phase 2 sample: a sample vault must be internally consistent (no
    link to a note it does not contain), so families, sources and properties are
    recomputed over the kept materials rather than copied whole.
    """
    materials = {i: m for i, m in records.materials.items() if i in keep}
    kept_notes = {m.note_id for m in materials.values()}
    cited = {sid for m in materials.values() for sid in m.cites}
    sources = {
        sid: replace(s, cited_by=[n for n in s.cited_by if n in kept_notes])
        for sid, s in records.sources.items()
        if sid in cited
    }
    families = {
        name: replace(f, members=[i for i in f.members if i in keep])
        for name, f in records.families.items()
        if any(m.family == name for m in materials.values())
    }
    names = {m.name for m in materials.values()}
    return replace(
        records,
        materials=materials,
        sources=sources,
        families=families,
        properties=_properties(materials, records.data_dictionary),
        notices=[n for n in records.notices if n.split(" (id ")[0] in names],
    )
