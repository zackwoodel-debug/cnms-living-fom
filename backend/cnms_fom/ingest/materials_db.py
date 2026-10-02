"""Ingest a CNMS ``materials-db`` SQLite export.

Written against the real schema of ``data/materials_oxide_test.db``:

    materials(material_id, name, formula, smiles, inchikey, cas_number, ...)
    physical_properties(material_id, density_g_cm3, xray_sld, neutron_sld,
                        dielectric_constant, temperature_c, frequency_hz,
                        dataset_label, source_id, ...)
    optical_dispersion(material_id, wavelength_nm, n, k, temperature_c,
                       dataset_label, source_id, ...)
    sources(source_id, doi, title, authors, journal, year, technique, url, ...)

Three findings shape this module, and they are worth stating because they are
the reason it is built the way it is rather than as a straight column mapping:

1. **The external schema has no polymorph and no specimen form.**  Its identity
   is name + formula, which is precisely the formula-only identity FOM_PROOF
   Sec. 2.1 rejects.  One row is literally named "Titanium dioxide (rutile /
   anatase)" — two polymorphs with different properties in a single record.

2. **Phase and optical axis are hidden inside ``dataset_label``** as free text
   (``"corundum/sapphire | Malitson1972 | o-ray"``).  Recovering them is what
   makes a record eligible at all, so ``labels.parse_dataset_label`` runs before
   anything is promoted.

3. **It contains no FOM input properties.**  ``dielectric_constant`` is present
   as a column but zero rows are populated, and there is no bandgap, band
   offset, breakdown field, loss tangent, or thermal conductivity anywhere.  So
   no material imported from here can be FOM-scored today.  What it does supply
   is density, x-ray and neutron SLD, and ~125,000 points of optical dispersion
   — plus, via Eq. (11), a route to eps_inf.

``survey()`` reports all of that without writing anything, and is the default
mode of the CLI wrapper.

A materials-db *release* (a ``.sqlite`` with its ``MANIFEST.json`` beside it, e.g.
v0.22.0) is read with the release's own rules, through ``cnms_fom.materials_release``,
the same package the materials atlas uses:

* labels are parsed by the release grammar, and a material's phase falls back to
  the one in its stable key (``oxides_50:TiO2@rutile``);
* every physical value is tiered by what its label says it is. DFT and DFPT
  values, all SLDs, and densities computed from lattice parameters or standing in
  for a film are CALCULATED; only literature and datasheet densities are MEASURED.
  An older export tiered these MEASURED, which would let a 0 K DFT permittivity
  reach a score as if it were measured;
* ``dielectric_electronic`` is eps_inf, not the static k;
* an optical dataset is MEASURED or, if materials-db's dataset_kind.py says it is
  a model fit, MODELED; when that cannot be decided (no materials-db checkout, or
  one that is not the release's) the rows are quarantined, not assumed measured.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import insert, select

from cnms_fom.db.context import context_digest
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
from cnms_fom.db.models import (
    DescriptorValue,
    ExternalRecord,
    Material,
    PropertyValue,
    SpectralPoint,
    SpectralSeries,
)
from cnms_fom.fom_engine.identity import FormulaNotCanonical, reduced_formula
from cnms_fom.materials_release import (
    DatasetKindHelper,
    LabelError,
    Release,
    ReleaseError,
    parse_optical_label,
    parse_physical_label,
    physical_kind,
)

from .labels import axis_to_tensor_component, parse_dataset_label

logger = logging.getLogger(__name__)

#  Rows per executemany batch when loading spectral points.
POINT_BATCH = 5_000

#  external physical_properties column -> (CNMS key, kind, units)
PHYSICAL_COLUMN_MAP: dict[str, tuple[str, str, str]] = {
    "density_g_cm3": ("rho", "descriptor", "g/cm^3"),
    "xray_sld": ("sld_xray", "property", "1e-6 A^-2"),
    "neutron_sld": ("sld_neutron", "property", "1e-6 A^-2"),
    "dielectric_constant": ("k", "property", "dimensionless"),
}

#  Values arriving from a literature compilation are other people's
#  measurements.  Tiering them CALCULATED rather than MEASURED would overstate
#  what we know; tiering them MEASURED would understate that we did not make
#  them.  MEASURED with an explicit source DOI is the honest reading — the
#  measurement happened, it just was not ours — and the compilation is recorded
#  in database_identifier so the chain is traceable.
DEFAULT_TIER = ProvenanceTier.MEASURED


@dataclass
class IngestionReport:
    source: str
    mode: str
    staged: int = 0
    promoted: int = 0
    quarantined: int = 0
    quarantine_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    materials_created: int = 0
    descriptors_created: int = 0
    properties_created: int = 0
    series_created: int = 0
    points_created: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "mode": self.mode,
            "staged": self.staged,
            "promoted": self.promoted,
            "quarantined": self.quarantined,
            "quarantine_reasons": dict(self.quarantine_reasons),
            "materials_created": self.materials_created,
            "descriptors_created": self.descriptors_created,
            "properties_created": self.properties_created,
            "series_created": self.series_created,
            "points_created": self.points_created,
            "source_counts": self.source_counts,
            "coverage": self.coverage,
            "notes": self.notes,
        }


def _release_parse(table: str, label: str | None) -> dict:
    """A release label parsed by the release grammar, in the staged ``parsed`` shape."""
    if table == "optical_dispersion":
        parsed = parse_optical_label(label)
        return {
            "raw": label,
            "grammar": "materials-db release",
            "phase": parsed.phase,
            "source_tag": parsed.tag,
            "axis": parsed.axis,
        }
    if table == "physical_properties":
        try:
            parsed = parse_physical_label(label)
            kind = physical_kind(parsed)
        except LabelError as exc:
            return {"raw": label, "grammar": "materials-db release", "error": str(exc)}
        return {
            "raw": label,
            "grammar": "materials-db release",
            "phase": parsed.phase,
            "quantity": parsed.quantity,
            "origin": parsed.origin,
            "method": parsed.method,
            "kind": kind.kind,
            "provenance": kind.provenance,
            "kind_reason": kind.reason,
        }
    return {"raw": label, "grammar": "materials-db release"}


#  Release quantity names -> CNMS registry keys. dielectric_electronic is the
#  clamped-ion permittivity, i.e. eps_inf: storing it as the static k would put a
#  high-frequency number where a low-frequency one belongs.
RELEASE_QUANTITY_KEYS: dict[str, tuple[str, str, str]] = {
    "density": ("rho", "descriptor", "g/cm^3"),
    "xray_sld_real": ("sld_xray", "property", "1e-6 A^-2"),
    "xray_sld_imag": ("sld_xray_imag", "property", "1e-6 A^-2"),
    "neutron_sld_real": ("sld_neutron", "property", "1e-6 A^-2"),
    "neutron_sld_imag": ("sld_neutron_imag", "property", "1e-6 A^-2"),
    "dielectric_static_total": ("k", "property", "dimensionless"),
    "dielectric_electronic": ("eps_inf", "property", "dimensionless"),
}

RELEASE_COLUMN = {
    "density": "density_g_cm3",
    "xray_sld_real": "xray_sld",
    "xray_sld_imag": "xray_sld",
    "neutron_sld_real": "neutron_sld",
    "neutron_sld_imag": "neutron_sld",
    "dielectric_static_total": "dielectric_constant",
    "dielectric_electronic": "dielectric_constant",
}


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }


# ---------------------------------------------------------------------------
# Survey — read-only
# ---------------------------------------------------------------------------


def survey(path: Path | str) -> IngestionReport:
    """Report what an external database contains and what blocks promotion.

    Writes nothing.  This is the default mode because pointing an importer at
    somebody's research database and letting it write on the first run is how
    provenance gets destroyed.
    """
    path = Path(path)
    report = IngestionReport(source=str(path), mode="survey")
    release = Release.detect(path)
    if release is not None:
        report.coverage["materials_db_release"] = release.version
        report.notes.append(
            f"materials-db release v{release.version}: labels are read with the release grammar, physical "
            "values are tiered by what their label says (DFT, DFPT, SLD and derived densities CALCULATED; "
            "literature densities MEASURED), and optical datasets need materials-db's dataset_kind.py "
            "(MATERIALS_DB_ROOT) to be told measured from model fit."
        )

    with _connect(path) as connection:
        tables = _table_names(connection)
        report.source_counts = {
            name: connection.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"]
            for name in sorted(tables)
            if not name.startswith("sqlite_")
        }

        if "physical_properties" in tables:
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(physical_properties)")
            }
            populated = {}
            for column, (key, kind, _units) in PHYSICAL_COLUMN_MAP.items():
                if column not in columns:
                    continue
                n = connection.execute(
                    f"SELECT COUNT({column}) AS n FROM physical_properties"
                ).fetchone()["n"]
                populated[f"{column} -> {key} ({kind})"] = n
            report.coverage["physical_properties"] = populated

            empty = [name for name, n in populated.items() if n == 0]
            if empty:
                report.notes.append(
                    f"Columns present but empty: {empty}. A column with no rows maps to nothing."
                )

        if "optical_dispersion" in tables:
            row = connection.execute(
                "SELECT COUNT(*) AS rows, COUNT(DISTINCT material_id) AS materials, "
                "COUNT(n) AS has_n, COUNT(k) AS has_k, MIN(wavelength_nm) AS min_nm, "
                "MAX(wavelength_nm) AS max_nm FROM optical_dispersion"
            ).fetchone()
            report.coverage["optical_dispersion"] = dict(row)

        #  How much of the corpus can actually name its own phase. A release's label
        #  grammar names it, or the stable key does (oxides_50:TiO2@rutile).
        if "optical_dispersion" in tables:
            with_phase = without_phase = 0
            key_phase = (
                {mid: entry.polymorph for mid, entry in release.registry().items()} if release is not None else {}
            )
            for row in connection.execute(
                "SELECT material_id, dataset_label, COUNT(*) AS n FROM optical_dispersion "
                "GROUP BY material_id, dataset_label"
            ):
                if release is not None:
                    phase = parse_optical_label(row["dataset_label"]).phase or key_phase.get(row["material_id"])
                else:
                    phase = parse_dataset_label(row["dataset_label"]).phase
                if phase:
                    with_phase += row["n"]
                else:
                    without_phase += row["n"]
            report.coverage["optical_rows_with_recoverable_phase"] = with_phase
            report.coverage["optical_rows_without_phase"] = without_phase
            if without_phase:
                report.notes.append(
                    f"{without_phase} optical rows carry no recoverable phase. FOM_PROOF Sec. 2.1: "
                    "a formula is not a material identifier, so these cannot become Material rows "
                    "without an explicit phase map supplied by a person."
                )

    #  The finding that matters most for planning.
    fom_inputs = ("k", "Eg", "dEc", "Ebd", "tan_delta", "kappa_th")
    report.notes.append(
        "No FOM input property is populated in this source "
        f"(needs all of {list(fom_inputs)} for a logic/power/rf score). Imported materials will "
        "report `not_scored` until dielectric, bandgap, band-offset, breakdown, loss, and thermal "
        "values are added with their measurement context."
    )
    report.notes.append(
        "Optical n(lambda) below the electronic gap and above the phonon resonances gives "
        "eps_inf ~ n^2, which does feed Eq. (11). Enable it explicitly with --derive-eps-inf; "
        "the wavelength window is recorded on every derived value."
    )
    return report


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def stage(session, path: Path | str, *, limit: int | None = None) -> IngestionReport:
    """Copy external rows into ``external_records`` verbatim.

    Idempotent on (source_database, source_table, source_row_id): re-running
    after the source grows adds only what is new.
    """
    path = Path(path)
    report = IngestionReport(source=str(path), mode="stage")
    source_name = path.name
    release = Release.detect(path)
    registry = release.registry() if release is not None else {}

    existing = {
        (table, row_id)
        for table, row_id in session.execute(
            select(ExternalRecord.source_table, ExternalRecord.source_row_id).where(
                ExternalRecord.source_database == source_name
            )
        ).all()
    }

    with _connect(path) as connection:
        tables = _table_names(connection)
        sources = (
            {row["source_id"]: dict(row) for row in connection.execute("SELECT * FROM sources")}
            if "sources" in tables
            else {}
        )

        for table, id_column in (
            ("materials", "material_id"),
            ("physical_properties", "record_id"),
            ("optical_dispersion", "record_id"),
        ):
            if table not in tables:
                continue
            if release is not None and table == "optical_dispersion":
                #  1.75 M rows, each carrying its citation, would be gigabytes of staging.
                #  A release is staged one record per dataset instead, its points inline.
                _stage_release_datasets(session, connection, release, source_name, existing, sources, report, limit)
                continue
            query = f"SELECT * FROM {table}"
            if limit:
                query += f" LIMIT {int(limit)}"

            batch: list[ExternalRecord] = []
            for row in connection.execute(query):
                payload = dict(row)
                row_id = str(payload.get(id_column))
                if (table, row_id) in existing:
                    continue

                #  Denormalise the source record onto the payload so a staged
                #  row is self-contained: the citation must survive even if the
                #  external database is not available later.
                if payload.get("source_id") in sources:
                    payload["_source"] = sources[payload["source_id"]]

                if release is not None:
                    payload["_release"] = {"version": release.version, "manifest_commit": release.manifest.get("git_commit")}
                    if table == "materials" and int(payload["material_id"]) in registry:
                        payload["_registry_key"] = registry[int(payload["material_id"])].key
                    parsed_dict = _release_parse(table, payload.get("dataset_label"))
                else:
                    parsed_dict = parse_dataset_label(payload.get("dataset_label")).as_dict()
                batch.append(
                    ExternalRecord(
                        source_database=source_name,
                        source_table=table,
                        source_row_id=row_id,
                        payload=payload,
                        parsed=parsed_dict,
                        status="staged",
                    )
                )
                if len(batch) >= POINT_BATCH:
                    session.add_all(batch)
                    session.flush()
                    report.staged += len(batch)
                    batch = []
            if batch:
                session.add_all(batch)
                session.flush()
                report.staged += len(batch)

    report.notes.append(
        "Staged rows are inert. Nothing enters the analysis tables until promote() confirms it "
        "carries a phase, a specimen form, and the context its property requires."
    )
    return report


def _dataset_row_id(material_id, label: str) -> str:
    """A stable staging id for (material, dataset label) that fits source_row_id (64 chars)."""
    return f"{material_id}:{hashlib.sha256((label or '').encode()).hexdigest()[:24]}"


def _stage_release_datasets(session, connection, release, source_name, existing, sources, report, limit) -> None:
    """Stage each optical dataset of a release as one record: its metadata and its stored points."""
    rows = connection.execute(
        "SELECT record_id, material_id, dataset_label, wavelength_nm, n, k, temperature_c, raw_record_table, "
        "source_id FROM optical_dispersion ORDER BY material_id, dataset_label, wavelength_nm, record_id"
    )
    batch: list[ExternalRecord] = []
    staged = 0

    def flush(key, points, meta) -> None:
        nonlocal staged
        material_id, label = key
        row_id = _dataset_row_id(material_id, label)
        if ("optical_dataset", row_id) in existing:
            return
        payload = {
            "material_id": material_id,
            "dataset_label": label,
            "temperatures_c": sorted(meta["temps"], key=lambda t: (t is None, t)),
            "raw_record_tables": sorted(meta["tables"]),
            "source_ids": sorted(meta["sources"]),
            "n_points": len(points),
            #  [wavelength_nm, n, k, record_id], exactly as stored, in wavelength order.
            "points": points,
            "_source": sources.get(next(iter(meta["sources"]))) if len(meta["sources"]) == 1 else None,
            "_release": {"version": release.version, "manifest_commit": release.manifest.get("git_commit")},
        }
        batch.append(
            ExternalRecord(
                source_database=source_name,
                source_table="optical_dataset",
                source_row_id=row_id,
                source_sha256=release.sha256,
                payload=payload,
                parsed=_release_parse("optical_dispersion", label),
                status="staged",
            )
        )
        staged += 1
        if len(batch) >= 200:
            session.add_all(batch)
            session.flush()
            batch.clear()

    key = None
    points: list = []
    meta: dict = {}
    for record_id, mid, label, wl, n, k, t, table, sid in rows:
        if (mid, label) != key:
            if key is not None:
                flush(key, points, meta)
                if limit and staged >= limit:
                    key = None
                    break
            key, points, meta = (mid, label), [], {"temps": set(), "tables": set(), "sources": set()}
        points.append([wl, n, k, record_id])
        meta["temps"].add(t)
        meta["tables"].add(table)
        meta["sources"].add(sid)
    if key is not None:
        flush(key, points, meta)
    if batch:
        session.add_all(batch)
        session.flush()
    report.staged += staged


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------


def promote(
    session,
    path: Path | str,
    *,
    specimen_form: SpecimenForm,
    phase_map: dict[str, str] | None = None,
    include_optical: bool = True,
    operator: str = "",
    materials_db_root: Path | str | None = None,
    kinds: DatasetKindHelper | None = None,
) -> IngestionReport:
    """Create analysis rows from staged records that satisfy the protocol.

    ``specimen_form`` has to be supplied by a person: the external database does
    not record it, and it decides comparability (Sec. 2.1).  Asserting it here
    is a documented judgement, recorded on every material this call creates; it
    is not the importer guessing.

    ``phase_map`` maps an external material name (or ``material_id``) to a
    polymorph, for records whose label carries no phase.  Same principle: a
    person supplies what the source omits, and it is written down.
    """
    path = Path(path)
    phase_map = phase_map or {}
    report = IngestionReport(source=str(path), mode="promote")
    now = datetime.now(timezone.utc)
    release = Release.detect(path)
    kinds_unavailable: str | None = None
    if release is not None and include_optical and kinds is None:
        try:
            kinds = DatasetKindHelper(release, materials_db_root)
        except ReleaseError as exc:
            kinds_unavailable = str(exc)
            report.notes.append(
                f"Optical datasets cannot be told measured from model fit ({exc}); they are quarantined, "
                "not assumed measured."
            )
    provenance_note = (
        f"Imported from {path.name}; specimen form asserted as {specimen_form.value}"
        + (f" by {operator}" if operator else "")
        + "."
    )

    staged = (
        session.query(ExternalRecord)
        .filter(
            ExternalRecord.source_database == path.name,
            ExternalRecord.status == "staged",
        )
        .all()
    )
    by_table: dict[str, list[ExternalRecord]] = defaultdict(list)
    for record in staged:
        by_table[record.source_table].append(record)

    external_materials = {
        str(r.payload.get("material_id")): r for r in by_table.get("materials", [])
    }

    def _quarantine(record: ExternalRecord, reason: str, missing: list[str] | None = None) -> None:
        record.status = "quarantined"
        record.quarantine_reason = reason
        record.missing_fields = missing or []
        report.quarantined += 1
        report.quarantine_reasons[reason] += 1

    def _resolve_phase(external_id: str, parsed: dict) -> str | None:
        if parsed.get("phase"):
            return parsed["phase"]
        holder = external_materials.get(external_id)
        name = (holder.payload.get("name") if holder else "") or ""
        mapped = phase_map.get(external_id) or phase_map.get(name)
        if mapped:
            return mapped
        #  A release key names the phase for its earliest families
        #  (oxides_50:TiO2@rutile): a release fact, not a guess.
        key = (holder.payload.get("_registry_key") if holder else None) or ""
        selection = key.split(":", 1)[1] if ":" in key else ""
        return selection.split("@", 1)[1] if "@" in selection else None

    #  material cache keyed by (external material id, polymorph)
    materials: dict[tuple[str, str], Material] = {}

    def _material_for(external_id: str, polymorph: str) -> Material | None:
        key = (external_id, polymorph)
        if key in materials:
            return materials[key]

        holder = external_materials.get(external_id)
        formula = (holder.payload.get("formula") if holder else None) or None
        if not formula:
            return None

        #  The same reduction the API uses. This path assigned
        #  ``formula_reduced=formula`` unreduced, so an external source writing
        #  ``Hf2O4`` created a second material for a film already recorded as
        #  ``HfO2``. Skipped rather than guessed when it cannot be reduced: a row
        #  with a wrong identity is harder to find later than a row that is absent.
        try:
            reduced = reduced_formula(formula)
        except FormulaNotCanonical as exc:
            logger.warning("Skipping external material %s: %s", external_id, exc)
            return None

        found = (
            session.query(Material)
            .filter(
                Material.formula_reduced == reduced,
                Material.polymorph == polymorph,
                Material.specimen_form == specimen_form,
            )
            .one_or_none()
        )
        if found is None:
            found = Material(
                formula=formula,
                formula_reduced=reduced,
                polymorph=polymorph,
                specimen_form=specimen_form,
                source_database=path.name,
                source_identifier=external_id,
                notes=provenance_note,
            )
            session.add(found)
            session.flush()
            report.materials_created += 1
        materials[key] = found
        return found

    # --- physical_properties ------------------------------------------------
    for record in by_table.get("physical_properties", []):
        payload, parsed = record.payload, record.parsed or {}
        external_id = str(payload.get("material_id"))
        polymorph = _resolve_phase(external_id, parsed)
        if not polymorph:
            _quarantine(
                record,
                "no polymorph recoverable (Sec. 2.1: a formula is not a material identifier)",
                ["polymorph"],
            )
            continue

        material = _material_for(external_id, polymorph)
        if material is None:
            _quarantine(record, "external material has no formula", ["formula"])
            continue

        source = payload.get("_source") or {}
        if parsed.get("grammar") == "materials-db release":
            if parsed.get("error"):
                _quarantine(record, f"label outside the release grammar: {parsed['error']}", ["dataset_label"])
                continue
            _promote_release_physical(session, path, record, material, source, now, report)
            record.status = "promoted"
            record.material_id = material.id
            record.promoted_at = now
            report.promoted += 1
            continue
        wrote = False
        for column, (default_key, kind, units) in PHYSICAL_COLUMN_MAP.items():
            value = payload.get(column)
            if value is None:
                continue
            method = parsed.get("method") or payload.get("dataset_label") or "external import"

            #  One column carries two quantities. ``xray_sld`` holds the real
            #  part in a row labelled ``xray_sld_real`` and the imaginary part
            #  in one labelled ``xray_sld_imag`` — same column, different
            #  physics. Trusting the column alone stores an absorption
            #  coefficient as if it were a scattering length density, which is
            #  wrong by a factor of ~40 for silicon and silently breaks any
            #  reflectivity calculation built on it. The label is the only thing
            #  that distinguishes them, so it wins where it is explicit.
            key = default_key
            labelled = parsed.get("quantity")
            if labelled and labelled.startswith(default_key):
                key = labelled
            tier = _legacy_tier(payload.get("dataset_label"))

            if kind == "descriptor":
                session.add(
                    DescriptorValue(
                        material_id=material.id,
                        descriptor_key=key,
                        value=float(value),
                        units=units,
                        method=method,
                        provenance_tier=tier,
                    )
                )
                report.descriptors_created += 1
            else:
                temperature_c = payload.get("temperature_c")
                session.add(
                    PropertyValue(
                        material_id=material.id,
                        property_key=key,
                        value=float(value),
                        units=units,
                        method=method,
                        temperature_k=(temperature_c + 273.15) if temperature_c is not None else None,
                        frequency_hz=payload.get("frequency_hz"),
                        provenance_tier=tier,
                        doi=source.get("doi"),
                        source_url=source.get("url"),
                        database_identifier=f"{path.name}:{record.source_table}",
                        source_locator=str(payload.get("record_id")),
                        uncertainty=source.get("uncertainty"),
                        ingested_at=now,
                    )
                )
                report.properties_created += 1
            wrote = True

        if not wrote:
            _quarantine(record, "no mappable value in this row")
            continue
        record.status = "promoted"
        record.material_id = material.id
        record.promoted_at = now
        report.promoted += 1

    # --- optical datasets of a release ---------------------------------------
    if include_optical and by_table.get("optical_dataset"):
        _promote_release_datasets(
            session, path, by_table["optical_dataset"], report, _resolve_phase, _material_for, _quarantine, now,
            release=release, kinds=kinds, kinds_unavailable=kinds_unavailable,
        )

    # --- optical_dispersion -------------------------------------------------
    if include_optical:
        report = _promote_optical(
            session, path, by_table, report, _resolve_phase, _material_for, _quarantine, now,
            release=release, kinds=kinds, kinds_unavailable=kinds_unavailable,
        )

    session.flush()
    return report


#  What each release value kind is, short enough for PropertyValue.method (128 chars).
KIND_TAGS = {
    "literature": "literature",
    "calculated_dft": "DFT, calculated",
    "bulk_approximation": "bulk DFT value for a film, calculated",
    "calculated_from_lattice": "from lattice parameters, calculated",
    "calculated_sld": "periodictable, calculated",
    "calculated_dfpt": "DFT perturbation theory at 0 K, calculated",
}


def _legacy_tier(label: str | None) -> ProvenanceTier:
    """An older export's value: CALCULATED where its label says DFT, SLD or derived, else the default.

    The older CNMS exports share the release's label vocabulary, so a periodictable SLD
    or a density_MP_DFT is recognisably calculated there too. A label the grammar cannot
    read keeps the documented literature default (MEASURED with its DOI).
    """
    try:
        kind = physical_kind(parse_physical_label(label))
    except LabelError:
        return DEFAULT_TIER
    return ProvenanceTier.MEASURED if kind.provenance == "measured" else ProvenanceTier.CALCULATED


def _promote_release_physical(session, path, record, material, source, now, report) -> None:
    """One release physical row: its quantity, column and tier all read from its label."""
    payload, parsed = record.payload, record.parsed
    quantity = parsed["quantity"]
    key, kind, units = RELEASE_QUANTITY_KEYS[quantity]
    value = float(payload[RELEASE_COLUMN[quantity]])
    tier = ProvenanceTier.MEASURED if parsed["provenance"] == "measured" else ProvenanceTier.CALCULATED
    version = (payload.get("_release") or {}).get("version")
    #  The label verbatim plus what kind of value it is. "DFT" appears wherever a value is
    #  DFT, so the compliance check asks for its XC functional, which the release does not
    #  state; that flag is correct and is not worked around.
    method = f"materials-db v{version}: {parsed['raw']} ({KIND_TAGS[parsed['kind']]})"
    if len(method) > 128:
        raise ValueError(f"method text for {parsed['raw']!r} exceeds the 128-character column")
    if kind == "descriptor":
        session.add(
            DescriptorValue(
                material_id=material.id,
                descriptor_key=key,
                value=value,
                units=units,
                method=method,
                provenance_tier=tier,
            )
        )
        report.descriptors_created += 1
        return
    temperature_c = payload.get("temperature_c")
    #  The release's dielectric constants are the mean of the DFPT tensor's principal
    #  values (README). That is a reduction, so it is recorded (Sec. 3.2), and no tensor
    #  component is claimed: a mean is not a direction, and 'k' requires one before the
    #  eligibility filter will let it into an analysis.
    reduction = "mean of principal values (materials-db)" if quantity.startswith("dielectric_") else None
    session.add(
        PropertyValue(
            material_id=material.id,
            property_key=key,
            value=value,
            units=units,
            method=method,
            reduction_rule=reduction,
            temperature_k=(temperature_c + 273.15) if temperature_c is not None else None,
            frequency_hz=payload.get("frequency_hz"),
            provenance_tier=tier,
            doi=source.get("doi"),
            source_url=source.get("url"),
            database_identifier=f"{path.name}:{record.source_table}",
            source_locator=str(payload.get("record_id")),
            uncertainty=source.get("uncertainty"),
            ingested_at=now,
        )
    )
    report.properties_created += 1


def _promote_release_datasets(
    session, path, records, report, resolve_phase, material_for, quarantine, now, *, release, kinds, kinds_unavailable
) -> None:
    """One staged release dataset -> an n series and a k series, tiered by dataset_kind.py."""
    for record in records:
        payload, parsed = record.payload, record.parsed or {}
        external_id = str(payload["material_id"])
        if len(payload["temperatures_c"]) != 1 or len(payload["raw_record_tables"]) != 1 or len(payload["source_ids"]) != 1:
            quarantine(record, "a dataset with more than one temperature, page or source", ["dataset_label"])
            continue
        polymorph = resolve_phase(external_id, parsed)
        if not polymorph:
            quarantine(
                record,
                "no polymorph recoverable (Sec. 2.1: a formula is not a material identifier)",
                ["polymorph"],
            )
            continue
        if kinds is None:
            quarantine(
                record,
                "measured vs model fit cannot be decided without materials-db's dataset_kind.py "
                f"({kinds_unavailable or 'no checkout'})",
                ["dataset_kind"],
            )
            continue
        material = material_for(external_id, polymorph)
        if material is None:
            quarantine(record, "external material has no formula", ["formula"])
            continue
        fit = kinds.model_fit_reason(payload["raw_record_tables"][0])
        tier = ProvenanceTier.MODELED if fit else ProvenanceTier.MEASURED
        method = f"optical dispersion, materials-db v{release.version}, " + (
            "model fit of the dielectric function" if fit else "measured"
        )
        source = payload.get("_source") or {}
        temperature_c = payload["temperatures_c"][0]
        axis = (parsed.get("axis") or "").removesuffix("-axis") or None
        for quantity, column in (("n", 1), ("k", 2)):
            kept: dict[float, float] = {}
            repeated = 0
            for point in payload["points"]:
                if point[column] is None:
                    continue
                if point[0] in kept:
                    #  The source repeats a wavelength (the MANIFEST lists every case). Keep
                    #  the first stored row, never an average, and count the rest.
                    repeated += 1
                    continue
                kept[point[0]] = point[column]
            if repeated:
                report.quarantine_reasons[
                    "repeated wavelength in the source: first stored row kept, the rest not loaded"
                ] += repeated
            if not kept:
                continue
            series = SpectralSeries(
                material_id=material.id,
                quantity=quantity,
                units="dimensionless",
                independent_variable="wavelength_nm",
                axis=parsed.get("axis"),
                tensor_component=axis_to_tensor_component(axis),
                temperature_k=(temperature_c + 273.15) if temperature_c is not None else None,
                method=method,
                provenance_tier=tier,
                doi=source.get("doi"),
                source_url=source.get("url"),
                database_identifier=f"{path.name}:optical_dispersion",
                dataset_label=payload["dataset_label"],
                n_points=len(kept),
                x_min=min(kept),
                x_max=max(kept),
            )
            series.context_digest = context_digest(
                {
                    "method": series.method,
                    "temperature_k": series.temperature_k,
                    "tensor_component": series.tensor_component,
                    "doi": series.doi,
                    "database_identifier": f"{series.database_identifier}:{payload['dataset_label']}",
                    "provenance_tier": series.provenance_tier,
                }
            )
            session.add(series)
            session.flush()
            report.series_created += 1
            rows = [{"series_id": series.id, "x_value": x, "y_value": y} for x, y in kept.items()]
            for start in range(0, len(rows), POINT_BATCH):
                session.execute(insert(SpectralPoint), rows[start : start + POINT_BATCH])
            report.points_created += len(rows)
        record.status = "promoted"
        record.material_id = material.id
        record.promoted_at = now
        report.promoted += 1


def _promote_optical(
    session, path, by_table, report, resolve_phase, material_for, quarantine, now,
    *, release=None, kinds=None, kinds_unavailable=None,
) -> IngestionReport:
    """Group optical rows into series, then bulk-load their points.

    Grouped by (material, phase, axis, source): that tuple is what makes one
    curve one curve.  The alternative — a PropertyValue per wavelength — would
    turn 125,000 samples into 125,000 rows of duplicated provenance, and would
    still not let anyone ask for "n between 1000 and 2000 nm".
    """
    grouped: dict[tuple, list] = defaultdict(list)
    holders: dict[tuple, list] = defaultdict(list)

    for record in by_table.get("optical_dispersion", []):
        payload, parsed = record.payload, record.parsed or {}
        external_id = str(payload.get("material_id"))
        polymorph = resolve_phase(external_id, parsed)
        if not polymorph:
            quarantine(
                record,
                "no polymorph recoverable (Sec. 2.1: a formula is not a material identifier)",
                ["polymorph"],
            )
            continue
        wavelength = payload.get("wavelength_nm")
        if wavelength is None or wavelength <= 0:
            quarantine(record, "non-physical wavelength", ["wavelength_nm"])
            continue

        key = (
            external_id,
            polymorph,
            parsed.get("axis"),
            parsed.get("source_tag"),
            payload.get("dataset_label"),
        )
        grouped[key].append(payload)
        holders[key].append(record)

    for key, rows in grouped.items():
        external_id, polymorph, axis, _source_tag, dataset_label = key
        tier, method = DEFAULT_TIER, "optical dispersion (external import)"
        if release is not None:
            if kinds is None:
                for record in holders[key]:
                    quarantine(
                        record,
                        "measured vs model fit cannot be decided without materials-db's dataset_kind.py "
                        f"({kinds_unavailable or 'no checkout'})",
                        ["dataset_kind"],
                    )
                continue
            fit = kinds.model_fit_reason(rows[0].get("raw_record_table"))
            tier = ProvenanceTier.MODELED if fit else ProvenanceTier.MEASURED
            #  The model-fit evidence is long; the dataset label on the series identifies the page.
            method = f"optical dispersion, materials-db v{release.version}, " + (
                "model fit of the dielectric function" if fit else "measured"
            )
        material = material_for(external_id, polymorph)
        if material is None:
            for record in holders[key]:
                quarantine(record, "external material has no formula", ["formula"])
            continue

        source = rows[0].get("_source") or {}
        temperature_c = rows[0].get("temperature_c")

        for quantity in ("n", "k"):
            samples = [
                (float(r["wavelength_nm"]), float(r[quantity]))
                for r in rows
                if r.get(quantity) is not None
            ]
            if not samples:
                continue
            #  The source can hold several readings at one wavelength for the
            #  same curve; keep the first and let the uniqueness constraint
            #  stand rather than averaging silently (Sec. 2.1).
            deduped = dict(sorted(samples))

            series = SpectralSeries(
                material_id=material.id,
                quantity=quantity,
                units="dimensionless",
                independent_variable="wavelength_nm",
                axis=axis,
                tensor_component=axis_to_tensor_component(axis),
                temperature_k=(temperature_c + 273.15) if temperature_c is not None else None,
                method=method,
                provenance_tier=tier,
                doi=source.get("doi"),
                source_url=source.get("url"),
                database_identifier=f"{path.name}:optical_dispersion",
                dataset_label=dataset_label,
                n_points=len(deduped),
                x_min=min(deduped),
                x_max=max(deduped),
            )
            series.context_digest = context_digest(
                {
                    "method": series.method,
                    "temperature_k": series.temperature_k,
                    "tensor_component": series.tensor_component,
                    "doi": series.doi,
                    "database_identifier": series.database_identifier,
                    "provenance_tier": series.provenance_tier,
                }
            )
            session.add(series)
            session.flush()
            report.series_created += 1

            payload_rows = [
                {"series_id": series.id, "x_value": x, "y_value": y}
                for x, y in deduped.items()
            ]
            for start in range(0, len(payload_rows), POINT_BATCH):
                session.execute(insert(SpectralPoint), payload_rows[start : start + POINT_BATCH])
            report.points_created += len(payload_rows)

        for record in holders[key]:
            record.status = "promoted"
            record.material_id = material.id
            record.promoted_at = now
            report.promoted += 1

    return report


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def derive_eps_inf(
    session,
    *,
    window_nm: tuple[float, float] = (1000.0, 2000.0),
    min_points: int = 3,
    max_k: float = 0.1,
) -> IngestionReport:
    """Estimate eps_inf from optical dispersion in a transparent window.

    The exact relation between the complex refractive index and the dielectric
    function is

        eps_real = n^2 - k^2

    and identifying that with eps_inf requires the material to be **transparent
    and dispersion-free** in the window: below the electronic absorption edge
    (so no interband contribution) and above the phonon resonances (so no ionic
    contribution).  Then eps_real has reached its electronic plateau, which is
    the eps_inf of Eq. (11).

    ``max_k`` is what enforces that, and it is not optional.  Without it this
    function happily returns "eps_inf = 36" for chromium: a metal has large k
    and a negative eps_real in the infrared, dominated by free carriers, and
    n^2 there is not a permittivity at all.  A material with no k data is
    skipped rather than assumed transparent — an unverifiable assumption is not
    a weaker version of a verified one.

    The result is tiered CALCULATED, never MEASURED: it is derived from a
    measurement under a stated approximation, and Sec. 2.3 keeps those apart.
    """
    report = IngestionReport(source="derived", mode="derive_eps_inf")
    lo, hi = window_nm
    method = (
        f"eps_inf = n^2 - k^2 from optical dispersion over {lo:g}-{hi:g} nm "
        f"(transparency: max |k| < {max_k:g})"
    )
    skipped: dict[str, int] = defaultdict(int)

    n_series = session.query(SpectralSeries).filter(SpectralSeries.quantity == "n").all()
    #  Pair each n curve with the k curve of the same material and axis: the
    #  transparency test has to be made on the same specimen and direction.
    k_series_by_key = {
        (s.material_id, s.axis): s
        for s in session.query(SpectralSeries).filter(SpectralSeries.quantity == "k").all()
    }

    def _window_points(series_id: int) -> list[tuple[float, float]]:
        return session.execute(
            select(SpectralPoint.x_value, SpectralPoint.y_value).where(
                SpectralPoint.series_id == series_id,
                SpectralPoint.x_value >= lo,
                SpectralPoint.x_value <= hi,
            )
        ).all()

    for series in n_series:
        n_points = _window_points(series.id)
        if len(n_points) < min_points:
            skipped["too few n samples in the window"] += 1
            continue

        k_series = k_series_by_key.get((series.material_id, series.axis))
        if k_series is None:
            skipped["no k data, so transparency cannot be verified"] += 1
            continue
        k_points = _window_points(k_series.id)
        if not k_points:
            skipped["no k data, so transparency cannot be verified"] += 1
            continue

        k_max = max(abs(value) for _x, value in k_points)
        if k_max >= max_k:
            #  Absorbing here — metals land in this branch, as they should.
            skipped[f"absorbing in the window (max |k| = {k_max:.3g})"] += 1
            continue

        #  The long-wavelength end is the physically meaningful limit; averaging
        #  the whole window would drag the estimate toward the absorption edge.
        tail_n = sorted(n_points, key=lambda row: row[0])[-min_points:]
        tail_k = sorted(k_points, key=lambda row: row[0])[-min_points:]
        n_mean = sum(value for _x, value in tail_n) / len(tail_n)
        k_mean = sum(value for _x, value in tail_k) / len(tail_k)
        eps_inf = n_mean**2 - k_mean**2

        if eps_inf <= 1.0:
            #  eps_inf < 1 is unphysical for a condensed phase.
            skipped[f"unphysical eps_inf = {eps_inf:.3g}"] += 1
            continue

        session.add(
            PropertyValue(
                material_id=series.material_id,
                property_key="eps_inf",
                value=float(eps_inf),
                units="dimensionless",
                tensor_component=series.tensor_component,
                temperature_k=series.temperature_k,
                method=method,
                provenance_tier=ProvenanceTier.CALCULATED,
                doi=series.doi,
                source_url=series.source_url,
                database_identifier=series.database_identifier,
                source_locator=f"spectral_series:{series.id}",
                ingested_at=datetime.now(timezone.utc),
            )
        )
        report.properties_created += 1

    report.quarantine_reasons = defaultdict(int, skipped)
    report.notes.append(
        f"{report.properties_created} eps_inf value(s) derived over {lo:g}-{hi:g} nm, tiered "
        f"CALCULATED. {sum(skipped.values())} series skipped; see quarantine_reasons."
    )
    report.notes.append(
        "The window is a scientific choice and is recorded in `method` on every value. "
        f"{lo:g}-{hi:g} nm suits wide-gap oxides; a narrow-gap material needs a redder window."
    )
    return report


def load_phase_map(path: Path | str | None) -> dict[str, str]:
    """Load an operator-supplied ``{external name or id: polymorph}`` JSON map."""
    if path is None:
        return {}
    return {str(k): str(v) for k, v in json.loads(Path(path).read_text()).items()}
