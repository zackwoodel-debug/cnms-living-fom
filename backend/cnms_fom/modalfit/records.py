"""Importing a ModalFit refinement into the platform as a measurement record.

The interface this module targets is ModalFit's **exported model JSON**, not its
HTTP API.  That is a deliberate choice, not a limitation we worked around: the
export format is documented and stable (README Sec. 4), the app's own
``session_store`` is explicitly in-process memory that "would move to redis/a
database" for anything shared, and the Flask layer was not part of the source
provided.  Writing a client against an API we have not seen would produce code
that looks integrated and is not.

What an import is allowed to infer
----------------------------------
Almost nothing.  A stack having an ``xray`` block does not mean XRR data was
ever loaded against it — ModalFit populates every block a technique *could*
read, and refinement is scoped to techniques that are both toggled on and have
data.  So ``techniques`` comes from the export's fit metadata or from the
caller; it is never reconstructed from which blocks happen to be filled.  The
same goes for chi-squared: absent stays absent rather than becoming zero, which
would read as a perfect fit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cnms_fom.db.enums import FitTechnique

from .slab_model import DEFAULT_LENGTH_UNITS, SlabModel, SlabModelError, parse_slab_model

logger = logging.getLogger(__name__)

#  Materials whose bundled n/k tables ModalFit's own README flags as
#  placeholders rather than digitized literature values.  An SE-derived number
#  resting on one of these is not a citable result, so the fit is marked and
#  ``promote`` refuses to export optical properties from it.
PLACEHOLDER_NK_MATERIALS = {"si", "au", "cr", "ti", "sio2", "bk7", "quartz"}

#  Which techniques read which blocks — used only to *warn*, never to infer the
#  technique list. See the module docstring.
BLOCKS_READ_BY: dict[str, tuple[str, ...]] = {
    "SE": ("optical",),
    "SPR": ("optical",),
    "QCM": ("viscoelastic",),
    "XRR": ("xray", "molecular"),
    "NR": ("neutron", "molecular"),
}

#  The x-axis each technique's data is measured against, for ``FitDataset``.
TECHNIQUE_X_UNITS: dict[str, str] = {
    "SE": "nm",        # wavelength
    "SPR": "deg",      # angle of incidence
    "QCM": "Hz",       # frequency offset from the overtone
    "XRR": "1/angstrom",
    "NR": "1/angstrom",
}


class FitImportError(ValueError):
    """The refinement cannot be recorded as a measurement."""


def sha256_of_payload(payload: dict) -> str:
    """Content hash of the model JSON, canonicalised.

    Canonical form rather than raw file bytes: the same fit re-exported with
    different key ordering or indentation is the same fit, and treating it as new
    would put two copies of one measurement in the table.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _coerce_techniques(values: Any) -> list[str]:
    """Validate a technique list against the closed vocabulary."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values:
        name = str(getattr(value, "value", value)).strip().upper()
        if not name:
            continue
        try:
            out.append(FitTechnique(name).value)
        except ValueError as exc:
            valid = ", ".join(t.value for t in FitTechnique)
            raise FitImportError(f"Unknown technique {name!r}. Expected one of: {valid}.") from exc
    #  Order-preserving dedupe: ["XRR", "SE"] is a co-refinement, ["XRR", "XRR"]
    #  is a typo.
    return list(dict.fromkeys(out))


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    for parse in (
        lambda t: datetime.fromisoformat(t),
        lambda t: datetime.strptime(t, "%Y%m%d_%H%M%S"),
        lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            return parse(text)
        except (ValueError, TypeError):
            continue
    logger.debug("Unparseable fit timestamp %r; left unset.", value)
    return None


def detect_placeholder_optics(model: SlabModel) -> list[str]:
    """Layers whose optical constants look like the bundled placeholders.

    A heuristic, and labelled as one: a constant (non-dispersive) n/k on one of
    the materials ModalFit ships placeholder tables for, with no ``nk_file``
    reference.  It over-reports rather than under-reports — a real measured
    constant-index layer gets flagged too, and a false "check your optical
    constants" costs a glance, while a missed one costs a wrong SE result.
    """
    flagged: list[str] = []
    for layer in model.layers:
        material = (layer.material or layer.label or "").strip().lower()
        if material not in PLACEHOLDER_NK_MATERIALS:
            continue
        optical = layer.extra.get("optical")
        if isinstance(optical, dict) and (optical.get("nk_file") or optical.get("model")):
            continue
        if "n" in layer.parameters or "k" in layer.parameters:
            flagged.append(layer.label or layer.material or f"layer_{layer.index}")
    return flagged


def missing_blocks_for(model: SlabModel, techniques: list[str]) -> dict[str, list[str]]:
    """Techniques listed as refined whose required blocks are absent.

    Reported, not corrected.  A fit that claims to have refined against NR while
    no layer carries a ``neutron`` block is describing something that did not
    happen, and the import should say so rather than quietly drop the technique.
    """
    problems: dict[str, list[str]] = {}
    for technique in techniques:
        required = BLOCKS_READ_BY.get(technique, ())
        present = {
            param.block
            for layer in model.layers
            if layer.role == "layer"
            for param in layer.parameters.values()
        }
        absent = [block for block in required if block not in present]
        if absent:
            problems[technique] = absent
    return problems


def import_fit(
    session,
    source: Path | str | dict,
    *,
    techniques: list[str] | None = None,
    algorithm: str | None = None,
    chi2_total: float | None = None,
    chi2_by_technique: dict[str, float] | None = None,
    technique_weights: dict[str, float] | None = None,
    datasets: list[dict] | None = None,
    length_units: str = DEFAULT_LENGTH_UNITS,
    sample_id: str | None = None,
    operator: str | None = None,
    notes: str | None = None,
    datafed_record_id: str | None = None,
    material_id: int | None = None,
    experiment_id: int | None = None,
    resolution_smearing_applied: bool = False,
    roughness_applied_to_spr: bool = False,
) -> dict:
    """Record one ModalFit refinement as ``FitRecord`` + layers + datasets.

    ``source`` is a path to an exported model JSON, or the parsed dict.
    Explicit arguments win over anything the export carried, so a fit exported
    before its metadata was complete can still be recorded accurately by hand.

    Idempotent by content hash: re-importing the same export returns the existing
    record.  Returns a summary dict including ``warnings`` — which is the part
    worth reading, since it is where "this fit claims NR but has no neutron
    block" and "these optical constants are placeholders" show up.

    The two ``*_applied`` flags default to False because that is what ModalFit
    does today: its refnx models are built with ``dq=0.0`` (no angular-resolution
    smearing) and its SPR path ignores layer roughness.  Both are recorded
    rather than assumed away, since both bias the fitted values.
    """
    from cnms_fom.db.models import FitDataset, FitLayer, FitRecord

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FitImportError(f"{path.name} is not valid JSON: {exc}") from exc
        source_filename: str | None = path.name
    else:
        payload = source
        source_filename = None

    try:
        model = parse_slab_model(payload, length_units=length_units)
    except SlabModelError as exc:
        raise FitImportError(str(exc)) from exc

    content_hash = sha256_of_payload(payload)
    existing = (
        session.query(FitRecord).filter(FitRecord.content_sha256 == content_hash).one_or_none()
    )
    if existing is not None:
        logger.info("Fit already imported (sha256 match): %s", source_filename or model.stack_id)
        return {
            "fit_record_id": existing.id,
            "sample_id": existing.sample_id,
            "stack_id": existing.stack_id,
            "techniques": existing.techniques,
            "skipped": True,
            "warnings": [],
        }

    exported = model.fit
    resolved_techniques = _coerce_techniques(techniques or exported.get("techniques"))
    if not resolved_techniques:
        raise FitImportError(
            "No techniques recorded for this fit. A refinement with no technique is not a "
            "measurement of anything: pass techniques=['XRR', ...] explicitly, or export the "
            "model from ModalFit with its fit metadata attached. The technique list is not "
            "inferred from which slab-model blocks are populated — ModalFit fills every block a "
            "technique could read, whether or not data was ever loaded for it."
        )

    warnings: list[str] = []
    missing = missing_blocks_for(model, resolved_techniques)
    for technique, blocks in missing.items():
        warnings.append(
            f"{technique} is recorded as refined, but no film layer carries a "
            f"{'/'.join(blocks)} block. Either the technique list is wrong or the stack is "
            f"incomplete; the fitted values attributed to {technique} are not trustworthy."
        )

    placeholders = detect_placeholder_optics(model)
    if placeholders:
        warnings.append(
            "Optical constants on "
            + ", ".join(placeholders)
            + " look like ModalFit's bundled placeholder n/k values rather than digitized "
            "literature or measured data. SE- and SPR-derived numbers from this fit are "
            "illustrative until they are replaced."
        )

    clamped = {
        (layer.label or f"layer_{layer.index}"): layer.clamped_parameters
        for layer in model.layers
        if layer.clamped_parameters
    }
    for label, names in clamped.items():
        warnings.append(
            f"On {label}, {', '.join(names)} finished on a fit bound. A clamped parameter has "
            "not converged — widen the bound and re-refine before reading it as a measurement."
        )

    chi2 = chi2_total if chi2_total is not None else exported.get("chi2_total")
    if chi2 is None:
        warnings.append(
            "No chi-squared recorded. The fit is stored, but without a goodness-of-fit there is "
            "nothing to distinguish it from an unrefined starting model."
        )

    record = FitRecord(
        stack_id=model.stack_id,
        sample_id=sample_id or model.sample_id,
        source_filename=source_filename,
        content_sha256=content_hash,
        datafed_record_id=datafed_record_id,
        techniques=resolved_techniques,
        technique_weights=technique_weights or exported.get("weights"),
        algorithm=algorithm or exported.get("algorithm"),
        chi2_total=chi2,
        chi2_by_technique=chi2_by_technique or exported.get("chi2_by_technique"),
        n_free_parameters=model.free_parameter_count,
        resolution_smearing_applied=resolution_smearing_applied,
        roughness_applied_to_spr=roughness_applied_to_spr,
        uses_placeholder_optical_constants=bool(placeholders),
        technique_settings=exported.get("settings"),
        raw_model=payload,
        fitted_at=_parse_timestamp(exported.get("fitted_at")) or datetime.now(timezone.utc),
        operator=operator or exported.get("operator"),
        notes=notes or exported.get("notes"),
        material_id=material_id,
        experiment_id=experiment_id,
    )
    session.add(record)
    session.flush()

    for layer in model.layers:
        session.add(
            FitLayer(
                fit_record_id=record.id,
                layer_index=layer.index,
                role=layer.role,
                label=layer.label,
                material=layer.material,
                formula=layer.formula,
                thickness_ang=layer.thickness_ang,
                roughness_ang=layer.roughness_ang,
                density_g_cm3=layer.density_g_cm3,
                parameters=layer.parameters_by_block(),
                free_parameters=layer.free_parameters,
                bounds=layer.bounds_dict(),
                uncertainties=layer.uncertainties_dict() or None,
            )
        )

    per_technique_chi2 = record.chi2_by_technique or {}
    weights = record.technique_weights or {}
    settings = record.technique_settings or {}
    supplied = {str(d.get("technique", "")).upper(): d for d in (datasets or [])}

    for technique in resolved_techniques:
        payload_for_technique = supplied.get(technique, {})
        session.add(
            FitDataset(
                fit_record_id=record.id,
                technique=FitTechnique(technique),
                source_filename=payload_for_technique.get("source_filename"),
                loader=payload_for_technique.get("loader"),
                datafed_record_id=payload_for_technique.get("datafed_record_id"),
                n_points=payload_for_technique.get("n_points"),
                x_min=payload_for_technique.get("x_min"),
                x_max=payload_for_technique.get("x_max"),
                x_units=payload_for_technique.get("x_units") or TECHNIQUE_X_UNITS.get(technique),
                chi2=payload_for_technique.get("chi2", per_technique_chi2.get(technique)),
                weight=payload_for_technique.get("weight", weights.get(technique)),
                settings=payload_for_technique.get("settings") or settings.get(technique),
            )
        )

    session.flush()
    logger.info(
        "Imported fit %s [%s]: %d layers, %d free parameters",
        record.sample_id or record.stack_id,
        "+".join(resolved_techniques),
        len(model.layers),
        model.free_parameter_count,
    )

    return {
        "fit_record_id": record.id,
        "sample_id": record.sample_id,
        "stack_id": record.stack_id,
        "techniques": resolved_techniques,
        "stack": model.describe(),
        "n_layers": len(model.layers),
        "n_free_parameters": model.free_parameter_count,
        "chi2_total": record.chi2_total,
        "length_units": model.length_units,
        "skipped": False,
        "warnings": warnings,
    }


def import_directory(
    session, directory: Path | str, *, length_units: str = DEFAULT_LENGTH_UNITS, **kwargs
) -> list[dict]:
    """Import every ``*.json`` under ``directory`` that parses as a slab model.

    One unreadable file must not stop the sweep, but it is reported rather than
    swallowed — a directory import that silently skipped half its files would be
    indistinguishable from one that worked.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    results: list[dict] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            results.append(import_fit(session, path, length_units=length_units, **kwargs))
        except (FitImportError, FileNotFoundError, OSError) as exc:
            logger.warning("Skipped %s: %s", path.name, exc)
            results.append({"filename": path.name, "error": str(exc)})
    return results
