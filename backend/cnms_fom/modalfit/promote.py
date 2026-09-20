"""Promoting fitted parameters into ``property_values`` / ``descriptor_values``.

This is the one path in the platform that turns a ModalFit refinement into an
analysable number, and it is gated hard.

The distinction that matters, and the reason this module is not in
``rag_backend``: a fitted SLD is *instrument-derived*.  A photon or neutron
bounced off the film, a forward model with no free interpretation of the data
reproduced the curve, and the number came out.  That is a measurement, and
FOM_PROOF Sec. 2.2 gives it the ``MEASURED`` tier.  Retrieval output is not —
``rag_backend.chains.assert_not_property_ingestion`` exists precisely to keep a
language model's summary of a paper out of these tables.  The two live in
different packages so that nothing can drift into treating them alike.

What gets refused, and why each refusal is not pedantry
------------------------------------------------------
``a parameter that was held fixed``
    It is an input to the fit.  Promoting it would report the operator's
    starting guess as a measurement.

``a parameter sitting on its fit bound``
    It is clamped, not converged.  The optimizer wanted to go further and was
    stopped by a number someone typed.

``a fit with no chi-squared``
    Nothing separates it from an unrefined starting model.

``a technique that cannot determine the parameter``
    An SLD carried through a QCM-only fit was never constrained by anything.

``a material identity we were not given``
    Sec. 2.1: identity is composition + polymorph + specimen form.  A slab-model
    layer records a formula and nothing else, and inventing "amorphous" to fill
    the polymorph column is exactly the fabrication the protocol forbids.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cnms_fom.db.enums import ProvenanceTier

logger = logging.getLogger(__name__)

#  (slab-model block, parameter) -> (registry key, kind, technique that must be
#  present for the value to have been constrained).
PROMOTABLE: dict[tuple[str, str], dict] = {
    ("xray", "sld_real"): {"key": "sld_xray", "kind": "property", "requires": "XRR", "units": "1e-6 A^-2"},
    ("xray", "sld_imag"): {"key": "sld_xray_imag", "kind": "property", "requires": "XRR", "units": "1e-6 A^-2"},
    ("neutron", "sld_real"): {"key": "sld_neutron", "kind": "property", "requires": "NR", "units": "1e-6 A^-2"},
    ("neutron", "sld_imag"): {"key": "sld_neutron_imag", "kind": "property", "requires": "NR", "units": "1e-6 A^-2"},
    #  Density is a structural descriptor, not a property. It reaches a fit
    #  through XRR/NR contrast or QCM acoustic loading; any of the three counts.
    ("molecular", "density"): {"key": "rho", "kind": "descriptor", "requires": ("XRR", "NR", "QCM"), "units": "g/cm^3"},
    ("viscoelastic", "density"): {"key": "rho", "kind": "descriptor", "requires": ("XRR", "NR", "QCM"), "units": "g/cm^3"},
}

#  Thickness and roughness are deliberately absent from PROMOTABLE. They are not
#  properties of a material — they are *context* for one, and Table 1 gives
#  PropertyValue a thickness_nm column for exactly that. A 103 Å film and a
#  1030 Å film of the same oxide are the same material measured under different
#  conditions, so thickness rides along on each promoted row instead of becoming
#  a row of its own.


class PromotionRefused(PermissionError):
    """A fitted value may not enter the analysis tables.

    ``PermissionError`` rather than ``ValueError``, matching
    ``assert_not_property_ingestion``: this is a rule about what the platform is
    allowed to assert, not a malformed input.
    """


def _method_string(record, technique: str) -> str:
    """The ``method`` context string, built from what the fit actually recorded.

    ``required_context=("method",)`` on every SLD spec means a promoted row needs
    this, and the registry caveats say why: X-ray SLD is energy-dependent and
    neutron SLD is isotope-dependent, so "XRR" alone does not identify the
    quantity.  The energy comes out of the stored settings when present, and is
    named as unrecorded when not — never defaulted to Cu K-alpha because that is
    the common case.
    """
    settings = (record.technique_settings or {}).get(technique) or {}
    bits = [f"ModalFit {technique} refinement"]
    if len(record.techniques or []) > 1:
        bits.append("co-refined with " + "+".join(t for t in record.techniques if t != technique))
    if technique == "XRR":
        energy = settings.get("energy_keV")
        wavelength = settings.get("wavelength_A")
        if energy or wavelength:
            bits.append(
                f"{energy} keV" if energy else f"lambda={wavelength} A"
            )
        else:
            bits.append("X-ray energy not recorded")
    if technique == "NR":
        wavelength = settings.get("wavelength_A")
        bits.append(f"lambda={wavelength} A" if wavelength else "neutron wavelength not recorded")
        bits.append("isotopic composition as given by the layer formula; enrichment not recorded")
    bits.append(f"algorithm {record.algorithm or 'not recorded'}")
    bits.append(f"fit record #{record.id}")
    return "; ".join(bits)


def _refusals_for(record, layer, block: str, parameter: str, spec: dict) -> list[str]:
    """Every reason this one value may not be promoted. Empty means it may."""
    reasons: list[str] = []
    techniques = set(record.techniques or [])

    required = spec["requires"]
    required_set = {required} if isinstance(required, str) else set(required)
    if not (techniques & required_set):
        reasons.append(
            f"none of {sorted(required_set)} is among this fit's techniques "
            f"({sorted(techniques)}), so nothing in the data constrained {parameter}"
        )

    if parameter not in (layer.free_parameters or []):
        reasons.append(f"{parameter} was held fixed — it is an input to the fit, not a result")

    value = ((layer.parameters or {}).get(block) or {}).get(parameter)
    if value is None:
        reasons.append(f"no {block}.{parameter} value on this layer")
    else:
        bounds = (layer.bounds or {}).get(parameter) or {}
        lower, upper = bounds.get("min"), bounds.get("max")
        if lower is not None and upper is not None and upper > lower:
            tol = 1e-6 * (upper - lower)
            if value <= lower + tol or value >= upper - tol:
                reasons.append(
                    f"{parameter}={value} sits on its bound [{lower}, {upper}] — clamped, not "
                    "converged"
                )

    if record.chi2_total is None and not (record.chi2_by_technique or {}):
        reasons.append(
            "the fit records no chi-squared, so nothing distinguishes it from an unrefined "
            "starting model"
        )
    return reasons


def promotion_plan(session, fit_record_id: int) -> dict:
    """What *would* be promoted from one fit, and what would be refused, and why.

    A dry run with no side effects.  Worth calling first every time: the refusal
    list is the actionable half — it names the parameters to free, the bounds to
    widen, and the fits to re-run.
    """
    from cnms_fom.db.models import FitRecord

    record = session.get(FitRecord, fit_record_id)
    if record is None:
        raise LookupError(f"No fit record {fit_record_id}.")

    eligible: list[dict] = []
    refused: list[dict] = []

    for layer in record.layers:
        if layer.role != "layer":
            #  Ambient and substrate are the experiment's fixtures, not the
            #  sample. Promoting a substrate SLD would file the wafer's
            #  properties against the film's material.
            continue
        for (block, parameter), spec in PROMOTABLE.items():
            value = ((layer.parameters or {}).get(block) or {}).get(parameter)
            if value is None:
                continue
            reasons = _refusals_for(record, layer, block, parameter, spec)
            entry = {
                "layer_index": layer.layer_index,
                "layer_label": layer.label or layer.material,
                "block": block,
                "parameter": parameter,
                "registry_key": spec["key"],
                "kind": spec["kind"],
                "value": value,
                "units": spec["units"],
            }
            if reasons:
                refused.append({**entry, "reasons": reasons})
            else:
                eligible.append(entry)

    return {
        "fit_record_id": record.id,
        "sample_id": record.sample_id,
        "techniques": record.techniques,
        "eligible": eligible,
        "refused": refused,
        "requires_material_id": True,
        "note": (
            "Promotion needs an explicit material_id. A slab-model layer carries a formula and "
            "nothing else, and FOM_PROOF Eq. (3) makes identity composition + polymorph + "
            "specimen form — so the polymorph has to come from someone who knows it (XRD, a "
            "growth record), not from this importer."
        ),
    }


def promote_fit(
    session,
    fit_record_id: int,
    *,
    material_id: int,
    layer_label: str | None = None,
    temperature_k: float | None = None,
    substrate: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Promote the eligible fitted values of one fit for one material.

    ``material_id`` is required and never inferred.  ``layer_label`` picks the
    film layer when the stack has more than one; with several films and no label
    this refuses rather than choosing, because attributing one layer's SLD to
    another material is not recoverable afterwards.

    Thickness and roughness from the same layer ride along as measurement
    context on every promoted row (Table 1), which is what makes two thicknesses
    of one oxide two rows rather than a contradiction.
    """
    from cnms_fom.db.models import DescriptorValue, FitRecord, Material, PropertyValue

    record = session.get(FitRecord, fit_record_id)
    if record is None:
        raise LookupError(f"No fit record {fit_record_id}.")
    material = session.get(Material, material_id)
    if material is None:
        raise PromotionRefused(
            f"No material {material_id}. Create the material record first, with its polymorph and "
            "specimen form — FOM_PROOF Sec. 2.1 does not let this importer guess either."
        )

    films = [layer for layer in record.layers if layer.role == "layer"]
    if layer_label:
        wanted = layer_label.strip().lower()
        films = [
            layer
            for layer in films
            if (layer.label or "").strip().lower() == wanted
            or (layer.material or "").strip().lower() == wanted
        ]
        if not films:
            raise PromotionRefused(f"No film layer labelled {layer_label!r} in fit #{record.id}.")
    elif len(films) > 1:
        labels = [layer.label or layer.material or f"layer_{layer.layer_index}" for layer in films]
        raise PromotionRefused(
            f"Fit #{record.id} has {len(films)} film layers ({', '.join(labels)}) and no "
            "layer_label was given. Refusing to guess which one is material "
            f"{material.formula_reduced}: a misattributed SLD is not recoverable later."
        )
    if not films:
        raise PromotionRefused(f"Fit #{record.id} has no film layers to promote.")

    plan = promotion_plan(session, fit_record_id)
    chosen_indices = {layer.layer_index for layer in films}
    eligible = [item for item in plan["eligible"] if item["layer_index"] in chosen_indices]
    refused = [item for item in plan["refused"] if item["layer_index"] in chosen_indices]

    if dry_run:
        return {**plan, "eligible": eligible, "refused": refused, "material_id": material_id, "written": 0}

    layer_by_index = {layer.layer_index: layer for layer in films}
    #  Thickness in nm for PropertyValue.thickness_nm; fits are stored in Å.
    written = 0
    created: list[dict] = []

    for item in eligible:
        layer = layer_by_index[item["layer_index"]]
        technique = _technique_for(record, item)
        thickness_nm = (layer.thickness_ang / 10.0) if layer.thickness_ang else None
        method = _method_string(record, technique)

        if item["kind"] == "property":
            row = PropertyValue(
                material_id=material_id,
                property_key=item["registry_key"],
                value=float(item["value"]),
                units=item["units"],
                uncertainty=(layer.uncertainties or {}).get(item["parameter"]),
                thickness_nm=thickness_nm,
                substrate=substrate or _substrate_label(record),
                interface=(
                    f"roughness {layer.roughness_ang:.3g} A (Nevot-Croce)"
                    if layer.roughness_ang is not None
                    else None
                ),
                temperature_k=temperature_k,
                method=method,
                software="ModalFit",
                provenance_tier=ProvenanceTier.MEASURED,
                source_locator=f"fit_record:{record.id}",
                database_identifier=record.datafed_record_id,
                ingested_at=datetime.now(timezone.utc),
                experiment_id=record.experiment_id,
            )
            #  A duplicate here means the same fit was promoted twice, or two
            #  fits produced the same value under identical context. The
            #  uniqueness constraint catches both; let it, rather than
            #  pre-checking and racing.
            session.add(row)
        else:
            row = DescriptorValue(
                material_id=material_id,
                descriptor_key=item["registry_key"],
                value=float(item["value"]),
                units=item["units"],
                uncertainty=(layer.uncertainties or {}).get(item["parameter"]),
                method=method,
                provenance_tier=ProvenanceTier.MEASURED,
            )
            session.add(row)

        written += 1
        created.append({**item, "technique": technique, "method": method})

    session.flush()
    logger.info(
        "Promoted %d value(s) from fit #%d to material %d; refused %d",
        written, record.id, material_id, len(refused),
    )
    return {
        "fit_record_id": record.id,
        "material_id": material_id,
        "written": written,
        "created": created,
        "refused": refused,
        "provenance_tier": ProvenanceTier.MEASURED.value,
        "note": (
            "These rows are instrument-derived and carry the MEASURED tier. That is a different "
            "path from retrieval: there is no code route from a RAG answer into these tables "
            "(see rag_backend.chains.assert_not_property_ingestion)."
        ),
    }


def _technique_for(record, item: dict) -> str:
    """Which technique to name in ``method`` for a promoted value."""
    required = {
        "sld_xray": ("XRR",),
        "sld_xray_imag": ("XRR",),
        "sld_neutron": ("NR",),
        "sld_neutron_imag": ("NR",),
        "rho": ("XRR", "NR", "QCM"),
    }[item["registry_key"]]
    present = [t for t in (record.techniques or []) if t in required]
    return "+".join(present) if present else (record.techniques or ["unknown"])[0]


def _substrate_label(record) -> str | None:
    """The substrate slab's material, for the ``substrate`` context column."""
    for layer in reversed(record.layers):
        if layer.role == "substrate":
            return layer.material or layer.label
    return None
