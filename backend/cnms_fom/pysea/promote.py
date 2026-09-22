"""Promoting a pySEA-derived scalar into ``property_values``.

The same gate as ``modalfit.promote``, applied to a different instrument. A number
that came off an electron microscope is instrument-derived, and FOM_PROOF Sec. 2.2
gives it the MEASURED tier. A number that came out of a multislice simulation is
MODELED, and no amount of agreement with a measurement changes that.

What gets refused, and why each refusal earns its place
-------------------------------------------------------
``a state that was not twin-reconstructed``
    Nominal lens strengths describe the setpoint, not the column. A collection
    angle derived from them is attached to an unknown solid angle, and the
    cross-section computed from it is wrong by an amount nobody can bound.

``a missing collection semi-angle``
    The solid angle scattered into is part of the quantity's identity. Two
    acquisitions at different collection angles measure different numbers and both
    are correct.

``a missing energy dispersion``
    The energy axis is detector channels. A peak position read off it is not in eV.

``a simulated scalar offered as MEASURED``
    ``record_kind`` decides the tier. The caller does not.

``a scalar with no uncertainty``
    It cannot be cross-checked against a second determination, and cross-platform
    comparison is what this integration exists for.

``a material identity we were not given``
    Sec. 2.1 again: composition + polymorph + specimen form. A sample label carries
    none of the three, and inventing a polymorph to fill the column is the
    fabrication the protocol forbids.

``a quantity with no registry key``
    A property this platform has no specification for has no required context, no
    direction, and no place in a FOM. Storing it would be storing a number nobody
    can score.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cnms_fom.db.enums import ProvenanceTier, PySeaPromotionStatus
from cnms_fom.pysea.contract import enum_value
from cnms_fom.pysea.instrument_state import (
    is_quantitative,
    parse_instrument_state,
    required_context,
    state_warnings,
)

logger = logging.getLogger(__name__)

#  Derivations that describe a number taken from data rather than produced by a
#  model. A fitted peak position on a measured spectrum is a measurement; the same
#  fit on a simulated spectrum is not, and record_kind catches that separately.
DATA_DERIVATIONS: frozenset[str] = frozenset({"measured", "fitted"})


class PromotionRefused(PermissionError):
    """A pySEA-derived value may not enter the analysis tables.

    ``PermissionError`` rather than ``ValueError``, matching ``modalfit.promote``:
    a rule about what the platform is allowed to assert, not a malformed input.
    """


def _tier_for(record_kind: str, derivation: str) -> ProvenanceTier:
    """The strongest tier this record and derivation together can support.

    A hybrid record is treated as a simulation. The safe reading of an envelope
    carrying both a measurement and its simulation is the weaker one, because
    nothing in the container says which the scalar came from beyond its source
    signals, and those may be either.
    """
    if record_kind != "experimental":
        return ProvenanceTier.MODELED
    if derivation not in DATA_DERIVATIONS:
        return ProvenanceTier.MODELED
    return ProvenanceTier.MEASURED


def _registry_context_gaps(property_key: str, context: dict) -> list[str]:
    """Required context fields the registry demands and this scalar lacks."""
    from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES

    spec = PHYSICAL_PROPERTIES.get(property_key)
    if spec is None:
        return []
    return [
        field_name
        for field_name in spec.required_context
        if context.get(field_name) in (None, "", [])
    ]


def _refusals_for(record, scalar, state, context: dict) -> list[str]:
    """Every reason this one scalar may not be promoted. Empty means it may."""
    reasons: list[str] = []

    if record.validation_status != "valid":
        reasons.append(
            "the record failed validation; a container whose axes, units or references "
            "do not hold together cannot support a measurement"
        )

    if not state.twin_reconstructed:
        reasons.append(
            "instrument state was not twin-reconstructed; values rest on "
            f"{state.lens_strength_source} lens strengths rather than the column's "
            "reconstructed optical configuration"
        )
    if state.collection_semi_angle_mrad is None:
        reasons.append(
            "collection semi-angle is missing; the solid angle scattered into is part of "
            "the quantity's identity and a cross-section cannot be identified without it"
        )
    if state.dispersion_ev_per_channel is None:
        reasons.append(
            "no energy dispersion is recorded; the energy axis is detector channels, so a "
            "position read off it is not in eV"
        )

    if scalar.value is None:
        reasons.append("the scalar carries no value")
    if scalar.uncertainty is None:
        reasons.append(
            "the scalar reports no uncertainty; it could not be cross-checked against a "
            "second determination, which is what this integration exists to do"
        )
    if not scalar.units:
        reasons.append("the scalar carries no units, and units are never assumed here")

    if not scalar.property_key:
        reasons.append(
            "the scalar maps onto no registry key, so the platform has no specification "
            "for it: no required context, no direction, no place in a FOM"
        )
    else:
        from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES

        if scalar.property_key not in PHYSICAL_PROPERTIES:
            reasons.append(
                f"{scalar.property_key!r} is not in the property registry; a key this "
                "platform does not define cannot be scored"
            )
        else:
            gaps = _registry_context_gaps(scalar.property_key, context)
            if gaps:
                reasons.append(
                    f"required context {sorted(gaps)} for {scalar.property_key!r} is absent; "
                    "FOM_PROOF Sec. 16 makes a value without its context unusable, and this "
                    "importer does not supply it"
                )

    if not scalar.source_signal_ids:
        reasons.append("the scalar names no source signal, so it cannot be traced to data")

    return reasons


def promotion_plan(session, pysea_record_id: int) -> dict:
    """What would be promoted from one record, what would be refused, and why.

    No writes. Worth calling first every time: the refusal list names the metadata
    to go and fetch, which is the actionable half.
    """
    from cnms_fom.db.models import PySeaRecord

    record = session.get(PySeaRecord, pysea_record_id)
    if record is None:
        raise LookupError(f"No pySEA record {pysea_record_id}.")

    state = parse_instrument_state(record.instrument_state)
    eligible: list[dict] = []
    refused: list[dict] = []

    for scalar in record.scalars:
        context = {**required_context(state), **(scalar.context or {})}
        tier = _tier_for(record.record_kind, scalar.derivation)
        reasons = _refusals_for(record, scalar, state, context)
        entry = {
            "scalar_id": scalar.id,
            "name": scalar.name,
            "property_key": scalar.property_key,
            "value": scalar.value,
            "uncertainty": scalar.uncertainty,
            "units": scalar.units,
            "derivation": enum_value(scalar.derivation),
            "proposed_tier": tier.value,
            "context": context,
            "source_signal_ids": scalar.source_signal_ids,
        }
        if reasons:
            refused.append({**entry, "reasons": reasons})
        else:
            eligible.append(entry)

    return {
        "pysea_record_id": record.id,
        "record_id": record.record_id,
        "record_kind": enum_value(record.record_kind),
        "sample_id": record.sample_id,
        "validation_status": enum_value(record.validation_status),
        "instrument_quantitative": is_quantitative(state),
        "state_warnings": state_warnings(state),
        "eligible": eligible,
        "refused": refused,
        "requires_material_id": True,
        "note": (
            "Promotion needs an explicit material_id. A sample label is not a composition, "
            "a polymorph and a specimen form, and FOM_PROOF Eq. (3) makes identity all "
            "three. The polymorph has to come from someone who knows it."
        ),
    }


def promote(
    session,
    pysea_record_id: int,
    *,
    material_id: int,
    scalar_ids: list[int] | None = None,
    dry_run: bool = True,
    operator: str | None = None,
) -> dict:
    """Promote the eligible scalars of one record for one material.

    ``dry_run`` defaults to True. Promotion writes into the tables the FOM is
    computed from, and a caller who has not said "commit" gets a preview.

    ``material_id`` is required and never inferred. ``scalar_ids`` narrows to a
    subset; omitted, every eligible scalar is promoted.
    """
    from cnms_fom.db.models import Material, PropertyValue, PySeaRecord

    record = session.get(PySeaRecord, pysea_record_id)
    if record is None:
        raise LookupError(f"No pySEA record {pysea_record_id}.")

    material = session.get(Material, material_id)
    if material is None:
        raise PromotionRefused(
            f"No material {material_id}. Create the material record first, with its "
            "polymorph and specimen form; this importer does not guess either."
        )

    if record.validation_status != "valid":
        raise PromotionRefused(
            f"pySEA record {record.record_id} failed validation "
            f"({len(record.validation_issues or [])} issue(s)). Fix the container, or "
            "promote nothing from it: a record whose axes and units do not hold together "
            "cannot support a measurement."
        )

    state = parse_instrument_state(record.instrument_state)
    if not is_quantitative(state):
        raise PromotionRefused(
            "Instrument state does not support a quantitative claim. "
            + " ".join(state_warnings(state))
        )

    plan = promotion_plan(session, pysea_record_id)
    eligible = plan["eligible"]
    if scalar_ids is not None:
        wanted = set(scalar_ids)
        eligible = [item for item in eligible if item["scalar_id"] in wanted]
        missing = wanted - {item["scalar_id"] for item in plan["eligible"]}
        if missing:
            refused_ids = {item["scalar_id"]: item for item in plan["refused"]}
            details = [
                f"#{sid}: {'; '.join(refused_ids[sid]['reasons'])}"
                if sid in refused_ids
                else f"#{sid}: not a scalar of this record"
                for sid in sorted(missing)
            ]
            raise PromotionRefused(
                "These scalars are not eligible: " + " | ".join(details)
            )

    if dry_run:
        return {
            **plan,
            "material_id": material_id,
            "eligible": eligible,
            "dry_run": True,
            "written": 0,
            "note": "Dry run. Nothing was written. Pass dry_run=False to commit.",
        }

    by_id = {scalar.id: scalar for scalar in record.scalars}
    written = 0
    created: list[dict] = []

    for item in eligible:
        scalar = by_id[item["scalar_id"]]
        tier = ProvenanceTier(item["proposed_tier"])
        context = item["context"]
        method = _method_string(record, scalar, state)

        row = PropertyValue(
            material_id=material_id,
            property_key=scalar.property_key,
            value=float(scalar.value),
            units=scalar.units,
            uncertainty=scalar.uncertainty,
            temperature_k=context.get("temperature_k"),
            frequency_hz=context.get("frequency_hz"),
            thickness_nm=context.get("thickness_nm"),
            substrate=context.get("substrate"),
            method=method,
            software=_software_label(record),
            provenance_tier=tier,
            source_locator=f"pysea_record:{record.id}",
            database_identifier=record.datafed_record_id,
            ingested_at=datetime.now(timezone.utc),
            experiment_id=record.experiment_id,
        )
        session.add(row)
        session.flush()

        scalar.promotion_status = PySeaPromotionStatus.PROMOTED.value
        scalar.promoted_property_value_id = row.id
        scalar.refusal_reasons = None
        written += 1
        created.append(
            {**item, "property_value_id": row.id, "method": method, "tier": tier.value}
        )

    for item in plan["refused"]:
        scalar = by_id.get(item["scalar_id"])
        if scalar is not None and scalar.promotion_status != PySeaPromotionStatus.PROMOTED.value:
            scalar.promotion_status = PySeaPromotionStatus.REFUSED.value
            scalar.refusal_reasons = item["reasons"]

    if record.material_id is None:
        #  Recorded now that a person has supplied it. The importer still never
        #  sets this.
        record.material_id = material_id

    session.flush()
    logger.info(
        "Promoted %d value(s) from pySEA record %s to material %d; refused %d",
        written, record.record_id, material_id, len(plan["refused"]),
    )
    return {
        "pysea_record_id": record.id,
        "record_id": record.record_id,
        "material_id": material_id,
        "dry_run": False,
        "written": written,
        "created": created,
        "refused": plan["refused"],
        "operator": operator,
        "note": (
            "These rows are instrument-derived and carry the tier their record_kind "
            "supports. There is no code path from retrieval output into these tables; see "
            "rag_backend.chains.assert_not_property_ingestion."
        ),
    }


def _method_string(record, scalar, state) -> str:
    """The ``method`` context string, built from what the record actually says.

    Every clause is conditional. A missing beam energy is named as unrecorded
    rather than defaulted, because "200 kV" is the common case and writing it in
    would be this file asserting something the instrument did not.
    """
    bits: list[str] = []
    technique = ((record.raw_metadata or {}).get("instrument") or {}).get("technique")
    bits.append(f"pySEA {technique}" if technique else "pySEA acquisition")

    if scalar.method:
        bits.append(scalar.method)
    bits.append(f"derivation {enum_value(scalar.derivation)}")

    bits.append(
        f"{state.beam_energy_kev} keV" if state.beam_energy_kev is not None
        else "beam energy not recorded"
    )
    bits.append(
        f"collection {state.collection_semi_angle_mrad} mrad"
        if state.collection_semi_angle_mrad is not None
        else "collection angle not recorded"
    )
    if state.convergence_semi_angle_mrad is not None:
        bits.append(f"convergence {state.convergence_semi_angle_mrad} mrad")
    if state.dispersion_ev_per_channel is not None:
        bits.append(f"dispersion {state.dispersion_ev_per_channel} eV/ch")
    bits.append(
        "twin-reconstructed column state" if state.twin_reconstructed
        else f"{state.lens_strength_source} lens strengths"
    )
    if record.record_kind != "experimental":
        simulation = record.simulation or {}
        code = simulation.get("code") or "unnamed code"
        bits.append(f"simulated with {code}")
    if scalar.source_signal_ids:
        bits.append("from signal(s) " + ", ".join(scalar.source_signal_ids))
    bits.append(f"pySEA record {record.record_id}")
    return "; ".join(bits)


def _software_label(record) -> str:
    """A short software string for the promoted row."""
    for entry in record.software or []:
        if isinstance(entry, dict) and entry.get("name"):
            version = entry.get("version")
            return f"{entry['name']} {version}" if version else str(entry["name"])
    return "pySEA"
