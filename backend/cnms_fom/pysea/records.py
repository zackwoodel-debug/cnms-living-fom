"""Importing a pySEA container into the database.

Idempotent by content hash, matching ``modalfit.records``: the same acquisition
re-exported with different key ordering is the same acquisition, and importing it
twice would put two copies of one measurement in the table.

Invalid records are stored. That is deliberate. A container with a missing unit or
an axis mismatch is still evidence that an acquisition happened, and refusing it
would leave the fact unrecorded and the operator with nothing to fix. What an
invalid record cannot do is promote: ``validation_status`` gates that, and the
issues travel with the row so the refusal can explain itself later.

Bulk arrays never reach Postgres. ``contract.strip_bulk`` removes them before
hashing and before storage, so a 4D-STEM scan contributes its shape and its
locator and leaves its gigabytes where pySEA and DataFed put them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from cnms_fom.db.enums import PySeaPromotionStatus
from cnms_fom.pysea.contract import (
    DERIVATIONS,
    assert_supported,
    canonical_envelope,
    content_sha256,
    enum_value,
    load_envelope,
    split_extensions,
)
from cnms_fom.pysea.instrument_state import parse_instrument_state, state_warnings
from cnms_fom.pysea.signal import parse_calibration_table, parse_signals
from cnms_fom.pysea.validate import validate_envelope

logger = logging.getLogger(__name__)


class PySeaImportError(ValueError):
    """The container cannot be read at all."""


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _summary(record, *, created: bool, warnings: list[str]) -> dict:
    return {
        "id": record.id,
        "record_id": record.record_id,
        "record_kind": enum_value(record.record_kind),
        "sample_id": record.sample_id,
        "instrument_id": record.instrument_id,
        "datafed_record_id": record.datafed_record_id,
        "content_sha256": record.content_sha256,
        "validation_status": enum_value(record.validation_status),
        "issues": record.validation_issues or [],
        "warnings": warnings,
        "n_signals": len(record.signals),
        "n_scalars": len(record.scalars),
        "created": created,
    }


def import_pysea_record(
    session,
    envelope_or_path: dict | str,
    *,
    source_filename: str | None = None,
    unit_hints: dict[str, str] | None = None,
) -> dict:
    """Read one pySEA container into ``pysea_records`` and its child tables.

    ``unit_hints`` maps axis kind to a unit the caller asserts, used only where the
    container is silent. Each axis records which source won, so a promoted value
    can be traced to an instrument-reported unit or to one typed at import.

    Returns a summary rather than the ORM object: the caller usually wants the
    validation issues, and handing back a live instance invites writes from places
    that should not be writing.
    """
    from cnms_fom.db.models import PySeaDerivedScalar, PySeaRecord, PySeaSignalRow

    envelope = load_envelope(envelope_or_path)
    version = assert_supported(envelope)

    envelope = split_extensions(envelope)
    canonical = canonical_envelope(envelope)
    digest = content_sha256(envelope)

    existing = (
        session.query(PySeaRecord).filter(PySeaRecord.content_sha256 == digest).one_or_none()
    )
    if existing is not None:
        logger.info(
            "pySEA record already imported (sha256 match): %s",
            source_filename or existing.record_id,
        )
        return _summary(existing, created=False, warnings=[])

    report = validate_envelope(envelope)

    sample = envelope.get("sample") or {}
    instrument = envelope.get("instrument") or {}
    datafed = envelope.get("datafed") or {}
    proposal = envelope.get("proposal") or {}

    calibration_table = parse_calibration_table(envelope)
    #  The state's dispersion and angles may live in a referenced calibration
    #  rather than in the state block itself, which is how a FAIR architecture
    #  separates a column configuration from a spectrometer calibration.
    primary_calibration = next(iter(calibration_table.values()), None)
    state = parse_instrument_state(
        envelope.get("instrument_state"),
        calibration=primary_calibration.as_dict() if primary_calibration else None,
    )

    record = PySeaRecord(
        record_id=str(envelope.get("record_id") or ""),
        contract_version=version,
        record_kind=str(envelope.get("record_kind") or "experimental").strip().lower(),
        sample_id=str(sample.get("sample_id") or ""),
        #  material_id is not read from the envelope even when present. Identity is
        #  supplied at promotion by someone who knows the polymorph.
        material_id=None,
        experiment_id=None,
        instrument_id=(
            str(instrument["instrument_id"]).strip() if instrument.get("instrument_id") else None
        ),
        proposal_id=(str(proposal["proposal_id"]).strip() if proposal.get("proposal_id") else None),
        operator=(str(envelope["operator"]).strip() if envelope.get("operator") else None),
        datafed_record_id=(
            str(datafed["record_id"]).strip() if datafed.get("record_id") else None
        ),
        source_filename=source_filename,
        content_sha256=digest,
        raw_metadata=canonical,
        instrument_state=state.as_dict(),
        calibrations={key: cal.as_dict() for key, cal in calibration_table.items()} or None,
        simulation=envelope.get("simulation") or None,
        software=envelope.get("software") or None,
        validation_status=report.status,
        validation_issues=report.as_dicts() or None,
        acquired_at=_timestamp(envelope.get("acquired_at")),
        ingested_at=datetime.now(timezone.utc),
    )
    session.add(record)
    session.flush()

    for signal in parse_signals(envelope, unit_hints=unit_hints):
        session.add(
            PySeaSignalRow(
                pysea_record_id=record.id,
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                shape=list(signal.shape),
                dtype=signal.dtype,
                units=signal.units,
                axes=[axis.as_dict() for axis in signal.axes],
                calibration=signal.calibration.as_dict() if signal.calibration else None,
                data_ref=signal.data_ref,
            )
        )

    skipped_scalars = 0
    for scalar in envelope.get("derived_scalars") or []:
        if not isinstance(scalar, dict):
            continue
        derivation = str(scalar.get("derivation") or "").strip().lower()
        if derivation not in DERIVATIONS:
            #  Not stored, and not coerced. Writing "calculated" over a derivation
            #  the container did not claim would invent metadata about how a number
            #  was obtained, which is the same class of fabrication as filling in a
            #  missing unit. The validator has already recorded the fault, so the
            #  scalar's existence survives in validation_issues.
            skipped_scalars += 1
            continue
        session.add(
            PySeaDerivedScalar(
                pysea_record_id=record.id,
                name=str(scalar.get("name") or ""),
                property_key=(
                    str(scalar["property_key"]).strip() if scalar.get("property_key") else None
                ),
                value=_float_or_none(scalar.get("value")),
                uncertainty=_float_or_none(scalar.get("uncertainty")),
                units=(str(scalar["units"]).strip() if scalar.get("units") else None),
                derivation=derivation,
                source_signal_ids=[str(s) for s in (scalar.get("source_signal_ids") or [])],
                method=(str(scalar["method"]).strip() if scalar.get("method") else None),
                context=scalar.get("context") or None,
                promotion_status=PySeaPromotionStatus.UNEXAMINED.value,
            )
        )

    session.flush()
    warnings = state_warnings(state)
    if skipped_scalars:
        warnings.append(
            f"{skipped_scalars} derived scalar(s) were not stored: their derivation is "
            "outside the closed vocabulary, and recording one this platform did not "
            "receive would invent metadata. See validation_issues."
        )
    logger.info(
        "Imported pySEA record %s (%s): %d signal(s), %d scalar(s), validation %s",
        record.record_id, record.record_kind, len(record.signals), len(record.scalars),
        record.validation_status,
    )
    return _summary(record, created=True, warnings=warnings)


def _float_or_none(value: Any) -> float | None:
    """A float, or None. An absent value stays absent rather than becoming zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_records(
    session,
    *,
    sample_id: str | None = None,
    record_kind: str | None = None,
    instrument_id: str | None = None,
    datafed_record_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Stored records, filtered and bounded."""
    from cnms_fom.db.models import PySeaRecord

    query = session.query(PySeaRecord)
    if sample_id:
        query = query.filter(PySeaRecord.sample_id == sample_id)
    if record_kind:
        query = query.filter(PySeaRecord.record_kind == record_kind)
    if instrument_id:
        query = query.filter(PySeaRecord.instrument_id == instrument_id)
    if datafed_record_id:
        query = query.filter(PySeaRecord.datafed_record_id == datafed_record_id)

    rows = query.order_by(PySeaRecord.id.desc()).limit(max(1, min(int(limit), 200))).all()
    return [_summary(row, created=False, warnings=[]) for row in rows]


def record_detail(session, pysea_record_id: int) -> dict:
    """One record with its signals and scalars, metadata only."""
    from cnms_fom.db.models import PySeaRecord

    record = session.get(PySeaRecord, pysea_record_id)
    if record is None:
        raise LookupError(f"No pySEA record {pysea_record_id}.")

    return {
        **_summary(record, created=False, warnings=state_warnings(
            parse_instrument_state(record.instrument_state)
        )),
        "contract_version": record.contract_version,
        "acquired_at": record.acquired_at.isoformat() if record.acquired_at else None,
        "ingested_at": record.ingested_at.isoformat() if record.ingested_at else None,
        "operator": record.operator,
        "proposal_id": record.proposal_id,
        "instrument_state": record.instrument_state,
        "calibrations": record.calibrations,
        "simulation": record.simulation,
        "software": record.software,
        "signals": [
            {
                "signal_id": row.signal_id,
                "signal_type": row.signal_type,
                "shape": row.shape,
                "dtype": row.dtype,
                "units": row.units,
                "axes": row.axes,
                "calibration": row.calibration,
                "data_ref": row.data_ref,
            }
            for row in record.signals
        ],
        "derived_scalars": [
            {
                "id": row.id,
                "name": row.name,
                "property_key": row.property_key,
                "value": row.value,
                "uncertainty": row.uncertainty,
                "units": row.units,
                "derivation": enum_value(row.derivation),
                "source_signal_ids": row.source_signal_ids,
                "method": row.method,
                "context": row.context,
                "promotion_status": enum_value(row.promotion_status),
                "refusal_reasons": row.refusal_reasons,
                "promoted_property_value_id": row.promoted_property_value_id,
            }
            for row in record.scalars
        ],
    }
