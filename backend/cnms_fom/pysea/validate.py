"""Structured validation of a pySEA envelope.

Returns issues, never prose, and never mutates the input. Each issue carries a
machine-usable ``code``, a ``field_path`` pointing at the offending value, and a
``remediation`` naming what would fix it. A caller can act on the list; a log line
saying "validation failed" cannot be acted on.

The split between error and warning is the split between "this record cannot be
stored as described" and "this record is storable and a number derived from it may
not be promotable". A missing collection angle is a warning here and a refusal in
``promote``: the acquisition is real and worth keeping, and what it cannot support
is a quantitative claim.

Nothing in this module guesses. Where a unit, a shape or a reference is absent, the
issue says so and the value stays absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cnms_fom.pysea.contract import (
    AXIS_KINDS,
    DERIVATIONS,
    RECORD_KINDS,
    supported_version,
)
from cnms_fom.pysea.instrument_state import parse_instrument_state, state_warnings
from cnms_fom.pysea.signal import parse_calibration_table, parse_signals

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One validation finding, addressed to whoever can fix it."""

    code: str
    severity: str
    field_path: str
    message: str
    remediation: str | None = None

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "field_path": self.field_path,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass
class ValidationReport:
    """Every issue found in one envelope, and whether it may be stored."""

    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        return "valid" if self.is_valid else "invalid"

    def codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    def as_dicts(self) -> list[dict]:
        return [issue.as_dict() for issue in self.issues]

    def add(
        self,
        code: str,
        severity: str,
        field_path: str,
        message: str,
        remediation: str | None = None,
    ) -> None:
        self.issues.append(Issue(code, severity, field_path, message, remediation))


def _acquisition_time(envelope: dict) -> datetime | None:
    """When the data was taken, for checking calibration validity against."""
    raw = envelope.get("acquired_at")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def validate_envelope(envelope: dict) -> ValidationReport:
    """Check one envelope against the canonical contract.

    Pure. The envelope is read and never written to, so a caller may validate
    before deciding whether to import.
    """
    report = ValidationReport()

    version = envelope.get("contract_version")
    if not supported_version(version):
        report.add(
            "contract.version_unsupported", ERROR, "contract_version",
            f"Contract version {version!r} is not readable by this build.",
            "Re-export under pysea-canonical/0.x, or extend this contract.",
        )
        #  Everything below interprets fields whose meaning is version-dependent.
        return report

    _check_identity(envelope, report)
    _check_instrument(envelope, report)
    _check_signals(envelope, report)
    _check_simulation(envelope, report)
    _check_scalars(envelope, report)
    _check_datafed(envelope, report)
    return report


def _check_identity(envelope: dict, report: ValidationReport) -> None:
    if not _nonempty(envelope.get("record_id")):
        report.add(
            "record.id_missing", ERROR, "record_id",
            "No record_id. Without a stable acquisition identifier the row cannot be "
            "traced back to pySEA or to DataFed.",
            "Carry pySEA's own record identifier through unchanged.",
        )

    kind = str(envelope.get("record_kind") or "").strip().lower()
    if kind not in RECORD_KINDS:
        report.add(
            "record.kind_invalid", ERROR, "record_kind",
            f"record_kind {envelope.get('record_kind')!r} is not one of "
            f"{sorted(RECORD_KINDS)}. It decides the provenance tier a derived value "
            "may claim and is never inferred.",
            "Set record_kind explicitly on export.",
        )

    sample = envelope.get("sample") or {}
    if not _nonempty(sample.get("sample_id")):
        report.add(
            "sample.id_missing", ERROR, "sample.sample_id",
            "No sample_id. Nothing ties this acquisition to a specimen.",
            "Include the sample identifier used at the instrument.",
        )
    if _nonempty(sample.get("material_id")):
        report.add(
            "sample.material_id_present", WARNING, "sample.material_id",
            "The envelope carries a material_id. This importer ignores it: identity is "
            "composition + polymorph + specimen form (FOM_PROOF Sec. 2.1), and a container "
            "cannot establish all three.",
            "Supply material_id at promotion time, from someone who knows the polymorph.",
        )

    if not _nonempty(envelope.get("acquired_at")):
        report.add(
            "record.acquired_at_missing", WARNING, "acquired_at",
            "No acquisition timestamp, so calibration validity cannot be checked against "
            "the date the data was taken.",
            "Record the acquisition time in ISO-8601.",
        )


def _check_instrument(envelope: dict, report: ValidationReport) -> None:
    instrument = envelope.get("instrument") or {}
    if not _nonempty(instrument.get("instrument_id")):
        report.add(
            "instrument.id_missing", ERROR, "instrument.instrument_id",
            "No instrument_id. A measurement context without an instrument cannot be "
            "reproduced or compared against another acquisition on the same column.",
            "Include the instrument identifier from the facility registry.",
        )
    if not _nonempty(instrument.get("technique")):
        report.add(
            "instrument.technique_missing", WARNING, "instrument.technique",
            "No technique named (STEM-EELS, 4D-STEM, vEELS).",
            "Name the technique so a reader knows what produced the signal.",
        )

    #  Resolve the calibration the same way ``records.import_pysea_record`` does.
    #  A FAIR architecture keeps the column state and the spectrometer calibration
    #  in separate records, so the dispersion usually lives in the referenced
    #  calibration rather than in the state block. Parsing without it made the
    #  validator report a missing dispersion on a record that has one, and
    #  disagree with what ingestion then stored.
    calibrations = parse_calibration_table(envelope)
    primary = next(iter(calibrations.values()), None)
    state = parse_instrument_state(
        envelope.get("instrument_state"),
        calibration=primary.as_dict() if primary else None,
    )

    for label, value in (
        ("beam_energy_kev", state.beam_energy_kev),
        ("convergence_semi_angle_mrad", state.convergence_semi_angle_mrad),
        ("collection_semi_angle_mrad", state.collection_semi_angle_mrad),
        ("camera_length_mm", state.camera_length_mm),
        ("aperture_um", state.aperture_um),
        ("dispersion_ev_per_channel", state.dispersion_ev_per_channel),
    ):
        if value is not None and value < 0:
            report.add(
                "state.negative_quantity", ERROR, f"instrument_state.{label}",
                f"{label} is {value}, which is not physical.",
                "Check the sign convention on export.",
            )

    #  A convergence angle wider than the collection angle means the detector is
    #  not collecting the whole probe, which is a real configuration and worth
    #  flagging rather than refusing.
    if (
        state.convergence_semi_angle_mrad is not None
        and state.collection_semi_angle_mrad is not None
        and state.convergence_semi_angle_mrad > state.collection_semi_angle_mrad
    ):
        report.add(
            "state.convergence_exceeds_collection", WARNING,
            "instrument_state.convergence_semi_angle_mrad",
            f"Convergence semi-angle ({state.convergence_semi_angle_mrad} mrad) exceeds the "
            f"collection semi-angle ({state.collection_semi_angle_mrad} mrad); part of the "
            "probe is outside the detector.",
            "Confirm both angles were recorded for the same aperture configuration.",
        )

    #  Calibration validity is checked against the acquisition date, not against
    #  the moment of import. A calibration that expired last month was still valid
    #  when the data was taken, and flagging it would tell an operator to re-take
    #  a measurement that was correctly calibrated.
    acquired = _acquisition_time(envelope)
    for text in state_warnings(state, at=acquired):
        report.add("state.quality", WARNING, "instrument_state", text, None)


def _check_signals(envelope: dict, report: ValidationReport) -> None:
    raw_signals = envelope.get("signals")
    if not raw_signals:
        report.add(
            "signals.none", ERROR, "signals",
            "The envelope carries no signals.",
            "Include at least one signal, even if the array itself lives elsewhere.",
        )
        return

    signals = parse_signals(envelope)
    seen: set[str] = set()

    for index, signal in enumerate(signals):
        path = f"signals[{index}]"

        if not _nonempty(signal.signal_id):
            report.add(
                "signal.id_missing", ERROR, f"{path}.signal_id",
                "Signal has no signal_id, so a derived scalar cannot name it as a source.",
                "Give every signal a stable id within the record.",
            )
        elif signal.signal_id in seen:
            report.add(
                "signal.id_duplicate", ERROR, f"{path}.signal_id",
                f"signal_id {signal.signal_id!r} appears more than once.",
                "Signal ids must be unique within a record.",
            )
        else:
            seen.add(signal.signal_id)

        if not signal.shape:
            report.add(
                "signal.shape_missing", ERROR, f"{path}.shape",
                "No shape. Axis consistency cannot be checked without it.",
                "Export the array shape alongside the reference.",
            )
        elif len(signal.axes) != len(signal.shape):
            report.add(
                "signal.axis_count_mismatch", ERROR, f"{path}.axes",
                f"{len(signal.axes)} axes for a rank-{len(signal.shape)} signal "
                f"{list(signal.shape)}. Every dimension needs an axis, or the data cannot "
                "be interpreted.",
                "Emit one axis per dimension, in dimension order.",
            )
        else:
            for axis_index, (axis, dim) in enumerate(zip(signal.axes, signal.shape, strict=True)):
                if axis.size is not None and axis.size != dim:
                    report.add(
                        "signal.axis_size_mismatch", ERROR,
                        f"{path}.axes[{axis_index}].size",
                        f"Axis {axis.name!r} declares size {axis.size} against dimension "
                        f"{axis_index} of length {dim}.",
                        "Check the axis order matches the array's dimension order.",
                    )

        for axis_index, axis in enumerate(signal.axes):
            axis_path = f"{path}.axes[{axis_index}]"
            if axis.kind == "other" and axis.name.lower() not in AXIS_KINDS:
                report.add(
                    "signal.axis_kind_unrecognised", WARNING, f"{axis_path}.kind",
                    f"Axis {axis.name!r} has no recognised kind and was read as 'other'.",
                    f"Use one of {sorted(AXIS_KINDS)} so momentum and energy axes are "
                    "distinguishable.",
                )
            if axis.needs_units and not axis.has_units:
                report.add(
                    "signal.axis_units_missing", ERROR, f"{axis_path}.units",
                    f"Axis {axis.name!r} is a {axis.kind} axis with no units. Units are never "
                    "assumed here: an unlabelled energy axis is not eV with the label missing.",
                    "Export the units, or pass them explicitly at import through unit_hints.",
                )
            elif axis.units_source == "caller":
                report.add(
                    "signal.axis_units_from_caller", WARNING, f"{axis_path}.units",
                    f"Units {axis.units!r} for axis {axis.name!r} were supplied at import, not "
                    "by the container. They record what an operator asserted.",
                    "Prefer units written by the instrument.",
                )

        if signal.data_ref is None:
            report.add(
                "signal.data_ref_missing", WARNING, f"{path}.data_ref",
                "No data_ref, so the array cannot be fetched from this row.",
                "Include a scheme and record_id, for example "
                '{"scheme": "datafed", "record_id": "d/12345"}.',
            )
        else:
            if not _nonempty(signal.data_ref.get("scheme")):
                report.add(
                    "signal.data_ref_scheme_missing", WARNING, f"{path}.data_ref.scheme",
                    "data_ref has no scheme, so how to resolve it is unstated.",
                    'Set scheme to "datafed", "file" or "uri".',
                )
            if not _nonempty(signal.data_ref.get("record_id")) and not _nonempty(
                signal.data_ref.get("uri")
            ):
                report.add(
                    "signal.data_ref_target_missing", WARNING, f"{path}.data_ref",
                    "data_ref names neither a record_id nor a uri.",
                    "Give the locator a target.",
                )


def _check_simulation(envelope: dict, report: ValidationReport) -> None:
    kind = str(envelope.get("record_kind") or "").strip().lower()
    simulation = envelope.get("simulation")

    if kind == "simulation" and not simulation:
        report.add(
            "simulation.metadata_missing", ERROR, "simulation",
            "record_kind is 'simulation' but no simulation block describes what produced "
            "it. A simulated spectrum with no code, potential or supercell is "
            "indistinguishable in the table from a measurement.",
            "Record the simulation code, its version, the interatomic potential and the "
            "supercell.",
        )
    if kind == "experimental" and simulation:
        report.add(
            "simulation.on_experimental_record", WARNING, "simulation",
            "An experimental record carries a simulation block. Stored, and ignored for "
            "tier purposes; the record_kind decides.",
            "Use record_kind 'hybrid' when an envelope carries both.",
        )
    if simulation and not _nonempty((simulation or {}).get("code")):
        report.add(
            "simulation.code_missing", WARNING, "simulation.code",
            "The simulation block names no code.",
            "Record the simulation package and version.",
        )


def _check_scalars(envelope: dict, report: ValidationReport) -> None:
    signal_ids = {
        str(raw.get("signal_id") or raw.get("id") or "")
        for raw in (envelope.get("signals") or [])
        if isinstance(raw, dict)
    }

    for index, scalar in enumerate(envelope.get("derived_scalars") or []):
        path = f"derived_scalars[{index}]"
        if not isinstance(scalar, dict):
            report.add(
                "scalar.malformed", ERROR, path, "Derived scalar is not an object.", None
            )
            continue

        if not _nonempty(scalar.get("name")):
            report.add(
                "scalar.name_missing", ERROR, f"{path}.name",
                "Derived scalar has no name.", "Name the quantity.",
            )

        derivation = str(scalar.get("derivation") or "").strip().lower()
        if derivation not in DERIVATIONS:
            report.add(
                "scalar.derivation_invalid", ERROR, f"{path}.derivation",
                f"derivation {scalar.get('derivation')!r} is not one of {sorted(DERIVATIONS)}.",
                "State how the number was obtained; it is never inferred from the value.",
            )

        sources = scalar.get("source_signal_ids") or []
        if not sources:
            report.add(
                "scalar.no_source_signal", ERROR, f"{path}.source_signal_ids",
                "Derived scalar names no source signal, so what it was computed from is "
                "unrecorded and the number cannot be traced.",
                "List the signal ids the value was derived from.",
            )
        else:
            for source in sources:
                if str(source) not in signal_ids:
                    report.add(
                        "scalar.source_signal_unknown", ERROR,
                        f"{path}.source_signal_ids",
                        f"Source signal {source!r} is not in this record.",
                        "Reference a signal that travels with the scalar.",
                    )

        if scalar.get("value") is not None and not _nonempty(scalar.get("units")):
            report.add(
                "scalar.units_missing", ERROR, f"{path}.units",
                f"Scalar {scalar.get('name')!r} has a value and no units.",
                "Units are never assumed; export them with the value.",
            )
        if scalar.get("value") is not None and scalar.get("uncertainty") is None:
            report.add(
                "scalar.uncertainty_missing", WARNING, f"{path}.uncertainty",
                f"Scalar {scalar.get('name')!r} reports no uncertainty. It cannot be "
                "cross-checked against a second determination and will not be promoted.",
                "Report an uncertainty, even a conservative one, with its basis in method.",
            )
        if not _nonempty(scalar.get("method")):
            report.add(
                "scalar.method_missing", WARNING, f"{path}.method",
                f"Scalar {scalar.get('name')!r} records no method.",
                "Describe how the number was extracted from the signal.",
            )


def _check_datafed(envelope: dict, report: ValidationReport) -> None:
    datafed = envelope.get("datafed") or {}
    if not _nonempty(datafed.get("record_id")):
        report.add(
            "datafed.record_id_missing", WARNING, "datafed.record_id",
            "No DataFed record id. The promoted value will carry no pointer back to the "
            "managed copy of the data.",
            "Include the DataFed record id once the container is deposited.",
        )
