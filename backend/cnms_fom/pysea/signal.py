"""Signals, axes and calibration, parsed without guessing.

The rule this module exists to enforce: **units are never inferred.** A vEELS
spectrum whose energy axis arrives without units is not an eV axis with the label
missing. It is an axis of unknown units, and a dispersion fitted from it cannot be
promoted. The alternative, defaulting to eV because that is the common case, turns
a labelling bug into a number with a unit stamped on it by this file rather than by
the instrument.

The caller may supply units explicitly through ``unit_hints``. Those lose to units
present in the container, and the parsed axis records which source won, so a reader
can tell an instrument-reported unit from one a human typed during import.

Four signal shapes are in scope, taken from what pySEA's abstract describes:

``EELS / vEELS``
    One energy-loss axis.
``momentum-resolved vEELS``
    Energy plus one or two momentum axes, in Å⁻¹. This is the case the integration
    is being built for, and the reason ``MOMENTUM`` is its own axis kind.
``4D-STEM``
    Two scan axes and two diffraction axes.
``simulated spectra``
    Same axes, different ``record_kind``, and that field alone decides the tier a
    derived number can reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cnms_fom.pysea.contract import AXIS_KINDS

#  Axis kinds that carry a physical scale and therefore must state their units
#  before anything derived from them can be promoted. "other" is exempt: an axis
#  indexing repeat number or detector channel has no unit to report.
UNIT_REQUIRED_KINDS: frozenset[str] = frozenset({"energy", "momentum", "spatial", "time"})


@dataclass(frozen=True)
class PySeaAxis:
    """One axis of a signal: what it indexes, how long it is, and its scale."""

    name: str
    kind: str
    units: str | None
    size: int | None
    offset: float | None = None
    scale: float | None = None
    #  "container" when the units came from the record, "caller" when they were
    #  supplied at import, None when absent. Promotion reads this: a unit typed by
    #  an operator is weaker evidence than one the instrument wrote.
    units_source: str | None = None

    @property
    def needs_units(self) -> bool:
        return self.kind in UNIT_REQUIRED_KINDS

    @property
    def has_units(self) -> bool:
        return bool(self.units and str(self.units).strip())

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "units": self.units,
            "size": self.size,
            "offset": self.offset,
            "scale": self.scale,
            "units_source": self.units_source,
        }


@dataclass(frozen=True)
class PySeaCalibration:
    """The numbers that turn detector channels into physical quantities.

    Every field is optional because every field is genuinely absent from some real
    acquisitions, and an absent calibration must stay absent. ``is_energy_calibrated``
    is the one the promotion gate asks about: without a dispersion, an energy axis
    is channels with a label.
    """

    dispersion_ev_per_channel: float | None = None
    energy_zero_channel: float | None = None
    camera_length_mm: float | None = None
    collection_angle_mrad: float | None = None
    convergence_angle_mrad: float | None = None
    source: str | None = None
    calibration_id: str | None = None

    @property
    def is_energy_calibrated(self) -> bool:
        return self.dispersion_ev_per_channel is not None

    def as_dict(self) -> dict:
        return {
            "dispersion_ev_per_channel": self.dispersion_ev_per_channel,
            "energy_zero_channel": self.energy_zero_channel,
            "camera_length_mm": self.camera_length_mm,
            "collection_angle_mrad": self.collection_angle_mrad,
            "convergence_angle_mrad": self.convergence_angle_mrad,
            "source": self.source,
            "calibration_id": self.calibration_id,
        }


@dataclass(frozen=True)
class PySeaSignal:
    """One signal: its shape, its axes, where the array lives, and its calibration."""

    signal_id: str
    signal_type: str | None
    shape: tuple[int, ...]
    dtype: str | None
    units: str | None
    axes: tuple[PySeaAxis, ...]
    calibration: PySeaCalibration | None = None
    record_kind: str | None = None
    data_ref: dict | None = None
    source_metadata: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def axis_of_kind(self, kind: str) -> PySeaAxis | None:
        """The first axis of a given kind, or None.

        Returns the first rather than raising on several: a 4D-STEM scan has two
        spatial axes and both are spatial.
        """
        for axis in self.axes:
            if axis.kind == kind:
                return axis
        return None

    @property
    def has_momentum_axis(self) -> bool:
        return any(axis.kind == "momentum" for axis in self.axes)

    def as_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "units": self.units,
            "axes": [axis.as_dict() for axis in self.axes],
            "calibration": self.calibration.as_dict() if self.calibration else None,
            "record_kind": self.record_kind,
            "data_ref": self.data_ref,
        }


def _number(value: Any) -> float | None:
    """A float, or None. Never a zero standing in for an absent number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_axis(raw: dict, *, unit_hint: str | None = None) -> PySeaAxis:
    """One axis from its container form.

    An unrecognised ``kind`` becomes ``"other"`` rather than raising, and the
    validator reports it. Parsing that refuses on a vocabulary mismatch would make
    the whole record unreadable over one field we may simply have named differently
    from pySEA.
    """
    kind = str(raw.get("kind") or raw.get("axis_kind") or "other").strip().lower()
    if kind not in AXIS_KINDS:
        kind = "other"

    units = raw.get("units") or raw.get("unit")
    units_source: str | None = None
    if units and str(units).strip():
        units, units_source = str(units).strip(), "container"
    elif unit_hint and str(unit_hint).strip():
        #  The caller's hint applies only where the container is silent. An
        #  instrument-reported unit always wins, and the source is recorded so the
        #  difference survives into the row.
        units, units_source = str(unit_hint).strip(), "caller"
    else:
        units = None

    return PySeaAxis(
        name=str(raw.get("name") or raw.get("label") or kind),
        kind=kind,
        units=units,
        size=_integer(raw.get("size") if raw.get("size") is not None else raw.get("length")),
        offset=_number(raw.get("offset")),
        scale=_number(raw.get("scale") if raw.get("scale") is not None else raw.get("step")),
        units_source=units_source,
    )


def parse_calibration(raw: dict | None) -> PySeaCalibration | None:
    """A calibration block, or None when the container carries none.

    None rather than an all-default instance. An empty calibration and an absent
    one read the same downstream, and the promotion gate has to be able to tell
    "this acquisition reports no dispersion" from "this field defaulted".
    """
    if not raw or not isinstance(raw, dict):
        return None
    return PySeaCalibration(
        dispersion_ev_per_channel=_number(raw.get("dispersion_ev_per_channel")),
        energy_zero_channel=_number(raw.get("energy_zero_channel")),
        camera_length_mm=_number(raw.get("camera_length_mm")),
        collection_angle_mrad=_number(raw.get("collection_angle_mrad")),
        convergence_angle_mrad=_number(raw.get("convergence_angle_mrad")),
        source=(str(raw["source"]).strip() if raw.get("source") else None),
        calibration_id=(str(raw["calibration_id"]).strip() if raw.get("calibration_id") else None),
    )


def parse_signal(
    raw: dict,
    *,
    record_kind: str | None = None,
    calibrations: dict[str, PySeaCalibration] | None = None,
    unit_hints: dict[str, str] | None = None,
) -> PySeaSignal:
    """One signal from its container form.

    ``unit_hints`` maps axis kind to a unit the caller is asserting, used only
    where the container is silent. ``calibrations`` resolves a signal's
    ``calibration_ref`` against the envelope's calibration table; an unresolvable
    reference leaves the calibration None and the validator reports it, rather than
    this function inventing one.
    """
    shape_raw = raw.get("shape") or []
    shape = tuple(int(dim) for dim in shape_raw if _integer(dim) is not None)

    hints = unit_hints or {}
    axes = tuple(
        parse_axis(axis_raw, unit_hint=hints.get(str(axis_raw.get("kind") or "").lower()))
        for axis_raw in (raw.get("axes") or [])
        if isinstance(axis_raw, dict)
    )

    calibration = parse_calibration(raw.get("calibration"))
    if calibration is None and calibrations:
        ref = raw.get("calibration_ref") or raw.get("calibration_id")
        if ref:
            calibration = calibrations.get(str(ref))

    return PySeaSignal(
        signal_id=str(raw.get("signal_id") or raw.get("id") or ""),
        signal_type=(str(raw["signal_type"]).strip() if raw.get("signal_type") else None),
        shape=shape,
        dtype=(str(raw["dtype"]).strip() if raw.get("dtype") else None),
        units=(str(raw["units"]).strip() if raw.get("units") else None),
        axes=axes,
        calibration=calibration,
        #  A signal may override the record's kind, which is how a hybrid record
        #  carries a measured spectrum next to its simulation.
        record_kind=str(raw.get("record_kind") or record_kind or "").strip().lower() or None,
        data_ref=raw.get("data_ref") if isinstance(raw.get("data_ref"), dict) else None,
        source_metadata={
            key: value
            for key, value in raw.items()
            if key not in {"signal_id", "id", "signal_type", "shape", "dtype", "units",
                           "axes", "calibration", "calibration_ref", "calibration_id",
                           "record_kind", "data_ref"}
        },
    )


def parse_calibration_table(envelope: dict) -> dict[str, PySeaCalibration]:
    """The envelope's calibrations, keyed by id, for signals to reference."""
    table: dict[str, PySeaCalibration] = {}
    raw = envelope.get("calibrations")
    #  Both shapes appear in the wild: a table keyed by id, and a list of blocks
    #  each carrying its own id. Normalised to pairs so the loop below reads once.
    items: list[tuple[str, Any]]
    if isinstance(raw, dict):
        items = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, list):
        items = [
            (str(item.get("calibration_id") or index), item)
            for index, item in enumerate(raw)
            if isinstance(item, dict)
        ]
    else:
        return table
    for key, value in items:
        parsed = parse_calibration(value if isinstance(value, dict) else None)
        if parsed is not None:
            table[str(key)] = parsed
    return table


def parse_signals(
    envelope: dict, *, unit_hints: dict[str, str] | None = None
) -> list[PySeaSignal]:
    """Every signal in an envelope, with calibration references resolved."""
    calibrations = parse_calibration_table(envelope)
    record_kind = str(envelope.get("record_kind") or "").strip().lower() or None
    return [
        parse_signal(
            raw, record_kind=record_kind, calibrations=calibrations, unit_hints=unit_hints
        )
        for raw in (envelope.get("signals") or [])
        if isinstance(raw, dict)
    ]
