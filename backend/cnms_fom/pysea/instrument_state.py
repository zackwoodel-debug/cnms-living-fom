"""The microscope state a number was measured under, and whether it is quantitative.

pySEA's second package is a ray-optics digital twin of the microscope. The reason
that matters here is narrow and specific: a twin-reconstructed state reports the
lens strengths the column actually ran at, and a nominal state reports the ones the
control software was asked for. A collection semi-angle derived from nominal lens
strengths is a plausible number attached to an unknown solid angle, and a
cross-section computed from it is wrong by an amount nobody can bound.

``is_quantitative`` is the gate. It answers one question: did this acquisition
record enough about the column to let a derived number be called a measurement? It
returns True only when the state was reconstructed, the collection angle is known,
and the energy axis has a dispersion. Any of those missing and the number is still
worth storing, still worth looking at, and not something this platform will file as
MEASURED.

The gate is deliberately narrow. It does not judge whether the science is good; it
judges whether the measurement context is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#  Where the lens strengths came from. "twin" is the reconstructed column state;
#  "nominal" is the setpoint; "user-entered" is a human filling a form. Only the
#  first supports a quantitative claim, and the vocabulary is closed so that an
#  unrecognised source cannot read as twin by accident.
LENS_STRENGTH_SOURCES: frozenset[str] = frozenset({
    "twin", "nominal", "user-entered", "unknown"
})


@dataclass(frozen=True)
class InstrumentState:
    """The column configuration for one acquisition or simulation.

    Every physical field is optional. An absent convergence angle is an absent
    convergence angle, and this class never substitutes a typical value for one.
    """

    beam_energy_kev: float | None = None
    camera_length_mm: float | None = None
    convergence_semi_angle_mrad: float | None = None
    collection_semi_angle_mrad: float | None = None
    lens_strengths: dict[str, float] = field(default_factory=dict)
    lens_strength_source: str = "unknown"
    monochromated: bool | None = None
    aperture_um: float | None = None
    detector: str | None = None
    dispersion_ev_per_channel: float | None = None
    calibration_validity_window: tuple[datetime, datetime] | None = None
    calibration_id: str | None = None
    #  The claim that pySEA's ray-optics twin produced this state rather than the
    #  control software's setpoints. Never inferred from the presence of lens
    #  strengths: a nominal state has those too.
    twin_reconstructed: bool = False
    temperature_k: float | None = None

    def as_dict(self) -> dict:
        window = self.calibration_validity_window
        return {
            "beam_energy_kev": self.beam_energy_kev,
            "camera_length_mm": self.camera_length_mm,
            "convergence_semi_angle_mrad": self.convergence_semi_angle_mrad,
            "collection_semi_angle_mrad": self.collection_semi_angle_mrad,
            "lens_strengths": dict(self.lens_strengths),
            "lens_strength_source": self.lens_strength_source,
            "monochromated": self.monochromated,
            "aperture_um": self.aperture_um,
            "detector": self.detector,
            "dispersion_ev_per_channel": self.dispersion_ev_per_channel,
            "calibration_validity_window": (
                [window[0].isoformat(), window[1].isoformat()] if window else None
            ),
            "calibration_id": self.calibration_id,
            "twin_reconstructed": self.twin_reconstructed,
            "temperature_k": self.temperature_k,
        }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_instrument_state(raw: dict | None, *, calibration: dict | None = None) -> InstrumentState:
    """An ``InstrumentState`` from the envelope's ``instrument_state`` block.

    ``calibration`` supplies the dispersion when the state block does not carry it,
    which is the common arrangement: the column state and the spectrometer
    calibration are separate records in a FAIR architecture.

    ``twin_reconstructed`` is read as an explicit boolean. A truthy string, a
    missing key, or an unrecognised lens-strength source all leave it False. The
    asymmetry is on purpose: the cost of treating a nominal state as reconstructed
    is a wrong number filed as a measurement, and the cost of the reverse is a
    refusal someone can override with better metadata.
    """
    raw = raw or {}
    source = str(raw.get("lens_strength_source") or "unknown").strip().lower()
    if source not in LENS_STRENGTH_SOURCES:
        source = "unknown"

    window: tuple[datetime, datetime] | None = None
    raw_window = raw.get("calibration_validity_window")
    if isinstance(raw_window, (list, tuple)) and len(raw_window) == 2:
        start, end = _datetime(raw_window[0]), _datetime(raw_window[1])
        if start and end:
            window = (start, end)

    strengths: dict[str, float] = {}
    for key, value in (raw.get("lens_strengths") or {}).items():
        number = _number(value)
        if number is not None:
            strengths[str(key)] = number

    dispersion = _number(raw.get("dispersion_ev_per_channel"))
    if dispersion is None and calibration:
        dispersion = _number(calibration.get("dispersion_ev_per_channel"))

    collection = _number(raw.get("collection_semi_angle_mrad"))
    if collection is None and calibration:
        collection = _number(calibration.get("collection_angle_mrad"))

    convergence = _number(raw.get("convergence_semi_angle_mrad"))
    if convergence is None and calibration:
        convergence = _number(calibration.get("convergence_angle_mrad"))

    monochromated = raw.get("monochromated")
    if not isinstance(monochromated, bool):
        monochromated = None

    return InstrumentState(
        beam_energy_kev=_number(raw.get("beam_energy_kev")),
        camera_length_mm=_number(raw.get("camera_length_mm")),
        convergence_semi_angle_mrad=convergence,
        collection_semi_angle_mrad=collection,
        lens_strengths=strengths,
        lens_strength_source=source,
        monochromated=monochromated,
        aperture_um=_number(raw.get("aperture_um")),
        detector=(str(raw["detector"]).strip() if raw.get("detector") else None),
        dispersion_ev_per_channel=dispersion,
        calibration_validity_window=window,
        calibration_id=(
            str(raw["calibration_id"]).strip() if raw.get("calibration_id") else None
        ),
        twin_reconstructed=raw.get("twin_reconstructed") is True,
        temperature_k=_number(raw.get("temperature_k")),
    )


def is_quantitative(state: InstrumentState) -> bool:
    """Whether a number derived under this state may be called a measurement.

    Three conditions, all required:

    1. The state was reconstructed by the digital twin. Nominal lens strengths
       describe what the column was asked to do, not what it did.
    2. A collection semi-angle is present. Without the solid angle collected into,
       a scattering cross-section is not identified.
    3. An energy dispersion is present. Without it the energy axis is channels.

    Returns a plain bool rather than a report; ``state_warnings`` says why.
    """
    return bool(
        state.twin_reconstructed
        and state.collection_semi_angle_mrad is not None
        and state.dispersion_ev_per_channel is not None
    )


def required_context(state: InstrumentState) -> dict:
    """The measurement context a promoted value must carry, from this state.

    Only fields the state actually reports. A key absent here is absent from the
    promoted row, which is what makes a missing temperature visible in the table
    rather than silently recorded as room temperature.
    """
    context: dict[str, Any] = {}
    if state.beam_energy_kev is not None:
        context["beam_energy_kev"] = state.beam_energy_kev
    if state.collection_semi_angle_mrad is not None:
        context["collection_semi_angle_mrad"] = state.collection_semi_angle_mrad
    if state.convergence_semi_angle_mrad is not None:
        context["convergence_semi_angle_mrad"] = state.convergence_semi_angle_mrad
    if state.dispersion_ev_per_channel is not None:
        context["dispersion_ev_per_channel"] = state.dispersion_ev_per_channel
    if state.temperature_k is not None:
        context["temperature_k"] = state.temperature_k
    if state.detector:
        context["detector"] = state.detector
    if state.monochromated is not None:
        context["monochromated"] = state.monochromated
    if state.calibration_id:
        context["calibration_id"] = state.calibration_id
    context["twin_reconstructed"] = state.twin_reconstructed
    context["lens_strength_source"] = state.lens_strength_source
    return context


def state_warnings(state: InstrumentState, *, at: datetime | None = None) -> list[str]:
    """Everything about this state a reader should know before trusting a number.

    Warnings, not errors. A record with all of these is still worth storing; what
    it is not is a source of MEASURED values.
    """
    warnings: list[str] = []

    if not state.twin_reconstructed:
        warnings.append(
            "Instrument state was not twin-reconstructed. Lens strengths are "
            f"{state.lens_strength_source}, so the optical configuration is the one the "
            "column was asked for rather than the one it ran at."
        )
    if state.collection_semi_angle_mrad is None:
        warnings.append(
            "No collection semi-angle. The solid angle scattered into is unknown, so a "
            "cross-section derived from this acquisition is not identified."
        )
    if state.dispersion_ev_per_channel is None:
        warnings.append(
            "No energy dispersion. The energy axis is detector channels with a label, and "
            "a peak position read off it is not in eV."
        )
    if state.beam_energy_kev is None:
        warnings.append(
            "No beam energy. Inelastic cross-sections and the relativistic correction both "
            "depend on it."
        )
    if state.lens_strength_source == "user-entered":
        warnings.append(
            "Lens strengths were entered by hand. They record what an operator believed, "
            "which is evidence of a different kind from an instrument reading."
        )

    window = state.calibration_validity_window
    if window is not None:
        now = at or datetime.now(timezone.utc)
        start, end = window
        if now > end:
            warnings.append(
                f"Calibration {state.calibration_id or '(unnamed)'} expired at "
                f"{end.isoformat()}; this acquisition is being read against it anyway."
            )
        elif now < start:
            warnings.append(
                f"Calibration {state.calibration_id or '(unnamed)'} does not begin until "
                f"{start.isoformat()}."
            )
    elif state.calibration_id:
        warnings.append(
            f"Calibration {state.calibration_id} has no validity window, so whether it "
            "applied on the acquisition date cannot be checked."
        )

    return warnings
