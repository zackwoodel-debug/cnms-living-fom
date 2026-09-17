"""CNMS instrument registry.

Everything here is a placeholder with the right *shape*. The capability
envelopes below are typical of the technique, not of any specific CNMS tool, and
they exist so the BO loop can be exercised end to end before facility data is
wired in.

TODO(CNMS): replace ``PLACEHOLDER_INSTRUMENTS`` with a live pull from the
facility instrument registry. Needed from that system:
  * canonical tool IDs and their technique;
  * the real parameter envelope per tool, with units;
  * calibration state and validity window;
  * current availability / scheduled downtime;
  * which proposals are authorised on which tool.
Until then ``source`` is ``"placeholder"`` on every record, and
``assert_real_registry`` blocks anything that must not run on made-up limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cnms_fom.config import get_settings
from cnms_fom.db.enums import SynthesisTechnique


@dataclass
class InstrumentRecord:
    """A growth or characterization tool and what it can physically reach."""

    instrument_id: str
    name: str
    technique: SynthesisTechnique
    location: str = ""
    #  {parameter: {"min": x, "max": y, "units": "..."}} — consumed by
    #  bo_engine.constraints.from_instrument_capabilities.
    capabilities: dict = field(default_factory=dict)
    available: bool = True
    source: str = "placeholder"

    def as_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "name": self.name,
            "technique": self.technique.value,
            "location": self.location,
            "capabilities": self.capabilities,
            "available": self.available,
            "source": self.source,
        }


PLACEHOLDER_INSTRUMENTS: tuple[InstrumentRecord, ...] = (
    InstrumentRecord(
        instrument_id="CNMS-PLD-01",
        name="Pulsed laser deposition chamber (placeholder)",
        technique=SynthesisTechnique.PLD,
        location="CNMS — TODO(CNMS): real building/room",
        capabilities={
            "substrate_temp_c": {"min": 25.0, "max": 900.0, "units": "degC"},
            "o2_pressure_mtorr": {"min": 1e-3, "max": 500.0, "units": "mTorr"},
            "laser_fluence_j_cm2": {"min": 0.3, "max": 4.0, "units": "J/cm^2"},
            "repetition_rate_hz": {"min": 1, "max": 50, "units": "Hz"},
            "substrate": {
                "choices": ["SrTiO3(001)", "MgO(001)", "Al2O3(0001)", "Si(001)", "LaAlO3(001)"]
            },
        },
    ),
    InstrumentRecord(
        instrument_id="CNMS-ALD-01",
        name="Thermal/plasma ALD reactor (placeholder)",
        technique=SynthesisTechnique.ALD,
        location="CNMS — TODO(CNMS): real building/room",
        capabilities={
            "substrate_temp_c": {"min": 50.0, "max": 400.0, "units": "degC"},
            "pulse_time_s": {"min": 0.01, "max": 10.0, "units": "s"},
            "purge_time_s": {"min": 1.0, "max": 60.0, "units": "s"},
            "n_cycles": {"min": 10, "max": 2000, "units": "cycles"},
        },
    ),
    InstrumentRecord(
        instrument_id="CNMS-MBE-01",
        name="Oxide molecular beam epitaxy system (placeholder)",
        technique=SynthesisTechnique.MBE,
        location="CNMS — TODO(CNMS): real building/room",
        capabilities={
            "substrate_temp_c": {"min": 200.0, "max": 1000.0, "units": "degC"},
            "ozone_pressure_torr": {"min": 1e-8, "max": 1e-5, "units": "Torr"},
            "growth_rate_a_per_s": {"min": 0.01, "max": 1.0, "units": "A/s"},
        },
    ),
)


def list_instruments(technique: SynthesisTechnique | None = None) -> list[InstrumentRecord]:
    """Available instruments, optionally filtered by technique.

    TODO(CNMS): hit ``CNMS_INSTRUMENT_REGISTRY_URL`` when it is configured and
    fall back to the placeholders only in development.
    """
    settings = get_settings()
    if settings.cnms_instrument_registry_url:
        raise NotImplementedError(
            "CNMS_INSTRUMENT_REGISTRY_URL is set but the client is not implemented yet. "
            "Implement the fetch here rather than silently serving placeholder envelopes."
        )
    records = list(PLACEHOLDER_INSTRUMENTS)
    if technique is not None:
        records = [r for r in records if r.technique is technique]
    return records


def get_instrument(instrument_id: str) -> InstrumentRecord:
    for record in list_instruments():
        if record.instrument_id == instrument_id:
            return record
    raise KeyError(f"Unknown instrument {instrument_id!r}.")


def assert_real_registry(action: str) -> None:
    """Block an action that must not depend on invented instrument limits.

    Call before anything that leaves the building: scheduling a run, emitting a
    recipe to an operator, or publishing a campaign plan. Proposing a recipe
    against a placeholder envelope is how you get a suggestion the tool cannot
    reach — or, worse, one it can reach but should not.
    """
    if get_settings().cnms_instrument_registry_url:
        return
    raise RuntimeError(
        f"Refusing to {action}: instrument envelopes are placeholders, not the real registry. "
        "Set CNMS_INSTRUMENT_REGISTRY_URL and implement the client in "
        "cnms_integration.instruments.list_instruments first."
    )
