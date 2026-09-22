"""Integration with the pySEA electron-microscopy ecosystem.

pySEA (Walker, Pfeifer, Lupini, Hachtel, Pantelides, Hoglund, *Microscopy &
Microanalysis* 2026) covers instrument configuration through scattering simulation
to experimental analysis: a FAIR data architecture for signals and calibration, a
ray-optics digital twin of the column, and a multislice framework for elastic and
inelastic scattering with MLIP-driven molecular dynamics for vEELS prediction.

This platform begins where that ends. pySEA determines a signal; this repository
turns a determined quantity into a property value, a figure of merit, and the next
recipe to try. The seam is a scalar with its measurement context attached.

**The contract here is provisional.** It was written from the published abstracts,
without sight of pySEA's container format or API. ``contract.CONTRACT_VERSION`` is
stamped on every stored row so that rows written under a guessed mapping can be
found and re-read once the real format is available. Nothing in this package should
be described to the pySEA authors as a format we require of them.

Two rules carry over unchanged from ``modalfit``:

* A simulation is MODELED. A multislice spectrum and a measured one are both arrays
  of counts against energy loss, and only ``record_kind`` separates them.
* Material identity is supplied, never inferred. A sample label is not a
  composition, a polymorph and a specimen form.
"""

from cnms_fom.pysea.contract import (  # noqa: F401
    CONTRACT_VERSION,
    ContractError,
    canonical_envelope,
    content_sha256,
    load_envelope,
)
from cnms_fom.pysea.instrument_state import (  # noqa: F401
    InstrumentState,
    is_quantitative,
    parse_instrument_state,
    required_context,
    state_warnings,
)
from cnms_fom.pysea.signal import (  # noqa: F401
    PySeaAxis,
    PySeaCalibration,
    PySeaSignal,
    parse_signals,
)

__all__ = [
    "CONTRACT_VERSION",
    "ContractError",
    "InstrumentState",
    "PySeaAxis",
    "PySeaCalibration",
    "PySeaSignal",
    "canonical_envelope",
    "content_sha256",
    "is_quantitative",
    "load_envelope",
    "parse_instrument_state",
    "parse_signals",
    "required_context",
    "state_warnings",
]
