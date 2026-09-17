"""CNMS facility integration.

    instruments.py  tool registry and capability envelopes  (placeholders)
    experiments.py  suggestion -> run -> measured property round trip
    provenance.py   run stamping for the Sec. 16 reproducibility checklist

Everything touching facility systems is a placeholder with a TODO(CNMS) marking
exactly what the real system must supply. ``instruments.assert_real_registry``
blocks outward-facing actions while the envelopes are still invented.
"""

from .experiments import ExperimentOutcome, ExperimentPlan  # noqa: F401
from .instruments import InstrumentRecord, get_instrument, list_instruments  # noqa: F401
from .provenance import RunProvenance  # noqa: F401
