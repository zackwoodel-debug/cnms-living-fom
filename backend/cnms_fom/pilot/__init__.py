"""The HfO2-on-Si pilot workflow — one complete loop, end to end.

    design → suggest → export → simulate → score → observe → suggest …

    stack.py       materials DB → a layer stack (JSON + n,k CSV)
    simulate.py    XRR by Parratt recursion; refnx optional
    properties.py  recipe → the property vector a FOM needs
    workflow.py    orchestration, and the provenance rules that bind it

Layer boundaries are deliberate: each stage takes a plain dataclass and returns
one, so a stage can be swapped for the real thing without touching the others.
Replacing ``simulate`` with a measurement is the whole point of the pilot.

Every property this package produces is tiered MODELED, so every FOM computed
from it comes back ILLUSTRATIVE. That is not a limitation to work around — a
simulated loop exercises the machinery, it does not produce a ranking, and
FOM_PROOF Sec. 2.3 requires the two to stay separable.
"""

from .properties import PILOT_RECIPE_KEYS, simulate_properties  # noqa: F401
from .simulate import XrrResult, simulate_xrr  # noqa: F401
from .stack import Stack, StackLayer, build_stack, stack_to_nk_csv  # noqa: F401
from .workflow import (  # noqa: F401
    PilotEvaluation,
    ensure_pilot_material,
    evaluate_recipe,
    run_pilot_iteration,
)
