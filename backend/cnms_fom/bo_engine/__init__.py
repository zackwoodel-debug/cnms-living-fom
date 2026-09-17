"""Bayesian optimization over growth recipes (BoTorch).

    space.py        parameter definitions and recipe encoding
    constraints.py  instrument envelope, safety limits, campaign restrictions
    surrogate.py    GP fitting; why the objective is ln F, not F
    acquisition.py  qLogEI / qLogNEI / qUCB and their optimization
    loop.py         ask/tell: observations in, next recipes out

BoTorch and PyTorch are imported lazily, so the API runs without the 'bo' extra.
"""

from .constraints import ConstraintSet, validate_recipe  # noqa: F401
from .loop import Observation, Suggestion, SuggestionBatch, suggest  # noqa: F401
from .space import ParameterSpec, SearchSpace, example_pld_space  # noqa: F401
