"""Acquisition functions and their optimization.

Defaults to the log-space variants (``qLogEI`` / ``qLogNEI``). The classic
``qEI`` computes an expectation that is numerically ~0 across most of the domain
once the model is confident, which flattens the gradient and makes the optimizer
return arbitrary points. The log variants are the current BoTorch recommendation
and behave far better late in a campaign, which is where the expensive runs are.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED = ("qLogEI", "qLogNEI", "qUCB")

#  Restarts / raw samples for the acquisition optimizer. Cheap relative to a
#  growth run, so generous defaults are the right trade.
NUM_RESTARTS = 10
RAW_SAMPLES = 256


def build_acquisition(
    model,
    *,
    kind: str = "qLogEI",
    best_f: float | None = None,
    x_baseline: np.ndarray | None = None,
    beta: float = 2.0,
):
    """Construct an acquisition function over a fitted model."""
    try:
        import torch
        from botorch.acquisition.logei import (
            qLogExpectedImprovement,
            qLogNoisyExpectedImprovement,
        )
        from botorch.acquisition.monte_carlo import qUpperConfidenceBound
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("BO needs the 'bo' extra: pip install -e '.[bo]'") from exc

    if kind == "qLogEI":
        if best_f is None:
            raise ValueError("qLogEI needs best_f, the incumbent objective value.")
        return qLogExpectedImprovement(model=model, best_f=float(best_f))
    if kind == "qLogNEI":
        if x_baseline is None:
            raise ValueError("qLogNEI needs X_baseline, the observed inputs.")
        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=torch.as_tensor(np.atleast_2d(x_baseline), dtype=torch.double),
        )
    if kind == "qUCB":
        return qUpperConfidenceBound(model=model, beta=float(beta))
    raise ValueError(f"Unknown acquisition {kind!r}; supported: {SUPPORTED}.")


def optimize(
    acquisition,
    bounds: np.ndarray,
    *,
    q: int = 1,
    integer_dims: list[int] | None = None,
    fixed_features: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximise the acquisition over the bounds; returns ``(candidates, values)``.

    Integer dimensions are optimized continuously and rounded afterwards. That is
    an approximation — the rounded point is not exactly the argmax over the
    integer lattice — and it is the right trade for parameters like repetition
    rate, where the acquisition surface is smooth over neighbouring integers.
    """
    try:
        import torch
        from botorch.optim import optimize_acqf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("BO needs the 'bo' extra: pip install -e '.[bo]'") from exc

    bounds_t = torch.as_tensor(np.asarray(bounds, dtype=float), dtype=torch.double)
    candidates, values = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds_t,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        fixed_features=fixed_features,
        sequential=q > 1,
    )

    result = candidates.detach().cpu().numpy()
    if integer_dims:
        for dim in integer_dims:
            result[:, dim] = np.round(result[:, dim])
            result[:, dim] = np.clip(result[:, dim], bounds[0][dim], bounds[1][dim])
    return result, np.atleast_1d(values.detach().cpu().numpy())
