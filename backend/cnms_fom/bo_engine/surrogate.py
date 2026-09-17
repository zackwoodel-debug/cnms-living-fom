"""GP surrogate model (BoTorch).

Objective choice, which matters more than the kernel: the FOM is a weighted
*geometric* mean (Eq. 29), so ``ln F`` (Eq. 30) is the additive quantity. A GP
assumes Gaussian residuals on the modelled scale, and ln F is far closer to that
than F — which is a product of bounded terms and is strongly skewed near the
floor. Optimizing ln F also makes the acquisition function's improvements
multiplicative in F, which is what "10% better" usually means for a score.

Since ln is monotone, argmax is unchanged: this is a better-behaved
parameterisation of the same problem, not a different one.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#  Below this many observations a GP's length scales are essentially prior; the
#  loop falls back to space-filling exploration instead of pretending otherwise.
MIN_OBSERVATIONS_FOR_GP = 5


def _require_botorch():
    try:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bayesian optimization needs the 'bo' extra: pip install -e '.[bo]'"
        ) from exc
    return (
        torch,
        SingleTaskGP,
        fit_gpytorch_mll,
        ExactMarginalLogLikelihood,
        Normalize,
        Standardize,
    )


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    *,
    noise: np.ndarray | None = None,
):
    """Fit a ``SingleTaskGP`` on model-space inputs and scalar outcomes.

    ``Normalize`` maps inputs to the unit cube using the declared bounds, and
    ``Standardize`` centres the outcome. Both live inside the model so that the
    acquisition optimizer and any later prediction apply them consistently —
    doing it by hand outside the model is a classic source of quietly wrong
    posteriors.
    """
    torch, SingleTaskGP, fit_gpytorch_mll, ExactMarginalLogLikelihood, Normalize, Standardize = (
        _require_botorch()
    )

    train_x = torch.as_tensor(np.atleast_2d(x), dtype=torch.double)
    train_y = torch.as_tensor(np.asarray(y, dtype=float).reshape(-1, 1), dtype=torch.double)
    bounds_t = torch.as_tensor(np.asarray(bounds, dtype=float), dtype=torch.double)

    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError(f"{train_x.shape[0]} inputs but {train_y.shape[0]} outcomes.")

    train_yvar = None
    if noise is not None:
        train_yvar = torch.as_tensor(
            np.asarray(noise, dtype=float).reshape(-1, 1) ** 2, dtype=torch.double
        ).clamp_min(1e-12)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds_t),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def posterior_summary(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation at ``x``, in outcome units."""
    torch, *_ = _require_botorch()
    query = torch.as_tensor(np.atleast_2d(x), dtype=torch.double)
    with torch.no_grad():
        posterior = model.posterior(query)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
    return mean, std


def sobol_points(bounds: np.ndarray, n: int, *, seed: int | None = None) -> np.ndarray:
    """Scrambled Sobol samples across the bounds — the cold-start design.

    Used when there are too few observations to fit a meaningful GP. Sobol beats
    uniform random here: it covers the space more evenly at small n, which is
    exactly the regime a first batch of growth runs is in.
    """
    torch, *_ = _require_botorch()
    from botorch.utils.sampling import draw_sobol_samples

    bounds_t = torch.as_tensor(np.asarray(bounds, dtype=float), dtype=torch.double)
    samples = draw_sobol_samples(bounds=bounds_t, n=n, q=1, seed=seed).squeeze(1)
    return samples.cpu().numpy()
