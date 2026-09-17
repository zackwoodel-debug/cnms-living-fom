"""Permutation inference and multiple-testing correction (FOM_PROOF Sec. 8).

Permutation test (Eq. 41):

    p_perm = (1 + #{ b : |r^(b)| >= |r_obs| }) / (B + 1)

The leading 1 in numerator and denominator is not cosmetic — it keeps the test
valid (p can never be 0) and is what makes B = 10,000 (Eq. 42) give a floor of
about 1e-4.

Benjamini-Hochberg (Eq. 44):

    q_(i) = min_{j >= i} ( m / j ) p_(j)

A structure-property matrix is dozens to hundreds of tests.  Sec. 8.2 is
explicit that raw p-values are not evidence without this correction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

#  Permutations per chunk.  Bounds peak memory at ~CHUNK x n floats regardless
#  of how large B gets.
_PERM_CHUNK = 2_000


def _unit_center(values: np.ndarray) -> np.ndarray | None:
    """Center and scale to unit norm, so that Pearson r is a plain dot product."""
    centered = values - values.mean()
    norm = float(np.linalg.norm(centered))
    if norm == 0.0:
        return None  # zero variance: correlation undefined
    return centered / norm


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    *,
    b: int = 10_000,
    method: str = "pearson",
    seed: int | None = None,
) -> tuple[float | None, float | None]:
    """Two-sided permutation p-value for the correlation of ``x`` and ``y``.

    Both arrays must already be complete-case aligned (Sec. 7.3) and of equal
    length.  Returns ``(r_obs, p_perm)``; both are ``None`` when the correlation
    is undefined (fewer than 3 points, or zero variance).

    Only ``y`` is permuted, which is the correct null: it destroys any
    association with ``x`` while preserving each marginal distribution exactly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must align; got {x.shape} and {y.shape}.")
    n = x.size
    if n < 3:
        return None, None

    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    elif method != "pearson":
        raise ValueError(f"Unknown method {method!r}; use 'pearson' or 'spearman'.")

    xn, yn = _unit_center(x), _unit_center(y)
    if xn is None or yn is None:
        return None, None

    r_obs = float(xn @ yn)
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    remaining = int(b)
    threshold = abs(r_obs) - 1e-12  # guard against floating-point self-exclusion

    while remaining > 0:
        size = min(_PERM_CHUNK, remaining)
        # argsort of uniform noise gives independent uniform permutations.
        perms = np.argsort(rng.random((size, n)), axis=1)
        r_null = yn[perms] @ xn            # (size,) correlations under the null
        at_least_as_extreme += int(np.count_nonzero(np.abs(r_null) >= threshold))
        remaining -= size

    p_perm = (1.0 + at_least_as_extreme) / (b + 1.0)
    return r_obs, float(p_perm)


def benjamini_hochberg(p_values) -> np.ndarray:
    """Eq. (44): FDR-adjusted q-values, in the input order.

    ``None``/NaN entries propagate as NaN and are excluded from the family size
    ``m`` — a cell whose correlation was undefined was never a test.
    """
    raw = np.asarray(
        [np.nan if p is None else float(p) for p in np.asarray(p_values, dtype=object)],
        dtype=float,
    )
    q = np.full(raw.shape, np.nan, dtype=float)

    valid = np.flatnonzero(np.isfinite(raw))
    m = valid.size
    if m == 0:
        return q

    order = valid[np.argsort(raw[valid], kind="mergesort")]  # stable: ties keep input order
    p_sorted = raw[order]
    ranks = np.arange(1, m + 1, dtype=float)

    scaled = (m / ranks) * p_sorted
    # min_{j >= i}: running minimum from the largest p-value downwards.
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def classify_outcome(
    r: float | None, q: float | None, predicted_sign: str | None, *, alpha: float = 0.05
) -> str:
    """Compare a result against its pre-registered sign (FOM_PROOF Table 6).

    Returns ``supports`` / ``contradicts`` / ``inconclusive``.

    Sec. 4.2: a result that contradicts the prediction is reported as a
    contradiction.  It is not retroactively explained away, so a significant
    result with the wrong sign returns ``contradicts`` — there is no branch here
    that can turn it into a success.
    """
    if r is None or q is None or not np.isfinite(r) or not np.isfinite(q):
        return "inconclusive"
    if predicted_sign in (None, "", "test"):
        return "inconclusive"  # pre-registered as "test empirically"; no directional claim
    if q > alpha:
        return "inconclusive"
    observed = "+" if r > 0 else "-"
    return "supports" if observed == predicted_sign else "contradicts"
