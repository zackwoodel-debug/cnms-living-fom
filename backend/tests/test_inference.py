"""FOM_PROOF Sec. 8: permutation inference and FDR control."""

from __future__ import annotations

import numpy as np
import pytest

from cnms_fom.fom_engine.inference import (
    benjamini_hochberg,
    classify_outcome,
    permutation_p_value,
)


def test_bh_matches_eq44_by_hand():
    """q_(i) = min_{j>=i} (m/j) p_(j), computed manually for m = 4."""
    p = [0.001, 0.01, 0.03, 0.2]
    q = benjamini_hochberg(p)
    assert q == pytest.approx([0.004, 0.02, 0.04, 0.2])


def test_bh_is_monotone_after_the_running_minimum():
    """The min_{j>=i} step enforces monotonicity even when m/j * p is not."""
    q = benjamini_hochberg([0.01, 0.02, 0.021, 0.9])
    assert all(a <= b + 1e-12 for a, b in zip(q, q[1:], strict=False))


def test_bh_never_exceeds_the_largest_p_value():
    """The min_{j>=i} step caps every q at p_(m), so q can never exceed 1.

    The clip to [0, 1] in the implementation is defensive: (m/m) p_(m) = p_(m),
    and the running minimum propagates that ceiling down to every smaller index.
    """
    q = benjamini_hochberg([0.8, 0.9])
    assert q == pytest.approx([0.9, 0.9])
    assert max(q) <= 1.0


def test_bh_excludes_undefined_cells_from_the_family():
    """A cell with no correlation was never a test, so m must not count it."""
    q = benjamini_hochberg([0.001, None, 0.01, float("nan")])
    assert np.isnan(q[1]) and np.isnan(q[3])
    #  m = 2, not 4: 2/1 * 0.001 = 0.002.
    assert q[0] == pytest.approx(0.002)


def test_permutation_p_has_the_eq41_floor():
    """p = (1 + count) / (B + 1) can never be zero."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 5.0 * x + rng.normal(size=60) * 0.01  # essentially perfect correlation
    r, p = permutation_p_value(x, y, b=999, seed=1)
    assert r > 0.99
    assert p == pytest.approx(1.0 / 1000.0)


def test_permutation_p_is_uniform_ish_under_the_null():
    """With no association, the p-value should not concentrate near zero."""
    rng = np.random.default_rng(7)
    p_values = []
    for _ in range(40):
        x, y = rng.normal(size=30), rng.normal(size=30)
        p_values.append(permutation_p_value(x, y, b=499, seed=int(rng.integers(1 << 30)))[1])
    #  Under the null roughly 5% should fall below 0.05; allow generous slack.
    assert np.mean(np.asarray(p_values) < 0.05) < 0.25


def test_permutation_returns_none_for_tiny_samples():
    """Sec. 15.2: two points do not establish a population correlation."""
    assert permutation_p_value(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == (None, None)


def test_spearman_permutation_uses_ranks():
    """A monotone but strongly nonlinear pair is significant on ranks."""
    x = np.arange(1, 21, dtype=float)
    y = np.exp(x / 3.0)
    r, p = permutation_p_value(x, y, b=999, method="spearman", seed=2)
    assert r == pytest.approx(1.0)
    assert p < 0.01


def test_contradiction_is_reported_as_contradiction():
    """Sec. 4.2: a wrong-signed significant result is a contradiction, not a pass."""
    assert classify_outcome(r=-0.8, q=0.001, predicted_sign="+") == "contradicts"
    assert classify_outcome(r=0.8, q=0.001, predicted_sign="+") == "supports"


def test_non_significant_result_is_inconclusive():
    assert classify_outcome(r=0.8, q=0.4, predicted_sign="+") == "inconclusive"


def test_test_empirically_makes_no_directional_claim():
    """Table 3 rows marked 'test' pre-register no sign, so nothing can support them."""
    assert classify_outcome(r=0.9, q=1e-6, predicted_sign="test") == "inconclusive"
