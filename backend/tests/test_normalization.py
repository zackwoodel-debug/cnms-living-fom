"""FOM_PROOF Sec. 5: transforms, standardization, normalization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnms_fom.db.enums import Direction, Transform
from cnms_fom.fom_engine.normalization import (
    NormalizationSpec,
    apply_transform,
    bounds_from_population,
    dlnz_dproperty,
    normalize_value,
    standardize,
)


def test_benefit_normalization_matches_eq23():
    spec = NormalizationSpec("Eg", lo=1.0, hi=9.0)
    assert normalize_value(1.0, spec).z == pytest.approx(spec.floor_eps)  # floored at the bound
    assert normalize_value(9.0, spec).z == pytest.approx(1.0)
    assert normalize_value(5.0, spec).z == pytest.approx(0.5)


def test_cost_normalization_matches_eq24():
    """Eq. (24): a detrimental property is inverted, so larger is worse."""
    spec = NormalizationSpec("alpha_th", lo=0.0, hi=10.0, direction=Direction.COST)
    assert normalize_value(0.0, spec).z == pytest.approx(1.0)
    assert normalize_value(2.5, spec).z == pytest.approx(0.75)


def test_log_cost_normalization_matches_eq25():
    """Eq. (25): loss tangent on a log scale, 1e-5 best and 1e-1 worst."""
    spec = NormalizationSpec(
        "tan_delta", lo=-5.0, hi=-1.0, direction=Direction.COST, transform=Transform.LOG10
    )
    assert normalize_value(1e-5, spec).z == pytest.approx(1.0)
    assert normalize_value(1e-3, spec).z == pytest.approx(0.5)


def test_floor_applied_and_flagged():
    """Eq. (26): z = max(z_hat, eps), and the caller can see it happened."""
    spec = NormalizationSpec("k", lo=0.0, hi=1.0, floor_eps=1e-3)
    result = normalize_value(-5.0, spec)
    assert result.z == pytest.approx(1e-3)
    assert result.floored
    assert result.out_of_bounds


def test_out_of_bounds_is_flagged_not_clipped():
    """A value above the declared range must surface, not be silently clipped."""
    spec = NormalizationSpec("Eg", lo=1.0, hi=9.0)
    result = normalize_value(12.0, spec)
    assert result.z > 1.0
    assert result.out_of_bounds


def test_log_transform_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        apply_transform(0.0, Transform.LOG10)


def test_standardize_uses_n_minus_1():
    """Eq. (22) specifies the sample standard deviation."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    z = standardize(values)
    assert z.mean() == pytest.approx(0.0)
    assert z.std(ddof=1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("transform", "direction", "value"),
    [
        (Transform.NONE, Direction.BENEFIT, 5.0),
        (Transform.NONE, Direction.COST, 5.0),
        (Transform.LOG10, Direction.BENEFIT, 25.0),
        (Transform.LOG10, Direction.COST, 2e-3),
    ],
)
def test_dlnz_matches_finite_difference(transform, direction, value):
    """The analytic d ln z / dP used by Gamma (Eq. 54) must match numerics."""
    lo, hi = (-5.0, 2.0) if transform is Transform.LOG10 else (0.0, 10.0)
    spec = NormalizationSpec("p", lo=lo, hi=hi, direction=direction, transform=transform)

    analytic = dlnz_dproperty(value, spec)
    h = value * 1e-6
    numeric = (
        math.log(normalize_value(value + h, spec).z) - math.log(normalize_value(value - h, spec).z)
    ) / (2 * h)
    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_dlnz_is_zero_where_floored():
    """A floored input makes the score locally flat; Gamma must reflect that."""
    spec = NormalizationSpec("k", lo=0.0, hi=1.0, floor_eps=1e-3)
    assert dlnz_dproperty(-2.0, spec) == 0.0


def test_bounds_refuse_small_populations():
    """Sec. 5.3: pairwise min-max is an artifact, not a ranking."""
    with pytest.raises(ValueError, match="Refusing to derive"):
        bounds_from_population([10.0, 20.0])


def test_bounds_from_adequate_population():
    lo, hi, n = bounds_from_population([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert (lo, hi, n) == (1.0, 6.0, 6)
