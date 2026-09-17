"""FOM_PROOF Sec. 3.2, 4.1, 6.1: tensor reduction and direct physical functions."""

from __future__ import annotations

import numpy as np
import pytest

from cnms_fom.descriptors.tensors import (
    dielectric_anisotropy,
    directional_component,
    isotropic_average,
    reduce_tensor,
)
from cnms_fom.fom_engine.physics import (
    K_SIO2,
    capacitance_density,
    equivalent_oxide_thickness,
    mode_dielectric_contribution,
    static_permittivity,
)

EPS = np.diag([10.0, 10.0, 40.0])


def test_isotropic_average_matches_eq7():
    result = isotropic_average(EPS)
    assert result.value == pytest.approx(60.0 / 3.0)
    assert result.rule == "trace/3"


def test_directional_component_matches_eq8():
    """eps_perp = n-hat^T eps n-hat, for a film normal along z."""
    result = directional_component(EPS, [0.0, 0.0, 1.0])
    assert result.value == pytest.approx(40.0)
    assert result.direction == (0.0, 0.0, 1.0)


def test_directional_component_normalises_the_direction():
    assert directional_component(EPS, [0.0, 0.0, 5.0]).value == pytest.approx(40.0)


def test_anisotropy_is_eigenvalue_ratio():
    assert dielectric_anisotropy(EPS).value == pytest.approx(4.0)


def test_reduction_requires_a_declared_rule():
    """Sec. 3.2: a tensor is never collapsed silently."""
    with pytest.raises(ValueError, match="Unknown reduction rule"):
        reduce_tensor(EPS, "whatever")
    with pytest.raises(ValueError, match="requires the interface normal"):
        reduce_tensor(EPS, "n.eps.n")


def test_static_permittivity_matches_eq11():
    assert static_permittivity(4.5, 20.5) == pytest.approx(25.0)


def test_mode_contribution_follows_eq12_scaling():
    """delta_eps ~ |Z*|^2 / (V mu omega^2): doubling Z* quadruples the contribution."""
    base = mode_dielectric_contribution(2.0, 30.0, 12.0, 100.0)
    doubled_charge = mode_dielectric_contribution(4.0, 30.0, 12.0, 100.0)
    halved_omega = mode_dielectric_contribution(2.0, 30.0, 12.0, 50.0)

    assert doubled_charge / base == pytest.approx(4.0)
    assert halved_omega / base == pytest.approx(4.0)


def test_eot_matches_eq28():
    """A 10 nm k=25 film is equivalent to 10 * 3.9/25 = 1.56 nm of SiO2."""
    assert equivalent_oxide_thickness(10.0, 25.0) == pytest.approx(10.0 * K_SIO2 / 25.0)


def test_capacitance_density_matches_eq27():
    """C/A = eps_0 k / t, with t in metres."""
    assert capacitance_density(25.0, 1e-8) == pytest.approx(8.8541878128e-12 * 25.0 / 1e-8)
