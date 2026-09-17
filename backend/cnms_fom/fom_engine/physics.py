"""Direct physical relations (FOM_PROOF Sec. 4.1 and 6.1).

These are closed-form physics, not fitted models.  Sec. 6.1 is blunt about what
they do and do not deliver:

    "These are direct physical functions. They do not, by themselves, predict
     leakage, breakdown, loss, interface stability, or manufacturability."

So they belong in the F layer as *direct* functions, separate from the composite
application scores in ``scores.py``.
"""

from __future__ import annotations

import math

#  SiO2 reference for equivalent oxide thickness (Eq. 28).
K_SIO2 = 3.9
#  Vacuum permittivity, F/m (CODATA).
EPSILON_0 = 8.8541878128e-12


def static_permittivity(eps_inf: float, eps_ionic: float) -> float:
    """Eq. (11): eps_static = eps_inf + eps_ionic.

    Keeping the split explicit matters — the lattice-polarization mechanism of
    Sec. 4.1 acts on ``eps_ionic`` only, so correlating a structural descriptor
    against total ``k`` dilutes the effect with the electronic term.
    """
    return float(eps_inf) + float(eps_ionic)


def mode_dielectric_contribution(
    z_star: float, volume: float, reduced_mass: float, omega_to: float
) -> float:
    """Eq. (12): delta_eps_m ~ |Z*_m|^2 / (V mu_m omega_TO,m^2).

    A *proportionality*, so the return value is in arbitrary units and is only
    meaningful in ratios between modes or materials computed the same way.  It
    is not a permittivity and must never be written into ``property_values`` as
    one; use it to rank candidates and to sanity-check the Eq. (17) coefficient
    pattern.
    """
    for name, value in (("volume", volume), ("reduced_mass", reduced_mass), ("omega_to", omega_to)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}.")
    return (float(z_star) ** 2) / (float(volume) * float(reduced_mass) * float(omega_to) ** 2)


def capacitance_density(k: float, thickness_m: float) -> float:
    """Eq. (27): C/A = eps_0 k / t, in F/m^2.  ``thickness_m`` is in metres."""
    if thickness_m <= 0:
        raise ValueError(f"Thickness must be positive, got {thickness_m!r}.")
    return EPSILON_0 * float(k) / float(thickness_m)


def equivalent_oxide_thickness(thickness: float, k_high_k: float, k_sio2: float = K_SIO2) -> float:
    """Eq. (28): EOT = t_high-k * (k_SiO2 / k_high-k).

    Returned in whatever length unit ``thickness`` was supplied in.
    """
    if k_high_k <= 0:
        raise ValueError(f"k must be positive, got {k_high_k!r}.")
    return float(thickness) * (float(k_sio2) / float(k_high_k))


def expected_elasticity_pattern() -> dict[str, float]:
    """Eq. (17): the coefficient pattern the oscillator picture predicts.

    (beta_Z, beta_omega, beta_V, beta_mu) ~ (+2, -2, -1, -1)
    """
    return {
        "log_Z_RMS_star": 2.0,
        "log_omega_TO_min": -2.0,
        "log_V_fu": -1.0,
        "log_mu_eff": -1.0,
    }


def softness(omega_to: float) -> float:
    """1 / omega_TO — the descriptor form used by the third row of Table 3.

    Registered as its own hypothesis because it expresses the same mechanism
    with the opposite expected sign, which is a useful consistency check on a
    fitted result.
    """
    if omega_to <= 0:
        raise ValueError(f"omega_TO must be positive, got {omega_to!r}.")
    return 1.0 / float(omega_to)


def log_ratio_sanity(value: float, reference: float) -> float:
    """log10(value / reference) — a scale-free comparison for screening output."""
    if value <= 0 or reference <= 0:
        raise ValueError("Both value and reference must be positive.")
    return math.log10(value / reference)
