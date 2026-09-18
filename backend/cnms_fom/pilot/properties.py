"""Recipe → property vector for the pilot.

This is the layer that turns a growth recipe into the P vector a FOM needs, and
it is the layer to read sceptically. Two very different kinds of relation live
here and the code keeps them apart, because replacing them has different costs:

**Derived physics** — exact given its inputs. The interfacial-layer series
capacitance is just two capacitors in series (Eq. 27 territory); it is why a
thinner HfO2 film does *not* keep improving EOT, and it needs no calibration.

**Heuristics** — stated functional forms with plausible coefficients, chosen so
the pilot has a non-trivial optimisation landscape. They are *not* measurements
and not fits to any dataset. Each carries a ``HEURISTIC`` marker and the
coefficients sit in one table at the top, so replacing them with a fit to CNMS
data is a single edit rather than an archaeology exercise.

Everything produced here is tiered MODELED, so any FOM built on it comes back
ILLUSTRATIVE (FOM_PROOF Sec. 2.3). That is the correct outcome for a simulated
loop: it exercises the machinery without producing a reportable ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#  Recipe parameters the pilot search space exposes.
PILOT_RECIPE_KEYS = ("thickness_ang", "roughness_ang", "dopant_fraction")

#  SiO2 reference permittivity, for EOT.
K_SIO2 = 3.9


@dataclass(frozen=True)
class PilotConstants:
    """Reference values and heuristic coefficients, all in one place.

    TODO(CNMS): replace every ``heuristic_*`` entry with a fit to measured CNMS
    data. The reference values are literature figures for monoclinic HfO2 on Si
    and are a starting point, not a dataset.
    """

    # --- reference film properties (literature, monoclinic HfO2) -------------
    k_film: float = 25.0            # dielectric constant
    eg_film: float = 5.7            # eV
    dec_film: float = 1.5           # eV, conduction-band offset to Si
    ebd_film: float = 4.0           # MV/cm, intrinsic breakdown field
    tan_delta_film: float = 2.0e-3
    kappa_film: float = 1.1         # W/(m K)

    # --- the dopant oxide (Al2O3) -------------------------------------------
    k_dopant: float = 9.0
    eg_dopant: float = 6.5
    dec_dopant: float = 2.1

    # --- interfacial SiO2 ----------------------------------------------------
    k_interfacial: float = K_SIO2
    interfacial_thickness_ang: float = 10.0

    # --- heuristic coefficients ---------------------------------------------
    #  Phase-stabilisation bump: a few percent Al or Si drives HfO2 toward the
    #  higher-k tetragonal phase before the low-k dopant oxide dominates.
    heuristic_phase_bump: float = 0.55       # peak fractional gain in k
    heuristic_phase_peak_x: float = 0.06     # dopant fraction at the peak
    #  Field enhancement at surface asperities reduces the usable breakdown
    #  field roughly as 1/(1 + beta*sigma/t).
    heuristic_roughness_beta: float = 6.0
    #  Breakdown scales steeply with bandgap in wide-gap oxides; exponent ~2.
    heuristic_ebd_gap_exponent: float = 2.0
    #  Disorder from doping and roughness raises dielectric loss.
    heuristic_loss_dopant_slope: float = 4.0
    heuristic_loss_roughness_slope: float = 0.08


CONSTANTS = PilotConstants()


@dataclass
class SimulatedProperties:
    """The P vector for one recipe, with its derivation recorded."""

    properties: dict[str, float]
    derived_physics: dict[str, str] = field(default_factory=dict)
    heuristics: dict[str, str] = field(default_factory=dict)
    intermediates: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "properties": self.properties,
            "derived_physics": self.derived_physics,
            "heuristics": self.heuristics,
            "intermediates": self.intermediates,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Derived physics
# ---------------------------------------------------------------------------


def series_effective_k(
    t_film_ang: float, k_film: float, t_interfacial_ang: float, k_interfacial: float
) -> float:
    """Effective permittivity of a film stacked on an interfacial oxide.

    Two capacitors in series:

        t_total / k_eff = t_film / k_film + t_IL / k_IL

    Exact given the inputs, and the single most important effect in this pilot:
    a native SiO2 interlayer with k = 3.9 caps what any high-k film can achieve,
    and its influence grows as the film thins. Omitting it would make thinner
    always better, which is the opposite of what gate stacks actually do.
    """
    if k_film <= 0 or k_interfacial <= 0:
        raise ValueError("Permittivities must be positive.")
    total = t_film_ang + t_interfacial_ang
    return total / (t_film_ang / k_film + t_interfacial_ang / k_interfacial)


def equivalent_oxide_thickness(
    t_film_ang: float, k_film: float, t_interfacial_ang: float
) -> float:
    """EOT in angstrom: the SiO2 thickness giving the same capacitance.

    The interlayer contributes its own thickness one-for-one, because it *is*
    SiO2 — which is why it sets the floor on achievable EOT.
    """
    return t_film_ang * (K_SIO2 / k_film) + t_interfacial_ang


# ---------------------------------------------------------------------------
# Heuristics — stated forms, plausible coefficients, no fit behind them
# ---------------------------------------------------------------------------


def _doped_k(dopant_fraction: float, c: PilotConstants) -> float:
    """HEURISTIC: k versus dopant fraction, with a phase-stabilisation peak.

    Linear mixing toward the low-k dopant oxide, plus a bump peaking at
    ``heuristic_phase_peak_x`` that stands in for stabilisation of the
    tetragonal phase. Captures the real experimental shape — k rises then falls
    — without claiming any particular dopant chemistry.
    """
    x = max(0.0, min(dopant_fraction, 1.0))
    mixed = (1 - x) * c.k_film + x * c.k_dopant
    peak = c.heuristic_phase_peak_x
    bump = c.heuristic_phase_bump * math.exp(-(((x - peak) / (peak or 1.0)) ** 2))
    return mixed * (1.0 + bump)


def _doped_gap(dopant_fraction: float, c: PilotConstants) -> tuple[float, float]:
    """HEURISTIC: bandgap and band offset by linear mixing.

    Al2O3 has both a wider gap and a larger offset than HfO2, so doping trades
    permittivity for reliability. That trade-off is the reason the pilot has an
    interior optimum rather than a corner solution.
    """
    x = max(0.0, min(dopant_fraction, 1.0))
    return (
        (1 - x) * c.eg_film + x * c.eg_dopant,
        (1 - x) * c.dec_film + x * c.dec_dopant,
    )


def _roughness_derated_breakdown(
    ebd_intrinsic: float, roughness_ang: float, thickness_ang: float, c: PilotConstants
) -> float:
    """HEURISTIC: field enhancement at asperities lowers usable breakdown.

        E_bd,eff = E_bd / (1 + beta * sigma / t)

    A rough surface concentrates field at the peaks, so breakdown occurs at a
    lower nominal field. The ratio sigma/t is what matters, which is why
    roughness hurts thin films disproportionately.
    """
    if thickness_ang <= 0:
        raise ValueError("Thickness must be positive.")
    return ebd_intrinsic / (1.0 + c.heuristic_roughness_beta * roughness_ang / thickness_ang)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def simulate_properties(
    recipe: dict, *, constants: PilotConstants = CONSTANTS
) -> SimulatedProperties:
    """Turn a recipe into the property vector a FOM consumes.

    Returns every value tiered-MODELED by construction; the caller is expected
    to store them as such.
    """
    missing = [key for key in ("thickness_ang", "roughness_ang") if key not in recipe]
    if missing:
        raise KeyError(f"Recipe is missing {missing}; expected {list(PILOT_RECIPE_KEYS)}.")

    thickness = float(recipe["thickness_ang"])
    roughness = float(recipe["roughness_ang"])
    dopant = float(recipe.get("dopant_fraction", 0.0))
    c = constants

    if thickness <= 0:
        raise ValueError(f"thickness_ang must be positive, got {thickness}.")
    if roughness < 0:
        raise ValueError(f"roughness_ang must be non-negative, got {roughness}.")

    # --- film properties under doping (heuristic) --------------------------
    k_film = _doped_k(dopant, c)
    eg, dec = _doped_gap(dopant, c)

    # --- the stack the device actually sees (derived physics) --------------
    k_eff = series_effective_k(
        thickness, k_film, c.interfacial_thickness_ang, c.k_interfacial
    )
    eot = equivalent_oxide_thickness(thickness, k_film, c.interfacial_thickness_ang)

    # --- reliability (heuristic) -------------------------------------------
    ebd_intrinsic = c.ebd_film * (eg / c.eg_film) ** c.heuristic_ebd_gap_exponent
    ebd = _roughness_derated_breakdown(ebd_intrinsic, roughness, thickness, c)

    tan_delta = c.tan_delta_film * (
        1.0
        + c.heuristic_loss_dopant_slope * dopant
        + c.heuristic_loss_roughness_slope * roughness
    )

    return SimulatedProperties(
        properties={
            "k": k_eff,
            "Eg": eg,
            "dEc": dec,
            "Ebd": ebd,
            "tan_delta": tan_delta,
            "kappa_th": c.kappa_film,
        },
        derived_physics={
            "k": (
                "Series capacitance of film and interfacial SiO2: "
                "t_total/k_eff = t_film/k_film + t_IL/k_IL. Exact given the inputs."
            ),
        },
        heuristics={
            "k_film": "Linear dopant mixing plus a phase-stabilisation peak. HEURISTIC.",
            "Eg": "Linear mixing between the host and dopant oxide gaps. HEURISTIC.",
            "dEc": "Linear mixing of band offsets. HEURISTIC.",
            "Ebd": (
                "E_bd ~ Eg^2, derated by roughness-driven field enhancement "
                "1/(1 + beta*sigma/t). HEURISTIC."
            ),
            "tan_delta": "Rises linearly with dopant disorder and roughness. HEURISTIC.",
            "kappa_th": "Held at the film reference value; no recipe dependence modelled.",
        },
        intermediates={
            "k_film_doped": k_film,
            "eot_ang": eot,
            "ebd_intrinsic_mv_cm": ebd_intrinsic,
            "interfacial_thickness_ang": c.interfacial_thickness_ang,
        },
        notes=[
            "All values are MODELED. Any FOM computed from them is ILLUSTRATIVE and must "
            "never be reported alongside measurement-based scores (FOM_PROOF Sec. 2.3).",
            "Only the effective-k series capacitance is derived physics; everything else is "
            "a stated heuristic awaiting a fit to CNMS data.",
        ],
    )
