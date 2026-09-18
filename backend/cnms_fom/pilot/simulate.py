"""X-ray reflectivity by Parratt recursion.

Real physics, implemented here rather than pulled in as a dependency so that CI
is deterministic and the pilot runs with no extra install. ``refnx`` is used
instead when it is available and ``prefer_refnx`` is set, which is the path to
take once fitting real data rather than simulating it.

The recursion (Parratt 1954), for layers indexed from ambient:

    k_j   = sqrt((q/2)^2 - 4 pi (rho_j - rho_ambient))
    r_j   = (k_j - k_{j+1}) / (k_j + k_{j+1}) * exp(-2 k_j k_{j+1} sigma_j^2)
    R_j   = (r_j + R_{j+1} e^{2 i k_{j+1} d_{j+1}})
            / (1 + r_j R_{j+1} e^{2 i k_{j+1} d_{j+1}})
    R     = |R_0|^2

The roughness factor is Névot-Croce, which treats an interface as a graded
error-function profile. It is the right correction for the few-angstrom
roughness this pilot explores and breaks down once roughness approaches layer
thickness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .stack import Stack

logger = logging.getLogger(__name__)

#  SLD is carried in 1e-6 A^-2 for readability; the maths wants A^-2.
SLD_SCALE = 1e-6


@dataclass
class XrrResult:
    """A simulated reflectivity curve and the quantities read off it."""

    q: np.ndarray              # A^-1
    reflectivity: np.ndarray   # unitless, 0-1
    critical_q: float          # A^-1, total-external-reflection edge
    engine: str = "parratt"
    notes: list[str] = field(default_factory=list)

    def as_dict(self, max_points: int = 200) -> dict:
        stride = max(1, len(self.q) // max_points)
        return {
            "engine": self.engine,
            "critical_q_ang-1": self.critical_q,
            "q_ang-1": [float(v) for v in self.q[::stride]],
            "reflectivity": [float(v) for v in self.reflectivity[::stride]],
            "notes": self.notes,
        }


def critical_q(sld_1e6: float) -> float:
    """q_c = 4 sqrt(pi * SLD) — the total-external-reflection edge.

    For silicon (SLD 20e-6 A^-2) this gives 0.0317 A^-1, which is the textbook
    value and a quick check that the units have not slipped.
    """
    return 4.0 * float(np.sqrt(np.pi * max(sld_1e6, 0.0) * SLD_SCALE))


def parratt_reflectivity(q: np.ndarray, stack: Stack) -> np.ndarray:
    """Specular reflectivity of ``stack`` on the momentum-transfer grid ``q``."""
    q = np.asarray(q, dtype=float)
    layers = stack.layers
    if len(layers) < 2:
        raise ValueError("A stack needs at least an ambient and a substrate.")

    #  Complex SLD. The minus sign on the imaginary part follows the
    #  n = 1 - delta + i*beta convention, so absorption damps the wave rather
    #  than amplifying it.
    sld = np.array(
        [(layer.sld_real - 1j * layer.sld_imag) * SLD_SCALE for layer in layers],
        dtype=complex,
    )
    thickness = np.array([layer.thickness_ang for layer in layers], dtype=float)
    roughness = np.array([layer.roughness_ang for layer in layers], dtype=float)

    #  k_j for every layer at every q. Ambient SLD is the reference, so k_0 = q/2.
    k = np.sqrt((q[:, None] / 2.0) ** 2 - 4.0 * np.pi * (sld[None, :] - sld[0]))
    #  Pick the branch with Im(k) >= 0: the wave must decay into the material,
    #  not grow. numpy's principal branch does not guarantee this below q_c.
    k = np.where(k.imag < 0, -k, k)

    #  Recurse from the substrate upwards.
    reflectance = np.zeros(q.shape, dtype=complex)
    for j in range(len(layers) - 2, -1, -1):
        k_here, k_next = k[:, j], k[:, j + 1]
        denominator = k_here + k_next
        fresnel = np.divide(
            k_here - k_next,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        #  Névot-Croce: the roughness of the interface at the *top* of layer j+1.
        fresnel = fresnel * np.exp(-2.0 * k_here * k_next * roughness[j + 1] ** 2)

        if j + 1 < len(layers) - 1:
            phase = np.exp(2j * k_next * thickness[j + 1])
            reflectance = (fresnel + reflectance * phase) / (
                1.0 + fresnel * reflectance * phase
            )
        else:
            #  The substrate is semi-infinite: nothing returns from below it.
            reflectance = fresnel

    return np.clip(np.abs(reflectance) ** 2, 0.0, 1.0)


def simulate_xrr(
    stack: Stack,
    *,
    q_min: float = 0.005,
    q_max: float = 0.35,
    n_points: int = 600,
    prefer_refnx: bool = False,
) -> XrrResult:
    """Simulate a reflectivity curve for ``stack``.

    The q range spans the critical edge (~0.03 A^-1 for these materials) through
    several Kiessig fringes, which is where thickness and roughness are actually
    determined.
    """
    q = np.linspace(q_min, q_max, n_points)
    notes: list[str] = []

    if prefer_refnx:
        try:
            return _simulate_with_refnx(q, stack)
        except ImportError:
            notes.append("refnx not installed; used the internal Parratt recursion.")

    reflectivity = parratt_reflectivity(q, stack)
    substrate = stack.layers[-1]
    return XrrResult(
        q=q,
        reflectivity=reflectivity,
        critical_q=critical_q(substrate.sld_real),
        engine="parratt",
        notes=notes,
    )


def _simulate_with_refnx(q: np.ndarray, stack: Stack) -> XrrResult:
    """refnx path, for when this is fitting measured data rather than simulating."""
    from refnx.reflect import SLD as RefnxSLD  # noqa: N811
    from refnx.reflect import ReflectModel

    structure = None
    for layer in stack.layers:
        material = RefnxSLD(layer.sld_real + 1j * layer.sld_imag)
        slab = material(layer.thickness_ang, layer.roughness_ang)
        structure = slab if structure is None else structure | slab

    model = ReflectModel(structure, bkg=0.0, dq=0.0)
    return XrrResult(
        q=q,
        reflectivity=np.asarray(model(q)),
        critical_q=critical_q(stack.layers[-1].sld_real),
        engine="refnx",
    )


def fringe_spacing_to_thickness(result: XrrResult, *, q_floor: float = 0.05) -> float | None:
    """Recover film thickness from the Kiessig fringe spacing: t = 2 pi / dq.

    Not used to feed the FOM — the pilot already knows the thickness it asked
    for. It exists as a closure check: if the simulated curve does not encode
    the thickness that went in, the simulation is wrong, and the test suite says
    so rather than the error propagating silently into a score.

    Approximate for a multilayer, by perhaps 10-15%. A film-on-interlayer stack
    produces two fringe frequencies — one from the total film, one from the
    high-k layer alone — which beat against each other, so a single mean spacing
    lands between the two. That is the physics, not a defect, and it is why this
    is a diagnostic rather than a fitting routine. Use refnx for a real fit.
    """
    mask = result.q > q_floor
    q, reflectivity = result.q[mask], result.reflectivity[mask]
    if q.size < 16:
        return None

    #  Work on log R, where the fringes are near-sinusoidal on a smooth decay.
    signal = np.log(np.clip(reflectivity, 1e-30, None))
    signal = signal - np.polyval(np.polyfit(q, signal, 3), q)

    peaks = [
        i
        for i in range(1, signal.size - 1)
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]
    ]
    if len(peaks) < 2:
        return None
    return float(2.0 * np.pi / np.mean(np.diff(q[peaks])))
