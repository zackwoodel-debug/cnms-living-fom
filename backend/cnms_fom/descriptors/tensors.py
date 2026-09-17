"""Tensor reduction (FOM_PROOF Sec. 3.2).

    "Tensor properties must not be converted to a scalar silently."

Every function here returns the value *and* the rule that produced it, so the
caller has no way to store a scalar without also storing how it was obtained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducedTensor:
    """A scalar derived from a tensor, carrying its own audit trail."""

    value: float
    rule: str
    direction: tuple[float, float, float] | None = None

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "reduction_rule": self.rule,
            "direction": list(self.direction) if self.direction else None,
        }


def _as_3x3(tensor) -> np.ndarray:
    arr = np.asarray(tensor, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 tensor, got shape {arr.shape}.")
    return arr


def isotropic_average(tensor) -> ReducedTensor:
    """Eq. (7): eps_iso = (eps_xx + eps_yy + eps_zz) / 3.

    Only appropriate when an isotropic average is physically justified —
    polycrystalline ceramics with no texture, or an explicitly random
    orientation.  For oriented films and single crystals use
    :func:`directional_component`.
    """
    arr = _as_3x3(tensor)
    return ReducedTensor(value=float(np.trace(arr) / 3.0), rule="trace/3")


def directional_component(tensor, normal) -> ReducedTensor:
    """Eq. (8): eps_perp = n-hat^T eps n-hat.

    ``normal`` is the electric-field direction normal to the relevant device
    interface.  It is normalised here, and retained in the result because the
    scalar is meaningless without it.
    """
    arr = _as_3x3(tensor)
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,):
        raise ValueError(f"Expected a 3-vector direction, got shape {n.shape}.")
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        raise ValueError("Direction vector must be non-zero.")
    n = n / norm
    return ReducedTensor(
        value=float(n @ arr @ n), rule="n.eps.n", direction=(float(n[0]), float(n[1]), float(n[2]))
    )


def dielectric_anisotropy(tensor) -> ReducedTensor:
    """Table 2: A_eps = eps_max / eps_min, from the tensor eigenvalues.

    Retains the directional information that Eq. (7) throws away, which is why
    the protocol lists it as a descriptor in its own right.
    """
    arr = _as_3x3(tensor)
    eigenvalues = np.linalg.eigvalsh(0.5 * (arr + arr.T))  # symmetrise; eps is symmetric
    lo, hi = float(eigenvalues.min()), float(eigenvalues.max())
    if lo <= 0.0:
        raise ValueError(
            f"Non-positive dielectric eigenvalue ({lo:g}); check the tensor before reducing it."
        )
    return ReducedTensor(value=hi / lo, rule="eig_max/eig_min")


def reduce_tensor(tensor, rule: str, normal=None) -> ReducedTensor:
    """Dispatch on a declared rule name.

    The rule must be named up front — there is deliberately no default.
    """
    if rule in ("trace/3", "iso", "isotropic"):
        return isotropic_average(tensor)
    if rule in ("n.eps.n", "directional", "perp"):
        if normal is None:
            raise ValueError("A directional reduction requires the interface normal n-hat.")
        return directional_component(tensor, normal)
    if rule in ("eig_max/eig_min", "anisotropy"):
        return dielectric_anisotropy(tensor)
    raise ValueError(
        f"Unknown reduction rule {rule!r}. Declare one of: trace/3, n.eps.n, eig_max/eig_min."
    )
