"""Transforms and normalization (FOM_PROOF Sec. 5).

Three separate operations live here, and conflating them is a common way to
produce a wrong score:

  * **Transform** (Eq. 18) — log10 for strongly right-skewed positives.
  * **Standardization** (Eqs. 20-22) — z-scores, for correlation and regression.
    Never used inside a composite score.
  * **Normalization** (Eqs. 23-26) — min-max onto [0, 1] with a direction
    correction and a numerical floor.  Used *only* to build the geometric FOM.

The bounds are part of the score definition, not a property of whatever
happens to be in the table today (Sec. 5.3).  ``bounds_from_population``
therefore refuses to derive bounds from a handful of materials, because
pairwise min-max forces values to exactly 0 and 1 and produces an artifact
rather than a ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cnms_fom.db.enums import Direction, Transform

LN10 = math.log(10.0)

#  Below this, min-max bounds are an artifact of the sample, not a scale.
MIN_MATERIALS_FOR_BOUNDS = 5


@dataclass(frozen=True)
class NormalizationSpec:
    """How one property is mapped onto the unit interval for a geometric score.

    ``lo``/``hi`` are expressed in *transformed* space: if ``transform`` is
    log10, they are log10 values.  Storing them that way means Eq. (25)'s
    log-scale loss normalization is the same code path as Eq. (23), not a
    special case.
    """

    property_key: str
    lo: float
    hi: float
    direction: Direction = Direction.BENEFIT
    transform: Transform = Transform.NONE
    floor_eps: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.lo) or not math.isfinite(self.hi):
            raise ValueError(f"{self.property_key}: bounds must be finite.")
        if self.hi <= self.lo:
            raise ValueError(
                f"{self.property_key}: hi ({self.hi}) must exceed lo ({self.lo})."
            )
        if not 0.0 < self.floor_eps < 1.0:
            raise ValueError(f"{self.property_key}: floor_eps must lie in (0, 1).")

    @classmethod
    def from_dict(cls, property_key: str, spec: dict) -> NormalizationSpec:
        return cls(
            property_key=property_key,
            lo=float(spec["lo"]),
            hi=float(spec["hi"]),
            direction=Direction(spec.get("direction", "benefit")),
            transform=Transform(spec.get("transform", "none")),
            floor_eps=float(spec.get("floor_eps", 1e-3)),
        )

    def as_dict(self) -> dict:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "direction": self.direction.value,
            "transform": self.transform.value,
            "floor_eps": self.floor_eps,
        }


@dataclass(frozen=True)
class NormalizedValue:
    """A normalized input together with the facts needed to audit it."""

    property_key: str
    raw: float
    transformed: float
    z: float
    floored: bool
    out_of_bounds: bool

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "transformed": self.transformed,
            "z": self.z,
            "floored": self.floored,
            "out_of_bounds": self.out_of_bounds,
        }


# ---------------------------------------------------------------------------
# Transform (Eq. 18)
# ---------------------------------------------------------------------------


def apply_transform(value: float, transform: Transform) -> float:
    """Eq. (18): X' = log10(X) for declared right-skewed positive variables."""
    if transform is Transform.NONE:
        return float(value)
    if transform is Transform.LOG10:
        if value <= 0.0:
            raise ValueError(
                f"log10 transform requires a strictly positive value, got {value!r}. "
                "A non-positive reading here is a data-quality problem, not something to clamp."
            )
        return math.log10(value)
    raise ValueError(f"Unsupported transform {transform!r}.")


def transform_derivative(value: float, transform: Transform) -> float:
    """du/dP for the declared transform — needed by the chain rule in Sec. 10.2."""
    if transform is Transform.NONE:
        return 1.0
    if transform is Transform.LOG10:
        return 1.0 / (value * LN10)
    raise ValueError(f"Unsupported transform {transform!r}.")


# ---------------------------------------------------------------------------
# Standardization (Eqs. 20-22)
# ---------------------------------------------------------------------------


def standardize(values: np.ndarray) -> np.ndarray:
    """Eqs. (20)-(22): z = (X - mean) / s, with the n-1 denominator.

    Makes regression coefficients comparable across descriptors.  Sec. 5.2 is
    explicit that it does *not* remove confounding or establish causality.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        raise ValueError("Standardization needs at least two finite observations.")
    mean = finite.mean()
    sd = finite.std(ddof=1)
    if sd == 0.0:
        raise ValueError("Zero variance: this variable cannot be standardized.")
    return (arr - mean) / sd


# ---------------------------------------------------------------------------
# Normalization for composite scores (Eqs. 23-26)
# ---------------------------------------------------------------------------


def normalize_value(value: float, spec: NormalizationSpec) -> NormalizedValue:
    """Map a raw property onto [floor_eps, ...] for use in a geometric score.

    Eq. (23) for benefit, Eq. (24) for cost, Eq. (25) when the cost is on a log
    scale (loss tangent), Eq. (26) for the floor.

    Values outside the declared bounds are flagged rather than clipped.  An
    out-of-bounds material is a signal that the bounds need re-freezing under a
    new FOM version — silently clipping would hide that.
    """
    u = apply_transform(value, spec.transform)
    t = (u - spec.lo) / (spec.hi - spec.lo)
    z_raw = t if spec.direction is Direction.BENEFIT else 1.0 - t
    out_of_bounds = not (0.0 <= t <= 1.0)
    z = max(z_raw, spec.floor_eps)  # Eq. (26)
    return NormalizedValue(
        property_key=spec.property_key,
        raw=float(value),
        transformed=float(u),
        z=float(z),
        floored=z_raw < spec.floor_eps,
        out_of_bounds=out_of_bounds,
    )


def dlnz_dproperty(value: float, spec: NormalizationSpec) -> float:
    """d ln z_q / dP_q — the inner factor of Gamma in Eq. (54).

    Chain rule through the declared transform and direction correction:

        u  = g(P)                      du/dP = 1  or  1/(P ln 10)
        t  = (u - lo) / (hi - lo)      dt/du = 1 / (hi - lo)
        z  = t   (benefit)   or   1 - t   (cost)
        d ln z / dP = (dz/dP) / z

    Returns 0 where the floor is active: the score is locally flat there, and
    pretending otherwise would put a spurious sensitivity into the mediated
    effect.
    """
    normalized = normalize_value(value, spec)
    if normalized.floored:
        return 0.0
    sign = 1.0 if spec.direction is Direction.BENEFIT else -1.0
    dz_dp = sign * transform_derivative(value, spec.transform) / (spec.hi - spec.lo)
    return dz_dp / normalized.z


# ---------------------------------------------------------------------------
# Deriving bounds — deliberately awkward to do casually
# ---------------------------------------------------------------------------


def bounds_from_population(
    values, *, transform: Transform = Transform.NONE, min_n: int = MIN_MATERIALS_FOR_BOUNDS
) -> tuple[float, float, int]:
    """Derive (lo, hi) in transformed space from an eligible population.

    Raises when the population is too small.  From Sec. 5.3:

        "Never normalize a two-material comparison by pairwise min-max scaling;
         that procedure forces values to 0 and 1 and produces an artifact rather
         than a meaningful ranking."

    The returned bounds still have to be written into a ``FomDefinition`` and
    frozen before any score computed with them is reportable.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_n:
        raise ValueError(
            f"Refusing to derive min-max bounds from {arr.size} material(s); "
            f"at least {min_n} are required (FOM_PROOF Sec. 5.3). "
            "Declare bounds explicitly from a defensible reference population instead."
        )
    transformed = np.array([apply_transform(float(v), transform) for v in arr])
    lo, hi = float(transformed.min()), float(transformed.max())
    if hi == lo:
        raise ValueError("Degenerate bounds: every eligible value is identical.")
    return lo, hi, int(arr.size)
