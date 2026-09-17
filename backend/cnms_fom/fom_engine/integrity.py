"""Composite-score covariance and integrity checks (FOM_PROOF Sec. 11).

Sec. 11.1 makes a point that is easy to get wrong in practice: a correlation
between two FOMs is *not* independent evidence that two applications share a
structural cause.  Since

    ln F = W ln z                                                    (Eq. 58)
    Cov(ln F) = W Cov(ln z) W^T                                      (Eq. 59)

any FOM-FOM correlation is manufactured jointly by (1) correlation among the
underlying property channels and (2) overlap between the application weight
vectors.  Eq. (60) isolates the second: under an equal-variance, independent-
property null, the expected score correlation is just the cosine similarity of
the weight vectors.  What is left over (Eq. 61) is the part worth discussing.

The two checks in Sec. 11.3 are cheap and catch real pipeline bugs:

  * direct vs reconstructed R_FF must agree (Eq. 62);
  * a structural descriptor must retain no association with a deterministic
    score once every score input is conditioned on (Eq. 63).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scores import FomSpec

#  Eq. (62) is an equality between two ways of computing the same thing, so the
#  tolerance is numerical, not statistical.
RECONSTRUCTION_ATOL = 1e-8
#  Eq. (63): ln F is exactly linear in ln z, so the residual should vanish.
LEAKAGE_RESIDUAL_ATOL = 1e-8


@dataclass
class IntegrityReport:
    """Outcome of the Sec. 11.3 checks."""

    passed: bool
    max_reconstruction_error: float | None = None
    residual_std: dict[str, float] = field(default_factory=dict)
    partial_correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_reconstruction_error": self.max_reconstruction_error,
            "residual_std": self.residual_std,
            "partial_correlations": self.partial_correlations,
            "failures": self.failures,
            "notes": self.notes,
        }


def weight_matrix(specs: list[FomSpec], properties: list[str] | None = None):
    """Assemble W from Eq. (58): applications (rows) x properties (columns)."""
    if properties is None:
        properties = sorted({q for spec in specs for q in spec.weights})
    w = np.zeros((len(specs), len(properties)), dtype=float)
    for ai, spec in enumerate(specs):
        for qi, q in enumerate(properties):
            w[ai, qi] = float(spec.weights.get(q, 0.0))
    return w, [spec.name for spec in specs], properties


def reconstruct_log_score_covariance(w: np.ndarray, cov_ln_z: np.ndarray) -> np.ndarray:
    """Eq. (59): Cov(ln F) = W Cov(ln z) W^T."""
    w = np.asarray(w, dtype=float)
    cov_ln_z = np.asarray(cov_ln_z, dtype=float)
    if cov_ln_z.shape[0] != w.shape[1]:
        raise ValueError(
            f"W has {w.shape[1]} property columns but Cov(ln z) is {cov_ln_z.shape}."
        )
    return w @ cov_ln_z @ w.T


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Normalise a covariance matrix to a correlation matrix."""
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    return np.where(np.isfinite(corr), corr, np.nan)


def null_score_correlation(specs: list[FomSpec], properties: list[str] | None = None):
    """Eq. (60): r_null_ab = (w_a . w_b) / (||w_a|| ||w_b||).

    The score correlation two applications would show even if every property
    channel were independent with equal variance — i.e. correlation that comes
    from sharing weights, not from physics.
    """
    w, names, props = weight_matrix(specs, properties)
    norms = np.linalg.norm(w, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("An application has an all-zero weight vector.")
    unit = w / norms[:, None]
    return unit @ unit.T, names, props


def excess_correlation(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Eq. (61): r_excess = r_observed - r_null.

    This is the quantity to report and interpret.  A large observed FOM-FOM
    correlation with near-zero excess is explained entirely by weight overlap
    and says nothing about shared physics.
    """
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if observed.shape != null.shape:
        raise ValueError(f"Shapes differ: {observed.shape} vs {null.shape}.")
    return observed - null


def _residualize(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residual of ``target`` after least-squares projection onto ``design`` + intercept."""
    n = target.shape[0]
    full = np.column_stack([np.ones(n), design])
    beta, *_ = np.linalg.lstsq(full, target, rcond=None)
    return target - full @ beta


def check_reconstruction(
    direct: np.ndarray, reconstructed: np.ndarray, *, atol: float = RECONSTRUCTION_ATOL
) -> tuple[bool, float]:
    """Eq. (62): the directly computed and reconstructed R_FF must agree."""
    direct = np.asarray(direct, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if direct.shape != reconstructed.shape:
        raise ValueError(f"Shapes differ: {direct.shape} vs {reconstructed.shape}.")
    both_finite = np.isfinite(direct) & np.isfinite(reconstructed)
    if not both_finite.any():
        return False, float("nan")
    max_error = float(np.max(np.abs(direct[both_finite] - reconstructed[both_finite])))
    return bool(max_error <= atol), max_error


def leakage_check(
    ln_scores: dict[str, np.ndarray],
    ln_z: dict[str, np.ndarray],
    descriptors: dict[str, np.ndarray] | None = None,
    *,
    atol: float = LEAKAGE_RESIDUAL_ATOL,
) -> IntegrityReport:
    """Eq. (63): corr(S_j, F_a | z) must be ~ 0.

    The check is run in log space because that is where the relationship is
    exact: Eq. (30) makes ``ln F_a`` a linear combination of ``ln z_q``, so
    regressing ``ln F_a`` on the ``ln z`` columns must leave a residual at
    machine precision.

    The check runs in two stages, and the order matters:

    1. **Is the score reproducible?**  Regress ln F on the ln z columns. A
       residual above machine precision means the score cannot be rebuilt from
       its declared inputs — an omitted input, an inconsistent normalization, or
       a score computed under bounds other than the ones recorded.  This is the
       pass/fail criterion.

    2. **Only if it is not:** correlate each descriptor against the unexplained
       part, to point at *which* omitted channel is responsible.

    Stage 2 is skipped on a passing pipeline rather than run and ignored.  A
    reproducible score leaves a residual of pure floating-point noise at ~1e-16,
    and correlating anything against noise yields a number — at small n,
    routinely a large one.  Those values are reported as NaN, because "there is
    no residual to correlate with" is the honest answer, not 0.0 and certainly
    not a leakage finding.
    """
    report = IntegrityReport(passed=True)
    if not ln_z:
        report.passed = False
        report.failures.append("No score inputs supplied; nothing to condition on.")
        return report

    z_keys = sorted(ln_z)
    design = np.column_stack([np.asarray(ln_z[k], dtype=float) for k in z_keys])

    for application, values in ln_scores.items():
        y = np.asarray(values, dtype=float)
        if y.shape[0] != design.shape[0]:
            raise ValueError(
                f"{application}: {y.shape[0]} scores but {design.shape[0]} input rows."
            )
        residual = _residualize(y, design)
        residual_std = float(np.std(residual, ddof=1)) if y.size > 1 else 0.0
        scale = max(float(np.std(y, ddof=1)) if y.size > 1 else 1.0, 1.0)
        report.residual_std[application] = residual_std
        reproducible = residual_std <= atol * scale

        if not reproducible:
            report.passed = False
            report.failures.append(
                f"{application}: ln F is not reproducible from its declared ln z inputs "
                f"(residual sd = {residual_std:.3e}). Sec. 11.3: suspect an omitted input, "
                "inconsistent normalization, or a stale FOM version."
            )

        if not descriptors:
            continue

        #  Only correlate against a residual that actually exists.
        #
        #  When ln F is reproducible the residual is numerical noise at ~1e-16.
        #  Correlating a descriptor against that noise produces a number, and at
        #  small n (few residual degrees of freedom) it is routinely large — the
        #  telltale being identical magnitudes across unrelated descriptors,
        #  because they are all correlating with the same noise vector. Reporting
        #  those as leakage would fail every correctly-built pipeline.
        if reproducible:
            report.partial_correlations[application] = dict.fromkeys(descriptors, float("nan"))
            continue

        partials: dict[str, float] = {}
        for descriptor, s_values in descriptors.items():
            s = np.asarray(s_values, dtype=float)
            if s.shape[0] != design.shape[0]:
                raise ValueError(
                    f"{descriptor}: {s.shape[0]} values but {design.shape[0]} input rows."
                )
            s_res = _residualize(s, design)
            if np.std(s_res) <= atol * max(float(np.std(s)), 1.0):
                partials[descriptor] = float("nan")  # descriptor fully explained by the inputs
                continue
            partials[descriptor] = float(np.corrcoef(s_res, residual)[0, 1])
            if abs(partials[descriptor]) > 0.1:
                report.failures.append(
                    f"{application} ~ {descriptor}: partial correlation "
                    f"{partials[descriptor]:.3f} against the unexplained part of the score. "
                    "Sec. 11.3: the omitted input is likely related to this descriptor."
                )
        report.partial_correlations[application] = partials

    if report.passed:
        report.notes.append(
            "Scores are exactly reproducible from their declared inputs; no residual association "
            "with structural descriptors (Eq. 63). NaN partial correlations mean there was no "
            "residual variance left to correlate — that is the expected pass."
        )
    return report
