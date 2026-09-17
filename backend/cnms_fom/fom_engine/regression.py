"""Multivariable regression and collinearity controls (FOM_PROOF Sec. 9).

    P_q = alpha + sum_j beta_j S_j + sum_l gamma_l C_l + e          (Eq. 45)
    VIF_j = 1 / (1 - R_j^2)                                          (Eq. 46)

Implemented on numpy rather than statsmodels: the model is a plain OLS fit, and
keeping it here means the coefficient convention, the degrees of freedom, and
the complete-case handling are visible and testable instead of inherited.

Sec. 9.2 is the reason ``vif`` exists and is easy to reach: coordination number,
density, volume, bonding, and phonon frequency are mutually correlated, and a
coefficient on one of them is not evidence about the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as student_t

from cnms_fom.db.enums import Transform

from .normalization import apply_transform, standardize

#  Above this, a coefficient should not be interpreted on its own (Sec. 9.2).
VIF_WARNING_THRESHOLD = 10.0


@dataclass
class RegressionFit:
    """An OLS fit with the diagnostics Sec. 9 asks for."""

    response: str
    predictors: list[str]
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    intercept: float
    n: int
    dof: int
    r_squared: float
    adj_r_squared: float
    vif: dict[str, float] = field(default_factory=dict)
    standardized: bool = False
    transform_note: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "response": self.response,
            "predictors": self.predictors,
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "t_values": self.t_values,
            "p_values": self.p_values,
            "intercept": self.intercept,
            "n": self.n,
            "dof": self.dof,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "vif": self.vif,
            "standardized": self.standardized,
            "transform_note": self.transform_note,
            "warnings": self.warnings,
        }


def _design_matrix(columns: dict[str, list], mask: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(v, dtype=float)[mask] for v in columns.values()])


def _complete_mask(response: list, predictors: dict[str, list]) -> np.ndarray:
    def finite(seq) -> np.ndarray:
        arr = np.asarray([np.nan if v is None else float(v) for v in seq], dtype=float)
        return np.isfinite(arr)

    mask = finite(response)
    for values in predictors.values():
        mask &= finite(values)
    return mask


def ols(
    response: list,
    predictors: dict[str, list],
    *,
    response_key: str = "y",
    standardize_inputs: bool = False,
    compute_vif: bool = True,
) -> RegressionFit:
    """Fit Eq. (45) by ordinary least squares on the complete cases.

    ``predictors`` holds both the structural descriptors S_j and the confounders
    C_l; the protocol treats them identically in the fit and distinguishes them
    only in interpretation.  Encode categorical confounders (anion family,
    polymorph, specimen form, provenance tier) as dummy columns before calling.
    """
    if not predictors:
        raise ValueError("At least one predictor is required.")

    mask = _complete_mask(response, predictors)
    n = int(mask.sum())
    p = len(predictors) + 1  # + intercept
    if n <= p:
        raise ValueError(
            f"Complete-case n={n} is not greater than the {p} parameters being fitted. "
            "Drop predictors or widen the eligible set; do not impute (Sec. 2.3)."
        )

    y = np.asarray([np.nan if v is None else float(v) for v in response], dtype=float)[mask]
    x_raw = _design_matrix(predictors, mask)
    names = list(predictors)

    warnings: list[str] = []
    if standardize_inputs:
        x_cols = []
        for j, name in enumerate(names):
            try:
                x_cols.append(standardize(x_raw[:, j]))
            except ValueError as exc:
                raise ValueError(f"Cannot standardize predictor {name!r}: {exc}") from exc
        x_raw = np.column_stack(x_cols)

    design = np.column_stack([np.ones(n), x_raw])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)

    fitted = design @ beta
    residuals = y - fitted
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    dof = n - p
    sigma2 = rss / dof

    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    p_values = 2.0 * student_t.sf(np.abs(t_stats), dof)

    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if tss > 0 else float("nan")

    vif_values: dict[str, float] = {}
    if compute_vif and len(names) > 1:
        vif_values = vif(dict(zip(names, x_raw.T, strict=True)))
        inflated = [k for k, v in vif_values.items() if v >= VIF_WARNING_THRESHOLD]
        if inflated:
            warnings.append(
                f"VIF >= {VIF_WARNING_THRESHOLD:g} for {inflated} (Eq. 46). Sec. 9.2: choose one "
                "descriptor on physical interpretability, or regularize; do not read these "
                "coefficients individually."
            )

    return RegressionFit(
        response=response_key,
        predictors=names,
        coefficients={name: float(b) for name, b in zip(names, beta[1:], strict=True)},
        std_errors={name: float(s) for name, s in zip(names, se[1:], strict=True)},
        t_values={name: float(v) for name, v in zip(names, t_stats[1:], strict=True)},
        p_values={name: float(v) for name, v in zip(names, p_values[1:], strict=True)},
        intercept=float(beta[0]),
        n=n,
        dof=int(dof),
        r_squared=float(r2),
        adj_r_squared=float(adj_r2),
        vif=vif_values,
        standardized=standardize_inputs,
        warnings=warnings,
    )


def vif(predictors: dict[str, np.ndarray]) -> dict[str, float]:
    """Eq. (46): VIF_j = 1 / (1 - R_j^2), regressing each predictor on the rest."""
    names = list(predictors)
    matrix = np.column_stack([np.asarray(predictors[name], dtype=float) for name in names])
    out: dict[str, float] = {}
    n = matrix.shape[0]

    for j, name in enumerate(names):
        target = matrix[:, j]
        others = np.delete(matrix, j, axis=1)
        design = np.column_stack([np.ones(n), others])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ beta
        tss = float(((target - target.mean()) ** 2).sum())
        if tss == 0.0:
            out[name] = float("inf")
            continue
        r2_j = 1.0 - float(residuals @ residuals) / tss
        out[name] = float("inf") if r2_j >= 1.0 else 1.0 / (1.0 - r2_j)
    return out


def fit_log_linear_lattice_model(
    eps_ionic: list,
    z_star: list,
    omega_to: list,
    v_fu: list,
    mu_eff: list | None = None,
) -> tuple[RegressionFit, dict[str, dict]]:
    """Fit Eq. (16) and compare against the predicted pattern of Eq. (17).

        log eps_ionic = b0 + bZ log Z* + bw log omega_TO + bV log V_fu + bmu log mu_eff

        (bZ, bw, bV, bmu) ~ (+2, -2, -1, -1)

    Returns the fit plus a per-coefficient comparison against the oscillator
    prediction.  A coefficient whose 95% interval excludes the predicted value is
    reported as a deviation — Sec. 4.2 requires contradictions to be reported as
    contradictions, so nothing here rescues a disagreeing fit.
    """
    predictors: dict[str, list] = {
        "log_Z_RMS_star": [None if v is None else apply_transform(v, Transform.LOG10) for v in z_star],
        "log_omega_TO_min": [
            None if v is None else apply_transform(v, Transform.LOG10) for v in omega_to
        ],
        "log_V_fu": [None if v is None else apply_transform(v, Transform.LOG10) for v in v_fu],
    }
    predicted = {"log_Z_RMS_star": 2.0, "log_omega_TO_min": -2.0, "log_V_fu": -1.0}

    if mu_eff is not None:
        predictors["log_mu_eff"] = [
            None if v is None else apply_transform(v, Transform.LOG10) for v in mu_eff
        ]
        predicted["log_mu_eff"] = -1.0

    response = [None if v is None else apply_transform(v, Transform.LOG10) for v in eps_ionic]
    fit = ols(response, predictors, response_key="log_eps_ionic")
    fit.transform_note = "All variables log10-transformed; coefficients are elasticities (Eq. 48)."

    t_crit = float(student_t.ppf(0.975, fit.dof))
    comparison: dict[str, dict] = {}
    for name, expected in predicted.items():
        beta = fit.coefficients[name]
        half_width = t_crit * fit.std_errors[name]
        lo, hi = beta - half_width, beta + half_width
        comparison[name] = {
            "estimated": beta,
            "predicted_by_eq17": expected,
            "ci95": [lo, hi],
            "consistent_with_prediction": bool(lo <= expected <= hi),
        }
    return fit, comparison
