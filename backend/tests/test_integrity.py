"""FOM_PROOF Sec. 11: covariance identity, weight overlap, and leakage."""

from __future__ import annotations

import numpy as np
import pytest

from cnms_fom.fom_engine.integrity import (
    check_reconstruction,
    covariance_to_correlation,
    excess_correlation,
    leakage_check,
    null_score_correlation,
    reconstruct_log_score_covariance,
    weight_matrix,
)


def test_covariance_identity_holds_exactly(draft_foms):
    """Eq. (59): Cov(ln F) = W Cov(ln z) W^T, computed both ways."""
    specs = [draft_foms["logic"], draft_foms["rf"]]
    w, names, properties = weight_matrix(specs)

    rng = np.random.default_rng(11)
    ln_z = rng.normal(size=(len(properties), 200)) * 0.4 - 0.6
    ln_f = w @ ln_z

    direct = np.cov(ln_f)
    reconstructed = reconstruct_log_score_covariance(w, np.cov(ln_z))
    ok, error = check_reconstruction(direct, reconstructed, atol=1e-10)
    assert ok, f"max |direct - reconstructed| = {error}"


def test_null_correlation_is_cosine_similarity_of_weights(draft_foms):
    """Eq. (60), checked against the definition."""
    specs = [draft_foms["logic"], draft_foms["power"]]
    null, names, properties = null_score_correlation(specs)

    w, _, _ = weight_matrix(specs, properties)
    expected = (w[0] @ w[1]) / (np.linalg.norm(w[0]) * np.linalg.norm(w[1]))
    assert null[0, 1] == pytest.approx(expected)
    assert null[0, 0] == pytest.approx(1.0)


def test_identical_weights_give_null_correlation_one(draft_foms):
    """Two applications with the same weights correlate perfectly for free."""
    spec = draft_foms["logic"]
    null, _, _ = null_score_correlation([spec, spec])
    assert null[0, 1] == pytest.approx(1.0)


def test_excess_correlation_subtracts_the_weight_artifact():
    observed = np.array([[1.0, 0.9], [0.9, 1.0]])
    null = np.array([[1.0, 0.85], [0.85, 1.0]])
    assert excess_correlation(observed, null)[0, 1] == pytest.approx(0.05)


def test_leakage_check_passes_on_a_correct_pipeline(draft_foms):
    """Eq. (63): ln F is exactly linear in ln z, so the residual must vanish."""
    spec = draft_foms["logic"]
    rng = np.random.default_rng(5)
    n = 40
    ln_z = {q: rng.normal(size=n) * 0.3 - 0.5 for q in spec.weights}
    ln_f = sum(spec.weights[q] * v for q, v in ln_z.items())

    report = leakage_check({"logic": ln_f}, ln_z, {"Z_RMS_star": rng.normal(size=n)})
    assert report.passed
    assert report.residual_std["logic"] < 1e-10


def test_leakage_check_passes_at_small_n_with_many_inputs(draft_foms):
    """A correct pipeline must pass even when residual degrees of freedom are tiny.

    Regression test. With n barely above the number of score inputs, the
    floating-point residual of a reproducible score correlates strongly with
    almost anything by construction. An implementation that correlates against
    that noise reports leakage on every correctly-built pipeline, and the
    giveaway is identical |r| across unrelated descriptors — they are all
    correlating with the same noise vector.
    """
    spec = draft_foms["logic"]
    rng = np.random.default_rng(17)
    n = 8  # 4 score inputs + intercept leaves 3 residual degrees of freedom
    ln_z = {q: rng.normal(size=n) * 0.3 - 0.5 for q in spec.weights}
    ln_f = sum(spec.weights[q] * v for q, v in ln_z.items())

    descriptors = {name: rng.normal(size=n) for name in ("V_fu", "rho", "CN", "d_M_O")}
    report = leakage_check({"logic": ln_f}, ln_z, descriptors)

    assert report.passed, report.failures
    assert report.residual_std["logic"] < 1e-12
    #  Nothing left to correlate with, so the partials are NaN rather than 0.0.
    assert all(np.isnan(v) for v in report.partial_correlations["logic"].values())


def test_leakage_check_names_the_descriptor_behind_an_omitted_input(draft_foms):
    """When the score is not reproducible, the partial correlation points at why."""
    spec = draft_foms["logic"]
    rng = np.random.default_rng(23)
    n = 60
    ln_z = {q: rng.normal(size=n) * 0.3 - 0.5 for q in spec.weights}
    hidden = rng.normal(size=n)
    ln_f = sum(spec.weights[q] * v for q, v in ln_z.items()) + 0.5 * hidden

    report = leakage_check(
        {"logic": ln_f}, ln_z, {"omega_TO_min": hidden, "CN": rng.normal(size=n)}
    )
    assert not report.passed
    partials = report.partial_correlations["logic"]
    #  The descriptor that *is* the omitted channel stands out from one that is not.
    assert abs(partials["omega_TO_min"]) > 0.9
    assert abs(partials["CN"]) < 0.5


def test_leakage_check_catches_an_omitted_input(draft_foms):
    """A score built on a channel not declared as an input must fail the check."""
    spec = draft_foms["logic"]
    rng = np.random.default_rng(6)
    n = 40
    ln_z = {q: rng.normal(size=n) * 0.3 - 0.5 for q in spec.weights}
    hidden = rng.normal(size=n)
    ln_f = sum(spec.weights[q] * v for q, v in ln_z.items()) + 0.5 * hidden

    report = leakage_check({"logic": ln_f}, ln_z)
    assert not report.passed
    assert "not reproducible" in report.failures[0]


def test_covariance_to_correlation_normalises_the_diagonal():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    corr = covariance_to_correlation(cov)
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[0, 1] == pytest.approx(2.0 / 6.0)
