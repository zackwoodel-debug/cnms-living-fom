"""FOM_PROOF Sec. 10: M = B Gamma, the primary mechanistic result."""

from __future__ import annotations

import numpy as np
import pytest

from cnms_fom.fom_engine.mediation import (
    THEORY_ELASTICITIES,
    gamma_matrix,
    mediated_effect,
    mediated_effect_from_theory,
    propagate_ionic_elasticities_to_k,
    sensitivity_from_elasticities,
)

REFERENCE_P = {"k": 25.0, "Eg": 5.7, "dEc": 1.5, "Ebd": 4.0, "kappa_th": 1.1, "eps_ionic": 20.5}
REFERENCE_S = {"Z_RMS_star": 4.5, "omega_TO_min": 140.0, "V_fu": 34.0, "mu_eff": 12.0}


def test_theory_elasticities_match_eqs_50_to_53():
    assert THEORY_ELASTICITIES["eps_ionic"] == {
        "Z_RMS_star": 2.0,
        "omega_TO_min": -2.0,
        "V_fu": -1.0,
        "mu_eff": -1.0,
    }


def test_elasticity_to_natural_units_conversion():
    """B_jq = (d ln P / d ln S) * P / S, evaluated at the reference point."""
    b, descriptors, properties = sensitivity_from_elasticities(
        {"eps_ionic": {"Z_RMS_star": 2.0}}, {"eps_ionic": 20.0}, {"Z_RMS_star": 4.0}
    )
    assert properties == ["eps_ionic"]
    assert b[descriptors.index("Z_RMS_star"), 0] == pytest.approx(2.0 * 20.0 / 4.0)


def test_gamma_is_weight_times_dlnz(draft_foms):
    """Eq. (54): Gamma_qa = w_aq * d ln z_q / dP_q."""
    from cnms_fom.fom_engine.normalization import dlnz_dproperty

    spec = draft_foms["logic"]
    gamma, properties = gamma_matrix(spec, REFERENCE_P)
    for index, key in enumerate(properties):
        expected = spec.weights[key] * dlnz_dproperty(REFERENCE_P[key], spec.normalization[key])
        assert gamma[index, 0] == pytest.approx(expected)


def test_contributions_sum_to_the_mediated_effect(draft_foms):
    """Eqs. (64)-(65): the per-channel decomposition must reconstruct M."""
    spec = draft_foms["logic"]
    properties = ["k", "Eg", "dEc", "Ebd"]
    descriptors = ["Z_RMS_star", "omega_TO_min"]
    b = np.array([[1.2, 0.0, 0.1, -0.3], [-0.9, 0.2, 0.0, 0.4]])

    result = mediated_effect(b, descriptors, properties, spec, REFERENCE_P, REFERENCE_S)
    for descriptor in descriptors:
        assert sum(result.contributions[descriptor].values()) == pytest.approx(
            result.mediated_effect[descriptor]
        )


def test_dominant_channel_is_the_largest_absolute_contribution(draft_foms):
    """Sec. 10.3: the point of the decomposition is to name the pathway."""
    spec = draft_foms["logic"]
    properties = ["k", "Eg", "dEc", "Ebd"]
    b = np.array([[10.0, 0.01, 0.01, 0.01]])
    result = mediated_effect(b, ["Z_RMS_star"], properties, spec, REFERENCE_P, REFERENCE_S)
    assert result.dominant_property["Z_RMS_star"] == "k"


def test_eq11_propagation_scales_by_the_ionic_fraction():
    """d ln k / d ln S = (d ln eps_ionic / d ln S) * eps_ionic / k."""
    propagated = propagate_ionic_elasticities_to_k({"Z_RMS_star": 2.0}, eps_ionic=20.0, k=25.0)
    assert propagated["Z_RMS_star"] == pytest.approx(2.0 * 0.8)


def test_eq11_propagation_rejects_impossible_split():
    """Eq. (11) requires eps_ionic <= eps_static."""
    with pytest.raises(ValueError, match="exceeds k"):
        propagate_ionic_elasticities_to_k({"Z_RMS_star": 2.0}, eps_ionic=30.0, k=25.0)


def test_theory_path_preserves_the_oscillator_sign_pattern(draft_foms):
    """The +2 / -2 / -1 / -1 pattern must survive the whole chain into M."""
    result = mediated_effect_from_theory(draft_foms["logic"], REFERENCE_P, REFERENCE_S)
    elasticity = result.mediated_elasticity

    assert elasticity["Z_RMS_star"] > 0
    assert elasticity["omega_TO_min"] < 0
    assert elasticity["V_fu"] < 0
    #  Ratios are preserved because every channel passes through the same Gamma.
    assert elasticity["Z_RMS_star"] / abs(elasticity["V_fu"]) == pytest.approx(2.0)
    assert elasticity["omega_TO_min"] == pytest.approx(-elasticity["Z_RMS_star"])


def test_uncovered_channels_are_omitted_not_zeroed(draft_foms):
    """The oscillator model says nothing about Eg/dEc/Ebd, so it must not claim zero."""
    result = mediated_effect_from_theory(draft_foms["logic"], REFERENCE_P, REFERENCE_S)
    assert result.properties == ["k"]
    assert any("No theoretical channel" in note for note in result.notes)


def test_application_weights_scale_the_mediated_effect(draft_foms):
    """Sec. 10.3: B is shared physics; Gamma is the application's policy.

    ``logic`` puts weight 1/4 on k and ``power`` puts 1/5 on it, so the same
    structural descriptor must produce mediated effects in exactly that ratio.
    This is what makes an application-specific answer application-specific.
    """
    logic = mediated_effect_from_theory(draft_foms["logic"], REFERENCE_P, REFERENCE_S)
    power = mediated_effect_from_theory(draft_foms["power"], REFERENCE_P, REFERENCE_S)

    ratio = power.mediated_effect["Z_RMS_star"] / logic.mediated_effect["Z_RMS_star"]
    assert ratio == pytest.approx(
        draft_foms["power"].weights["k"] / draft_foms["logic"].weights["k"]
    )


def test_identical_weights_give_identical_mediated_effects(draft_foms):
    """``logic`` and ``rf`` both weight k at 1/4, so their k-channel effects coincide.

    Worth pinning: it shows the difference between applications comes from the
    weights alone, not from anything incidental in how each score is assembled.
    """
    logic = mediated_effect_from_theory(draft_foms["logic"], REFERENCE_P, REFERENCE_S)
    rf = mediated_effect_from_theory(draft_foms["rf"], REFERENCE_P, REFERENCE_S)
    assert logic.mediated_effect["Z_RMS_star"] == pytest.approx(
        rf.mediated_effect["Z_RMS_star"]
    )
