"""Physical plausibility: the three tiers, and keeping them apart.

The whole value of this module is that a violation, an inconsistency, and a
heuristic flag mean different things. Collapsing them turns a rule of thumb into
a law, which is how a real finding gets discarded for being surprising — so most
of this file is about which tier a given problem lands in.
"""

from __future__ import annotations

import pytest

from cnms_fom.fom_engine.plausibility import (
    check_values,
    electron_fraction,
    parse_formula,
    xray_sld_from_density,
)

# --- composition -----------------------------------------------------------


def test_parses_simple_and_fractional_formulas():
    assert parse_formula("HfO2") == {"Hf": 1.0, "O": 2.0}
    assert parse_formula("SrTiO3") == {"Sr": 1.0, "Ti": 1.0, "O": 3.0}
    assert parse_formula("Hf0.5Zr0.5O2") == {"Hf": 0.5, "Zr": 0.5, "O": 2.0}


def test_unparseable_formula_returns_none_not_a_partial_parse():
    """A missing element gives a wrong electron count, then a confident wrong SLD."""
    assert parse_formula("HfO2·2H2O") is None
    assert parse_formula("Ba(TiO3)") is None
    assert parse_formula("Xx2O3") is None
    assert parse_formula("") is None


def test_xray_sld_matches_the_literature_for_silicon():
    """Si at 2.33 g/cm^3 is ~20.07e-6 A^-2 in the tables.

    The formula gives the electron count only; the remaining couple of percent is
    the dispersion correction, which is why the SLD/density check runs at a wide
    tolerance rather than a tight one.
    """
    predicted = xray_sld_from_density("Si", 2.33)
    assert predicted == pytest.approx(19.7, abs=0.5)
    assert abs(predicted - 20.07) / 20.07 < 0.03


def test_xray_sld_scales_with_density():
    assert xray_sld_from_density("HfO2", 9.68) == pytest.approx(68.7, abs=0.5)
    assert electron_fraction("HfO2") == pytest.approx(0.418, abs=0.002)
    assert xray_sld_from_density("HfO2", 0.0) is None


# --- violations: a number that cannot be true ------------------------------


def test_permittivity_below_one_is_a_violation():
    report = check_values({"k": 0.4})
    assert not report.physical
    assert report.violations
    assert "polarizes against" in report.violations[0].message


def test_a_unit_error_of_six_orders_is_caught_and_named():
    report = check_values({"Ebd": 4.0e6})
    assert report.violations
    #  The message should say what the usual cause is, because it usually is.
    assert "V/cm" in report.violations[0].message
    assert "unit" in report.violations[0].resolution.lower()


def test_optical_permittivity_above_static_is_a_violation():
    """The static response contains everything the optical one does, plus ionic."""
    report = check_values({"k": 4.0, "eps_inf": 9.0})
    assert not report.physical
    finding = report.violations[0]
    assert set(finding.keys) == {"k", "eps_inf"}
    assert "cannot be the smaller" in finding.message


def test_band_offset_larger_than_the_gap_is_a_violation():
    report = check_values({"Eg": 1.2, "dEc": 3.0})
    assert not report.physical
    assert "below its own" in report.violations[0].message


def test_a_negative_band_offset_is_allowed():
    """Type-II alignment is real, and a real finding."""
    report = check_values({"Eg": 5.7, "dEc": -0.3})
    assert not report.violations


def test_nan_is_missing_data_not_a_value():
    report = check_values({"k": float("nan")})
    assert report.violations
    assert "NaN" in report.violations[0].message


# --- inconsistencies: two values that must agree ---------------------------


def test_sld_and_density_disagreeing_is_an_inconsistency():
    """For a fixed composition these two are one measurement, not two."""
    report = check_values({"sld_xray": 40.1, "rho": 9.1}, formula="HfO2")
    assert report.inconsistencies
    finding = report.inconsistencies[0]
    assert set(finding.keys) == {"sld_xray", "rho"}
    assert "64" in finding.message  # the electron density implies ~64.6
    #  The resolution has to name the actual cause, which is the XRR degeneracy.
    assert "degenerate" in finding.resolution


def test_consistent_sld_and_density_pass():
    report = check_values({"sld_xray": 64.6, "rho": 9.1}, formula="HfO2")
    assert report.physical
    assert not report.inconsistencies


def test_sld_density_check_is_skipped_without_a_formula():
    """No electron count, no check — and it says so rather than passing silently."""
    report = check_values({"sld_xray": 40.1, "rho": 9.1})
    assert not report.inconsistencies
    assert "sld_xray_vs_rho" in report.not_checked


def test_sld_density_check_is_skipped_for_an_unparseable_formula():
    report = check_values({"sld_xray": 40.1, "rho": 9.1}, formula="Ba(TiO3)")
    assert "could not parse" in report.not_checked["sld_xray_vs_rho"]


def test_eq_11_decomposition_is_an_inconsistency_not_a_violation():
    """k = eps_inf + eps_ionic holds to measurement error, not exactly."""
    report = check_values({"k": 25.0, "eps_inf": 4.0, "eps_ionic": 5.0})
    assert report.inconsistencies
    assert "Eq. (11)" in report.inconsistencies[0].message

    consistent = check_values({"k": 25.0, "eps_inf": 4.0, "eps_ionic": 21.0})
    assert not consistent.inconsistencies


# --- heuristics: look again, do not exclude --------------------------------


def test_the_k_eg_tradeoff_flags_the_corner_that_is_actually_rare():
    """High k and wide gap together — not a product band, which is the wrong shape."""
    report = check_values({"k": 45.0, "Eg": 7.0})
    assert report.heuristics
    #  The crucial property: a heuristic does not vote on `physical`.
    assert report.physical is True
    finding = report.heuristics[0]
    assert "trade" in finding.message
    assert "a result, not an error" in finding.resolution


def test_ordinary_high_k_oxides_do_not_trip_the_tradeoff_flag():
    """HfO2, ZrO2, TiO2, SiO2, Al2O3 all sit on the normal side of the trend."""
    for k, gap in ((25.0, 5.7), (25.0, 5.8), (80.0, 3.05), (3.9, 9.0), (9.0, 8.8)):
        report = check_values({"k": k, "Eg": gap})
        assert report.heuristics == [], f"k={k}, Eg={gap} should not flag"


def test_a_low_band_offset_is_a_device_finding_not_a_data_problem():
    report = check_values({"dEc": 0.4, "Eg": 5.0})
    assert report.physical is True
    assert any("thermionic" in f.message for f in report.heuristics)
    assert any("not in an exclusion" in f.resolution for f in report.heuristics)


def test_ald_growth_above_a_monolayer_is_flagged_as_not_self_limiting():
    report = check_values(
        {}, context={"growth_technique": "ald", "growth_per_cycle_ang": 6.0}
    )
    assert report.heuristics
    assert "CVD-like" in report.heuristics[0].message
    assert report.physical is True


def test_the_report_states_that_heuristics_are_not_grounds_for_exclusion():
    payload = check_values({"k": 45.0, "Eg": 7.0}).as_dict()
    assert "never grounds to exclude" in payload["note"]
    assert payload["n_heuristic_flags"] >= 1
    assert payload["physical"] is True


def test_unknown_keys_are_reported_as_unchecked_not_as_passing():
    report = check_values({"some_new_quantity": 3.0})
    assert "some_new_quantity" in report.not_checked
    assert report.physical is True


def test_a_realistic_hfo2_row_is_clean():
    report = check_values(
        {"k": 25.0, "eps_inf": 4.0, "Eg": 5.7, "dEc": 1.5, "rho": 9.68, "sld_xray": 68.7},
        formula="HfO2",
    )
    assert report.physical
    assert report.heuristics == []
