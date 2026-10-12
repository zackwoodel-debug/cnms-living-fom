"""Rubrics 6.1, 6.2, 6.3, 6.5: every branch on built cases."""

from __future__ import annotations

import pytest
from atlas.authors import GroupResult
from atlas.evidence import (
    GRADE_RULES,
    GRADES,
    dataset_kind,
    density_evidence,
    fmt_number,
    grade_n633,
    independence_value,
    is_ambient,
    k_status,
    material_k_status,
    validation_relation,
)

from cnms_fom.materials_release import LabelError

TWO = GroupResult("determined", 2, "2 groups", ((1,), (2,)))
ONE = GroupResult("determined", 1, "connected", ((1, 2),))
UNKNOWN = GroupResult("unknown", None, "source(s) 9 have no parseable authors")

# --- 6.1 ------------------------------------------------------------------------


def test_every_grade_has_a_published_rule():
    assert set(GRADES) == set(GRADE_RULES)


def test_a_needs_excellent_and_two_independent_groups():
    grade = grade_n633(consensus_class="excellent", n_voting=2, n_voting_sources=2, groups=TWO)
    assert grade.grade == "A"
    assert "2 independent author groups" in grade.reason


def test_excellent_within_one_group_is_b():
    grade = grade_n633(consensus_class="excellent", n_voting=3, n_voting_sources=2, groups=ONE)
    assert grade.grade == "B"
    assert "one author group" in grade.reason


def test_excellent_with_unknown_independence_is_b_never_a():
    grade = grade_n633(consensus_class="excellent", n_voting=2, n_voting_sources=2, groups=UNKNOWN)
    assert grade.grade == "B"
    assert "independence unknown" in grade.reason and "no parseable authors" in grade.reason


def test_excellent_with_independence_not_computed_is_b():
    assert grade_n633(consensus_class="excellent", n_voting=2, groups=None).grade == "B"


def test_warning_is_b_even_with_independent_groups():
    grade = grade_n633(consensus_class="warning", n_voting=2, n_voting_sources=2, groups=TWO)
    assert grade.grade == "B"
    assert "warning" in grade.reason


def test_single_source_is_c():
    grade = grade_n633(consensus_class="single_source", n_voting=1, n_voting_sources=1)
    assert grade.grade == "C"


def test_suspicious_is_s():
    grade = grade_n633(consensus_class="suspicious", n_voting=2, n_voting_sources=1, groups=ONE)
    assert grade.grade == "S"
    assert "disagree" in grade.reason


def test_only_model_fits_is_d():
    grade = grade_n633(consensus_class=None, ambient_model_fits=["Adachi1993"])
    assert grade.grade == "D"
    assert "Adachi1993" in grade.reason


def test_model_fit_at_ambient_outranks_a_measured_set_at_another_temperature():
    grade = grade_n633(
        consensus_class=None,
        ambient_model_fits=["dry air | Ciddor1996"],
        other_temperatures=[("dry air | Peck1972", 14.85)],
    )
    assert grade.grade == "D"
    assert "14.85 degC" in grade.reason


def test_only_other_temperatures_is_t():
    grade = grade_n633(
        consensus_class=None,
        other_temperatures=[("liquid | Sinnock1969-90K", -183.15), ("solid | Sinnock1969-20K", -253.15)],
    )
    assert grade.grade == "T"
    assert "-253.15 degC, -183.15 degC" in grade.reason


def test_nothing_covers_633_is_none():
    assert grade_n633(consensus_class=None).grade == "none"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"consensus_class": "great", "n_voting": 2},
        {"consensus_class": "excellent", "n_voting": 1},
        {"consensus_class": "single_source", "n_voting": 2},
    ],
)
def test_impossible_inputs_raise_rather_than_grade(kwargs):
    with pytest.raises(ValueError):
        grade_n633(**kwargs)


@pytest.mark.parametrize(
    ("temperature", "ambient"),
    [(None, True), (15.0, True), (30.0, True), (19.85, True), (14.85, False), (0.0, False), (599.85, False)],
)
def test_ambient_window_is_the_releases(temperature, ambient):
    assert is_ambient(temperature) is ambient


# --- 6.2 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "kind"),
    [
        ("MP_DFT", "calculated_dft"),
        ("bulk_elemental_approximation", "bulk_approximation"),
        ("experimental lattice params", "calculated_from_lattice"),
        ("literature", "literature"),
        ("literature (manufacturer datasheet)", "literature"),
        ("literature (density stated on the RI.info page)", "literature"),
    ],
)
def test_density_kind_from_the_label(origin, kind):
    assert density_evidence(origin, present=True).kind == kind


def test_absent_density_carries_its_reason():
    evidence = density_evidence(None, present=False, absent_reason="PubChem records are inconsistent")
    assert (evidence.kind, evidence.reason) == ("absent", "PubChem records are inconsistent")


def test_absent_density_without_a_recorded_reason_says_so():
    assert "gives no reason" in density_evidence(None, present=False).reason


def test_an_undocumented_density_origin_is_an_error_not_a_guess():
    with pytest.raises(LabelError):
        density_evidence("estimated", present=True)


# --- 6.3 ------------------------------------------------------------------------


def test_dataset_kind():
    assert dataset_kind(None).kind == "measured"
    fit = dataset_kind("RI.info calculation script (values generated from a published model)")
    assert fit.kind == "model_fit" and "calculation script" in fit.reason


# --- 6.4 summary value ---------------------------------------------------------------


def test_independence_value():
    assert independence_value(TWO) == 2
    assert independence_value(ONE) == 1
    assert independence_value(UNKNOWN) == "unknown"
    assert independence_value(None) == "unknown"


# --- 6.5 ------------------------------------------------------------------------


def test_k_status_per_dataset():
    assert k_status(10, 10) == "measured"
    assert k_status(10, 3) == "partially_measured"
    assert k_status(10, 0) == "not_measured"
    with pytest.raises(ValueError):
        k_status(0, 0)
    with pytest.raises(ValueError):
        k_status(3, 4)


def test_k_status_per_material():
    assert material_k_status(["measured", "measured"]) == "measured"
    assert material_k_status(["not_measured"]) == "not_measured"
    assert material_k_status(["measured", "not_measured"]) == "partially_measured"
    with pytest.raises(ValueError):
        material_k_status([])


# --- validation pairs, formatting ------------------------------------------------------


def test_validation_relations():
    assert validation_relation("excellent") == "agrees_with"
    assert validation_relation("warning") == "partly_agrees_with"
    assert validation_relation("suspicious") == "disagrees_with"
    with pytest.raises(ValueError):
        validation_relation("fine")


@pytest.mark.parametrize(
    ("value", "text"),
    [(210.0, "210"), (1.457012124643149, "1.457012124643149"), (-0.14999999999997726, "-0.14999999999997726"), (7, "7"), (1e-05, "1e-05")],
)
def test_numbers_are_printed_exactly(value, text):
    assert fmt_number(value) == text
    assert float(text) == value


def test_non_finite_values_are_refused():
    with pytest.raises(ValueError):
        fmt_number(float("nan"))


# --- contested voters (a label naming a non-ambient temperature, rows storing none) -----


def test_tag_temperature():
    from atlas.evidence import tag_temperature_c

    assert tag_temperature_c("Boyd1971-120C") == 120.0
    assert tag_temperature_c("Wu1993-25.1C") == 25.1
    assert abs(tag_temperature_c("Kerl1995-293K") - 19.85) < 1e-9
    assert tag_temperature_c("Malitson1965") is None
    assert tag_temperature_c("3C (beta, zincblende)") is None  # a polytype, not 3 degC: not a series suffix


def test_a_contested_voter_is_excluded_and_the_rest_graded():
    contested = [("Boyd1971-120C | e-ray", 120.0)]
    grade = grade_n633(consensus_class="excellent", n_voting=2, n_voting_sources=1, groups=ONE, contested=contested)
    assert grade.grade == "C"
    assert "Boyd1971-120C | e-ray names 120 degC" in grade.reason
    three = grade_n633(consensus_class="excellent", n_voting=3, groups=TWO, contested=contested)
    assert three.grade == "B" and "cannot be re-classed" in three.reason
    assert grade_n633(consensus_class="single_source", n_voting=1, contested=contested).grade == "T"
    with pytest.raises(ValueError):
        grade_n633(consensus_class=None, contested=contested)
