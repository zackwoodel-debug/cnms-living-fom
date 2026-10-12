"""Section 5: every sentence template branch."""

from __future__ import annotations

import pytest
from atlas.authors import GroupResult
from atlas.templates import (
    DensityFact,
    LimitFacts,
    agreement_sentence,
    independence_sentence,
    limit_sentences,
    summary_sentence,
)

# --- 5.1 ------------------------------------------------------------------------------


def base(**overrides):
    kwargs = dict(
        name="Silicon dioxide (fused silica)",
        formula="SiO2",
        material_id=38,
        n_datasets=1,
        wavelength_min_nm=210.0,
        wavelength_max_nm=6700.0,
        n_model_fits=0,
        density=DensityFact(2.2, "density_literature", "literature"),
    )
    kwargs.update(overrides)
    return summary_sentence(**kwargs)


def test_summary_with_a_literature_density():
    assert base() == (
        "Silicon dioxide (fused silica) (SiO2, id 38) has 1 optical dataset covering 210-6700 nm and "
        "a literature density of 2.2 g/cm3 (density_literature)."
    )


def test_summary_counts_model_fits():
    assert ", which is a model fit and" in base(n_model_fits=1)
    assert "has 3 optical datasets covering 210-6700 nm, 1 of them a model fit and" in base(n_datasets=3, n_model_fits=1)
    assert "2 of them model fits" in base(n_datasets=3, n_model_fits=2)


@pytest.mark.parametrize(
    ("kind", "adjective"),
    [("calculated_dft", "DFT-calculated"), ("bulk_approximation", "bulk-approximation"), ("calculated_from_lattice", "lattice-derived")],
)
def test_summary_names_the_density_kind(kind, adjective):
    assert f"a {adjective} density of 5.3 g/cm3" in base(density=DensityFact(5.3, "density_x", kind))


def test_summary_without_a_density_gives_the_reason():
    text = base(density=None, density_absent_reason="no stated density and no supported single MP entry")
    assert text.endswith("and no density in this release (no stated density and no supported single MP entry).")
    assert "no reason recorded" in base(density=None)


def test_summary_without_a_formula():
    assert "(no formula in this release, id 1840)" in base(formula=None, material_id=1840)


def test_summary_without_datasets_and_bad_inputs():
    assert "has no optical dataset and" in base(n_datasets=0, wavelength_min_nm=None, wavelength_max_nm=None)
    with pytest.raises(ValueError):
        base(wavelength_min_nm=None)
    with pytest.raises(ValueError):
        base(density=DensityFact(1.0, "density_x", "guessed"))


# --- 5.2 ------------------------------------------------------------------------------


def test_agreement_sentences():
    assert agreement_sentence(grade="C", consensus_class="single_source", n_voting=1, n_sources=1) == (
        "Only one measured dataset gives n at 633 nm, so its agreement cannot be checked."
    )
    assert agreement_sentence(grade="A", consensus_class="excellent", n_voting=8, n_sources=5) == (
        "8 measured datasets from 5 sources give n at 633 nm and agree within 2% of their median (excellent)."
    )
    assert "within 10% of their median (warning)" in agreement_sentence(
        grade="B", consensus_class="warning", n_voting=2, n_sources=2
    )
    assert "disagree by 10% of their median or more (suspicious)" in agreement_sentence(
        grade="S", consensus_class="suspicious", n_voting=2, n_sources=1
    )
    assert "model fits never vote" in agreement_sentence(grade="D", consensus_class=None, n_model_fits=1)
    assert "only at other stated temperatures" in agreement_sentence(
        grade="T", consensus_class=None, n_other_temperature=3
    )
    assert agreement_sentence(grade="none", consensus_class=None) == "No dataset covers 633 nm."
    with pytest.raises(ValueError):
        agreement_sentence(grade="A", consensus_class=None)
    with pytest.raises(ValueError):
        agreement_sentence(grade="A", consensus_class="perfect")


# --- 5.3 ------------------------------------------------------------------------------


def test_independence_sentences():
    assert independence_sentence(None) is None
    assert independence_sentence(GroupResult("determined", 3, "", ())) == "They come from 3 distinct author groups."
    assert "within one research group" in independence_sentence(GroupResult("determined", 1, "", ()))
    assert independence_sentence(GroupResult("unknown", None, "source(s) 182 have no parseable authors", ())) == (
        "Independence cannot be determined: source(s) 182 have no parseable authors."
    )


# --- 5.4 ------------------------------------------------------------------------------


def test_limits_cover_k_density_sld_dielectric_fits_temperature_and_phase():
    limits = limit_sentences(
        LimitFacts(
            k_status="not_measured",
            density_kind="calculated_dft",
            has_sld=True,
            has_dielectric=True,
            n_model_fits=2,
            unstated_temperature_labels=["rutile | Devore1951 | o-ray"],
            phase_unstated=True,
        )
    )
    assert limits[0] == "k is not measured in any dataset; this does not mean k = 0."
    assert "The density is a DFT calculation (Materials Project), not a measurement." in limits
    assert any("periodictable" in line for line in limits)
    assert any("0 K DFPT" in line for line in limits)
    assert "2 datasets are model fits of the dielectric function, not measured data." in limits
    assert any("rutile | Devore1951 | o-ray states no temperature" in line for line in limits)
    assert "The crystal phase of the optical samples is not stated in this release." in limits
    assert limits[-1].startswith("These are reference values from materials-db")


def test_limits_for_the_other_branches():
    partial = limit_sentences(LimitFacts(k_status="partially_measured", density_kind="absent", density_absent_reason="no reliable value"))
    assert "unknown, not 0" in partial[0]
    assert "no density for this material (no reliable value), so it has no SLD either." in partial[1]
    assert any("standing in for a film" in s for s in limit_sentences(LimitFacts("measured", "bulk_approximation")))
    assert any("experimental lattice" in s for s in limit_sentences(LimitFacts("measured", "calculated_from_lattice")))
    measured = limit_sentences(LimitFacts("measured", "literature"))
    assert len(measured) == 1  # only the reference-data limit
    withheld = limit_sentences(LimitFacts("measured", "literature", dielectric_withheld_reason="its band gap is narrow"))
    assert any("stores no calculated dielectric constant here: its band gap is narrow." in s for s in withheld)
    with pytest.raises(ValueError):
        limit_sentences(LimitFacts("unknown", "literature"))
    with pytest.raises(ValueError):
        limit_sentences(LimitFacts("measured", "guessed"))
