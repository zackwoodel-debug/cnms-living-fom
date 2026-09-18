"""Recovering phase, source, and axis from packed ``dataset_label`` strings.

Every literal in this file is a real label shape taken from
``materials_oxide_test.db`` — 124,547 optical rows across three different field
counts. The parser has to survive all of them.
"""

from __future__ import annotations

import pytest

from cnms_fom.ingest.labels import axis_to_tensor_component, parse_dataset_label


def test_three_field_label_phase_source_axis():
    parsed = parse_dataset_label("corundum/sapphire | Malitson1972 | o-ray")
    assert parsed.phase == "corundum/sapphire"
    assert parsed.source_tag == "Malitson1972"
    assert parsed.axis == "o-ray"
    assert parsed.unclassified == ()


def test_two_field_label_source_and_axis_only():
    """The phase slot is simply absent — position cannot be relied on."""
    parsed = parse_dataset_label("Pestryakov1997 | alpha-axis")
    assert parsed.phase is None
    assert parsed.source_tag == "Pestryakov1997"
    assert parsed.axis == "alpha"


def test_quantity_and_method_label():
    parsed = parse_dataset_label("hcp | xray_sld_real | periodictable_CuKalpha")
    assert parsed.phase == "hcp"
    assert parsed.quantity == "sld_xray"
    assert parsed.method == "periodictable_CuKalpha"


def test_single_token_packs_quantity_and_method():
    parsed = parse_dataset_label("density_MP_DFT")
    assert parsed.quantity == "rho"
    assert parsed.method == "MP_DFT"
    assert parsed.phase is None


def test_neutron_sld_label():
    parsed = parse_dataset_label("neutron_sld_real | periodictable_thermal")
    assert parsed.quantity == "sld_neutron"
    assert parsed.method == "periodictable_thermal"


@pytest.mark.parametrize("label", ["", None, "   "])
def test_empty_labels_are_not_an_error(label):
    parsed = parse_dataset_label(label)
    assert parsed.phase is None and parsed.axis is None


def test_unclassifiable_second_phase_is_surfaced_not_guessed():
    """Two phase-looking tokens is an ambiguity the caller must see."""
    parsed = parse_dataset_label("rutile | anatase | Malitson1972")
    assert parsed.phase == "rutile"
    assert "anatase" in parsed.unclassified


def test_citation_pattern_beats_phase_fallback():
    """An author-year token is a source, not a polymorph called 'Edwards1991'."""
    assert parse_dataset_label("Edwards1991").source_tag == "Edwards1991"
    assert parse_dataset_label("Edwards1991").phase is None


@pytest.mark.parametrize(
    ("axis", "component"),
    [("o-ray", "xx"), ("e-ray", "zz"), ("alpha", "11"), ("gamma", "33"), ("isotropic", "iso")],
)
def test_axis_maps_to_tensor_component(axis, component):
    """Sec. 3.2: the optical axis is what makes a scalar directional."""
    assert axis_to_tensor_component(axis) == component


def test_unknown_axis_yields_no_component():
    """Better to store nothing than a tensor component we cannot justify."""
    assert axis_to_tensor_component("sideways") is None
    assert axis_to_tensor_component(None) is None
