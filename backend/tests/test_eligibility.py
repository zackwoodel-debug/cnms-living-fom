"""FOM_PROOF Sec. 2: eligibility, context matching, and the missing-data rule.

Uses lightweight stand-ins rather than ORM rows: the eligibility logic is pure
and should be testable without a database.
"""

from __future__ import annotations

from types import SimpleNamespace

from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
from cnms_fom.fom_engine.eligibility import (
    ContextFilter,
    build_analysis_table,
    is_eligible,
    missing_context_fields,
)


def make_property(**overrides):
    base = dict(
        property_key="k",
        value=25.0,
        provenance_tier=ProvenanceTier.MEASURED,
        temperature_k=300.0,
        frequency_hz=1e4,
        tensor_component="zz",
        thickness_nm=10.0,
        electrode="TiN",
        area_cm2=1e-4,
        failure_criterion="1 mA/cm^2",
        field_amplitude_v_per_cm=1e4,
        xc_functional=None,
        material=SimpleNamespace(specimen_form=SpecimenForm.CRYSTALLINE_FILM),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_material(descriptors=(), properties=(), polymorph="monoclinic"):
    return SimpleNamespace(
        formula_reduced="HfO2",
        polymorph=polymorph,
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
        descriptors=list(descriptors),
        properties=list(properties),
    )


def make_descriptor(key, value, tier=ProvenanceTier.CALCULATED, method="pymatgen"):
    return SimpleNamespace(
        descriptor_key=key, value=value, provenance_tier=tier, method=method
    )


def test_na_value_is_never_eligible():
    """Eq. (4): a missing value stays missing."""
    ok, reason = is_eligible(make_property(value=None), "k", ContextFilter())
    assert not ok
    assert "NA" in reason


def test_modeled_tier_excluded_by_default():
    """The default context admits measured and calculated only."""
    ok, reason = is_eligible(
        make_property(provenance_tier=ProvenanceTier.MODELED), "k", ContextFilter()
    )
    assert not ok
    assert "modeled" in reason


def test_specimen_form_mismatch_is_excluded():
    """Sec. 2.1: a ceramic and a film are different records."""
    ok, reason = is_eligible(
        make_property(),
        "k",
        ContextFilter(specimen_forms=(SpecimenForm.BULK_SINGLE_CRYSTAL,)),
    )
    assert not ok
    assert "specimen form" in reason


def test_temperature_outside_context_is_excluded():
    ok, _ = is_eligible(
        make_property(temperature_k=900.0), "k", ContextFilter(temperature_k=(290.0, 310.0))
    )
    assert not ok


def test_tensor_component_mismatch_is_excluded():
    """Sec. 3.2: eps_zz and eps_iso are not interchangeable."""
    ok, reason = is_eligible(
        make_property(tensor_component="xx"), "k", ContextFilter(tensor_component="zz")
    )
    assert not ok
    assert "tensor component" in reason


def test_breakdown_without_failure_criterion_is_ineligible():
    """Sec. 16 item 6: E_bd needs thickness, electrode, area, and failure criterion."""
    value = make_property(property_key="Ebd", value=4.0, failure_criterion=None)
    assert "failure_criterion" in missing_context_fields(value, "Ebd")
    ok, reason = is_eligible(value, "Ebd", ContextFilter())
    assert not ok
    assert "required context" in reason


def test_analysis_table_keeps_missing_as_none():
    material = make_material(
        descriptors=[make_descriptor("V_fu", 34.0)],
        properties=[make_property()],
    )
    table = build_analysis_table([material], ["k", "Eg"], ["V_fu"], ContextFilter())

    assert table.columns["V_fu"] == [34.0]
    assert table.columns["k"] == [25.0]
    assert table.columns["Eg"] == [None]  # absent, not imputed
    assert table.coverage() == {"V_fu": 1, "k": 1, "Eg": 0}


def test_ambiguous_duplicates_are_refused_not_averaged():
    """Sec. 2.1: merging needs an explicit, defensible aggregation rule."""
    material = make_material(
        properties=[make_property(value=25.0), make_property(value=30.0)]
    )
    table = build_analysis_table([material], ["k"], [], ContextFilter())

    assert table.columns["k"] == [None]
    assert any("aggregation rule" in e.reason for e in table.exclusions)


def test_material_key_encodes_the_full_identity():
    """A formula alone is not an identifier (Sec. 2.1)."""
    table = build_analysis_table([make_material()], [], [], ContextFilter())
    assert table.material_keys == ["HfO2|monoclinic|crystalline_film"]
