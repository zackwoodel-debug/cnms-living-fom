"""Search-space encoding and constraint handling for the BO engine.

BoTorch is not exercised here — these are the pure parts, which is where a
recipe gets silently mangled if anything is wrong.
"""

from __future__ import annotations

import pytest

from cnms_fom.bo_engine.constraints import (
    ConstraintSet,
    apply,
    from_instrument_capabilities,
    validate_recipe,
)
from cnms_fom.bo_engine.space import ParameterSpec, SearchSpace, example_pld_space


def test_log_scale_roundtrip():
    """A pressure spanning decades is modelled in log10 and must come back intact."""
    parameter = ParameterSpec(
        name="p", kind="continuous", lower=1e-2, upper=1e2, log_scale=True
    )
    assert parameter.to_model_space(1.0) == pytest.approx(0.0)
    assert parameter.from_model_space(parameter.to_model_space(37.0)) == pytest.approx(37.0)


def test_integer_parameters_round():
    parameter = ParameterSpec(name="n", kind="integer", lower=1, upper=20)
    value = parameter.from_model_space(7.4)
    assert value == 7
    assert isinstance(value, int)


def test_encode_decode_roundtrip():
    space = example_pld_space()
    recipe = {
        "substrate_temp_c": 700.0,
        "o2_pressure_mtorr": 10.0,
        "laser_fluence_j_cm2": 1.5,
        "repetition_rate_hz": 5,
        "substrate": "SrTiO3(001)",
    }
    decoded = space.decode(space.encode(recipe), {"substrate": "SrTiO3(001)"})
    assert decoded["substrate_temp_c"] == pytest.approx(700.0)
    assert decoded["o2_pressure_mtorr"] == pytest.approx(10.0)
    assert decoded["repetition_rate_hz"] == 5
    assert decoded["substrate"] == "SrTiO3(001)"


def test_log_scale_parameter_rejects_non_positive_lower_bound():
    with pytest.raises(ValueError, match="positive lower bound"):
        ParameterSpec(name="p", kind="continuous", lower=0.0, upper=10.0, log_scale=True)


def test_duplicate_parameter_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        SearchSpace(
            parameters=[
                ParameterSpec(name="t", kind="continuous", lower=0, upper=1),
                ParameterSpec(name="t", kind="continuous", lower=0, upper=2),
            ]
        )


def test_instrument_envelope_tightens_the_space():
    space = SearchSpace(
        parameters=[
            ParameterSpec(name="substrate_temp_c", kind="continuous", lower=400.0, upper=1200.0)
        ]
    )
    constraints = from_instrument_capabilities(
        {"substrate_temp_c": {"min": 25.0, "max": 900.0, "units": "degC"}}
    )
    tightened = apply(space, constraints)
    assert tightened.parameters[0].upper == pytest.approx(900.0)
    assert tightened.parameters[0].lower == pytest.approx(400.0)


def test_empty_intersection_fails_loudly():
    """An unsatisfiable campaign must stop here, not emit unrunnable recipes."""
    space = SearchSpace(
        parameters=[
            ParameterSpec(name="substrate_temp_c", kind="continuous", lower=1000.0, upper=1200.0)
        ]
    )
    constraints = ConstraintSet(bounds={"substrate_temp_c": (25.0, 900.0)})
    with pytest.raises(ValueError, match="does not overlap"):
        apply(space, constraints)


def test_categorical_choices_are_intersected():
    space = SearchSpace(
        parameters=[
            ParameterSpec(name="substrate", kind="categorical", choices=("STO", "MgO", "Si"))
        ]
    )
    tightened = apply(space, ConstraintSet(allowed_choices={"substrate": ["MgO", "Si"]}))
    assert tightened.parameters[0].choices == ("MgO", "Si")


def test_validate_recipe_reports_every_problem():
    space = example_pld_space()
    violations = validate_recipe(
        {"substrate_temp_c": 2000.0, "substrate": "Diamond"}, space
    )
    reasons = {v.parameter: v.reason for v in violations}
    assert "outside" in reasons["substrate_temp_c"]
    assert "not among" in reasons["substrate"]
    assert "missing from the recipe" in reasons["laser_fluence_j_cm2"]


def test_categorical_combination_cap():
    space = SearchSpace(
        parameters=[
            ParameterSpec(name=f"c{i}", kind="categorical", choices=tuple(range(4)))
            for i in range(4)  # 4^4 = 256 > cap
        ]
    )
    with pytest.raises(ValueError, match="exceeds the cap"):
        space.categorical_combinations()
