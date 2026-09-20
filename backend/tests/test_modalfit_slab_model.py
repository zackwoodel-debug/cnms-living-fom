"""Parsing ModalFit slab models.

The two things that can silently corrupt a fit record are the unit convention
and the vary flag, so most of this file is about those. A factor-of-ten thickness
and a fixed parameter reported as a measurement both look completely plausible
downstream.
"""

from __future__ import annotations

import pytest

from cnms_fom.modalfit.slab_model import (
    SlabModelError,
    load_slab_model,
    parse_slab_model,
)

NESTED = {
    "stack_id": "20260901_hfo2_v3",
    "sample_id": "HFO2-PILOT-07",
    "stack": [
        {"role": "ambient", "label": "air", "optical": {"n": 1.0, "k": 0.0}},
        {
            "role": "layer",
            "label": "hfo2_film",
            "material": "HfO2",
            "structural": {
                "thickness": {"value": 103.4, "min": 50.0, "max": 200.0, "vary": True},
                "roughness": {"value": 4.2, "min": 0.0, "max": 20.0, "vary": True},
            },
            "xray": {"sld_real": {"value": 40.1, "min": 30.0, "max": 50.0, "vary": True}},
            "molecular": {"formula": "HfO2", "density": {"value": 9.1}},
        },
        {"role": "substrate", "label": "silicon", "material": "Si", "xray": {"sld_real": 20.07}},
    ],
}

#  substrates/library.json shape: flat scalars, nanometres.
FLAT = {
    "stack_id": "lib_si_native_oxide",
    "stack": [
        {"role": "layer", "material": "SiO2", "thickness": 2.0, "roughness": 0.3, "n": 1.46,
         "sld": 3.47},
        {"role": "substrate", "material": "Si", "roughness": 0.3, "n": 3.88, "sld": 2.07},
    ],
}


def test_parses_nested_form_and_identifiers():
    model = parse_slab_model(NESTED)
    assert model.sample_id == "HFO2-PILOT-07"
    assert model.stack_id == "20260901_hfo2_v3"
    assert [layer.role for layer in model.layers] == ["ambient", "layer", "substrate"]

    film = model.film_layers[0]
    assert film.thickness_ang == pytest.approx(103.4)
    assert film.roughness_ang == pytest.approx(4.2)
    assert film.formula == "HfO2"
    assert film.free_parameters == ["roughness", "sld_real", "thickness"]


def test_parses_flat_library_form():
    """substrates/library.json puts everything at the top level of a layer."""
    model = parse_slab_model(FLAT)
    oxide = model.film_layers[0]
    assert oxide.material == "SiO2"
    #  Default unit is the angstrom convention of the fitting engine, so a file
    #  written in nm parses as-is until told otherwise — which is the next test.
    assert oxide.thickness_ang == pytest.approx(2.0)
    assert oxide.value_of("n") == pytest.approx(1.46)
    assert oxide.value_of("sld_real") == pytest.approx(3.47)


def test_nanometre_files_are_converted_not_guessed():
    """The format records no unit, so it is an argument, and it scales bounds too."""
    model = parse_slab_model(FLAT, length_units="nm")
    assert model.film_layers[0].thickness_ang == pytest.approx(20.0)
    assert model.length_units == "nm"

    nested_nm = parse_slab_model(NESTED, length_units="nm")
    film = nested_nm.film_layers[0]
    assert film.thickness_ang == pytest.approx(1034.0)
    #  Bounds scale with the value. Scaling one without the other would turn a
    #  converged parameter into one that looks clamped.
    bounds = film.bounds_dict()["thickness"]
    assert bounds == {"min": pytest.approx(500.0), "max": pytest.approx(2000.0)}


def test_a_file_can_declare_its_own_unit():
    payload = {**FLAT, "length_units": "nm"}
    assert parse_slab_model(payload).film_layers[0].thickness_ang == pytest.approx(20.0)


def test_unknown_unit_is_refused():
    with pytest.raises(SlabModelError, match="Unknown length unit"):
        parse_slab_model(FLAT, length_units="furlongs")


def test_a_layer_without_a_role_is_refused():
    """Without roles the stack order is unattributable."""
    broken = {"stack": [{"label": "mystery", "structural": {"thickness": {"value": 10}}}]}
    with pytest.raises(SlabModelError, match="role"):
        parse_slab_model(broken)


def test_missing_stack_is_refused():
    with pytest.raises(SlabModelError, match="stack"):
        parse_slab_model({"sample_id": "x"})


def test_fixed_parameters_are_not_counted_as_free():
    model = parse_slab_model(NESTED)
    film = model.film_layers[0]
    #  Density has a value and no vary flag: an input to the fit, not a result.
    assert film.value_of("density") == pytest.approx(9.1)
    assert "density" not in film.free_parameters
    assert model.free_parameter_count == 3


def test_at_bound_detects_a_clamped_parameter():
    payload = {
        "sample_id": "clamped",
        "stack": [
            {
                "role": "layer",
                "label": "film",
                "structural": {
                    "thickness": {"value": 200.0, "min": 50.0, "max": 200.0, "vary": True},
                    "roughness": {"value": 4.0, "min": 0.0, "max": 20.0, "vary": True},
                },
            }
        ],
    }
    film = parse_slab_model(payload).film_layers[0]
    assert film.get("thickness").at_bound is True
    assert film.get("roughness").at_bound is False
    assert film.clamped_parameters == ["thickness"]


def test_total_thickness_is_none_when_a_layer_is_missing_one():
    """A stack total that quietly omits a layer is worse than no total."""
    payload = {
        "sample_id": "partial",
        "stack": [
            {"role": "layer", "label": "a", "structural": {"thickness": {"value": 50.0}}},
            {"role": "layer", "label": "b", "structural": {"roughness": {"value": 2.0}}},
        ],
    }
    assert parse_slab_model(payload).total_film_thickness_ang is None

    payload["stack"][1]["structural"]["thickness"] = {"value": 30.0}
    assert parse_slab_model(payload).total_film_thickness_ang == pytest.approx(80.0)


def test_parameter_name_variants_normalise_to_one_key():
    payload = {
        "sample_id": "aliases",
        "stack": [
            {
                "role": "layer",
                "label": "film",
                "structural": {"thickness_ang": {"value": 12.0, "vary": True}},
                "xray": {"sldReal": {"value": 33.0}, "SLD_im": {"value": 0.5}},
            }
        ],
    }
    film = parse_slab_model(payload).film_layers[0]
    assert film.thickness_ang == pytest.approx(12.0)
    assert film.value_of("sld_real") == pytest.approx(33.0)
    assert film.value_of("sld_imag") == pytest.approx(0.5)


def test_reversed_bounds_are_normalised_without_dropping_the_parameter():
    payload = {
        "sample_id": "reversed",
        "stack": [
            {
                "role": "layer",
                "label": "film",
                "structural": {"thickness": {"value": 100.0, "min": 200.0, "max": 50.0, "vary": True}},
            }
        ],
    }
    param = parse_slab_model(payload).film_layers[0].get("thickness")
    assert (param.lower, param.upper) == (50.0, 200.0)
    assert param.at_bound is False


def test_unparseable_value_is_absent_not_zero():
    """A thickness that failed to parse is unknown; zero is a physical claim."""
    payload = {
        "sample_id": "junk",
        "stack": [
            {"role": "layer", "label": "film",
             "structural": {"thickness": {"value": "n/a"}, "roughness": {"value": "3.5"}}}
        ],
    }
    film = parse_slab_model(payload).film_layers[0]
    assert film.thickness_ang is None
    assert film.roughness_ang == pytest.approx(3.5)


def test_fit_metadata_is_read_but_never_invented():
    """A populated xray block does not mean XRR data was ever loaded."""
    assert parse_slab_model(NESTED).fit.get("techniques", []) == []

    with_meta = {
        **NESTED,
        "fit": {"techniques": ["xrr", "SE"], "algorithm": "L-BFGS-B", "chi2": 1.84},
    }
    fit = parse_slab_model(with_meta).fit
    assert fit["techniques"] == ["XRR", "SE"]
    assert fit["algorithm"] == "L-BFGS-B"
    assert fit["chi2_total"] == pytest.approx(1.84)


def test_technique_toggles_become_a_technique_list():
    payload = {**NESTED, "fit": {"techniques": {"XRR": True, "SE": False, "NR": True}}}
    assert set(parse_slab_model(payload).fit["techniques"]) == {"XRR", "NR"}


def test_load_from_disk(tmp_path):
    import json

    path = tmp_path / "model.json"
    path.write_text(json.dumps(NESTED))
    assert load_slab_model(path).sample_id == "HFO2-PILOT-07"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SlabModelError, match="not valid JSON"):
        load_slab_model(bad)


def test_describe_renders_the_stack():
    assert parse_slab_model(NESTED).describe() == "air / hfo2_film (103.4 Å) / silicon"
