"""The digital-twin state, and the gate that decides what counts as quantitative.

``is_quantitative`` is the single most consequential function in this package. It
decides whether a number off an electron microscope may be filed as a measurement,
and it answers one question only: did this acquisition record enough about the
column? Not whether the science is good. Whether the context exists.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cnms_fom.pysea.instrument_state import (
    InstrumentState,
    is_quantitative,
    parse_instrument_state,
    required_context,
    state_warnings,
)

QUANTITATIVE = {
    "beam_energy_kev": 60.0,
    "collection_semi_angle_mrad": 2.4,
    "dispersion_ev_per_channel": 0.0021,
    "lens_strength_source": "twin",
    "twin_reconstructed": True,
}


# --- the gate --------------------------------------------------------------


def test_a_twin_reconstructed_calibrated_state_is_quantitative():
    assert is_quantitative(parse_instrument_state(QUANTITATIVE))


def test_a_nominal_state_is_not_quantitative():
    """Nominal lens strengths describe the setpoint, not the column."""
    state = parse_instrument_state({**QUANTITATIVE, "twin_reconstructed": False,
                                    "lens_strength_source": "nominal"})
    assert not is_quantitative(state)


@pytest.mark.parametrize("missing", ["collection_semi_angle_mrad", "dispersion_ev_per_channel"])
def test_a_missing_calibration_quantity_blocks_the_gate(missing):
    raw = {key: value for key, value in QUANTITATIVE.items() if key != missing}
    assert not is_quantitative(parse_instrument_state(raw))


def test_twin_reconstructed_is_read_as_an_explicit_boolean():
    """A truthy string must not read as reconstructed.

    The asymmetry is deliberate: treating a nominal state as reconstructed files a
    wrong number as a measurement, while the reverse is a refusal someone can
    overturn with better metadata.
    """
    for truthy in ("true", "yes", 1, "1"):
        state = parse_instrument_state({**QUANTITATIVE, "twin_reconstructed": truthy})
        assert not state.twin_reconstructed
        assert not is_quantitative(state)


def test_an_unrecognised_lens_strength_source_is_unknown():
    state = parse_instrument_state({**QUANTITATIVE, "lens_strength_source": "vibes"})
    assert state.lens_strength_source == "unknown"


def test_the_dispersion_may_arrive_from_a_referenced_calibration():
    """A FAIR architecture keeps column state and spectrometer calibration apart."""
    raw = {key: value for key, value in QUANTITATIVE.items()
           if key != "dispersion_ev_per_channel"}
    state = parse_instrument_state(raw, calibration={"dispersion_ev_per_channel": 0.0021})
    assert state.dispersion_ev_per_channel == 0.0021
    assert is_quantitative(state)


# --- context ---------------------------------------------------------------


def test_required_context_reports_only_what_the_state_has():
    """A key absent here is absent from the promoted row, which is the point."""
    state = parse_instrument_state(QUANTITATIVE)
    context = required_context(state)
    assert context["beam_energy_kev"] == 60.0
    assert context["twin_reconstructed"] is True
    #  No temperature was recorded, so none is reported.
    assert "temperature_k" not in context


def test_required_context_never_invents_a_temperature():
    state = parse_instrument_state({**QUANTITATIVE, "temperature_k": None})
    assert "temperature_k" not in required_context(state)


# --- warnings --------------------------------------------------------------


def test_a_quantitative_state_warns_about_nothing():
    assert state_warnings(parse_instrument_state(QUANTITATIVE)) == []


def test_each_missing_piece_produces_its_own_warning():
    state = parse_instrument_state({"lens_strength_source": "nominal"})
    warnings = " ".join(state_warnings(state))
    assert "not twin-reconstructed" in warnings
    assert "collection semi-angle" in warnings
    assert "energy dispersion" in warnings
    assert "beam energy" in warnings


def test_an_expired_calibration_is_flagged_against_the_acquisition_date():
    window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    window_end = datetime(2026, 3, 1, tzinfo=timezone.utc)
    state = parse_instrument_state({
        **QUANTITATIVE,
        "calibration_id": "cal-1",
        "calibration_validity_window": [window_start.isoformat(), window_end.isoformat()],
    })

    during = state_warnings(state, at=window_end - timedelta(days=1))
    assert during == []

    after = state_warnings(state, at=window_end + timedelta(days=1))
    assert any("expired" in text for text in after)


def test_a_calibration_with_no_window_says_so():
    state = parse_instrument_state({**QUANTITATIVE, "calibration_id": "cal-1"})
    assert any("no validity window" in text for text in state_warnings(state))


def test_hand_entered_lens_strengths_are_called_out():
    state = parse_instrument_state({**QUANTITATIVE, "lens_strength_source": "user-entered"})
    assert any("entered by hand" in text for text in state_warnings(state))


def test_a_bare_state_round_trips_through_as_dict():
    state = InstrumentState()
    payload = state.as_dict()
    assert payload["twin_reconstructed"] is False
    assert payload["beam_energy_kev"] is None
    assert payload["calibration_validity_window"] is None
