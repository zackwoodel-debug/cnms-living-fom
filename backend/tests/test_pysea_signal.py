"""Signals, axes, calibration. The rule under test is that units are never guessed.

An energy axis with no units is not an eV axis with the label missing. Defaulting
it would turn a labelling bug into a number with a unit stamped on it by the parser
rather than by the instrument, and every downstream comparison would inherit that.
"""

from __future__ import annotations

from cnms_fom.pysea.signal import (
    parse_axis,
    parse_calibration,
    parse_calibration_table,
    parse_signal,
    parse_signals,
)
from tests.pysea_fixtures import EXPERIMENTAL, SIMULATION, fixture_envelope

# --- axes ------------------------------------------------------------------


def test_a_momentum_axis_keeps_its_kind_and_units():
    """Momentum-resolved vEELS is the case this integration exists for."""
    axis = parse_axis(
        {"name": "qx", "kind": "momentum", "units": "1/angstrom", "size": 64}
    )
    assert axis.kind == "momentum"
    assert axis.units == "1/angstrom"
    assert axis.units_source == "container"
    assert axis.needs_units and axis.has_units


def test_an_axis_without_units_stays_without_units():
    axis = parse_axis({"name": "energy_loss", "kind": "energy", "size": 1024})
    assert axis.units is None
    assert axis.units_source is None
    assert axis.needs_units and not axis.has_units


def test_a_caller_hint_fills_only_where_the_container_is_silent():
    axis = parse_axis({"name": "e", "kind": "energy", "size": 8}, unit_hint="eV")
    assert axis.units == "eV"
    assert axis.units_source == "caller"


def test_the_container_beats_the_caller():
    """An instrument-reported unit outranks one typed during import."""
    axis = parse_axis(
        {"name": "e", "kind": "energy", "units": "meV", "size": 8}, unit_hint="eV"
    )
    assert axis.units == "meV"
    assert axis.units_source == "container"


def test_an_unrecognised_axis_kind_becomes_other_rather_than_raising():
    """Parsing must not fail over a field we may have named differently."""
    axis = parse_axis({"name": "repeat", "kind": "sausage", "size": 3})
    assert axis.kind == "other"
    assert not axis.needs_units


def test_an_other_axis_needs_no_units():
    """An axis indexing repeat number has no unit to report."""
    axis = parse_axis({"name": "repeat", "kind": "other", "size": 3})
    assert not axis.needs_units


# --- calibration -----------------------------------------------------------


def test_an_absent_calibration_is_none_rather_than_defaults():
    """An empty calibration and an absent one must not read the same."""
    assert parse_calibration(None) is None
    assert parse_calibration({}) is None


def test_a_calibration_without_dispersion_is_not_energy_calibrated():
    calibration = parse_calibration({"collection_angle_mrad": 2.4})
    assert calibration is not None
    assert calibration.dispersion_ev_per_channel is None
    assert not calibration.is_energy_calibrated


def test_the_calibration_table_is_keyed_for_signals_to_reference():
    table = parse_calibration_table(fixture_envelope(EXPERIMENTAL))
    assert "cal-veels-2026-Q2" in table
    assert table["cal-veels-2026-Q2"].dispersion_ev_per_channel == 0.0021


# --- signals ---------------------------------------------------------------


def test_the_momentum_resolved_signal_parses_with_three_axes():
    signals = parse_signals(fixture_envelope(EXPERIMENTAL))
    veels = next(s for s in signals if s.signal_id == "veels-q-resolved")
    assert veels.rank == 3
    assert veels.shape == (64, 64, 1024)
    assert veels.has_momentum_axis
    assert veels.axis_of_kind("energy").units == "eV"


def test_a_signal_resolves_its_calibration_reference():
    signals = parse_signals(fixture_envelope(EXPERIMENTAL))
    veels = next(s for s in signals if s.signal_id == "veels-q-resolved")
    assert veels.calibration is not None
    assert veels.calibration.is_energy_calibrated


def test_an_unresolvable_calibration_reference_leaves_it_none():
    """The validator reports it; the parser does not invent one."""
    signal = parse_signal(
        {"signal_id": "s", "shape": [4], "axes": [], "calibration_ref": "nope"},
        calibrations={"real": parse_calibration({"dispersion_ev_per_channel": 0.1})},
    )
    assert signal.calibration is None


def test_a_signal_inherits_the_records_kind():
    signals = parse_signals(fixture_envelope(SIMULATION))
    assert all(s.record_kind == "simulation" for s in signals)


def test_a_signal_may_override_the_records_kind():
    """How a hybrid record carries a measurement next to its simulation."""
    signal = parse_signal(
        {"signal_id": "s", "shape": [4], "axes": [], "record_kind": "experimental"},
        record_kind="hybrid",
    )
    assert signal.record_kind == "experimental"


def test_a_4d_stem_signal_parses_with_two_scan_and_two_diffraction_axes():
    signal = parse_signal({
        "signal_id": "4dstem",
        "signal_type": "4D-STEM",
        "shape": [128, 128, 256, 256],
        "axes": [
            {"name": "x", "kind": "spatial", "units": "nm", "size": 128},
            {"name": "y", "kind": "spatial", "units": "nm", "size": 128},
            {"name": "kx", "kind": "momentum", "units": "1/angstrom", "size": 256},
            {"name": "ky", "kind": "momentum", "units": "1/angstrom", "size": 256},
        ],
    })
    assert signal.rank == 4
    assert sum(1 for a in signal.axes if a.kind == "spatial") == 2
    assert sum(1 for a in signal.axes if a.kind == "momentum") == 2


def test_source_metadata_keeps_fields_the_parser_does_not_model():
    signal = parse_signal(
        {"signal_id": "s", "shape": [2], "axes": [], "drift_corrected": True}
    )
    assert signal.source_metadata["drift_corrected"] is True
