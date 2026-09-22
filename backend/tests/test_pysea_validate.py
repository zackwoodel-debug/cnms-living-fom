"""Validation reports; it does not guess and it does not mutate."""

from __future__ import annotations

import copy

from cnms_fom.pysea.validate import ERROR, WARNING, validate_envelope
from tests.pysea_fixtures import EXPERIMENTAL, INVALID, SIMULATION, fixture_envelope


def test_the_experimental_fixture_is_clean():
    report = validate_envelope(fixture_envelope(EXPERIMENTAL))
    assert report.is_valid
    assert report.issues == []


def test_the_simulation_fixture_is_clean():
    report = validate_envelope(fixture_envelope(SIMULATION))
    assert report.is_valid


def test_validation_does_not_mutate_its_input():
    envelope = fixture_envelope(INVALID)
    before = copy.deepcopy(envelope)
    validate_envelope(envelope)
    assert envelope == before


def test_the_invalid_fixture_reports_every_planted_fault():
    report = validate_envelope(fixture_envelope(INVALID))
    codes = report.codes()

    assert not report.is_valid
    assert report.status == "invalid"
    #  One axis for a rank-3 signal, and that axis has no units.
    assert "signal.axis_count_mismatch" in codes
    assert "signal.axis_units_missing" in codes
    #  A scalar with no source signal, and one pointing at a signal that is absent.
    assert "scalar.no_source_signal" in codes
    assert "scalar.source_signal_unknown" in codes
    #  A value with no units, and a derivation outside the vocabulary.
    assert "scalar.units_missing" in codes
    assert "scalar.derivation_invalid" in codes


def test_every_issue_carries_a_code_severity_and_path():
    """A caller acts on the list; a prose failure cannot be acted on."""
    for issue in validate_envelope(fixture_envelope(INVALID)).issues:
        assert issue.code
        assert issue.severity in {ERROR, WARNING}
        assert issue.field_path
        assert issue.message


def test_an_unsupported_version_stops_before_reading_anything_else():
    """Reading a 1.x envelope under 0.x would interpret redefined fields."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["contract_version"] = "pysea-canonical/1.0"
    report = validate_envelope(envelope)
    assert [issue.code for issue in report.issues] == ["contract.version_unsupported"]


def test_a_simulation_without_metadata_is_an_error():
    envelope = fixture_envelope(SIMULATION)
    del envelope["simulation"]
    report = validate_envelope(envelope)
    assert "simulation.metadata_missing" in report.codes()
    assert not report.is_valid


def test_an_experimental_record_carrying_a_simulation_is_warned_not_refused():
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["simulation"] = {"code": "pySEA multislice"}
    report = validate_envelope(envelope)
    assert "simulation.on_experimental_record" in report.codes()
    assert report.is_valid


def test_a_material_id_in_the_container_is_warned_about():
    """Identity is supplied at promotion, by someone who knows the polymorph."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["sample"]["material_id"] = 17
    report = validate_envelope(envelope)
    assert "sample.material_id_present" in report.codes()
    assert report.is_valid


def test_a_missing_uncertainty_is_a_warning_not_an_error():
    """The acquisition is real; what it cannot support is promotion."""
    envelope = fixture_envelope(EXPERIMENTAL)
    del envelope["derived_scalars"][0]["uncertainty"]
    report = validate_envelope(envelope)
    assert "scalar.uncertainty_missing" in report.codes()
    assert report.is_valid


def test_a_negative_angle_is_an_error():
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["instrument_state"]["collection_semi_angle_mrad"] = -2.4
    report = validate_envelope(envelope)
    assert "state.negative_quantity" in report.codes()
    assert not report.is_valid


def test_convergence_wider_than_collection_is_a_warning():
    """A real configuration worth flagging rather than refusing."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["instrument_state"]["convergence_semi_angle_mrad"] = 30.0
    report = validate_envelope(envelope)
    assert "state.convergence_exceeds_collection" in report.codes()
    assert report.is_valid


def test_axis_size_disagreeing_with_the_shape_is_an_error():
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["signals"][0]["axes"][0]["size"] = 63
    report = validate_envelope(envelope)
    assert "signal.axis_size_mismatch" in report.codes()


def test_duplicate_signal_ids_are_an_error():
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["signals"][1]["signal_id"] = envelope["signals"][0]["signal_id"]
    report = validate_envelope(envelope)
    assert "signal.id_duplicate" in report.codes()


def test_units_supplied_by_the_caller_are_recorded_as_such():
    """A unit typed at import is weaker evidence than one the instrument wrote."""
    envelope = fixture_envelope(EXPERIMENTAL)
    del envelope["signals"][0]["axes"][2]["units"]
    report = validate_envelope(envelope)
    assert "signal.axis_units_missing" in report.codes()


def test_a_record_with_no_signals_is_an_error():
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["signals"] = []
    report = validate_envelope(envelope)
    assert "signals.none" in report.codes()


def test_a_missing_datafed_id_is_a_warning():
    envelope = fixture_envelope(EXPERIMENTAL)
    del envelope["datafed"]
    report = validate_envelope(envelope)
    assert "datafed.record_id_missing" in report.codes()
    assert report.is_valid
