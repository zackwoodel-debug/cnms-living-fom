"""The canonical envelope: versioning, canonicalisation, idempotency.

Everything here guards a provisional mapping. ``pysea-canonical/0.1`` was written
from pySEA's published abstracts without sight of the container format, so the
tests that matter most are the ones about *not* over-reading an envelope: refusing
a major version this build cannot interpret, and keeping unmapped fields instead of
dropping them.
"""

from __future__ import annotations

import json

import pytest

from cnms_fom.pysea.contract import (
    CONTRACT_VERSION,
    ContractError,
    assert_supported,
    canonical_envelope,
    content_sha256,
    load_envelope,
    split_extensions,
    strip_bulk,
    supported_version,
)
from tests.pysea_fixtures import EXPERIMENTAL, fixture_envelope


def test_the_shipped_contract_version_is_supported():
    assert supported_version(CONTRACT_VERSION)


@pytest.mark.parametrize(
    "version",
    ["pysea-canonical/0.1", "pysea-canonical/0.2", "pysea-canonical/0.99"],
)
def test_minor_versions_are_readable(version):
    """0.x changes are additive by construction, so a 0.1 build reads 0.2."""
    assert supported_version(version)


@pytest.mark.parametrize(
    "version",
    ["pysea-canonical/1.0", "pysea/0.1", "", None, "0.1", 1.0],
)
def test_an_unreadable_version_is_refused(version):
    """A 1.x envelope read by a 0.x parser would interpret redefined fields."""
    assert not supported_version(version)
    with pytest.raises(ContractError):
        assert_supported({"contract_version": version})


def test_key_order_does_not_change_the_hash():
    """The same container re-exported differently is the same container."""
    envelope = fixture_envelope(EXPERIMENTAL)
    shuffled = json.loads(json.dumps(dict(reversed(list(envelope.items())))))
    assert content_sha256(envelope) == content_sha256(shuffled)


def test_a_changed_value_changes_the_hash():
    envelope = fixture_envelope(EXPERIMENTAL)
    other = fixture_envelope(EXPERIMENTAL)
    other["derived_scalars"][0]["value"] = 99.0
    assert content_sha256(envelope) != content_sha256(other)


def test_axis_order_is_preserved_by_canonicalisation():
    """Dict keys sort; lists do not. Axis order is the signal's dimension order."""
    envelope = fixture_envelope(EXPERIMENTAL)
    canonical = canonical_envelope(envelope)
    kinds = [axis["kind"] for axis in canonical["signals"][0]["axes"]]
    assert kinds == ["momentum", "momentum", "energy"]


def test_bulk_arrays_never_reach_the_hash_or_the_row():
    """A 4D-STEM scan is gigabytes; only shape and locator are stored."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["signals"][0]["data"] = [[1.0] * 8] * 8

    stripped = strip_bulk(envelope)
    assert "data" not in stripped["signals"][0]
    assert stripped["signals"][0]["shape"] == [64, 64, 1024]

    #  And two exports differing only in their arrays are one acquisition.
    without = fixture_envelope(EXPERIMENTAL)
    assert content_sha256(envelope) == content_sha256(without)


def test_unmapped_fields_are_kept_rather_than_dropped():
    """The fields we failed to anticipate are the ones worth keeping."""
    envelope = fixture_envelope(EXPERIMENTAL)
    envelope["aberration_coefficients"] = {"C30_um": 1.2}

    split = split_extensions(envelope)
    assert "aberration_coefficients" not in split
    assert split["extensions"]["_unmapped"]["aberration_coefficients"] == {"C30_um": 1.2}


def test_known_sections_survive_the_split():
    envelope = fixture_envelope(EXPERIMENTAL)
    split = split_extensions(envelope)
    for section in ("record_id", "sample", "instrument", "signals", "derived_scalars"):
        assert section in split


def test_load_envelope_accepts_a_dict_or_a_path(tmp_path):
    envelope = fixture_envelope(EXPERIMENTAL)
    assert load_envelope(envelope) == envelope

    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert load_envelope(str(path))["record_id"] == envelope["record_id"]


def test_a_non_object_envelope_is_refused(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ContractError, match="JSON object"):
        load_envelope(str(path))


def test_malformed_json_is_refused(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ContractError, match="not valid JSON"):
        load_envelope(str(path))
