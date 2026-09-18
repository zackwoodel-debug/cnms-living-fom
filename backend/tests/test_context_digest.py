"""Measurement-context identity (FOM_PROOF Eq. 3).

The digest is what lets the database enforce "these two rows are the same
measurement" as a uniqueness constraint. These tests pin the two things that
make it usable: equivalent contexts must collide, and genuinely different ones
must not.
"""

from __future__ import annotations

from cnms_fom.db.context import CONTEXT_FIELDS, context_digest, describe_context
from cnms_fom.db.enums import ProvenanceTier


def test_digest_is_stable_and_short():
    digest = context_digest({"temperature_k": 300.0, "method": "CV"})
    assert len(digest) == 32
    assert digest == context_digest({"temperature_k": 300.0, "method": "CV"})


def test_field_order_does_not_matter():
    a = {"temperature_k": 300.0, "frequency_hz": 1e4, "method": "CV"}
    b = {"method": "CV", "frequency_hz": 1e4, "temperature_k": 300.0}
    assert context_digest(a) == context_digest(b)


def test_float_round_tripping_noise_collapses():
    """300.0 and 300.00000000000006 are one temperature, not two."""
    assert context_digest({"temperature_k": 300.0}) == context_digest(
        {"temperature_k": 300.00000000000006}
    )


def test_empty_string_and_null_are_both_not_recorded():
    """A blank electrode field and a missing one mean the same thing."""
    assert context_digest({"electrode": ""}) == context_digest({"electrode": None})


def test_genuinely_different_contexts_differ():
    base = {"temperature_k": 300.0, "frequency_hz": 1e4}
    assert context_digest(base) != context_digest({**base, "temperature_k": 301.0})
    assert context_digest(base) != context_digest({**base, "frequency_hz": 1e6})


def test_tensor_component_is_part_of_identity():
    """Sec. 3.2: eps_zz and eps_iso are not the same quantity."""
    assert context_digest({"tensor_component": "zz"}) != context_digest(
        {"tensor_component": "iso"}
    )


def test_provenance_tier_is_part_of_identity():
    """A measured and a calculated value under the same conditions are two records."""
    assert context_digest({"provenance_tier": ProvenanceTier.MEASURED}) != context_digest(
        {"provenance_tier": ProvenanceTier.CALCULATED}
    )


def test_enum_and_its_string_value_agree():
    """So an ingestion script can pass either form and get the same digest."""
    assert context_digest({"provenance_tier": ProvenanceTier.MEASURED}) == context_digest(
        {"provenance_tier": "measured"}
    )


def test_source_identity_keeps_independent_reports_distinct():
    """Sec. 2.1: two papers reporting the same quantity are two records.

    They must both be storable — the ambiguity is resolved at analysis time with
    a declared aggregation rule, not by the database silently keeping one.
    """
    base = {"temperature_k": 300.0, "method": "CV"}
    assert context_digest({**base, "doi": "10.1/aaa"}) != context_digest(
        {**base, "doi": "10.1/bbb"}
    )


def test_value_is_not_part_of_identity():
    """A second reading under the same context is a duplicate, not a new record."""
    assert "value" not in CONTEXT_FIELDS
    assert "uncertainty" not in CONTEXT_FIELDS


def test_describe_context_is_readable():
    text = describe_context({"temperature_k": 300.0, "tensor_component": "zz", "electrode": None})
    assert "temperature_k=300.0" in text
    assert "electrode" not in text  # unset fields are omitted, not shown as None
