"""FOM_PROOF Sec. 6.2: the weighted geometric score."""

from __future__ import annotations

import math

import pytest

from cnms_fom.db.enums import ProvenanceTier, ScoreStatus
from cnms_fom.fom_engine.definitions import draft_fom
from cnms_fom.fom_engine.normalization import NormalizationSpec
from cnms_fom.fom_engine.scores import FomSpec, score_material


def test_weights_must_sum_to_one():
    """Eq. (29) constrains the weights to a simplex."""
    with pytest.raises(ValueError, match="must be 1"):
        FomSpec(
            name="bad",
            application="bad",
            version=1,
            weights={"k": 0.5, "Eg": 0.2},
            normalization={
                "k": NormalizationSpec("k", 0.0, 1.0),
                "Eg": NormalizationSpec("Eg", 0.0, 1.0),
            },
        )


def test_negative_weights_rejected():
    with pytest.raises(ValueError, match="negative weights"):
        FomSpec(
            name="bad",
            application="bad",
            version=1,
            weights={"k": 1.5, "Eg": -0.5},
            normalization={
                "k": NormalizationSpec("k", 0.0, 1.0),
                "Eg": NormalizationSpec("Eg", 0.0, 1.0),
            },
        )


def test_score_equals_weighted_geometric_mean(hfo2_properties, draft_foms):
    """Eq. (29) and Eq. (30) must agree: F = exp(sum w ln z)."""
    spec = draft_foms["logic"]
    result = score_material(hfo2_properties, spec)

    product = 1.0
    for key, component in result.components.items():
        product *= component["z"] ** spec.weights[key]

    assert result.value == pytest.approx(product)
    assert result.log_value == pytest.approx(math.log(product))


def test_missing_input_is_not_scored(hfo2_properties, draft_foms):
    """Sec. 6.2: report 'not scored' rather than a manufactured score."""
    incomplete = {**hfo2_properties, "Ebd": None}
    result = score_material(incomplete, draft_foms["logic"])
    assert result.status is ScoreStatus.NOT_SCORED
    assert result.missing_inputs == ["Ebd"]
    assert result.value is None


def test_unavailable_tier_counts_as_missing(hfo2_properties, draft_foms):
    result = score_material(
        hfo2_properties,
        draft_foms["logic"],
        provenance={"Eg": ProvenanceTier.UNAVAILABLE},
    )
    assert result.status is ScoreStatus.NOT_SCORED
    assert "Eg" in result.missing_inputs


def test_modeled_input_marks_score_illustrative(hfo2_properties, draft_foms):
    """Sec. 2.3: a modeled scenario is labelled and kept separate."""
    result = score_material(
        hfo2_properties, draft_foms["logic"], provenance={"k": ProvenanceTier.MODELED}
    )
    assert result.status is ScoreStatus.ILLUSTRATIVE
    assert result.uses_modeled_inputs
    assert "ILLUSTRATIVE" in result.note


def test_draft_definitions_are_unapproved():
    """Table 4 values are examples; weights need formal approval before use."""
    spec = draft_fom("power")
    assert spec.approved is False
    assert "DRAFT" in score_material(
        {"k": 25.0, "Eg": 5.7, "dEc": 1.5, "Ebd": 4.0, "kappa_th": 1.1}, spec
    ).note


def test_rf_score_penalises_loss(draft_foms):
    """tan_delta is a cost: a lossier material must score lower, all else equal."""
    base = {"k": 25.0, "Ebd": 4.0, "kappa_th": 1.1, "tan_delta": 1e-4}
    lossy = {**base, "tan_delta": 1e-2}
    assert score_material(lossy, draft_foms["rf"]).value < score_material(
        base, draft_foms["rf"]
    ).value
