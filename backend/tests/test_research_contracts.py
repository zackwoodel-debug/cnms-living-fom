"""The research contracts, and what they refuse to represent.

These run with no database and no model server: the invariants are structural, and
that is the point — an extraction that cannot describe itself as a measurement is
safer than one that is merely never asked to.
"""

from __future__ import annotations

import pytest

from cnms_fom.db.enums import ClaimStatus, ClaimTier, ContextStatus, ProvenanceTier, StatementKind
from cnms_fom.research.contracts import (
    BoundProposal,
    Contradiction,
    DataGap,
    EvidenceItem,
    ExtractedClaim,
    LabelledStatement,
    ProposedBOContext,
    ResearchBrief,
    ResearchContractError,
)

SPACE = {
    "parameters": [
        {"name": "substrate_temp_c", "kind": "continuous", "lower": 150.0, "upper": 400.0},
        {"name": "purge_s", "kind": "continuous", "lower": 1.0, "upper": 20.0},
        {"name": "substrate", "kind": "categorical", "choices": ["Si(100)", "SiO2", "Ge"]},
    ]
}


def _evidence(**kwargs) -> EvidenceItem:
    defaults = {
        "document_id": 1,
        "document_title": "Kim 2024",
        "page": 7,
        "quote": "growth per cycle saturated at 0.98 angstrom per cycle",
        "content_sha256": "ab" * 32,
    }
    return EvidenceItem(**{**defaults, **kwargs})


def _claim(**kwargs) -> ExtractedClaim:
    defaults = {
        "field_name": "k",
        "value": 25.0,
        "units": "1",
        "evidence": [_evidence()],
        "context": {"temperature_k": 300.0, "frequency_hz": 10_000.0},
    }
    return ExtractedClaim(**{**defaults, **kwargs})


# --- the tier separation ---------------------------------------------------


def test_claim_tier_is_not_the_analysis_provenance_tier():
    """Collapsing them would let a literature value inherit MEASURED on a coercion."""
    assert ClaimTier is not ProvenanceTier
    #  The give-away member: no analysis tier calls anything "reported", because a
    #  value that only has a source's word for it is not eligible to be scored.
    assert "reported" in {m.value for m in ClaimTier}
    assert "reported" not in {m.value for m in ProvenanceTier}
    #  And ProvenanceTier's "unavailable" has no meaning for an extraction.
    assert "unavailable" not in {m.value for m in ClaimTier}


def test_a_claim_never_serialises_as_a_measurement():
    payload = _claim(tier=ClaimTier.MEASURED).as_dict()
    #  Even when the *source* says it measured it.
    assert payload["tier"] == "measured"
    assert payload["is_measurement"] is False
    assert payload["status"] == "candidate"
    #  There is no field a consumer could mistake for an analysis tier.
    assert "provenance_tier" not in payload


def test_claim_status_has_no_accepted_member():
    """Acceptance means entering the analysis tables, which happens elsewhere."""
    assert "accepted" not in {m.value for m in ClaimStatus}
    assert "measured" not in {m.value for m in ClaimStatus}


# --- evidence -------------------------------------------------------------


def test_evidence_without_a_quote_is_refused():
    """A citation with no supporting text cannot be checked against its page."""
    with pytest.raises(ResearchContractError, match="quote"):
        _evidence(quote="   ")


def test_evidence_without_a_page_is_not_locatable():
    assert _evidence().is_locatable is True
    assert _evidence(page=None).is_locatable is False
    #  A title alone points at a document, not at a claim.
    assert _evidence(document_id=None, content_sha256=None).is_locatable is False


def test_evidence_carries_a_content_hash_as_well_as_an_id():
    """An id is only meaningful inside one database; a brief outlives the row."""
    item = _evidence()
    assert item.as_dict()["content_sha256"] == "ab" * 32
    assert item.citation == "Kim 2024, p. 7"


# --- claims ---------------------------------------------------------------


def test_a_claim_without_evidence_is_refused():
    with pytest.raises(ResearchContractError, match="no evidence"):
        _claim(evidence=[])


def test_a_claim_with_no_value_at_all_is_refused():
    with pytest.raises(ResearchContractError, match="neither a numeric"):
        _claim(value=None, value_text=None)


def test_a_text_only_claim_is_allowed():
    """A paper's most useful statement is often not a number."""
    claim = _claim(field_name="ald_window", value=None, value_text="200-300 C")
    assert claim.is_comparable is False  # readable, not comparable


def test_a_normalised_value_needs_its_conversion_recorded():
    with pytest.raises(ResearchContractError, match="how it"):
        _claim(normalized_value=25.0)

    ok = _claim(normalized_value=2.5e-11, normalized_units="F/m",
                normalization_note="relative permittivity x eps_0")
    assert ok.as_dict()["normalization_note"]


def test_missing_required_context_is_reported_never_filled():
    """Sec. 16: context is part of what a value is, not an optional annotation."""
    incomplete = _claim(context={"temperature_k": 300.0})  # k also needs frequency
    assert incomplete.missing_context == ["frequency_hz"]
    assert incomplete.is_context_complete is False
    assert incomplete.is_comparable is False
    #  Nothing was invented to fill it.
    assert "frequency_hz" not in incomplete.context


def test_a_complete_claim_is_comparable():
    assert _claim().is_comparable is True


def test_a_claim_needs_locatable_evidence_to_be_comparable():
    assert _claim(evidence=[_evidence(page=None)]).is_comparable is False


def test_an_unrecognised_context_key_is_flagged_not_dropped():
    """Usually a typo, which would otherwise silently stop matching."""
    claim = _claim(context={"temperature_k": 300.0, "frequency_hz": 1e4, "tempreature": 350})
    assert "tempreature" in claim.context  # kept
    assert "unrecognised context keys" in claim.notes  # and flagged


# --- contradictions -------------------------------------------------------


def test_a_contradiction_keeps_both_claims_and_has_no_resolved_value():
    """Sec. 2.1: two sources disagreeing do not have a mean worth reporting."""
    left = _claim(field_name="growth_per_cycle_ang", value=0.98, units="A/cycle",
                  context={"technique": "ald", "temperature_k": 523.0,
                           "precursor": "TDMAH", "chamber": "hot-wall"})
    right = _claim(field_name="growth_per_cycle_ang", value=1.42, units="A/cycle",
                   context={"technique": "ald", "temperature_k": 523.0,
                            "precursor": "TDMAH", "chamber": "cross-flow"})
    contradiction = Contradiction(
        field_name="growth_per_cycle_ang", left=left, right=right,
        basis="same chemistry and temperature range, different reactor geometry",
        differing_context={"chamber": ("hot-wall", "cross-flow")},
    )
    payload = contradiction.as_dict()
    assert payload["left"]["value"] == 0.98 and payload["right"]["value"] == 1.42
    #  Relative to the midpoint: 0.44 / 1.20 = 0.367, matching modalfit.compare.
    assert contradiction.relative_spread == pytest.approx(0.367, abs=0.005)
    assert "resolved_value" not in payload
    assert "mean" not in payload and "average" not in payload


def test_a_contradiction_without_a_basis_is_refused():
    with pytest.raises(ResearchContractError, match="basis"):
        Contradiction(field_name="k", left=_claim(), right=_claim(value=30.0), basis="")


# --- data gaps ------------------------------------------------------------


def test_a_data_gap_must_say_what_would_resolve_it():
    """A gap with no route out of it is a complaint, not a finding."""
    with pytest.raises(ResearchContractError, match="what would resolve"):
        DataGap(question="What is the ALD window?", what_was_searched="ald corpus",
                what_would_resolve_it="")

    gap = DataGap(
        question="What is the ALD window on Ge?",
        what_was_searched="ald partition, 12 candidates, none graded useful",
        what_would_resolve_it="a TDMAH/H2O growth study on Ge, or a run on the CNMS ALD tool",
    )
    assert gap.as_dict()["what_would_resolve_it"]


# --- proposed BO context --------------------------------------------------


def test_a_narrowing_bound_is_accepted():
    context = ProposedBOContext(
        bo_run_id=1,
        recommended_bounds=[BoundProposal(
            parameter="substrate_temp_c", lower=200.0, upper=300.0,
            rationale="ALD window reported at 200-300 C for this chemistry",
        )],
    )
    assert context.widening_violations(SPACE) == []
    assert context.changes_search_behaviour is True


def test_a_widening_bound_is_refused_with_the_reason():
    """No paper is a source for what an instrument can physically reach."""
    context = ProposedBOContext(
        bo_run_id=1,
        recommended_bounds=[BoundProposal(
            parameter="substrate_temp_c", lower=100.0, upper=500.0,
            rationale="a paper grew films at 450 C",
        )],
    )
    problems = context.widening_violations(SPACE)
    assert len(problems) == 2
    assert any("below the live lower bound" in p for p in problems)
    assert any("above the live upper bound" in p for p in problems)
    assert all("never widen" in p for p in problems)


def test_a_bound_on_an_unknown_or_categorical_parameter_is_refused():
    unknown = ProposedBOContext(bo_run_id=1, recommended_bounds=[
        BoundProposal(parameter="laser_fluence", lower=1.0, upper=2.0, rationale="x")])
    assert "not a parameter of this campaign" in unknown.widening_violations(SPACE)[0]

    categorical = ProposedBOContext(bo_run_id=1, recommended_bounds=[
        BoundProposal(parameter="substrate", lower=1.0, upper=2.0, rationale="x")])
    assert "categorical" in categorical.widening_violations(SPACE)[0]


def test_excluding_every_choice_is_refused():
    context = ProposedBOContext(
        bo_run_id=1, excluded_choices={"substrate": ["Si(100)", "SiO2", "Ge"]}
    )
    assert any("no choices at all" in p for p in context.widening_violations(SPACE))


def test_excluding_an_unknown_choice_is_refused():
    context = ProposedBOContext(bo_run_id=1, excluded_choices={"substrate": ["GaAs"]})
    assert any("not choices of this parameter" in p for p in context.widening_violations(SPACE))


def test_an_inverted_bound_proposal_is_refused_at_construction():
    with pytest.raises(ResearchContractError, match="must exceed"):
        BoundProposal(parameter="purge_s", lower=10.0, upper=5.0, rationale="x")


def test_a_bound_proposal_needs_a_rationale():
    with pytest.raises(ResearchContractError, match="rationale"):
        BoundProposal(parameter="purge_s", lower=5.0, upper=10.0, rationale="  ")


def test_advisory_content_becomes_notes_and_never_a_numeric_input():
    """A literature prior silently steering a GP is a result nobody can attribute."""
    context = ProposedBOContext(
        bo_run_id=1,
        soft_priors=["density likely below bulk 9.68 g/cm3 for low-temperature growth"],
        process_window_hints=["purge under 4 s left unreacted precursor"],
        uncertainty_notes=["only one source for the upper limit"],
    )
    patch = context.as_constraint_patch()
    assert patch["bounds"] == {}
    assert context.changes_search_behaviour is False
    assert all("advisory" in note for note in patch["notes"])
    #  Every advisory item survives into notes, none into bounds.
    assert len(patch["notes"]) == 3


def test_a_constraint_patch_matches_the_bo_engine_shape():
    """The bridge hands BO a shape it already parses, not a parallel one."""
    from cnms_fom.bo_engine.constraints import ConstraintSet

    context = ProposedBOContext(
        bo_run_id=1,
        recommended_bounds=[BoundProposal(
            parameter="substrate_temp_c", lower=200.0, upper=300.0, rationale="window")],
        soft_priors=["prefer the low end"],
    )
    parsed = ConstraintSet.from_dict(context.as_constraint_patch())
    assert parsed.bounds["substrate_temp_c"] == (200.0, 300.0)
    assert any("advisory" in n for n in parsed.notes)


def test_a_proposal_starts_proposed_and_cannot_be_born_applied():
    context = ProposedBOContext(bo_run_id=1)
    assert context.status is ContextStatus.PROPOSED
    assert context.is_empty is True
    #  APPLIED exists, but reaching it is the bridge's job under review, not a
    #  constructor argument a generator could set.
    assert ContextStatus.APPLIED.value == "applied"


# --- statements and the brief --------------------------------------------


def test_an_evidence_statement_must_carry_its_evidence():
    with pytest.raises(ResearchContractError, match="must carry its evidence"):
        LabelledStatement(kind=StatementKind.EVIDENCE, text="The window is 200-300 C.")

    #  Interpretations and proposals may reason over the brief as a whole.
    LabelledStatement(kind=StatementKind.INTERPRETATION, text="The reactors differ.")
    LabelledStatement(kind=StatementKind.PROPOSAL, text="Measure on both substrates.")


def test_a_brief_needs_a_question():
    with pytest.raises(ResearchContractError, match="research question"):
        ResearchBrief(research_question="  ")


def test_a_brief_fingerprint_is_stable_across_reruns_and_ignores_timestamps():
    """How a benchmark tells a policy change from noise."""
    first = ResearchBrief(research_question="ALD window?", claims=[_claim()], evidence=[_evidence()])
    second = ResearchBrief(research_question="ALD window?", claims=[_claim()], evidence=[_evidence()])
    assert first.fingerprint() == second.fingerprint()

    changed = ResearchBrief(
        research_question="ALD window?", claims=[_claim(value=30.0)], evidence=[_evidence()]
    )
    assert changed.fingerprint() != first.fingerprint()


def test_a_brief_reports_unsupported_statements_when_it_has_no_evidence():
    empty = ResearchBrief(
        research_question="ALD window?",
        statements=[LabelledStatement(kind=StatementKind.INTERPRETATION, text="probably 250 C")],
    )
    assert len(empty.unsupported_statements) == 1

    grounded = ResearchBrief(
        research_question="ALD window?",
        evidence=[_evidence()],
        statements=[LabelledStatement(kind=StatementKind.INTERPRETATION, text="probably 250 C")],
    )
    assert grounded.unsupported_statements == []


def test_a_brief_states_that_nothing_in_it_is_a_measurement():
    payload = ResearchBrief(research_question="ALD window?", claims=[_claim()]).as_dict()
    assert "not a measurement" in payload["disclaimer"]
    assert "no code path" in payload["disclaimer"]
    assert payload["n_comparable_claims"] == 1


def test_a_brief_separates_comparable_from_incomplete_claims(  ):
    brief = ResearchBrief(
        research_question="k of HfO2?",
        claims=[_claim(), _claim(context={"temperature_k": 300.0})],
    )
    assert len(brief.comparable_claims) == 1
    assert len(brief.incomplete_claims) == 1


def test_citation_is_a_property_on_both_evidence_types():
    """Two classes with the same attribute, one callable and one not, is a trap.

    A bare ``.citation`` on the method form formats a bound method into the string,
    producing a citation that reads like a bug report. Caught while writing a
    diagnostic against ``ChunkHit``.
    """
    from cnms_fom.rag_backend.vectorstore import ChunkHit

    chunk = ChunkHit(
        chunk_id=1, document_id=1, document_title="Kim 2024", technique="ald",
        page=7, text="...", similarity=0.8, doi="10.0/x",
    )
    assert isinstance(chunk.citation, str)
    assert chunk.citation == "Kim 2024, p. 7, doi:10.0/x"
    assert isinstance(_evidence().citation, str)
    #  Both are data attributes, not callables.
    assert not callable(type(chunk).citation)
    assert not callable(type(_evidence()).citation)


# --- unit-bearing context fields ------------------------------------------


def test_a_celsius_range_is_moved_out_of_a_kelvin_field():
    """Observed live: an extractor put "200 to 300 degC" in temperature_k.

    Worse than a missing temperature, because the field name asserts kelvin and
    anything trusting it reads 200-300 K.
    """
    claim = _claim(context={"temperature_k": "200 to 300 degC", "frequency_hz": 1e4})

    assert "temperature_k" not in claim.context
    assert claim.context["temperature_k_as_stated"] == "200 to 300 degC"
    #  The field is now honestly absent, so the gap is reported...
    assert claim.missing_context == ["temperature_k"]
    assert claim.is_comparable is False
    #  ...and the source's own wording is not lost.
    assert "not a number in kelvin" in claim.notes


def test_a_numeric_string_is_accepted_and_coerced():
    claim = _claim(context={"temperature_k": "300", "frequency_hz": "10000"})
    assert claim.context["temperature_k"] == 300.0
    assert claim.context["frequency_hz"] == 10000.0
    assert claim.is_comparable is True


def test_a_real_number_passes_through_untouched():
    claim = _claim(context={"temperature_k": 300.0, "frequency_hz": 1e4})
    assert claim.context == {"temperature_k": 300.0, "frequency_hz": 1e4}
    assert claim.notes == ""


def test_every_unit_bearing_field_is_guarded():
    from cnms_fom.research.contracts import UNIT_BEARING_CONTEXT

    for field_name, unit in UNIT_BEARING_CONTEXT.items():
        claim = _claim(
            field_name="some_quantity", context={field_name: f"about 5 {unit}"}
        )
        assert field_name not in claim.context, field_name
        assert claim.context[f"{field_name}_as_stated"] == f"about 5 {unit}"


def test_a_non_unit_bearing_field_keeps_its_text():
    """`chamber` and `precursor` are words, not numbers."""
    claim = _claim(context={"temperature_k": 300.0, "frequency_hz": 1e4,
                            "chamber": "cross-flow", "precursor": "TDMAH"})
    assert claim.context["chamber"] == "cross-flow"
    assert claim.context["precursor"] == "TDMAH"


# --- Celsius sources ------------------------------------------------------
#
# REQUIRED_CONTEXT demands temperature_k for every growth claim and the extraction
# prompt forbids converting. Together those made a claim from any source stating
# degC permanently incomparable — which is nearly every ALD and PLD paper.


def test_a_celsius_temperature_becomes_a_comparable_kelvin_one():
    claim = _claim(context={"temperature_c": 250})

    assert claim.context["temperature_k"] == pytest.approx(523.15)
    #  What the source actually said is preserved, not consumed by the conversion.
    assert claim.context["temperature_c"] == 250
    assert "derived from the stated 250 degC" in claim.notes


def test_a_stated_kelvin_value_wins_over_a_derived_one():
    """If the source states kelvin, that is the record; nothing is recomputed."""
    claim = _claim(context={"temperature_k": 500.0, "temperature_c": 250})
    assert claim.context["temperature_k"] == 500.0
    assert "derived" not in claim.notes


def test_a_celsius_range_does_not_become_a_kelvin_number():
    """A window is not a value, and inventing one from it would be a fabrication."""
    claim = _claim(context={"temperature_c": "200 to 300"})
    assert "temperature_k" not in claim.context
    assert claim.context["temperature_c_as_stated"] == "200 to 300"


def test_the_conversion_makes_a_growth_claim_comparable():
    """The point of the whole thing, end to end."""
    claim = _claim(
        field_name="growth_per_cycle_ang",
        context={"technique": "ald", "precursor": "TDMAH", "chamber": "cross-flow",
                 "temperature_c": 250},
    )
    assert claim.missing_context == []
    assert claim.is_comparable is True


# --- a claim whose units contradict its own field --------------------------
#
# Found on a real corpus: asked a compound question, the extractor filed a
# passage's ALD cycle timings ("0.2 s TDMAH dose, 6 s purge") as four
# growth_per_cycle_ang claims, and the interpretation step then reported "growth
# per cycle values vary widely (0.2 s, 6.0 s, 0.1 s, 6.0 s), indicating a lack of
# consistency in the literature" — a fabricated finding built on dose times.


def test_a_dose_time_cannot_be_a_growth_per_cycle():
    with pytest.raises(ResearchContractError, match="growth per cycle"):
        _claim(field_name="growth_per_cycle_ang", value=0.2, units="s")


def test_the_real_growth_per_cycle_is_accepted():
    """The guard must not reject the values it exists to protect.

    Spellings that already *mean* angstrom-per-cycle are left exactly as written; only a
    different magnitude is rewritten, which the next test covers.
    """
    for units in ("A/cycle", "angstrom per cycle", "\u00c5/cy"):
        claim = _claim(field_name="growth_per_cycle_ang", value=1.42, units=units)
        assert claim.units == units
        assert claim.value == pytest.approx(1.42)
        assert claim.magnitude_unverified is False


def test_a_dimensionless_property_rejects_a_physical_unit():
    with pytest.raises(ResearchContractError, match="dimensionless"):
        _claim(field_name="k", value=18.5, units="Torr")


def test_a_dimensionless_property_accepts_its_conventional_unit():
    """`"1"` is how this codebase writes dimensionless, and it must survive the guard.

    `is_comparable` separately requires non-empty units, so `None` is not comparable
    by a pre-existing rule that has nothing to do with dimensions.
    """
    claim = _claim(field_name="k", value=18.5, units="1",
                   context={"temperature_k": 300.0, "frequency_hz": 1e4})
    assert claim.is_comparable is True

    from cnms_fom.research.contracts import classify_unit

    #  "1" must not classify as any physical dimension, or every k would be rejected.
    assert classify_unit("1") is None


def test_an_unrecognised_unit_is_kept_not_rejected():
    """Act on knowledge, abstain on ignorance: a spelling we do not know is not wrong."""
    claim = _claim(field_name="rho", value=8.7, units="arb. units")
    assert claim.units == "arb. units"


def test_a_descriptive_field_name_is_unconstrained():
    """Only registry keys carry a required dimension."""
    claim = _claim(field_name="purge_duration_s", value=6.0, units="s")
    assert claim.value == 6.0


def test_short_unit_markers_do_not_match_as_substrings():
    """`"s" in "angstrom"` is True, which classified a length as a time.

    That bug would have rejected exactly the claims this guard protects.
    """
    from cnms_fom.research.contracts import classify_unit

    assert classify_unit("angstrom") == "length"
    assert classify_unit("angstroms") == "length"
    assert classify_unit("s") == "time"
    assert classify_unit("0.2 s") == "time"
    assert classify_unit("widgets") is None


def test_a_category_cannot_have_a_numeric_value():
    """`material = 2` was read downstream as "a growth per cycle of 2.0"."""
    with pytest.raises(ResearchContractError, match="names a category"):
        _claim(field_name="material", value=2.0, units=None)

    for name in ("technique", "precursor", "chamber", "substrate", "oxidant"):
        with pytest.raises(ResearchContractError, match="names a category"):
            _claim(field_name=name, value=1.0, units=None)


def test_a_field_that_is_both_context_and_quantity_is_untouched():
    """thickness_nm, temperature_c, pressure_torr and frequency_hz are legitimately both."""
    assert _claim(field_name="thickness_nm", value=12.0, units="nm").value == 12.0
    assert _claim(field_name="temperature_c", value=250.0, units="degC").value == 250.0
    assert _claim(field_name="pressure_torr", value=1.5, units="Torr").value == 1.5


def test_a_non_numeric_category_statement_is_still_allowed():
    """A claim *about* the material with no number is not what this guard is for."""
    claim = _claim(field_name="material", value=None, units=None,
                   value_text="the film is monoclinic HfO2")
    assert claim.value_text == "the film is monoclinic HfO2"


# --- value_text scalar recovery (bug 20) ----------------------------------
#
# Observed on a real extraction: {"value": null, "value_text": "0.98 angstrom per
# cycle"}. A claim with no numeric value is invisible to is_comparable, to the
# benchmark's matcher and to contradiction detection, so the number was read from the
# source and then silently discarded.


def test_a_single_scalar_in_value_text_is_recovered():
    claim = _claim(
        field_name="growth_per_cycle_ang", value=None, units="angstrom per cycle",
        value_text="0.98 angstrom per cycle",
    )
    assert claim.value == pytest.approx(0.98)
    #  What the model said is preserved; recovery adds, it does not rewrite.
    assert claim.value_text == "0.98 angstrom per cycle"
    assert claim.units == "angstrom per cycle"
    assert "recovered from value_text" in claim.notes


def test_recovery_makes_the_claim_visible_to_comparison():
    """The point of the fix: a value_text-only claim could not be compared at all."""
    claim = _claim(
        field_name="growth_per_cycle_ang", value=None, units="A/cycle",
        value_text="0.98 A/cycle",
        context={"technique": "ald", "precursor": "TDMAH", "chamber": "hot-wall",
                 "temperature_c": 250},
    )
    assert claim.is_comparable is True


@pytest.mark.parametrize(
    "value_text",
    [
        "0.9 to 1.1 angstrom per cycle",
        "0.9-1.1 angstrom per cycle",
        "0.9 – 1.1 angstrom per cycle",
        "0.98 ± 0.05 angstrom per cycle",
        "< 1.0 angstrom per cycle",
        "> 1.0 angstrom per cycle",
        "about 1 to 2 angstrom per cycle",
        "approximately 1.0 angstrom per cycle",
        "between 0.9 and 1.1 angstrom per cycle",
        "up to 1.0 angstrom per cycle",
        "~1.0 angstrom per cycle",
        "1.0 or 1.2 angstrom per cycle",
    ],
)
def test_a_range_or_bound_is_never_recovered(value_text):
    """A citation attached to a number nobody wrote is worse than a missing value."""
    claim = _claim(
        field_name="growth_per_cycle_ang", value=None, units="angstrom per cycle",
        value_text=value_text,
    )
    assert claim.value is None, f"recovered a scalar from {value_text!r}"
    assert claim.is_comparable is False


def test_a_scalar_with_an_incompatible_unit_is_not_recovered():
    """A dose time must not become a growth per cycle by way of value_text."""
    claim = _claim(field_name="rho", value=None, units=None, value_text="6 s")
    assert claim.value is None


def test_an_unrecognised_unit_is_not_recovered_for_a_dimensioned_field():
    """Recovery adds a number, so it abstains where it cannot verify the unit.

    Stricter than the rejection guard, which keeps an unclassifiable unit.
    """
    claim = _claim(field_name="rho", value=None, units=None, value_text="8.7 widgets")
    assert claim.value is None


def test_a_chemical_formula_digit_is_never_recovered_as_a_value():
    """Regression for the near-miss: "monoclinic HfO2" yields the number 2.

    Recovering that reproduced bug 17 exactly — `material = 2` was read downstream as
    "a growth per cycle of 2.0".
    """
    claim = _claim(
        field_name="material", value=None, units=None,
        value_text="the film is monoclinic HfO2",
    )
    assert claim.value is None

    #  And for a field that *could* hold a number, the formula digit still must not.
    unconstrained = _claim(
        field_name="phase_note", value=None, units=None,
        value_text="monoclinic HfO2 throughout",
    )
    assert unconstrained.value is None


def test_a_dimensionless_field_rejects_a_united_value_text():
    claim = _claim(field_name="k", value=None, units=None, value_text="18.5 Torr")
    assert claim.value is None


def test_a_unit_attached_to_the_number_still_recovers():
    claim = _claim(field_name="thickness_nm", value=None, units=None, value_text="12nm")
    assert claim.value == pytest.approx(12.0)


def test_an_existing_value_is_never_overwritten():
    claim = _claim(
        field_name="growth_per_cycle_ang", value=1.42, units="A/cycle",
        value_text="0.98 A/cycle",
    )
    assert claim.value == pytest.approx(1.42)
    assert "recovered" not in claim.notes


def test_two_numbers_leave_the_value_alone():
    claim = _claim(
        field_name="growth_per_cycle_ang", value=None, units="A/cycle",
        value_text="1.42 and 0.98 A/cycle",
    )
    assert claim.value is None


# --- unit magnitude, not just dimension ------------------------------------
#
# The dimensional guard checks the *kind* of quantity and never its scale, so a field
# named thickness_nm could hold a value in angstrom and still read as comparable.
# Observed on the real corpus: the extractor emitted thickness_nm = 12 'nm' from the
# quote "the interfacial oxide measured 12 angstrom". 12 angstrom is 1.2 nm.


def test_a_length_in_angstrom_is_converted_to_the_nm_the_field_name_declares():
    claim = _claim(field_name="thickness_nm", value=12.0, units="angstrom")
    assert claim.value == pytest.approx(1.2)
    assert claim.units == "nm"
    assert "converted 12.0" in claim.notes


def test_a_pressure_in_millitorr_is_converted_to_torr():
    claim = _claim(field_name="pressure_torr", value=100.0, units="mTorr")
    assert claim.value == pytest.approx(0.1)
    assert claim.units == "torr"


def test_a_growth_rate_in_nm_per_cycle_becomes_angstrom_per_cycle():
    """The flagship number: 0.098 nm/cycle IS 0.98 A/cycle.

    Comparing 0.098 against 1.42 would report a 93% disagreement where the real one
    is 31%.
    """
    claim = _claim(field_name="growth_per_cycle_ang", value=0.098, units="nm/cycle")
    assert claim.value == pytest.approx(0.98)
    assert claim.units == "a/cycle"


def test_kelvin_becomes_celsius_by_offset_not_by_a_factor():
    claim = _claim(field_name="temperature_c", value=523.15, units="K")
    assert claim.value == pytest.approx(250.0)
    assert claim.units == "degC"


def test_a_canonical_value_is_left_completely_alone():
    claim = _claim(field_name="thickness_nm", value=12.0, units="nm")
    assert claim.value == pytest.approx(12.0)
    assert claim.units == "nm"
    assert claim.notes == ""


def test_an_unconvertible_spelling_is_flagged_and_not_comparable():
    """The honest middle: readable, and explicitly not comparable.

    Silently trusting a number whose scale nobody established is the failure mode this
    whole guard exists to prevent, so an unknown spelling must not simply pass.
    """
    claim = _claim(
        field_name="thickness_nm", value=12.0, units="furlongs",
        context={"temperature_k": 300.0, "frequency_hz": 1e4},
    )
    assert claim.value == pytest.approx(12.0), "the number must not be rescaled by guesswork"
    assert claim.magnitude_unverified is True
    assert claim.is_comparable is False
    assert "no conversion is known" in claim.notes


def test_a_field_whose_name_declares_no_unit_is_untouched():
    """Only a unit-declaring name asserts a scale."""
    claim = _claim(field_name="rho", value=8.7, units="g/cm3")
    assert claim.value == pytest.approx(8.7)
    assert claim.units == "g/cm3"
    assert claim.magnitude_unverified is False


def test_a_claim_with_no_units_is_not_rescaled():
    claim = _claim(field_name="thickness_nm", value=12.0, units=None)
    assert claim.value == pytest.approx(12.0)
    assert claim.magnitude_unverified is False


def test_conversion_applies_after_value_text_recovery():
    """A recovered scalar must be normalised too, not left in the source's unit."""
    claim = _claim(
        field_name="thickness_nm", value=None, units=None,
        value_text="12 angstrom",
    )
    assert claim.value == pytest.approx(1.2)
    assert claim.units == "nm"


def test_every_canonical_unit_has_an_identity_factor():
    """A canonical unit missing from its own table would convert nothing."""
    from cnms_fom.research.contracts import CANONICAL_UNIT, UNIT_FACTORS

    for field_name, canonical in CANONICAL_UNIT.items():
        assert canonical in UNIT_FACTORS, f"{field_name} declares {canonical}, untabulated"
        assert UNIT_FACTORS[canonical][canonical] == 1.0, canonical


def test_every_canonical_unit_classifies_to_its_field_dimension():
    """The two tables must agree, or a conversion would fight the rejection guard."""
    from cnms_fom.research.contracts import (
        CANONICAL_UNIT,
        FIELD_DIMENSION,
        classify_unit,
    )

    for field_name, canonical in CANONICAL_UNIT.items():
        expected = FIELD_DIMENSION.get(field_name)
        if expected is None:
            continue
        assert classify_unit(canonical) == expected, (
            f"{field_name}: canonical {canonical!r} classifies as "
            f"{classify_unit(canonical)}, but the field expects {expected}"
        )


# --- the quote outranks the declared units ---------------------------------
#
# The live failure: {"field": "thickness_nm", "value": 12, "units": "nm"} from the
# quote "the interfacial oxide measured 12 angstrom". The declared unit and the field
# name agreed with each other and were both wrong about the source, so no guard fired
# and the claim read `ok` while being ten times the truth.


def _quoted(field_name: str, value: float, units: str, quote: str):
    return ExtractedClaim(
        field_name=field_name, value=value, units=units, tier=ClaimTier.MEASURED,
        evidence=[_evidence(quote=quote)],
    )


def test_a_declared_unit_contradicting_the_quote_is_corrected():
    claim = _quoted(
        "thickness_nm", 12.0, "nm", "the interfacial oxide measured 12 angstrom"
    )
    assert claim.value == pytest.approx(1.2)
    assert "contradict the quote" in claim.notes


def test_a_declared_unit_agreeing_with_the_quote_is_untouched():
    claim = _quoted(
        "growth_per_cycle_ang", 1.42, "A/cycle",
        "the growth per cycle saturated at 1.42 angstrom per cycle",
    )
    assert claim.value == pytest.approx(1.42)
    assert claim.units == "A/cycle"
    assert "contradict" not in claim.notes


def test_a_value_appearing_twice_in_the_quote_is_left_alone():
    """Two occurrences give two candidate units, so there is no single answer."""
    claim = _quoted(
        "thickness_nm", 12.0, "nm", "12 angstrom initially, then 12 nm after anneal"
    )
    assert claim.value == pytest.approx(12.0)
    assert "contradict" not in claim.notes


def test_a_value_absent_from_the_quote_is_left_alone():
    claim = _quoted("thickness_nm", 99.0, "nm", "the oxide measured 12 angstrom")
    assert claim.value == pytest.approx(99.0)


def test_an_unrecognised_trailing_token_is_left_alone():
    claim = _quoted("thickness_nm", 12.0, "nm", "the oxide measured 12 widgets across")
    assert claim.value == pytest.approx(12.0)
    assert "contradict" not in claim.notes


def test_a_cross_dimension_quote_does_not_rewrite_the_value():
    """A time in the quote must not silently become a length."""
    claim = _quoted("thickness_nm", 6.0, "nm", "a 6 s purge followed each dose")
    assert claim.value == pytest.approx(6.0)
    assert "contradict" not in claim.notes


def test_a_field_with_no_declared_unit_is_not_reconciled():
    claim = _quoted("rho", 8.7, "g/cm3", "density from XRR was 8.7 g/cm3")
    assert claim.value == pytest.approx(8.7)
    assert claim.units == "g/cm3"
