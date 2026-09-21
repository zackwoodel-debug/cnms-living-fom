"""Brief generation: assembly, abstention, contradictions, and what it will not do.

Runs on SQLite with a scripted provider and the dense retriever disabled, so the
whole path is exercised without Ollama, Postgres, or a network.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import (
    BriefStatus,
    CardCategory,
    ClaimTier,
    StatementKind,
    SynthesisTechnique,
)
from cnms_fom.db.models import (
    BoObservation,
    BoRun,
    Document,
    DocumentChunk,
    FomDefinition,
)
from cnms_fom.research.brief import generate_brief
from cnms_fom.research.policy import BASELINE
from cnms_fom.research.store import claims_for_field, load_brief, review_brief, save_brief
from tests.fakes import ResearchProvider

HOT_WALL = (
    "Between 200 and 300 degC the growth per cycle was constant at 0.98 angstrom per cycle "
    "in a Beneq TFS-200 hot-wall reactor using TDMAH and water at 1.5 Torr."
)
CROSS_FLOW = (
    "Over the range 200 to 300 degC the growth per cycle saturated at 1.42 angstrom per cycle "
    "in a cross-flow reactor at 0.3 Torr using TDMAH and water."
)

CLAIM_HOT = {
    "field": "growth_per_cycle_ang", "value": 0.98, "units": "A/cycle", "tier": "measured",
    "context": {"technique": "ald", "temperature_k": 523.0, "precursor": "TDMAH",
                "chamber": "hot-wall"},
    "quote": "growth per cycle was constant at 0.98 angstrom per cycle", "confidence": 0.9,
}
CLAIM_CROSS = {
    "field": "growth_per_cycle_ang", "value": 1.42, "units": "A/cycle", "tier": "measured",
    "context": {"technique": "ald", "temperature_k": 523.0, "precursor": "TDMAH",
                "chamber": "cross-flow"},
    "quote": "growth per cycle saturated at 1.42 angstrom per cycle", "confidence": 0.9,
}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'brief.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()

    for index, (title, text) in enumerate(
        (("ALD of HfO2 hot-wall", HOT_WALL), ("ALD of HfO2 cross-flow", CROSS_FLOW)), start=1
    ):
        document = Document(
            title=title, filename=f"{title}.pdf", content_sha256=f"hash{index}",
            technique=SynthesisTechnique.ALD, doi=f"10.0000/doc{index}", n_pages=1,
        )
        session.add(document)
        session.flush()
        session.add(DocumentChunk(
            document_id=document.id, chunk_index=0, page=index, text=text,
        ))
    session.commit()
    yield session
    session.close()
    engine.dispose()


#  Lexical-only, so no embedder is needed anywhere in this file.
POLICY = BASELINE.evolve(name="test", use_dense=False)


def _campaign(db, *, approved=False):
    definition = FomDefinition(
        name="logic", version=1, application="logic", weights={"k": 1.0},
        normalization={}, floor_eps=1e-3, approved=approved,
        approved_by="Z. Woodel" if approved else None,
    )
    db.add(definition)
    db.flush()
    run = BoRun(
        name="hfo2_logic", fom_definition_id=definition.id,
        search_space={"parameters": [
            {"name": "substrate_temp_c", "kind": "continuous", "lower": 150.0, "upper": 400.0}
        ]},
        constraints={"bounds": {}, "allowed_choices": {}, "notes": []},
    )
    db.add(run)
    db.flush()
    db.add(BoObservation(bo_run_id=run.id, parameters={"substrate_temp_c": 250.0},
                         objective_value=-1.2, is_feasible=True))
    db.commit()
    return run


# --- assembly -------------------------------------------------------------


def test_a_brief_assembles_evidence_claims_and_a_fingerprint(db):
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "What is the growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    )

    assert brief.evidence
    assert brief.claims
    claim = brief.claims[0]
    assert claim.field_name == "growth_per_cycle_ang"
    assert claim.value == pytest.approx(0.98)
    assert claim.tier is ClaimTier.MEASURED   # what the *source* said
    assert claim.prompt_version == POLICY.extraction_prompt_version
    assert brief.policy_version == POLICY.version
    assert brief.fingerprint()
    assert brief.abstained is False


def test_a_brief_records_every_tool_call(db):
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    tools = [call["tool"] for call in brief.tool_calls]
    assert "search_cards" in tools
    assert "retrieve" in tools
    assert "extract_claims" in tools


def test_a_brief_is_read_only(db):
    """Generating one must not touch a scientific table."""
    from cnms_fom.db.models import DescriptorValue, FomScore, PropertyValue

    provider = ResearchProvider(claims=[CLAIM_HOT])
    generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    assert db.query(PropertyValue).count() == 0
    assert db.query(DescriptorValue).count() == 0
    assert db.query(FomScore).count() == 0


def test_the_extracted_claim_never_reads_as_a_measurement(db):
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    payload = brief.as_dict()
    assert payload["claims"][0]["is_measurement"] is False
    assert "not a measurement" in payload["disclaimer"]


# --- contradictions -------------------------------------------------------


def test_two_documents_disagreeing_produce_a_contradiction_not_an_average(db):
    provider = ResearchProvider(claims_by_page={1: [CLAIM_HOT], 2: [CLAIM_CROSS]})
    brief = generate_brief(
        db, "growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    )

    assert len(brief.contradictions) == 1
    contradiction = brief.contradictions[0]
    assert {contradiction.left.value, contradiction.right.value} == {0.98, 1.42}
    #  The explanation is named, not guessed at.
    assert "chamber" in contradiction.differing_context
    assert "relative spread" in contradiction.basis
    #  And no merged value exists anywhere.
    assert any("no average was taken" in w for w in brief.warnings)


def test_one_document_stating_two_values_is_not_a_contradiction(db):
    """A range or two conditions in one passage is one source, not a disagreement."""
    provider = ResearchProvider(claims=[CLAIM_HOT, CLAIM_CROSS])
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    #  Both claims came from the same passage each time, so every pair shares a page.
    same_page_pairs = [
        c for c in brief.contradictions
        if c.left.evidence[0].page == c.right.evidence[0].page
    ]
    assert same_page_pairs == []


# --- context completeness -------------------------------------------------


def test_a_claim_missing_required_context_is_flagged_and_becomes_a_gap(db):
    incomplete = {**CLAIM_HOT, "context": {"technique": "ald", "temperature_k": 523.0}}
    provider = ResearchProvider(claims=[incomplete])
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)

    claim = brief.claims[0]
    assert set(claim.missing_context) == {"precursor", "chamber"}
    assert claim.is_comparable is False
    assert any("missing required context" in w for w in brief.warnings)
    assert any("Under what" in g.question for g in brief.data_gaps)
    #  Nothing was invented to fill it.
    assert "chamber" not in claim.context


# --- abstention -----------------------------------------------------------


def test_a_question_the_corpus_cannot_answer_abstains_with_a_route_out(db):
    provider = ResearchProvider(grade=0, rewrite=None)
    brief = generate_brief(
        db, "molecular beam epitaxy of gallium arsenide on germanium", provider=provider,
        policy=POLICY,
    )
    assert brief.abstained is True
    assert brief.claims == []
    assert brief.data_gaps
    gap = brief.data_gaps[0]
    assert gap.what_would_resolve_it
    assert "diagnostics=true" in gap.what_would_resolve_it
    #  Abstaining means no narrative was invented.
    assert brief.statements == []
    assert "interpret" not in provider.calls


def test_abstention_is_recorded_as_an_outcome_not_an_error(db):
    provider = ResearchProvider(grade=0, rewrite=None)
    brief = generate_brief(db, "gallium arsenide on germanium", provider=provider, policy=POLICY)
    assert any("abstains" in w for w in brief.warnings)
    assert brief.status is BriefStatus.PROPOSED


# --- campaign awareness ---------------------------------------------------


def test_campaign_warnings_are_attached_without_the_model_noticing_them(db):
    """The warnings are computed, so they cannot depend on the narrative."""
    run = _campaign(db, approved=False)
    #  A provider that produces no statements at all.
    provider = ResearchProvider(claims=[CLAIM_HOT], statements=[], actions=[])
    brief = generate_brief(
        db, "growth per cycle?", bo_run_id=run.id, provider=provider, policy=POLICY
    )

    assert any("UNAPPROVED" in w and "not a result" in w for w in brief.warnings)
    assert brief.statements == []          # the model said nothing
    assert brief.fom_definition == "logic v1"
    assert any(c["tool"] == "campaign_snapshot" for c in brief.tool_calls)


def test_an_approved_campaign_does_not_raise_the_unapproved_warning(db):
    run = _campaign(db, approved=True)
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", bo_run_id=run.id, provider=provider, policy=POLICY
    )
    assert not any("UNAPPROVED" in w for w in brief.warnings)
    #  Approved but unfrozen is still worth saying.
    assert any("not frozen" in w for w in brief.warnings)


# --- cards ----------------------------------------------------------------


def test_a_proposed_card_is_flagged_and_not_used_as_evidence(db):
    from cnms_fom.knowledge.cards import upsert_card

    upsert_card(
        db, slug="concepts/ald-window-hfo2", title="ALD window for HfO2 growth per cycle",
        body="GPC saturates at 0.98 A/cycle.", category=CardCategory.PROCESS_WINDOW,
        sources=[{"kind": "document", "document_id": 1, "page": 1}],
    )
    db.commit()

    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(db, "ALD window for HfO2", provider=provider, policy=POLICY)

    assert any("PROPOSED" in w and "nothing in this brief rests on it" in w
               for w in brief.warnings)
    assert not any(item.document_title.startswith("card:") for item in brief.evidence)


def test_a_reviewed_card_becomes_evidence(db):
    from cnms_fom.knowledge.cards import review_card, upsert_card

    upsert_card(
        db, slug="concepts/ald-window-hfo2", title="ALD window for HfO2 growth per cycle",
        body="GPC saturates at 0.98 A/cycle.", summary="0.98 A/cycle between 200 and 300 C.",
        category=CardCategory.PROCESS_WINDOW,
        sources=[{"kind": "document", "document_id": 1, "page": 1}],
    )
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()

    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(db, "ALD window for HfO2", provider=provider, policy=POLICY)

    cards = [i for i in brief.evidence if i.document_title.startswith("card:")]
    assert len(cards) == 1
    assert cards[0].grade == 3
    assert "Z. Woodel" in cards[0].grade_reason


def test_a_stale_reviewed_card_is_not_used_as_evidence(db):
    from cnms_fom.knowledge.cards import review_card, upsert_card

    upsert_card(
        db, slug="concepts/ald-window-hfo2", title="ALD window for HfO2 growth per cycle",
        body="GPC saturates at 0.98 A/cycle.",
        sources=[{"kind": "document", "document_id": 1, "page": 1}],
    )
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    #  Edited after review: the review no longer covers what it says.
    upsert_card(
        db, slug="concepts/ald-window-hfo2", title="ALD window for HfO2 growth per cycle",
        body="Revised: 1.4 A/cycle.",
    )
    db.commit()

    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(db, "ALD window for HfO2", provider=provider, policy=POLICY)

    assert any("edited since" in w for w in brief.warnings)
    assert not any(i.document_title.startswith("card:") for i in brief.evidence)


# --- extraction robustness -----------------------------------------------


def test_a_quote_not_present_in_the_passage_is_rejected(db):
    """A model that paraphrases its evidence has broken the only link back."""
    fabricated = {**CLAIM_HOT, "quote": "we grew the film at 900 degrees in a furnace"}
    provider = ResearchProvider(claims=[fabricated])
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)

    assert any("quoted text that is not in the passage" in w for w in brief.warnings)
    #  The claim survives but is demoted, and its quote is the passage's own.
    claim = brief.claims[0]
    assert claim.model_confidence <= POLICY.low_confidence_threshold
    assert "0.98 angstrom per cycle" in claim.evidence[0].quote


def test_an_unparseable_extraction_reply_loses_no_brief(db):
    provider = ResearchProvider(malformed_extraction=True)
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    assert brief.claims == []
    assert any("Unparseable extraction" in w for w in brief.warnings)
    assert brief.evidence  # the passages still stand


def test_an_extraction_backend_failure_is_reported_not_raised(db):
    provider = ResearchProvider(raise_on_extraction=True)
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    assert any("Extraction failed" in w for w in brief.warnings)


def test_a_low_confidence_claim_is_kept_and_marked(db):
    """A discarded extraction is invisible; an underconfident extractor would look
    like an empty corpus."""
    unsure = {**CLAIM_HOT, "confidence": 0.1}
    provider = ResearchProvider(claims=[unsure])
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    #  Extraction runs once per passage, and both documents match this query, so
    #  every retrieved passage yields its own claim. All of them are kept.
    assert len(brief.claims) == 2
    assert all("below the policy threshold" in c.notes for c in brief.claims)


# --- interpretation ------------------------------------------------------


def test_interpretation_statements_are_labelled(db):
    provider = ResearchProvider(
        claims=[CLAIM_HOT],
        statements=[
            {"kind": "evidence", "text": "The hot-wall reactor gives 0.98 A/cycle [p. 1]."},
            {"kind": "interpretation", "text": "The reactors differ in pressure."},
            {"kind": "proposal", "text": "Run both at one pressure."},
        ],
        actions=["Measure at 1.5 and 0.3 Torr on one tool."],
    )
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)

    kinds = [s.kind for s in brief.statements]
    assert kinds == [StatementKind.EVIDENCE, StatementKind.INTERPRETATION, StatementKind.PROPOSAL]
    assert brief.statements[0].evidence  # an evidence statement carries it
    assert brief.proposed_actions == ["Measure at 1.5 and 0.3 Torr on one tool."]


def test_a_refused_interpretation_leaves_the_evidence_intact(db):
    provider = ResearchProvider(claims=[CLAIM_HOT], refuse_interpretation=True)
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    assert brief.claims  # unaffected
    assert any("declined to write the interpretation" in w for w in brief.warnings)


def test_interpret_false_skips_the_narrative(db):
    """What the benchmark scores: the numbers and the abstention, not the prose."""
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", provider=provider, policy=POLICY, interpret=False
    )
    assert brief.claims
    assert brief.statements == []
    assert "interpret" not in provider.calls


# --- retrieval unavailable ----------------------------------------------


def test_retrieval_failing_is_not_reported_as_an_empty_literature(db, no_dense_retrieval):
    """Saying the corpus is silent about a search that did not run is a wrong
    scientific conclusion."""
    dense_only = BASELINE.evolve(name="dense_only_test", use_lexical=False)
    provider = ResearchProvider()
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=dense_only)

    assert any("did not run" in w or "retrieval failed" in w.lower() for w in brief.warnings)
    assert any("must not be read as gaps in the literature" in w for w in brief.warnings)


# --- persistence --------------------------------------------------------


def test_a_brief_round_trips_through_the_database(db):
    provider = ResearchProvider(claims_by_page={1: [CLAIM_HOT], 2: [CLAIM_CROSS]})
    brief = generate_brief(db, "growth per cycle?", provider=provider, policy=POLICY)
    record = save_brief(db, brief)
    db.commit()

    loaded = load_brief(db, record.id)
    assert loaded.research_question == brief.research_question
    assert len(loaded.claims) == len(brief.claims)
    assert len(loaded.contradictions) == len(brief.contradictions)
    assert loaded.policy_version == brief.policy_version
    #  The provenance survived, which is the whole point of storing it.
    claim = loaded.claims[0]
    assert claim.evidence[0].page in (1, 2)
    assert claim.evidence[0].quote


def test_stored_claims_are_queryable_by_field_without_aggregation(db):
    """The query the corpus exists to answer, and it returns every claim separately."""
    provider = ResearchProvider(claims_by_page={1: [CLAIM_HOT], 2: [CLAIM_CROSS]})
    save_brief(db, generate_brief(
        db, "growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    ))
    db.commit()

    rows = claims_for_field(db, "growth_per_cycle_ang")
    assert len(rows) == 2
    assert {row["value"] for row in rows} == {0.98, 1.42}
    assert all(row["is_measurement"] is False for row in rows)
    assert all(row["citation"] and row["quote"] for row in rows)
    #  Nothing aggregated.
    assert all("mean" not in row for row in rows)


def test_missing_context_is_stored_as_a_queryable_column(db):
    incomplete = {**CLAIM_HOT, "context": {"technique": "ald"}}
    provider = ResearchProvider(claims=[incomplete])
    save_brief(db, generate_brief(
        db, "growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    ))
    db.commit()

    from cnms_fom.db.models import ExtractedClaimRecord

    rows = db.query(ExtractedClaimRecord).all()
    assert rows
    assert all(
        set(row.missing_context) == {"temperature_k", "precursor", "chamber"} for row in rows
    )


def test_reviewing_a_brief_needs_a_named_reviewer(db):
    from cnms_fom.research.store import ReviewRefused

    provider = ResearchProvider(claims=[CLAIM_HOT])
    record = save_brief(db, generate_brief(
        db, "growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    ))
    db.commit()

    with pytest.raises(ReviewRefused, match="named reviewer"):
        review_brief(db, record.id, reviewed_by="  ")

    reviewed = review_brief(db, record.id, reviewed_by="Z. Woodel")
    db.commit()
    assert reviewed.status is BriefStatus.REVIEWED
    assert reviewed.reviewed_by == "Z. Woodel"


def test_reviewing_a_brief_promotes_nothing(db):
    """A review records that someone read it, not that anything became measured."""
    from cnms_fom.db.models import PropertyValue

    provider = ResearchProvider(claims=[CLAIM_HOT])
    record = save_brief(db, generate_brief(
        db, "growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    ))
    review_brief(db, record.id, reviewed_by="Z. Woodel")
    db.commit()
    assert db.query(PropertyValue).count() == 0


# --- records: fits, plausibility, properties ------------------------------


SAMPLE = "HFO2-PILOT-07"


def _import_fits(db, *, xrr_thickness=103.4, se_thickness=152.0, density=9.1, sld=64.6):
    """Two single-technique fits on one film, so they can be compared."""
    from cnms_fom.modalfit.records import import_fit

    base = {
        "sample_id": SAMPLE,
        "stack": [
            {"role": "ambient", "label": "air"},
            {"role": "layer", "label": "hfo2_film", "material": "HfO2",
             "structural": {"thickness": {"value": xrr_thickness, "min": 50.0, "max": 250.0,
                                          "vary": True},
                            "roughness": {"value": 4.2, "min": 0.0, "max": 20.0, "vary": True}},
             "xray": {"sld_real": {"value": sld, "min": 55.0, "max": 75.0, "vary": True}},
             "molecular": {"formula": "HfO2",
                           "density": {"value": density, "min": 8.0, "max": 10.0, "vary": True}}},
            {"role": "substrate", "label": "silicon", "material": "Si"},
        ],
    }
    import_fit(db, {**base, "stack_id": "xrr",
                    "fit": {"techniques": ["XRR"], "algorithm": "L-BFGS-B", "chi2": 1.84}})

    se = {k: v for k, v in base.items()}
    se["stack_id"] = "se"
    se["fit"] = {"techniques": ["SE"], "algorithm": "Nelder-Mead", "chi2": 3.02}
    import copy
    se["stack"] = copy.deepcopy(base["stack"])
    se["stack"][1]["structural"]["thickness"]["value"] = se_thickness
    se["stack"][1]["optical"] = {"n": {"value": 2.05, "min": 1.5, "max": 2.5, "vary": True}}
    import_fit(db, se)
    db.commit()


def test_a_brief_surfaces_a_cross_technique_disagreement_from_the_fits(db):
    _import_fits(db)
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", sample_id=SAMPLE, provider=provider, policy=POLICY
    )

    warning = next(w for w in brief.warnings if "Techniques disagree on thickness" in w)
    assert "nothing was averaged" in warning
    tools = [c["tool"] for c in brief.tool_calls]
    assert "list_sample_fits" in tools
    assert "fit_disagreements" in tools
    assert "check_fit_plausibility" in tools


def test_a_brief_surfaces_a_fit_that_disagrees_with_itself(db):
    """An SLD and a density that cannot both be right for this composition."""
    _import_fits(db, density=4.0, sld=64.6)  # 4.0 g/cm3 implies ~28e-6, not 64.6
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", sample_id=SAMPLE, provider=provider, policy=POLICY
    )
    assert any("disagrees with itself" in w for w in brief.warnings)


def test_a_brief_surfaces_a_clamped_fit_parameter(db):
    _import_fits(db, xrr_thickness=250.0)  # == the upper bound
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", sample_id=SAMPLE, provider=provider, policy=POLICY
    )
    warning = next(w for w in brief.warnings if "clamped" in w)
    assert "somebody typed" in warning


def test_fit_records_never_become_literature_claims(db):
    """A fitted thickness is not a thing a paper said."""
    _import_fits(db)
    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", sample_id=SAMPLE, provider=provider, policy=POLICY
    )
    assert all(c.field_name == "growth_per_cycle_ang" for c in brief.claims)
    assert not any(c.field_name in ("thickness_ang", "sld_xray") for c in brief.claims)
    #  The fits are in the trace, where they are identifiable as records.
    assert any(c["tool"] == "list_sample_fits" for c in brief.tool_calls)


def test_stored_property_values_are_read_when_a_material_is_named(db):
    from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
    from cnms_fom.db.models import Material, PropertyValue

    material = Material(formula="HfO2", formula_reduced="HfO2", polymorph="monoclinic",
                        specimen_form=SpecimenForm.CRYSTALLINE_FILM)
    db.add(material)
    db.flush()
    db.add(PropertyValue(material_id=material.id, property_key="k", value=25.0, units="1",
                         temperature_k=300.0, frequency_hz=1e4,
                         provenance_tier=ProvenanceTier.MEASURED))
    db.commit()

    provider = ResearchProvider(claims=[CLAIM_HOT])
    brief = generate_brief(
        db, "growth per cycle?", material="HfO2", target_property="k",
        provider=provider, policy=POLICY,
    )
    call = next(c for c in brief.tool_calls if c["tool"] == "lookup_property_values")
    assert call["result"]["n_values"] == 1
    assert call["result"]["values"][0]["provenance_tier"] == "measured"


# --- an unreachable model is not a data gap -------------------------------
#
# Found on the first real-provider run: a benchmark case whose every model call
# raised still abstained, and the abstention was scored as correct. In production
# the same shape means an Ollama outage reports "insufficient evidence" about the
# corpus, which is a claim about the science rather than about the infrastructure.


class _UnreachableProvider(ResearchProvider):
    """Every call raises, the way an unreachable Ollama behaves."""

    def send(self, system, messages, *, tools=None, temperature: float = 0.0):
        raise RuntimeError("connection refused")


def test_a_brief_whose_model_calls_failed_is_marked_degraded(db):
    brief = generate_brief(
        db, "What is the growth per cycle for HfO2 ALD?",
        provider=_UnreachableProvider(claims=[CLAIM_HOT]), policy=POLICY,
    )

    assert brief.degraded is True
    assert "failed to reach the model" in brief.degraded_reason
    #  The distinction that matters: this is not a statement about the corpus.
    assert "not evidence of absence" in brief.degraded_reason


def test_a_degraded_brief_says_it_abstained_for_want_of_a_model_not_evidence(db):
    brief = generate_brief(
        db, "What is the growth per cycle for HfO2 ALD?",
        provider=_UnreachableProvider(claims=[CLAIM_HOT]), policy=POLICY,
    )

    joined = " ".join(brief.warnings)
    if brief.abstained:
        assert "for want of a working model rather than for want of evidence" in joined
        #  And the reasoned-abstention wording must NOT appear; that would assert a
        #  finding about the evidence that was never actually checked.
        assert "the evidence did not clear the policy's threshold" not in joined


def test_a_healthy_brief_is_not_degraded(db):
    brief = generate_brief(
        db, "What is the growth per cycle for HfO2 ALD?",
        provider=ResearchProvider(claims=[CLAIM_HOT]), policy=POLICY,
    )
    assert brief.degraded is False
    assert brief.degraded_reason == ""


def test_the_cost_report_counts_failures_alongside_calls(db):
    brief = generate_brief(
        db, "growth per cycle?", provider=_UnreachableProvider(claims=[CLAIM_HOT]),
        policy=POLICY,
    )
    report = next(c["result"] for c in brief.tool_calls if c["tool"] == "cost_report")

    assert "grading_failed" in report and "extraction_failed" in report
    assert report["grading_failed"] + report["extraction_failed"] > 0


def test_degraded_survives_the_round_trip(db):
    """The flag has to reach whoever reads the brief later, not just the logs."""
    brief = generate_brief(
        db, "growth per cycle?", provider=_UnreachableProvider(claims=[CLAIM_HOT]),
        policy=POLICY,
    )
    assert brief.as_dict()["degraded"] is True
    assert brief.as_dict()["degraded_reason"] == brief.degraded_reason


# --- retrieved passages with no numbers in them ---------------------------
#
# Found on a real-provider benchmark run: the GaAs-on-Ge question retrieved
# GaAs-on-GaAs passages (close enough to pass grading), extraction correctly found
# nothing in them that answered the question, and the brief did not abstain — so it
# went on to interpret passages it had extracted nothing from.


def test_passages_with_no_extractable_claims_abstain(db):
    """`_should_abstain` took `claims` and never looked at it. This is why it needed to."""
    provider = ResearchProvider(claims=[])
    brief = generate_brief(
        db, "What is the growth per cycle for HfO2 ALD?", provider=provider, policy=POLICY
    )

    assert brief.claims == []
    assert brief.evidence, "the point of the case is that passages WERE retrieved"
    assert brief.abstained is True
    #  And it is a reasoned abstention, not an infrastructure one.
    assert brief.degraded is False


def test_a_brief_assembled_without_a_provider_is_not_an_abstention_about_the_corpus(db):
    """No provider means extraction never ran, so zero claims says nothing about the corpus."""
    brief = generate_brief(db, "growth per cycle?", provider=None, policy=POLICY)

    assert brief.claims == []
    #  Whatever it decides, it must not be claiming the corpus lacks the evidence on
    #  the strength of an extraction that never happened.
    if brief.abstained:
        assert not any("did not clear the policy's threshold" in w for w in brief.warnings) \
            or brief.evidence == []


def test_claims_present_means_no_abstention_on_that_ground(db):
    brief = generate_brief(
        db, "What is the growth per cycle for HfO2 ALD?",
        provider=ResearchProvider(claims=[CLAIM_HOT]), policy=POLICY,
    )
    assert brief.claims
    assert brief.abstained is False
