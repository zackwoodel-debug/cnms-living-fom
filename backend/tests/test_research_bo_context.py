"""The BO context bridge: propose, review, apply — and every refusal in between.

This is the only path by which evidence may influence the optimizer, so almost
every test here is a refusal. No BoTorch, no model server, no Postgres.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import CardCategory, ContextStatus
from cnms_fom.db.models import BoObservation, BoRun, CampaignContextProposal
from cnms_fom.knowledge.cards import review_card, upsert_card
from cnms_fom.research.bo_context import (
    ContextRefused,
    ProposalNotFound,
    apply_context,
    list_proposals,
    propose_context,
    revert_context,
    review_context,
    validate_supporting_cards,
)
from cnms_fom.research.campaign import snapshot
from cnms_fom.research.contracts import BoundProposal, ProposedBOContext

SPACE = {
    "parameters": [
        {"name": "substrate_temp_c", "kind": "continuous", "lower": 150.0, "upper": 400.0},
        {"name": "purge_s", "kind": "continuous", "lower": 1.0, "upper": 20.0},
        {"name": "substrate", "kind": "categorical", "choices": ["Si(100)", "SiO2", "Ge"]},
    ]
}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'bridge.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def run(db):
    campaign = BoRun(
        name="hfo2_logic", search_space=SPACE,
        constraints={"bounds": {}, "allowed_choices": {}, "notes": []},
    )
    db.add(campaign)
    db.flush()
    db.add(BoObservation(bo_run_id=campaign.id, parameters={"substrate_temp_c": 250.0},
                         objective_value=-1.2, is_feasible=True))
    db.commit()
    return campaign


@pytest.fixture
def card(db):
    """A reviewed, sourced, process-window card — the only kind the bridge reads."""
    upsert_card(
        db, slug="concepts/ald-window-hfo2", title="ALD window for HfO2",
        body="GPC saturates at 0.98 A/cycle between 200 and 300 C [1].",
        category=CardCategory.PROCESS_WINDOW,
        sources=[{"kind": "document", "document_id": 1, "page": 7}],
    )
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()
    return "concepts/ald-window-hfo2"


def _narrowing(run_id, card_slug=None, **kwargs) -> ProposedBOContext:
    return ProposedBOContext(
        bo_run_id=run_id,
        recommended_bounds=[BoundProposal(
            parameter="substrate_temp_c", lower=200.0, upper=300.0,
            rationale="ALD window reported at 200-300 C",
        )],
        supporting_card_slugs=[card_slug] if card_slug else [],
        rationale="narrow to the reported window",
        **kwargs,
    )


# --- card validation ------------------------------------------------------


def test_only_reviewed_sourced_bo_relevant_cards_qualify(db, card):
    accepted, problems = validate_supporting_cards(db, [card])
    assert problems == []
    assert accepted[0]["slug"] == card
    assert accepted[0]["body_sha256"]


def test_an_unreviewed_card_is_refused(db):
    upsert_card(db, slug="concepts/draft", title="Draft", body="x",
                category=CardCategory.PROCESS_WINDOW,
                sources=[{"kind": "document", "document_id": 1, "page": 1}])
    db.commit()
    _, problems = validate_supporting_cards(db, ["concepts/draft"])
    assert any("not reviewed" in p and "Sec. 15.2" in p for p in problems)


def test_a_stale_card_is_refused(db, card):
    upsert_card(db, slug=card, title="ALD window for HfO2", body="Revised: 1.4 A/cycle.")
    db.commit()
    _, problems = validate_supporting_cards(db, [card])
    assert any("edited since" in p for p in problems)


def test_a_card_in_an_unread_category_is_refused(db):
    upsert_card(db, slug="concepts/caveat", title="Caveat", body="x",
                category=CardCategory.MEASUREMENT_CAVEAT,
                sources=[{"kind": "document", "document_id": 1, "page": 1}])
    review_card(db, "concepts/caveat", reviewed_by="Z. Woodel")
    db.commit()
    _, problems = validate_supporting_cards(db, ["concepts/caveat"])
    assert any("does not read" in p for p in problems)


def test_a_missing_card_is_refused(db):
    _, problems = validate_supporting_cards(db, ["concepts/nope"])
    assert any("does not exist" in p for p in problems)


# --- propose --------------------------------------------------------------


def test_a_narrowing_proposal_is_recorded_and_changes_nothing(db, run, card):
    before = snapshot(db, run.id).fingerprint()
    proposal = propose_context(db, _narrowing(run.id, card))
    db.commit()

    assert proposal.status is ContextStatus.PROPOSED
    assert proposal.campaign_fingerprint_before == before
    #  The campaign is untouched.
    assert snapshot(db, run.id).fingerprint() == before
    assert run.constraints["bounds"] == {}


def test_an_empty_proposal_is_refused(db, run):
    with pytest.raises(ContextRefused, match="empty"):
        propose_context(db, ProposedBOContext(bo_run_id=run.id))


def test_a_widening_proposal_is_refused_at_propose_time(db, run, card):
    context = ProposedBOContext(
        bo_run_id=run.id,
        recommended_bounds=[BoundProposal(
            parameter="substrate_temp_c", lower=100.0, upper=500.0,
            rationale="a paper grew films at 450 C",
        )],
        supporting_card_slugs=[card],
    )
    with pytest.raises(ContextRefused, match="never widen"):
        propose_context(db, context)


def test_a_bound_change_without_a_reviewed_card_is_refused(db, run):
    """A brief is unreviewed by construction, so it cannot warrant a bound."""
    with pytest.raises(ContextRefused, match="at least one reviewed, sourced card"):
        propose_context(db, _narrowing(run.id))


def test_advisory_only_proposals_need_no_card(db, run):
    """A note a human reads is a different risk from a bound the optimizer obeys."""
    context = ProposedBOContext(
        bo_run_id=run.id,
        soft_priors=["density likely below bulk for low-temperature growth"],
        uncertainty_notes=["only one source for the upper limit"],
    )
    proposal = propose_context(db, context)
    db.commit()
    assert proposal.status is ContextStatus.PROPOSED
    assert proposal.supporting_cards is None


def test_a_proposal_resting_on_an_unreviewed_card_is_refused(db, run):
    upsert_card(db, slug="concepts/draft", title="Draft", body="x",
                category=CardCategory.PROCESS_WINDOW,
                sources=[{"kind": "document", "document_id": 1, "page": 1}])
    db.commit()
    with pytest.raises(ContextRefused, match="not reviewed"):
        propose_context(db, _narrowing(run.id, "concepts/draft"))


# --- review ---------------------------------------------------------------


def test_review_needs_a_named_reviewer(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    db.commit()
    with pytest.raises(ContextRefused, match="named reviewer"):
        review_context(db, proposal.id, reviewed_by="   ")


def test_review_changes_no_campaign(db, run, card):
    before = snapshot(db, run.id).fingerprint()
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    db.commit()

    assert proposal.status is ContextStatus.REVIEWED
    assert snapshot(db, run.id).fingerprint() == before


def test_rejecting_a_proposal_blocks_it_permanently(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel", accept=False, note="not convinced")
    db.commit()
    assert proposal.status is ContextStatus.REJECTED
    assert proposal.review_note == "not convinced"

    with pytest.raises(ContextRefused, match="already rejected"):
        review_context(db, proposal.id, reviewed_by="Z. Woodel")
    with pytest.raises(ContextRefused, match="only a REVIEWED proposal"):
        apply_context(db, proposal.id, applied_by="Z. Woodel")


def test_a_card_edited_before_review_blocks_acceptance(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    db.commit()
    upsert_card(db, slug=card, title="ALD window for HfO2", body="Revised: 1.4 A/cycle.")
    db.commit()

    with pytest.raises(ContextRefused, match="edited since"):
        review_context(db, proposal.id, reviewed_by="Z. Woodel")


def test_an_unknown_proposal_raises(db):
    with pytest.raises(ProposalNotFound):
        review_context(db, 999, reviewed_by="Z. Woodel")


# --- apply ----------------------------------------------------------------


def test_an_unreviewed_proposal_cannot_be_applied(db, run, card):
    """The single most important invariant in the bridge."""
    proposal = propose_context(db, _narrowing(run.id, card))
    db.commit()
    with pytest.raises(ContextRefused, match="only a REVIEWED proposal"):
        apply_context(db, proposal.id, applied_by="Z. Woodel")
    assert run.constraints["bounds"] == {}


def test_apply_needs_a_named_person(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    db.commit()
    with pytest.raises(ContextRefused, match="named person"):
        apply_context(db, proposal.id, applied_by="")


def test_applying_narrows_the_constraints_and_moves_the_fingerprint(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    db.commit()

    result = apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()

    assert result["fingerprint_changed"] is True
    assert result["constraints_after"]["bounds"]["substrate_temp_c"] == [200.0, 300.0]
    assert proposal.status is ContextStatus.APPLIED
    assert proposal.applied_by == "Z. Woodel"
    assert proposal.campaign_fingerprint_before != proposal.campaign_fingerprint_after
    assert "Only BoRun.constraints changed" in result["note"]


def test_applying_touches_only_the_constraints(db, run, card):
    space_before = dict(run.search_space)
    objective_before = (run.acquisition, run.objective_sense, run.fom_definition_id)
    observations_before = db.query(BoObservation).count()

    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()

    assert run.search_space == space_before
    assert (run.acquisition, run.objective_sense, run.fom_definition_id) == objective_before
    assert db.query(BoObservation).count() == observations_before


def test_advisory_content_lands_in_notes_not_in_bounds(db, run):
    context = ProposedBOContext(
        bo_run_id=run.id,
        soft_priors=["prefer the low end of the window"],
        process_window_hints=["purge under 4 s left unreacted precursor"],
    )
    proposal = propose_context(db, context)
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    result = apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()

    assert result["constraints_after"]["bounds"] == {}
    notes = result["constraints_after"]["notes"]
    assert len(notes) == 2
    assert all("advisory" in n for n in notes)
    #  Every note names the proposal and the person, so the campaign's notes are an
    #  attribution log rather than free text.
    assert all(f"proposal #{proposal.id}" in n and "Z. Woodel" in n for n in notes)


def test_a_card_edited_between_review_and_apply_marks_the_proposal_stale(db, run, card):
    """The window where an unnoticed edit does the most damage."""
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    db.commit()

    upsert_card(db, slug=card, title="ALD window for HfO2", body="Revised: 1.4 A/cycle.")
    db.commit()

    with pytest.raises(ContextRefused, match="no longer applicable"):
        apply_context(db, proposal.id, applied_by="Z. Woodel")
    assert proposal.status is ContextStatus.STALE
    assert run.constraints["bounds"] == {}


def test_a_card_unreviewed_between_review_and_apply_also_blocks(db, run, card):
    """A body-hash check alone would miss this: the text did not change."""
    from cnms_fom.db.enums import CardStatus
    from cnms_fom.db.models import KnowledgeCard

    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    db.commit()

    row = db.query(KnowledgeCard).filter(KnowledgeCard.slug == card).one()
    row.status = CardStatus.PROPOSED
    db.commit()

    with pytest.raises(ContextRefused, match="not reviewed"):
        apply_context(db, proposal.id, applied_by="Z. Woodel")


def test_two_proposals_intersect_rather_than_overwrite(db, run, card):
    """A replacement would let the second silently undo the first."""
    first = propose_context(db, _narrowing(run.id, card))
    review_context(db, first.id, reviewed_by="Z. Woodel")
    apply_context(db, first.id, applied_by="Z. Woodel")
    db.commit()

    tighter = ProposedBOContext(
        bo_run_id=run.id,
        recommended_bounds=[BoundProposal(
            parameter="substrate_temp_c", lower=220.0, upper=280.0,
            rationale="a second source narrows it further",
        )],
        supporting_card_slugs=[card],
    )
    second = propose_context(db, tighter)
    review_context(db, second.id, reviewed_by="Z. Woodel")
    result = apply_context(db, second.id, applied_by="Z. Woodel")
    db.commit()

    assert result["constraints_after"]["bounds"]["substrate_temp_c"] == [220.0, 280.0]


def test_excluding_a_categorical_choice_narrows_the_allowed_list(db, run, card):
    context = ProposedBOContext(
        bo_run_id=run.id,
        excluded_choices={"substrate": ["Ge"]},
        supporting_card_slugs=[card],
        rationale="no reported growth on Ge",
    )
    proposal = propose_context(db, context)
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    result = apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()

    allowed = result["constraints_after"]["allowed_choices"]["substrate"]
    assert "Ge" not in allowed
    assert set(allowed) == {"Si(100)", "SiO2"}


def test_a_proposal_that_would_make_the_campaign_unsatisfiable_is_refused(db, run, card):
    """Better to fail here than at the next suggest(), where it looks like a bug."""
    run.constraints = {"bounds": {"substrate_temp_c": [150.0, 180.0]},
                       "allowed_choices": {}, "notes": []}
    db.commit()

    proposal = propose_context(db, _narrowing(run.id, card))  # 200-300, disjoint from 150-180
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    db.commit()

    with pytest.raises(ContextRefused, match="unsatisfiable"):
        apply_context(db, proposal.id, applied_by="Z. Woodel")


def test_an_applied_proposal_cannot_be_re_reviewed(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()
    with pytest.raises(ContextRefused, match="already been applied"):
        review_context(db, proposal.id, reviewed_by="Someone Else")


# --- revert and log -------------------------------------------------------


def test_an_applied_proposal_can_be_reverted(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    before = snapshot(db, run.id).fingerprint()
    apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()

    result = revert_context(db, proposal.id, reverted_by="Z. Woodel")
    db.commit()
    assert result["campaign_fingerprint_after"] == before
    assert run.constraints["bounds"] == {}
    assert proposal.status is ContextStatus.SUPERSEDED
    assert "reverted by Z. Woodel" in proposal.review_note


def test_reverting_out_of_order_is_refused(db, run, card):
    """It would silently discard everything applied after it."""
    first = propose_context(db, _narrowing(run.id, card))
    review_context(db, first.id, reviewed_by="Z. Woodel")
    apply_context(db, first.id, applied_by="Z. Woodel")
    db.commit()

    second = propose_context(db, ProposedBOContext(
        bo_run_id=run.id, soft_priors=["a later note"],
    ))
    review_context(db, second.id, reviewed_by="Z. Woodel")
    apply_context(db, second.id, applied_by="Z. Woodel")
    db.commit()

    with pytest.raises(ContextRefused, match="since this one"):
        revert_context(db, first.id, reverted_by="Z. Woodel")


def test_the_proposal_log_is_the_campaign_change_history(db, run, card):
    proposal = propose_context(db, _narrowing(run.id, card))
    review_context(db, proposal.id, reviewed_by="Z. Woodel")
    apply_context(db, proposal.id, applied_by="Z. Woodel")
    db.commit()

    log = list_proposals(db, run.id)
    assert len(log) == 1
    entry = log[0]
    assert entry["status"] == "applied"
    assert entry["applied_by"] == "Z. Woodel"
    assert entry["changes_search_behaviour"] is True
    assert entry["campaign_fingerprint_before"] != entry["campaign_fingerprint_after"]
    assert entry["supporting_cards"][0]["slug"] == card


def test_the_database_refuses_an_applied_proposal_with_no_applier(db, run):
    """Belt and braces: the constraint holds even against raw SQL."""
    import sqlalchemy.exc

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        db.add(CampaignContextProposal(
            bo_run_id=run.id, status=ContextStatus.APPLIED,
            proposed_by="assistant", reviewed_by="Z. Woodel",
        ))
        db.flush()
    db.rollback()
