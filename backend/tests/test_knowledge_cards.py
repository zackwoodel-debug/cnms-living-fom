"""Knowledge cards: the review gate, typed edges, and what they refuse.

A card is where a language model's synthesis of the corpus gets written down, so
almost everything worth testing here is a refusal — the states that would let
model output pass as checked evidence.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import CardRelation, CardStatus, CardType
from cnms_fom.db.models import KnowledgeCard  # noqa: F401 - registers mappers
from cnms_fom.knowledge.cards import (
    CardError,
    ReviewRefused,
    card_graph,
    card_stats,
    link_cards,
    read_card,
    review_card,
    search_cards,
    upsert_card,
)

DOC_SOURCE = [{"kind": "document", "document_id": 1, "page": 7, "doi": "10.0000/ald"}]


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'cards.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


def _card(db, slug="concepts/ald-window-hfo2", **kwargs):
    defaults = {
        "title": "ALD window for HfO2",
        "body": "GPC saturates at 0.98 A/cycle between 200 and 300 C [1].",
        "card_type": CardType.CONCEPT,
        "sources": DOC_SOURCE,
    }
    return upsert_card(db, slug=slug, **{**defaults, **kwargs})


# --- the review gate -------------------------------------------------------


def test_a_new_card_is_proposed_and_not_citable(db):
    """An assistant's synthesis is not evidence until a person checks it."""
    card = _card(db)
    db.commit()
    assert card.status is CardStatus.PROPOSED
    assert card.citable is False
    assert card.authored_by == "assistant"


def test_review_makes_a_sourced_card_citable(db):
    _card(db)
    card = review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()
    assert card.status is CardStatus.REVIEWED
    assert card.citable is True
    assert card.reviewed_by == "Z. Woodel"
    assert card.reviewed_body_sha256


def test_review_without_a_named_reviewer_is_refused(db):
    _card(db)
    with pytest.raises(ReviewRefused, match="named reviewer"):
        review_card(db, "concepts/ald-window-hfo2", reviewed_by="   ")


def test_review_of_an_unsourced_card_is_refused(db):
    """A factual claim with no source is an assertion, not knowledge."""
    _card(db, slug="concepts/unsourced", sources=None)
    with pytest.raises(ReviewRefused, match="no resolved source"):
        review_card(db, "concepts/unsourced", reviewed_by="Z. Woodel")


def test_a_free_text_locator_alone_cannot_satisfy_review(db):
    """Sec. 2.2 wants provenance a reader can actually follow."""
    _card(db, slug="concepts/freetext-only", sources=["Kim 2024, Table 2"])
    with pytest.raises(ReviewRefused, match="free-text"):
        review_card(db, "concepts/freetext-only", reviewed_by="Z. Woodel")


def test_editing_after_review_makes_the_review_stale(db):
    """Otherwise a card could be approved, then rewritten, and still read as checked."""
    _card(db)
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()

    card = _card(db, body="Actually GPC saturates at 1.4 A/cycle.")
    db.commit()
    assert card.status is CardStatus.REVIEWED  # status untouched
    assert card.review_is_stale is True
    assert card.citable is False
    assert "edited since" in read_card(db, card.slug)["caution"]


def test_an_update_unions_sources_rather_than_replacing_them(db):
    """A second source for a claim is more evidence, not a replacement."""
    _card(db)
    card = _card(db, sources=[{"kind": "fit_record", "fit_record_id": 3}])
    db.commit()
    kinds = {s["kind"] for s in card.sources}
    assert kinds == {"document", "fit_record"}


# --- slugs and validation --------------------------------------------------


def test_a_bad_slug_is_refused_rather_than_normalised(db):
    """The slug is the citation handle; rewriting one breaks every reference."""
    for bad in ("Concepts/ALD Window", "concepts//x", "has spaces", ""):
        with pytest.raises(CardError, match="usable slug"):
            _card(db, slug=bad)


def test_an_empty_body_is_refused(db):
    with pytest.raises(CardError, match="needs a body"):
        _card(db, slug="concepts/empty", body="   ")


def test_confidence_outside_the_unit_interval_is_refused(db):
    with pytest.raises(CardError, match=r"\[0, 1\]"):
        _card(db, slug="concepts/overconfident", confidence=1.4)


# --- typed edges -----------------------------------------------------------


def test_typed_links_are_readable_from_both_ends(db):
    """The incoming edges are often the useful ones."""
    _card(db)
    _card(db, slug="sources/kim-2024", title="Kim 2024", card_type=CardType.SOURCE)
    link_cards(db, "concepts/ald-window-hfo2", "sources/kim-2024", relation=CardRelation.FED_BY)
    db.commit()

    concept = read_card(db, "concepts/ald-window-hfo2")
    assert concept["links_out"][0] == {
        "relation": "fed_by",
        "slug": "sources/kim-2024",
        "title": "Kim 2024",
        "status": "proposed",
        "note": None,
    }
    source = read_card(db, "sources/kim-2024")
    assert source["links_in"][0]["slug"] == "concepts/ald-window-hfo2"


def test_a_contradicts_link_requires_a_note(db):
    """Recording that two cards disagree while dropping what they disagree about
    leaves nobody able to resolve it."""
    _card(db)
    _card(db, slug="concepts/ald-window-hfo2-alt", title="Alternative window")

    with pytest.raises(CardError, match="needs a note"):
        link_cards(
            db,
            "concepts/ald-window-hfo2",
            "concepts/ald-window-hfo2-alt",
            relation=CardRelation.CONTRADICTS,
        )

    link_cards(
        db,
        "concepts/ald-window-hfo2",
        "concepts/ald-window-hfo2-alt",
        relation=CardRelation.CONTRADICTS,
        note="0.98 vs 1.4 A/cycle over the same 200-300 C range.",
    )
    db.commit()


def test_contradictions_are_surfaced_on_both_cards(db):
    _card(db)
    _card(db, slug="concepts/other", title="Other")
    link_cards(
        db, "concepts/ald-window-hfo2", "concepts/other",
        relation=CardRelation.CONTRADICTS, note="GPC disagrees.",
    )
    db.commit()

    assert read_card(db, "concepts/ald-window-hfo2")["unresolved_contradictions"]
    assert read_card(db, "concepts/other")["unresolved_contradictions"]


def test_linking_is_idempotent_and_self_links_are_refused(db):
    _card(db)
    _card(db, slug="concepts/other", title="Other")
    first = link_cards(db, "concepts/ald-window-hfo2", "concepts/other", relation="relates_to")
    again = link_cards(db, "concepts/ald-window-hfo2", "concepts/other", relation="relates_to")
    db.commit()
    assert first.id == again.id

    with pytest.raises(CardError, match="cannot link to itself"):
        link_cards(db, "concepts/other", "concepts/other", relation="relates_to")


def test_linking_an_unknown_card_raises_lookup_error(db):
    _card(db)
    with pytest.raises(LookupError):
        link_cards(db, "concepts/ald-window-hfo2", "concepts/nope", relation="relates_to")


# --- superseding -----------------------------------------------------------


def test_superseding_marks_the_old_card_and_keeps_it(db):
    """The audit trail needs the old claim, not just the new one."""
    _card(db, slug="concepts/v1", title="First take")
    new = _card(db, slug="concepts/v2", title="Second take", supersedes_slug="concepts/v1")
    db.commit()

    old = read_card(db, "concepts/v1")
    assert old["status"] == "superseded"
    assert "superseded" in old["caution"]
    assert new.supersedes_id is not None


# --- search, graph, stats --------------------------------------------------


def test_search_matches_body_and_filters_by_citability(db):
    _card(db)
    _card(db, slug="concepts/pld-oxygen", title="PLD oxygen pressure",
          body="Deposition at 100 mTorr O2.")
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()

    assert search_cards(db, "0.98")["n_cards"] == 1
    assert search_cards(db, "mTorr")["n_cards"] == 1
    assert search_cards(db)["n_cards"] == 2

    citable = search_cards(db, citable_only=True)
    assert citable["n_cards"] == 1
    assert citable["cards"][0]["slug"] == "concepts/ald-window-hfo2"


def test_search_results_omit_the_body_but_keep_the_caution(db):
    _card(db)
    db.commit()
    card = search_cards(db)["cards"][0]
    assert "body" not in card
    assert "not been reviewed" in card["caution"]


def test_search_filters_by_type_and_tags(db):
    _card(db, tags=["ald", "hfo2"])
    _card(db, slug="sources/kim-2024", title="Kim 2024", card_type=CardType.SOURCE,
          tags=["ald"])
    db.commit()

    assert search_cards(db, card_type="source")["n_cards"] == 1
    assert search_cards(db, tags=["hfo2"])["n_cards"] == 1
    assert search_cards(db, tags=["ald"])["n_cards"] == 2


def test_graph_returns_typed_edges_and_names_orphans(db):
    _card(db)
    _card(db, slug="sources/kim-2024", title="Kim 2024", card_type=CardType.SOURCE)
    _card(db, slug="concepts/lonely", title="Unconnected note")
    link_cards(db, "concepts/ald-window-hfo2", "sources/kim-2024", relation="fed_by")
    db.commit()

    whole = card_graph(db)
    assert whole["n_nodes"] == 3
    assert whole["n_edges"] == 1
    assert whole["edges"][0]["relation"] == "fed_by"
    #  An orphan is knowledge that failed to connect to anything.
    assert whole["orphans"] == ["concepts/lonely"]

    rooted = card_graph(db, "concepts/ald-window-hfo2", depth=1)
    assert {n["slug"] for n in rooted["nodes"]} == {
        "concepts/ald-window-hfo2", "sources/kim-2024"
    }


def test_graph_of_an_unknown_root_reports_it(db):
    assert "error" in card_graph(db, "concepts/nope")


def test_stats_report_the_review_backlog_and_contradictions(db):
    _card(db)
    _card(db, slug="concepts/other", title="Other")
    link_cards(db, "concepts/ald-window-hfo2", "concepts/other",
               relation="contradicts", note="GPC disagrees.")
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()

    stats = card_stats(db)
    assert stats["total_cards"] == 2
    assert stats["citable"] == 1
    assert stats["awaiting_review"] == 1
    assert stats["unresolved_contradictions"] == 1
    assert stats["by_status"] == {"proposed": 1, "reviewed": 1}
    assert "knowledge or" in stats["note"]


def test_stale_reviews_are_counted(db):
    _card(db)
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()
    _card(db, body="Revised claim.")
    db.commit()
    assert card_stats(db)["stale_reviews"] == 1


def test_reading_an_unknown_card_reports_not_found(db):
    result = read_card(db, "concepts/nope")
    assert result["found"] is False
