"""Bug 18: the lexical leg returned nothing for any natural-language question.

``websearch_to_tsquery`` ANDs its terms, so a 14-word question demands all fourteen
stems in one chunk. Measured in §10e: every real question returned 0 lexical hits while
dense returned 7-12, so dense alone was carrying the system.

The pure tokenisation and coverage functions are tested here on any backend. The
two-stage SQL behaviour is Postgres-only and lives in
``test_postgres_schema_and_isolation.py``-style skips at the bottom.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from cnms_fom.db import models  # noqa: F401 - registers the mappers
from cnms_fom.db.base import Base
from cnms_fom.db.enums import SynthesisTechnique
from cnms_fom.db.models import Document, DocumentChunk
from cnms_fom.rag_backend.hybrid import (
    lexical_search,
    lexical_terms,
    term_coverage,
)

# --- tokenisation: backend-independent ------------------------------------


def test_a_compound_question_keeps_only_its_content_terms():
    terms = lexical_terms(
        "What growth per cycle and film density are reported for HfO2 ALD, "
        "and do the sources agree?"
    )
    assert "HfO2" in terms and "ALD" in terms
    assert "growth" in terms and "density" in terms
    #  Scaffolding that would otherwise be part of the coverage denominator.
    for noise in ("what", "are", "reported", "sources", "agree", "and", "do", "for"):
        assert noise not in [t.lower() for t in terms], noise


def test_rare_exact_tokens_are_not_split():
    """`HfO2` must not become `hfo` + `2`: the rare token is the point of lexical search."""
    for token in ("HfO2", "TDMAH", "Nevot-Croce", "SrTiO3", "MgO(001)", "S12_a"):
        terms = lexical_terms(f"What is the value for {token} here?")
        assert any(token.lower().startswith(t.lower()[:4]) for t in terms), token
        assert token.split("(")[0].split("_")[0] in " ".join(terms), token


def test_a_stopword_only_question_yields_no_terms():
    assert lexical_terms("What is it?") == []
    assert lexical_terms("") == []


def test_single_characters_and_punctuation_are_dropped():
    assert lexical_terms("a b, c. -- /") == []


def test_terms_are_deduplicated_case_insensitively():
    terms = lexical_terms("HfO2 and hfo2 and HFO2 growth growth")
    assert len([t for t in terms if t.lower() == "hfo2"]) == 1
    assert len([t for t in terms if t.lower() == "growth"]) == 1


# --- coverage -------------------------------------------------------------


def test_coverage_is_the_fraction_of_query_terms_present():
    terms = ["HfO2", "growth", "density"]
    assert term_coverage("HfO2 growth and density here", terms) == pytest.approx(1.0)
    assert term_coverage("HfO2 growth only", terms) == pytest.approx(2 / 3)
    assert term_coverage("nothing relevant", terms) == pytest.approx(0.0)


def test_coverage_prefers_the_passage_sharing_more_terms():
    """The property that stops OR from ranking a one-generic-word match first."""
    terms = lexical_terms("What growth per cycle is reported for HfO2 ALD?")
    specific = "HfO2 ALD growth per cycle was 0.98 angstrom per cycle"
    generic = "The cycle was repeated for each sample in the study"
    assert term_coverage(specific, terms) > term_coverage(generic, terms)


def test_coverage_with_no_terms_is_zero_not_a_crash():
    assert term_coverage("anything", []) == 0.0


# --- the two-stage SQL behaviour: Postgres only ---------------------------

POSTGRES_URL = os.environ.get("CNMS_TEST_POSTGRES_URL")

postgres_only = pytest.mark.skipif(
    not POSTGRES_URL,
    reason=(
        "Set CNMS_TEST_POSTGRES_URL to exercise the two-stage lexical query. Skipped "
        "means UNVERIFIED: bug 18 was invisible to the SQLite fallback, which already "
        "scores by term overlap."
    ),
)


@pytest.fixture
def pg_session():
    import uuid

    name = f"cnms_lex_{uuid.uuid4().hex[:12]}"
    admin_url = str(make_url(POSTGRES_URL).set(database="postgres"))
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = str(make_url(POSTGRES_URL).set(database=name))
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    try:
        document = Document(
            title="ALD of HfO2 on Si(100) from TDMAH and water",
            filename="hfo2.pdf",
            content_sha256="c" * 64,
            technique=SynthesisTechnique.ALD,
        )
        session.add(document)
        session.flush()
        session.add_all([
            DocumentChunk(
                document_id=document.id, chunk_index=0, page=1,
                text="The growth per cycle was constant at 0.98 angstrom per cycle "
                     "and the film density from XRR was 8.7 g/cm3.",
            ),
            DocumentChunk(
                document_id=document.id, chunk_index=1, page=2,
                text="Each cycle was repeated for every sample in the study.",
            ),
        ])
        #  create_all makes no triggers, so populate the title copy by hand.
        session.flush()
        session.execute(
            text("UPDATE document_chunks SET search_title = :t"), {"t": document.title}
        )
        session.commit()
        yield session
    finally:
        session.close()
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": name},
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


LONG_QUESTION = (
    "What growth per cycle and film density are reported for HfO2 ALD, "
    "and do the sources agree?"
)


@postgres_only
def test_a_long_question_finds_nothing_precisely_and_something_relaxed(pg_session):
    """Bug 18 in one assertion."""
    assert lexical_search(pg_session, LONG_QUESTION, k=10) == []
    relaxed = lexical_search(pg_session, LONG_QUESTION, k=10, relaxed=True)
    assert relaxed, "the relaxed stage must recover the leg"
    assert "0.98" in relaxed[0].text


@postgres_only
def test_relaxed_ranks_the_specific_passage_above_the_generic_one(pg_session):
    hits = lexical_search(pg_session, LONG_QUESTION, k=10, relaxed=True)
    texts = [h.text for h in hits]
    assert "0.98" in texts[0], (
        "term coverage should put the passage sharing HfO2/growth/density first, "
        f"got {texts}"
    )


@postgres_only
def test_a_short_precise_question_is_not_regressed(pg_session):
    """Stage 1 must be untouched: the relaxed stage only runs when it finds nothing."""
    precise = lexical_search(pg_session, "growth per cycle", k=10)
    assert precise, "the short question must still match precisely"
    with_relaxed = lexical_search(pg_session, "growth per cycle", k=10, relaxed=True)
    assert [h.chunk_id for h in with_relaxed] == [h.chunk_id for h in precise]


@postgres_only
def test_a_rare_token_query_still_matches_exactly(pg_session):
    assert lexical_search(pg_session, "TDMAH", k=10, relaxed=True)
    assert lexical_search(pg_session, "HfO2", k=10, relaxed=True)


@postgres_only
def test_a_stopword_only_question_returns_nothing_rather_than_everything(pg_session):
    """No terms must mean no relaxed match, not an unfiltered table scan."""
    assert lexical_search(pg_session, "What is it?", k=10, relaxed=True) == []


@postgres_only
def test_a_genuinely_absent_term_returns_nothing(pg_session):
    assert lexical_search(pg_session, "ruthenium tetroxide plasma", k=10, relaxed=True) == []


@postgres_only
def test_a_technique_filter_still_applies_in_the_relaxed_stage(pg_session):
    assert lexical_search(
        pg_session, LONG_QUESTION, k=10, relaxed=True,
        techniques=[SynthesisTechnique.ALD],
    )
    assert lexical_search(
        pg_session, LONG_QUESTION, k=10, relaxed=True,
        techniques=[SynthesisTechnique.PLD],
    ) == []


@postgres_only
def test_the_title_contributes_to_relaxed_matching(pg_session):
    """Migration 0007's whole point, through the relaxed path.

    "Si(100)" appears only in the document title, never in a chunk's text.
    """
    hits = lexical_search(pg_session, "Si(100) substrate deposition", k=10, relaxed=True)
    assert hits, "a term present only in the title should still match"


@postgres_only
def test_a_query_containing_tsquery_syntax_is_treated_as_data(pg_session):
    """`&`, `!` and a stray quote are a user's punctuation, not an expression."""
    for hostile in ("HfO2 & !density", "growth ' per cycle", "HfO2 <-> ALD", "a | b & c"):
        hits = lexical_search(pg_session, hostile, k=10, relaxed=True)
        assert isinstance(hits, list)
