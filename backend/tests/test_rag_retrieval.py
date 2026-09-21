"""Hybrid retrieval, rank fusion, and the corrective loop.

These run without Ollama on purpose. The dense leg is unavailable here, which
exercises the degradation path — lexical-only retrieval still finds the exact
tokens a synthesis corpus is mostly made of — and the grading and rewrite passes
run against a scripted provider so the interesting branches are reachable.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import SynthesisTechnique
from cnms_fom.db.models import Document, DocumentChunk
from cnms_fom.rag_backend.grading import (
    MIN_USEFUL_GRADE,
    grade_and_rerank,
    retrieve_with_correction,
    rewrite_query,
)
from cnms_fom.rag_backend.hybrid import (
    ChunkHit,
    hybrid_search,
    lexical_search,
    reciprocal_rank_fusion,
    retrieval_diagnostics,
    tokenise,
)
from tests.fakes import GradingProvider

PASSAGES = [
    (
        "ALD of HfO2 on Si",
        SynthesisTechnique.ALD,
        [
            "Films were grown by atomic layer deposition using tetrakis(dimethylamido)hafnium "
            "and water at a substrate temperature of 250 C in a Beneq TFS-200.",
            "The growth per cycle saturated at 0.98 A/cycle between 200 and 300 C, defining the "
            "ALD window for this precursor combination.",
        ],
    ),
    (
        "PLD of SrTiO3",
        SynthesisTechnique.PLD,
        [
            "Pulsed laser deposition was carried out at 700 C in 100 mTorr of oxygen with a "
            "KrF excimer laser at 2 J/cm2.",
        ],
    ),
    (
        "XRR fitting practice",
        SynthesisTechnique.CNMS_USER_DOC,
        [
            "Interfacial width is handled with a Nevot-Croce roughness factor applied to each "
            "Fresnel coefficient; it is not a graded layer and must not be read as one.",
        ],
    ),
]


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'rag.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()

    for title, technique, chunks in PASSAGES:
        document = Document(
            title=title,
            filename=f"{title}.pdf",
            content_sha256=f"hash-{title}",
            technique=technique,
            doi=f"10.0000/{title.replace(' ', '-')}",
            n_pages=len(chunks),
        )
        session.add(document)
        session.flush()
        for index, text in enumerate(chunks):
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    page=index + 1,
                    text=text,
                    n_tokens=len(text.split()),
                )
            )
    session.commit()
    yield session
    session.close()
    engine.dispose()


# --- tokenisation ----------------------------------------------------------


def test_chemical_formulas_survive_tokenisation():
    """Splitting HfO2 into 'hf' and '2' is what makes a formula unfindable."""
    assert "hfo2" in tokenise("What is the ALD window for HfO2?")
    assert "ald" in tokenise("What is the ALD window for HfO2?")
    #  Stopwords out, but the list stays short so rare tokens survive.
    assert "the" not in tokenise("the growth of the film")


# --- lexical retrieval -----------------------------------------------------


def test_lexical_search_finds_a_rare_exact_token(db):
    hits = lexical_search(db, "Nevot-Croce roughness factor")
    assert hits
    assert "Nevot-Croce" in hits[0].text


def test_lexical_search_respects_the_technique_filter(db):
    hits = lexical_search(db, "excimer laser oxygen", techniques=[SynthesisTechnique.PLD])
    assert hits
    assert all(hit.technique == "pld" for hit in hits)

    #  The same query scoped to a partition that does not hold it returns nothing,
    #  rather than the next-best thing from elsewhere.
    assert lexical_search(db, "excimer laser oxygen", techniques=[SynthesisTechnique.ALD]) == []


def test_lexical_search_returns_nothing_for_an_absent_term(db):
    assert lexical_search(db, "molecular beam epitaxy of gallium arsenide") == []


# --- fusion ----------------------------------------------------------------


def _hit(chunk_id: int, similarity: float = 0.5) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id=1,
        document_title="doc",
        technique="ald",
        page=1,
        text=f"chunk {chunk_id}",
        similarity=similarity,
    )


def test_fusion_promotes_what_both_retrievers_found():
    """The property that makes fusion worth doing over either list alone."""
    fused = reciprocal_rank_fusion(
        {
            "vector": [_hit(1), _hit(2), _hit(3)],
            "lexical": [_hit(3), _hit(4), _hit(1)],
        }
    )
    #  Chunk 3 is rank 3 + rank 1; chunk 1 is rank 1 + rank 3. Equal RRF, both
    #  found twice, so they lead — ahead of 2 and 4, which each appear once.
    top_two = {f.hit.chunk_id for f in fused[:2]}
    assert top_two == {1, 3}
    assert all(f.retriever_count == 2 for f in fused[:2])
    assert {f.hit.chunk_id for f in fused[2:]} == {2, 4}


def test_fusion_records_which_retriever_found_each_hit():
    fused = reciprocal_rank_fusion({"vector": [_hit(1)], "lexical": [_hit(2)]})
    by_id = {f.hit.chunk_id: f for f in fused}
    assert by_id[1].vector_rank == 1 and by_id[1].lexical_rank is None
    assert by_id[2].lexical_rank == 1 and by_id[2].vector_rank is None
    assert by_id[2].as_dict()["found_by"] == ["lexical"]


def test_fusion_ordering_is_deterministic():
    """An audit trail that reorders itself between runs is not one."""
    lists = {"vector": [_hit(i) for i in (5, 6, 7)], "lexical": [_hit(i) for i in (7, 8, 5)]}
    first = [f.hit.chunk_id for f in reciprocal_rank_fusion(lists)]
    second = [f.hit.chunk_id for f in reciprocal_rank_fusion(lists)]
    assert first == second


# --- degradation -----------------------------------------------------------


def test_hybrid_search_degrades_to_lexical_when_dense_is_unavailable(db, no_dense_retrieval):
    """Half a retriever beats none; the failure is logged, not fatal."""
    hits = hybrid_search(db, "Nevot-Croce roughness", k=3)
    assert hits
    assert all(hit.lexical_rank is not None for hit in hits)
    assert all(hit.vector_rank is None for hit in hits)


def test_hybrid_search_raises_when_nothing_retrieved_at_all(db, no_dense_retrieval):
    """With no lexical match and no dense leg, the caller needs the real error."""
    with pytest.raises(ImportError, match="rag"):
        hybrid_search(db, "molecular beam epitaxy of gallium arsenide")


def test_diagnostics_show_each_retriever_separately(db, no_dense_retrieval):
    report = retrieval_diagnostics(db, "ALD window HfO2")
    assert report["lexical_backend"] == "python_term_overlap"
    assert report["dense_error"]  # the fixture makes the embedder unreachable
    assert report["lexical"]
    assert report["fused"]
    assert "hfo2" in report["terms"]


# --- grading ---------------------------------------------------------------


def test_grading_reorders_by_grade_then_by_fusion_score(db):
    candidates = hybrid_search(db, "ALD window HfO2 growth per cycle", k=4)
    assert len(candidates) >= 2

    graded, cost = grade_and_rerank(GradingProvider(grade=3), "ALD window?", candidates)
    assert [g.grade for g in graded] == [3] * len(candidates)
    assert all(g.useful for g in graded)
    #  Without a db there is no cache, so every candidate cost a call.
    assert cost == {
        "cached": 0, "called": len(candidates), "failed": 0,
        #  Zero unless the per-conjunct policy is on, which it is not here.
        "conjunct_rescued": 0,
    }


def test_an_unparseable_grade_fails_closed(db):
    """Grade 1 keeps the passage out of the answer but visible in the trail."""
    candidates = hybrid_search(db, "Nevot-Croce", k=2)
    graded, _ = grade_and_rerank(
        GradingProvider(malformed_grades=True), "what is it?", candidates
    )
    assert all(g.grade == 1 for g in graded)
    assert all(not g.useful for g in graded)
    assert all("not parseable" in g.reason for g in graded)
    assert MIN_USEFUL_GRADE == 2


def test_rewrite_discards_a_non_rewrite():
    assert rewrite_query(GradingProvider(rewrite=None), "ALD window?") is None


def test_rewrite_discards_an_over_long_reply():
    """A 'rewrite' many times the original is the model answering, not rephrasing."""
    provider = GradingProvider(rewrite="x " * 2000)
    assert rewrite_query(provider, "ALD window for HfO2?") is None


# --- the corrective loop ---------------------------------------------------


def test_sufficient_retrieval_returns_on_the_first_attempt(db):
    outcome = retrieve_with_correction(
        db, "Nevot-Croce roughness factor", provider=GradingProvider(grade=3), k=3
    )
    assert outcome.sufficient is True
    assert outcome.rewritten is False
    assert len(outcome.attempts) == 1


def test_an_insufficient_first_attempt_retries_with_a_rewritten_query(db):
    """The common failure is vocabulary: the user's words are not the paper's."""
    provider = GradingProvider(grade=3, rewrite="Nevot-Croce roughness factor Fresnel")
    outcome = retrieve_with_correction(
        db,
        "beam divergence smearing",  # no lexical overlap with any indexed chunk
        provider=provider,
        k=3,
        use_vector=False,
    )
    assert "rewrite" in provider.calls
    assert outcome.effective_query == "Nevot-Croce roughness factor Fresnel"
    assert outcome.rewritten is True
    assert outcome.sufficient is True


def test_everything_graded_irrelevant_is_insufficient(db):
    outcome = retrieve_with_correction(
        db, "Nevot-Croce roughness", provider=GradingProvider(grade=0, rewrite=None), k=3
    )
    assert outcome.sufficient is False
    #  The graded-but-rejected hits are kept: "searched and came up short" is a
    #  more useful refusal than an empty one.
    assert outcome.hits
    assert outcome.useful == []


def test_ungraded_retrieval_is_marked_as_such(db):
    outcome = retrieve_with_correction(
        db, "Nevot-Croce roughness", provider=GradingProvider(), grade=False, k=3
    )
    assert outcome.graded is False
    assert outcome.sufficient is True
    assert all(hit.reason == "not graded" for hit in outcome.hits)


def test_a_term_the_corpus_lacks_is_an_insufficient_outcome_not_an_error(db):
    """Lexical-only, asked for explicitly: an empty result is a real finding."""
    outcome = retrieve_with_correction(
        db,
        "gallium arsenide molecular beam epitaxy",
        provider=GradingProvider(grade=3, rewrite=None),
        k=3,
        use_vector=False,
    )
    assert outcome.sufficient is False
    assert outcome.hits == []
    assert outcome.attempts[0]["n_candidates"] == 0


def test_a_failed_search_is_an_error_not_a_data_gap(db, no_dense_retrieval):
    """Reporting "the corpus lacks this" about a search that could not run is worse
    than failing: it is a wrong scientific conclusion dressed as a clean answer."""
    with pytest.raises(ImportError, match="rag"):
        retrieve_with_correction(
            db,
            "gallium arsenide molecular beam epitaxy",
            provider=GradingProvider(grade=3, rewrite=None),
            k=3,
        )


# --- the single-shot path --------------------------------------------------


def test_answer_question_returns_the_answer_with_its_retrieval_trace(db):
    from cnms_fom.rag_backend.chains import answer_question

    result = answer_question(
        db,
        "Nevot-Croce roughness factor",
        provider=GradingProvider(grade=3, answer="It is applied per Fresnel coefficient [1]."),
        k=3,
    )
    assert result.insufficient_context is False
    assert result.sources
    assert result.provider == "grading-stub"
    #  The search that produced the answer travels with it: without this, "why
    #  did it not find X?" is unanswerable from the response alone.
    assert result.retrieval["attempts"]
    assert result.retrieval["candidates"][0]["grade"] == 3
    assert "lexical" in result.retrieval["candidates"][0]["found_by"]


def test_answer_question_refuses_when_nothing_grades_useful(db):
    """Retrieved-but-irrelevant gets a different message from nothing-retrieved."""
    from cnms_fom.rag_backend.chains import answer_question

    result = answer_question(
        db, "Nevot-Croce roughness", provider=GradingProvider(grade=0, rewrite=None), k=3
    )
    assert result.insufficient_context is True
    assert "[DATA GAP" in result.answer
    assert "adjacent material, not the answer" in result.answer


def test_answer_question_surfaces_a_provider_refusal_as_such(db):
    """A refusal is not a data gap: the corpus is fine and the passages still stand."""
    from cnms_fom.rag_backend.chains import answer_question
    from cnms_fom.rag_backend.providers import ChatResult

    class RefusingGrader(GradingProvider):
        def send(self, system, messages, *, tools=None, temperature: float = 0.0):
            body = " ".join(str(m.get("content", "")) for m in messages)
            if "Grade this passage" in body:
                return super().send(system, messages, tools=tools, temperature=temperature)
            return ChatResult(text="", refused=True, refusal_category="cyber", model=self.model)

    result = answer_question(db, "Nevot-Croce roughness", provider=RefusingGrader(grade=3), k=3)
    assert result.refused_by_provider is True
    assert "declined to answer" in result.answer
    #  The evidence is returned regardless, so the question is still answerable
    #  by a human reading the passages.
    assert result.sources


def test_the_prompts_survive_formatting_with_retrieved_text(db):
    """Document text containing braces must not break prompt interpolation."""
    from cnms_fom.rag_backend.chains import (
        SYNTHESIS_SYSTEM_PROMPT,
        USER_PROMPT,
        format_context,
    )

    hits = [fused.hit for fused in hybrid_search(db, "Nevot-Croce roughness", k=1)]
    rendered = SYNTHESIS_SYSTEM_PROMPT.format(context=format_context(hits))
    assert "Nevot-Croce" in rendered
    assert "{context}" not in rendered
    assert USER_PROMPT.format(question="what {is} this?") .endswith("[n] citations.")



# --- pgvector is a property of the database, not of the config -------------
#
# Caught by CI's Postgres job, which installs the `vector` extra and sets
# PGVECTOR_ENABLED=true while this test file still uses SQLite. `cosine_distance`
# compiles to the Postgres-only `<=>` operator, so a reasonable deployment setting
# emitted it at SQLite and failed with "near '>': syntax error" — surfacing to the
# assistant as `search_corpus failed`, a config flag masquerading as a statement
# about the corpus.


def test_pgvector_is_not_used_against_a_sqlite_session(db, monkeypatch):
    from cnms_fom.rag_backend import vectorstore

    monkeypatch.setattr(
        vectorstore, "get_settings",
        lambda: type("S", (), {"pgvector_enabled": True, "embedding_dim": 768})(),
    )
    pytest.importorskip("pgvector.sqlalchemy")

    #  Configured, yes; usable against this session, no.
    assert vectorstore._pgvector_available() is True
    assert vectorstore._pgvector_available(db) is False


def test_a_vector_search_on_sqlite_returns_results_rather_than_raising(
    db, monkeypatch
):
    from cnms_fom.rag_backend import vectorstore

    monkeypatch.setattr(
        vectorstore, "get_settings",
        lambda: type("S", (), {"pgvector_enabled": True, "embedding_dim": 768})(),
    )
    #  Must not raise: the portable path handles it.
    hits = vectorstore.search_chunks(db, [0.0] * 768, k=3)
    assert isinstance(hits, list)
