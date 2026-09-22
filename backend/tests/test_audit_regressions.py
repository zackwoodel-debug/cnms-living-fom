"""Regressions for the second audit pass.

Five defects, found by reading for the shape of the ones already fixed rather than by
reading for mistakes: a component that is optional by design taking something
non-optional down with it, and a report that describes intent rather than behaviour.

Each test is named for the consequence, not the mechanism, because the mechanism is
the part that will be refactored.
"""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import SynthesisTechnique
from cnms_fom.db.models import Document, DocumentChunk, LlmCacheEntry


@pytest.fixture
def db_session(tmp_path):
    #  A file-backed SQLite database rather than :memory:, so SAVEPOINT behaviour is
    #  the same as the one these tests are about. Matches test_llm_cache.py.
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'audit.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()

# ---------------------------------------------------------------------------
# Bug 23: an optional cache write destroyed the caller's transaction
# ---------------------------------------------------------------------------


def _document(sha: str, title: str = "real work") -> Document:
    return Document(
        title=title, filename=f"{sha}.pdf", content_sha256=sha,
        technique=SynthesisTechnique.ALD,
    )


def test_a_failed_cache_write_does_not_discard_the_callers_work(db_session, monkeypatch):
    """The cache is infrastructure. It may miss; it may not delete data.

    ``get`` and ``put`` promised never to raise and kept that promise with
    ``db.rollback()`` — the *caller's* transaction. They are called from the
    extraction and grading loops, where the session holds a half-built brief and its
    claims, so one failed cache write silently discarded real work and the following
    commit wrote nothing. No error surfaced: the failure had already been logged as a
    cache miss.

    The realistic trigger is not exotic. ``llm_max_parallel`` is 4 and cases are
    designed to share passages, so two workers racing to insert the same key is the
    expected case, not the unlucky one.
    """
    from cnms_fom.rag_backend import cache

    monkeypatch.setattr(cache, "enabled", lambda: True)
    key = hashlib.sha256(b"a shared passage").hexdigest()

    #  Another worker already committed this key.
    cache.put(db_session, "extraction", key, {"first": True}, model="m")
    db_session.commit()

    db_session.add(_document("sha-real"))
    db_session.flush()

    #  Force the write to fail the way a race does, inside the cache's own statement.
    original = cache._match

    def exploding(*args, **kwargs):
        raise RuntimeError("simulated unique-constraint race")

    monkeypatch.setattr(cache, "_match", exploding)
    cache.put(db_session, "extraction", key, {"second": True}, model="m")
    monkeypatch.setattr(cache, "_match", original)

    assert db_session.query(Document).count() == 1, (
        "The caller's pending Document was rolled back by a failed cache write. "
        "A cache that can delete the request's work is worse than no cache."
    )
    db_session.commit()
    assert db_session.query(Document).count() == 1


def test_a_failed_cache_lookup_does_not_discard_the_callers_work(db_session, monkeypatch):
    """Same for ``get``, which writes too — it bumps ``hit_count``."""
    from cnms_fom.rag_backend import cache

    monkeypatch.setattr(cache, "enabled", lambda: True)
    db_session.add(_document("sha-lookup"))
    db_session.flush()

    def exploding(*args, **kwargs):
        raise RuntimeError("simulated failure inside the lookup")

    monkeypatch.setattr(cache, "_match", exploding)
    assert cache.get(db_session, "extraction", "k" * 64, model="m") is None
    assert db_session.query(Document).count() == 1


def test_the_cache_still_works_after_a_failure(db_session, monkeypatch):
    """Containment must not come at the cost of the cache being unusable afterwards.

    The savepoint has to be released, not left open: a dangling one would make every
    later cache call fail, turning a contained failure into a permanent outage.
    """
    from cnms_fom.rag_backend import cache

    monkeypatch.setattr(cache, "enabled", lambda: True)
    original = cache._match

    def exploding(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cache, "_match", exploding)
    cache.put(db_session, "extraction", "a" * 64, {"x": 1}, model="m")
    monkeypatch.setattr(cache, "_match", original)

    cache.put(db_session, "extraction", "b" * 64, {"y": 2}, model="m")
    assert cache.get(db_session, "extraction", "b" * 64, model="m") == {"y": 2}


def test_a_cache_hit_counts_in_sql_rather_than_read_modify_write(db_session, monkeypatch):
    """Two workers sharing a passage must not each overwrite the other's increment."""
    from cnms_fom.rag_backend import cache

    monkeypatch.setattr(cache, "enabled", lambda: True)
    key = "c" * 64
    cache.put(db_session, "extraction", key, {"x": 1}, model="m")
    for _ in range(3):
        cache.get(db_session, "extraction", key, model="m")
    db_session.commit()
    row = db_session.query(LlmCacheEntry).filter(LlmCacheEntry.cache_key == key).one()
    assert row.hit_count == 3, f"hit_count is {row.hit_count}, expected 3"


# ---------------------------------------------------------------------------
# Bug 24: embeddings from two models were compared as if commensurable
# ---------------------------------------------------------------------------


def test_chunks_from_another_embedding_model_are_not_ranked_against_this_query(
    db_session, monkeypatch
):
    """A cosine similarity across two embedding models is not weak evidence.

    It is a number with no meaning: the two models put their coordinates in
    unrelated bases. ``nomic-embed-text`` is 768-dimensional and so are plenty of
    others, so the width check in ``_rank_in_python`` cannot notice, and
    ``embedding_model`` was recorded on every chunk and read by nothing.
    """
    from cnms_fom.config import get_settings
    from cnms_fom.rag_backend.vectorstore import search_chunks

    current = get_settings().ollama_embed_model
    document = _document("sha-models", title="two models")
    db_session.add(document)
    db_session.flush()

    #  Identical vectors, so anything that reaches the ranker scores identically and
    #  only the model label can separate them.
    vector = [1.0] + [0.0] * 767
    db_session.add_all([
        DocumentChunk(
            document_id=document.id, chunk_index=0, text="from the current model",
            embedding_model=current, embedding=vector,
        ),
        DocumentChunk(
            document_id=document.id, chunk_index=1, text="from some other model",
            embedding_model="some-other-model", embedding=vector,
        ),
        DocumentChunk(
            document_id=document.id, chunk_index=2, text="unlabelled, predates the column",
            embedding_model=None, embedding=vector,
        ),
    ])
    db_session.flush()

    texts = {hit.text for hit in search_chunks(db_session, vector, k=10)}
    assert "from the current model" in texts
    assert "from some other model" not in texts, (
        "A chunk embedded by a different model was ranked against this query vector."
    )
    #  An unlabelled chunk predates the labelling, so excluding it would silently
    #  drop an older corpus. Included deliberately, and reported by corpus_stats.
    assert "unlabelled, predates the column" in texts


def test_corpus_stats_reports_the_embedding_models_present(db_session):
    """A mixed corpus has to be visible somewhere, and this is the somewhere."""
    from cnms_fom.rag_backend.vectorstore import corpus_stats

    document = _document("sha-stats", title="mixed")
    db_session.add(document)
    db_session.flush()
    db_session.add_all([
        DocumentChunk(document_id=document.id, chunk_index=0, text="a",
                      embedding_model="model-a", embedding=[0.1] * 768),
        DocumentChunk(document_id=document.id, chunk_index=1, text="b",
                      embedding_model="model-b", embedding=[0.2] * 768),
        DocumentChunk(document_id=document.id, chunk_index=2, text="c",
                      embedding_model=None, embedding=[0.3] * 768),
    ])
    db_session.flush()

    by_model = corpus_stats(db_session)["embeddings_by_model"]
    assert by_model == {"model-a": 1, "model-b": 1, "(unlabelled)": 1}


def test_corpus_stats_does_not_claim_pgvector_where_ranking_is_in_python(db_session):
    """The capability report has to describe behaviour, not configuration.

    ``_pgvector_available()`` called without a session answers from the flag alone,
    so ``pgvector: true`` was served on SQLite and over a json column — through the
    API and into the assistant's tool output — while ranking happened in numpy.
    """
    from cnms_fom.rag_backend.vectorstore import corpus_stats

    claimed = corpus_stats(db_session)["pgvector"]
    dialect = db_session.get_bind().dialect.name
    if dialect != "postgresql":
        assert claimed is False, (
            f"corpus_stats claims pgvector on {dialect}, which has no vector type."
        )


# ---------------------------------------------------------------------------
# Bug 25: material identity depended on an optional package
# ---------------------------------------------------------------------------


def test_an_unreducible_formula_is_refused_rather_than_stored_as_a_new_material():
    """Identity is not guessed, and an identity that varies by machine is guessed.

    ``POST /materials`` reduced with pymatgen and, on ImportError, used the formula
    exactly as typed. pymatgen is the optional ``descriptors`` extra, so whether
    ``Hf2O4`` and ``HfO2`` are one material or two depended on the installation — and
    a corpus written with the extra, then added to without it, accumulates duplicates
    the unique constraint cannot see.
    """
    from cnms_fom.fom_engine.identity import FormulaNotCanonical, reduced_formula

    #  Already canonical: the same answer with or without pymatgen.
    assert reduced_formula("HfO2") == "HfO2"
    assert reduced_formula("TiO2") == "TiO2"
    #  Fractional occupancies have no meaningful GCD and are common here.
    assert reduced_formula("Hf0.5Zr0.5O2") == "Hf0.5Zr0.5O2"

    pymatgen_available = True
    try:
        import pymatgen.core  # noqa: F401
    except ImportError:
        pymatgen_available = False

    if pymatgen_available:
        assert reduced_formula("Hf2O4") == "HfO2"
    else:
        with pytest.raises(FormulaNotCanonical) as caught:
            reduced_formula("Hf2O4")
        #  The refusal must name the fix, not merely decline.
        assert "descriptors" in str(caught.value)

    with pytest.raises(FormulaNotCanonical):
        reduced_formula("")


def test_both_material_writers_use_the_same_reduction():
    """The API and the external-database ingest must agree on identity.

    ``ingest/materials_db.py`` assigned ``formula_reduced=formula`` with no reduction
    at all, so an external source writing an unreduced formula created a second
    material for a film already recorded under its reduced one.
    """
    import inspect

    from cnms_fom.ingest import materials_db
    from cnms_fom.routers import materials

    for module in (materials, materials_db):
        source = inspect.getsource(module)
        assert "reduced_formula(" in source, (
            f"{module.__name__} does not go through fom_engine.identity, so it can "
            "compute a different identity than the other writer."
        )
        assert "formula_reduced=formula," not in source, (
            f"{module.__name__} still stores an unreduced formula as the identity."
        )
