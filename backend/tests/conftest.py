"""Test configuration.

Environment is set before any ``cnms_fom`` import: ``db.models`` decides the
embedding column type at class-definition time, so ``PGVECTOR_ENABLED`` has to
be in place first.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("PGVECTOR_ENABLED", "false")
os.environ.setdefault("RANDOM_SEED", "20260823")

import pytest  # noqa: E402

from cnms_fom.fom_engine.definitions import all_draft_foms  # noqa: E402


@pytest.fixture
def draft_foms():
    return all_draft_foms()


@pytest.fixture
def hfo2_properties() -> dict:
    """A complete, plausible property row for a monoclinic HfO2 film.

    Illustrative values for exercising the pipeline — not a citable dataset.
    """
    return {
        "k": 25.0,
        "Eg": 5.7,
        "dEc": 1.5,
        "Ebd": 4.0,
        "kappa_th": 1.1,
        "tan_delta": 2.0e-3,
    }


@pytest.fixture
def no_dense_retrieval(monkeypatch):
    """Make dense retrieval fail, deterministically.

    Several retrieval tests are about the degraded path: lexical-only search, and
    the rule that a search which could not *run* is an error rather than a data
    gap. Those were originally exercised by the ``rag`` extra simply being absent
    — which meant they passed on CI and broke on any machine that had the extra
    installed and Ollama running. The behaviour under test is "the embedder is
    unreachable", so the test has to cause that rather than hope for it.
    """

    def _unavailable(*args, **kwargs):  # noqa: ARG001
        raise ImportError("RAG needs the 'rag' extra: pip install -e '.[rag]'")

    monkeypatch.setattr(
        "cnms_fom.rag_backend.embeddings.embed_query", _unavailable, raising=False
    )
    return _unavailable
