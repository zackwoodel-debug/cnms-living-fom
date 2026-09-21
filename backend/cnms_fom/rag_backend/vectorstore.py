"""Similarity search over ``document_chunks``.

We own this table rather than delegating to a generic LangChain vector store for
one reason: a retrieved passage has to come back with a citation (document,
page, DOI), and a store that treats metadata as an opaque blob makes that
awkward exactly when it matters.

Two backends, same interface:
  * pgvector cosine distance, indexed, when the extension is enabled;
  * a numpy fallback that ranks in Python, which is fine into the low tens of
    thousands of chunks and keeps the scaffold runnable without the extension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from cnms_fom.config import get_settings
from cnms_fom.db.enums import SynthesisTechnique

logger = logging.getLogger(__name__)


@dataclass
class ChunkHit:
    """A retrieved passage with everything needed to cite it."""

    chunk_id: int
    document_id: int
    document_title: str
    technique: str
    page: int | None
    text: str
    similarity: float
    doi: str | None = None
    source_url: str | None = None

    @property
    def citation(self) -> str:
        """A compact human-readable locator, e.g. ``ALD of HfO2, p. 12 (doi:10.x/y)``.

        A property rather than a method so that it matches
        ``research.contracts.EvidenceItem.citation``. Two classes carrying the same
        attribute, one callable and one not, is a trip hazard: a bare ``.citation``
        on the method form silently formats a bound method into a string, which
        produces a citation that looks like a bug report.
        """
        parts = [self.document_title]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.doi:
            parts.append(f"doi:{self.doi}")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "technique": self.technique,
            "page": self.page,
            "text": self.text,
            "similarity": self.similarity,
            "doi": self.doi,
            "source_url": self.source_url,
            "citation": self.citation,
        }


def _pgvector_available(session=None) -> bool:
    """Whether the pgvector query path can actually run.

    ``session`` matters. The setting and the package say pgvector is *configured*;
    only the session says which database is in front of us, and ``cosine_distance``
    compiles to the Postgres-only ``<=>`` operator. Without the dialect check a
    perfectly reasonable deployment — ``PGVECTOR_ENABLED=true`` for the production
    Postgres — emitted ``<=>`` at any SQLite session in the same process and failed
    with "near '>': syntax error". That reaches the assistant as
    ``search_corpus failed``, so a configuration flag became a statement about the
    corpus.

    Called without a session it answers the old question, "is pgvector configured",
    which is what the capability report wants.
    """
    if not get_settings().pgvector_enabled:
        return False
    try:
        import pgvector.sqlalchemy  # noqa: F401
    except ImportError:
        return False
    if session is not None:
        try:
            return session.get_bind().dialect.name == "postgresql"
        except Exception:  # noqa: BLE001 - an unbound session is not Postgres
            return False
    return True


def search_chunks(
    session,
    query_vector: list[float],
    *,
    k: int = 6,
    techniques: list[SynthesisTechnique] | None = None,
    min_similarity: float = 0.0,
) -> list[ChunkHit]:
    """Return the ``k`` most similar chunks, highest cosine similarity first.

    ``min_similarity`` is worth setting above 0 in practice: an empty corpus or
    an off-topic question otherwise returns the k least-bad chunks, which is how
    a confident answer gets built on irrelevant context.
    """
    from cnms_fom.db.models import Document, DocumentChunk

    query = session.query(DocumentChunk, Document).join(
        Document, DocumentChunk.document_id == Document.id
    )
    if techniques:
        query = query.filter(Document.technique.in_(list(techniques)))

    if _pgvector_available(session):
        #  pgvector's cosine_distance is 1 - cosine_similarity.
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        rows = query.add_columns(distance.label("distance")).order_by(distance).limit(k).all()
        hits = [
            _to_hit(chunk, document, similarity=1.0 - float(dist))
            for chunk, document, dist in rows
            if dist is not None
        ]
    else:
        logger.debug("pgvector unavailable; ranking in Python.")
        hits = _rank_in_python(query.all(), query_vector, k)

    return [hit for hit in hits if hit.similarity >= min_similarity]


def _to_hit(chunk, document, *, similarity: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        technique=document.technique.value,
        page=chunk.page,
        text=chunk.text,
        similarity=similarity,
        doi=document.doi,
        source_url=document.source_url,
    )


def _rank_in_python(rows, query_vector: list[float], k: int) -> list[ChunkHit]:
    """Cosine similarity in numpy, for the no-pgvector path."""
    q = np.asarray(query_vector, dtype=float)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []

    scored: list[tuple[float, object, object]] = []
    for chunk, document in rows:
        if chunk.embedding is None:
            continue
        v = np.asarray(chunk.embedding, dtype=float)
        if v.shape != q.shape:
            #  A chunk embedded with a different model. Skip rather than compare.
            continue
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            continue
        scored.append((float(q @ v) / (q_norm * v_norm), chunk, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_to_hit(c, d, similarity=s) for s, c, d in scored[:k]]


def corpus_stats(session) -> dict:
    """Document and chunk counts by technique — the corpus coverage at a glance."""
    from sqlalchemy import func

    from cnms_fom.db.models import Document, DocumentChunk

    by_technique = (
        session.query(Document.technique, func.count(Document.id))
        .group_by(Document.technique)
        .all()
    )
    total_chunks = session.query(func.count(DocumentChunk.id)).scalar() or 0
    embedded = (
        session.query(func.count(DocumentChunk.id))
        .filter(DocumentChunk.embedding.isnot(None))
        .scalar()
        or 0
    )
    return {
        "documents_by_technique": {t.value: int(n) for t, n in by_technique},
        "total_documents": sum(int(n) for _, n in by_technique),
        "total_chunks": int(total_chunks),
        "embedded_chunks": int(embedded),
        "pgvector": _pgvector_available(),
    }
