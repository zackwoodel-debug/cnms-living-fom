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
import weakref
from dataclasses import dataclass

import numpy as np
from sqlalchemy import or_

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

    The dialect is not sufficient on its own. ``embedding`` is only a pgvector
    column when the schema was built with pgvector switched on; a database migrated
    while it was off has a ``json`` column, and ``<=>`` against JSON fails with
    "operator does not exist: json <=> unknown". That is a real deployment: turn
    ``PGVECTOR_ENABLED`` on for a database that predates it and dense retrieval
    stops working, silently, while the capability report still says pgvector is
    available. So the column's own type is checked too.

    Called without a session it answers the narrower question, "is pgvector
    configured", which is what the capability report wants.
    """
    if not get_settings().pgvector_enabled:
        return False
    try:
        import pgvector.sqlalchemy  # noqa: F401
    except ImportError:
        return False
    if session is None:
        return True
    try:
        bind = session.get_bind()
        if bind.dialect.name != "postgresql":
            return False
        return _embedding_is_vector(bind)
    except Exception:  # noqa: BLE001 - an unbound session is not Postgres
        return False


#  Cached per engine. Reflecting a column type on every search would put a
#  catalogue query in front of each query; the schema does not change under a
#  running process, and a migration restarts it.
#  Keyed on the engine object, not ``id(engine)``: CPython reuses an id once the
#  object is collected, so an id-keyed cache can answer for a disposed engine on
#  behalf of a new one pointing at a different database. A weak key also lets the
#  entry disappear with the engine rather than pinning it for the process lifetime.
_EMBEDDING_IS_VECTOR: weakref.WeakKeyDictionary[object, bool] = (
    weakref.WeakKeyDictionary()
)


def _embedding_is_vector(bind) -> bool:
    """Whether ``document_chunks.embedding`` is a pgvector column in this database.

    Asks Postgres for the column's underlying type name rather than reflecting the
    table. SQLAlchemy's reflection maps ``vector`` to ``NullType`` and warns "Did not
    recognize type \'vector\'" unless pgvector has registered itself in the dialect's
    type registry first, so a type-name comparison on the reflected column reads a
    real vector column as "not a vector" — which would report a mismatch on a
    correctly configured deployment and take retrieval down to protect it from
    nothing. ``udt_name`` is the same answer without the dependency on import order.

    Reflected once per engine: the column type cannot change under a running process
    without a migration, and a migration restarts it.
    """
    engine = getattr(bind, "engine", bind)
    key = engine
    cached = _EMBEDDING_IS_VECTOR.get(key)
    if cached is not None:
        return cached

    from sqlalchemy import text as sa_text

    try:
        with engine.connect() as connection:
            row = connection.execute(
                sa_text(
                    #  information_schema rather than a regclass cast: a missing table
                    #  returns no rows instead of raising. current_schemas(false)
                    #  honours the search path, so this reads the table the session
                    #  would actually query.
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_name = 'document_chunks' AND column_name = 'embedding' "
                    "AND table_schema = ANY(current_schemas(false)) LIMIT 1"
                )
            ).first()
    except Exception:  # noqa: BLE001 - a check that cannot run must not block a request
        return False

    if row is None:
        #  No table yet: nothing to disagree with, and nothing to cache either, since
        #  a migration may create it while this process is running.
        return False

    is_vector = str(row[0]).lower() == "vector"
    if not is_vector:
        logger.warning(
            "PGVECTOR_ENABLED is set, but document_chunks.embedding is %s, not a vector "
            "column, in this database. Every read of a chunk will fail: the ORM maps the "
            "column as a vector and Postgres returns %s. Migrate the column, or unset "
            "the flag.",
            row[0],
            row[0],
        )
    _EMBEDDING_IS_VECTOR[key] = is_vector
    return is_vector


def embedding_storage_mismatch(session_or_bind) -> str | None:
    """Describe the ORM/column type disagreement, or None when there is none.

    ``PGVECTOR_ENABLED`` selects the ORM type at import, before anything has looked at
    the database, and migration 0001 created ``embedding`` as ``json`` regardless of
    it — so both directions occur in real deployments.

    Neither is fatal any more. ``db/embedding_type.py`` makes both column shapes
    *readable* under either mapping, so retrieval keeps working through a mismatch
    instead of failing on every chunk row. What remains is worth saying out loud:

    * flag on, ``json`` column — correct results, no ANN index. A performance ceiling
      that looks like nothing at all until the corpus grows.
    * flag off, ``vector`` column — reads fine, but an **ingest will fail**: Postgres
      refuses a ``json`` bind into a ``vector`` column, and the cast cannot be
      suppressed from the type. Reading working while writing does not is the kind of
      split that gets diagnosed as a corpus problem.

    Returns a message rather than raising, because neither case justifies refusing a
    query. Startup and ``/health/ready`` report it.
    """
    settings = get_settings()
    try:
        bind = getattr(session_or_bind, "get_bind", lambda: session_or_bind)()
        if bind.dialect.name != "postgresql":
            #  SQLite has no vector type, so the JSON mapping is the only shape
            #  available and cannot disagree with anything.
            return None
        column_is_vector = _embedding_is_vector(bind)
    except Exception:  # noqa: BLE001 - a check that cannot run must not block a request
        return None

    if column_is_vector == settings.pgvector_enabled:
        return None

    if settings.pgvector_enabled:
        return (
            "PGVECTOR_ENABLED is true, but this database stores document_chunks."
            "embedding as json. Retrieval works and ranks in Python; what is missing is "
            "the ANN index, so dense search scans every chunk. Run 'alembic upgrade "
            "head' to convert the column (migration 0009), or set "
            "PGVECTOR_ENABLED=false to stop claiming an index that is not there."
        )
    return (
        "PGVECTOR_ENABLED is false, but this database stores document_chunks.embedding "
        "as a pgvector column. Existing rows read correctly, but ingesting new ones "
        "will fail: Postgres refuses a json bind into a vector column. Set "
        "PGVECTOR_ENABLED=true to match the schema."
    )


def search_chunks(
    session,
    query_vector: list[float],
    *,
    k: int = 6,
    techniques: list[SynthesisTechnique] | None = None,
    min_similarity: float = 0.0,
    embedding_model: str | None = None,
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

    #  Compare only vectors that live in the same space. Two embedding models
    #  produce coordinates in unrelated bases, so a cosine similarity between them
    #  is not a weak signal — it is a number with no meaning, and `nomic-embed-text`
    #  shares its 768 dimensions with plenty of other models, so the shape check in
    #  `_rank_in_python` cannot catch the mix. A NULL model is included because it
    #  predates the labelling and excluding it would silently drop an older corpus;
    #  `corpus_stats` reports the breakdown so a mix is visible rather than assumed.
    model = embedding_model or get_settings().ollama_embed_model
    query = query.filter(
        or_(DocumentChunk.embedding_model == model, DocumentChunk.embedding_model.is_(None))
    )

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
    skipped_dim = 0
    for chunk, document in rows:
        if chunk.embedding is None:
            continue
        v = np.asarray(chunk.embedding, dtype=float)
        if v.shape != q.shape:
            #  A different embedding width. Skipping is right — the vectors are not
            #  comparable — but doing it silently is how a changed EMBEDDING_DIM
            #  turns into a corpus that answers from a shrinking subset of itself
            #  while every metric still looks healthy. Counted and reported below.
            skipped_dim += 1
            continue
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            continue
        scored.append((float(q @ v) / (q_norm * v_norm), chunk, document))

    if skipped_dim:
        logger.warning(
            "Skipped %d of %d chunks whose embedding width is not %d: they cannot be "
            "compared against this query vector. Retrieval is searching part of the "
            "corpus. Re-embed those chunks, or point EMBEDDING_DIM at the model that "
            "produced them.",
            skipped_dim,
            len(rows),
            q.shape[0],
        )

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
    #  One row per embedding model present. A corpus holding two is a corpus whose
    #  similarities are only meaningful within each group, and nothing else in the
    #  system was reporting it.
    by_model = (
        session.query(DocumentChunk.embedding_model, func.count(DocumentChunk.id))
        .filter(DocumentChunk.embedding.isnot(None))
        .group_by(DocumentChunk.embedding_model)
        .all()
    )

    return {
        "documents_by_technique": {t.value: int(n) for t, n in by_technique},
        "total_documents": sum(int(n) for _, n in by_technique),
        "total_chunks": int(total_chunks),
        "embedded_chunks": int(embedded),
        #  With the session, not without it. Called bare, `_pgvector_available`
        #  answers from the flag alone and reported `true` on SQLite and over a json
        #  column — a capability claim served to the API and to the assistant's
        #  tools while ranking was happening in Python.
        "pgvector": _pgvector_available(session),
        "embeddings_by_model": {(m or "(unlabelled)"): int(n) for m, n in by_model},
    }
