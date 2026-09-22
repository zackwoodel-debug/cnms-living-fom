"""Convert ``document_chunks.embedding`` to pgvector, where pgvector exists.

Migration 0001 creates the column as ``sa.JSON()`` unconditionally, while the ORM
picks its type at import from ``PGVECTOR_ENABLED`` (``db/base.embedding_column_type``).
Nothing reconciled the two, so ``PGVECTOR_ENABLED=true`` was unusable on **every**
Alembic-migrated database: the flag described a schema no migration produced. The
vector column only ever appeared where the tables were built by
``Base.metadata.create_all`` with the flag already set, which is tests and nothing
else. Migration 0002's HNSW index has been silently skipping itself ever since for
the same reason — it checks ``udt_name = 'vector'`` and gives up.

Both halves of the disagreement break retrieval, in opposite ways:

* flag on, json column — pgvector's result parser raises on every chunk row, so
  retrieval stops outright;
* flag off, vector column — embeddings read back as ``'[0.1,0.2]'`` strings, the
  dense leg is dropped as an optional retriever, and the system quietly serves
  lexical-only results.

This migration removes the first case and creates the second deliberately, which is
the trade: after it runs, a pgvector-equipped database wants ``PGVECTOR_ENABLED=true``,
and ``embedding_storage_mismatch`` says so at startup and in ``/health/ready`` rather
than leaving it to be discovered.

**Conditional on the extension *and* on the flag**, and the second half took a
correction to get right. Keying it on the extension alone looked more principled — a
schema that depends on a runtime setting is how the original divergence happened — but
it converts the column for every deployment that merely *has* pgvector installed,
including those running ``PGVECTOR_ENABLED=false``. Those then cannot insert a chunk at
all: with the JSON mapping Postgres renders the bind as ``%(embedding)s::JSON`` and
refuses ``json`` into a ``vector`` column, even for NULL. That broke ingestion and the
test suite.

The flag is not incidental configuration here: it is what ``embedding_column_type``
reads to choose the ORM type. Following it is therefore the only thing that makes the
schema and the mapping agree *by construction* rather than by coincidence. If the flag
is changed later, ``alembic downgrade 0008 && alembic upgrade head`` re-evaluates it —
so there is a way back, which is what the one-shot objection to flag-keyed migrations
is really about.

**No values are altered.** ``embedding::text::vector`` reparses what is already
there. A row whose array length disagrees with ``embedding_dim`` cannot be cast, and
rather than dropping those embeddings the migration refuses and names the count:
re-embedding is a decision about the corpus, not something to do silently inside a
schema change.

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

#  Same index name migration 0002 uses, so the two cannot both create one.
HNSW_INDEX = "ix_chunk_embedding_hnsw"


def _embedding_udt(bind) -> str | None:
    row = bind.execute(
        sa.text(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'document_chunks' AND column_name = 'embedding' "
            "AND table_schema = ANY(current_schemas(false)) LIMIT 1"
        )
    ).first()
    return str(row[0]).lower() if row else None


def _has_vector_extension(bind) -> bool:
    return bool(
        bind.execute(sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).first()
    )


def _embedding_dim() -> int:
    from cnms_fom.config import get_settings

    return int(get_settings().embedding_dim)


def _pgvector_enabled() -> bool:
    from cnms_fom.config import get_settings

    return bool(get_settings().pgvector_enabled)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite has no vector type; the json column is the only shape.
    if not _has_vector_extension(bind):
        return  # No pgvector here. json is correct, and the flag should stay false.
    if not _pgvector_enabled():
        #  The ORM will map JSON, and a vector column cannot accept a json bind.
        #  Converting here would leave a database that reads but cannot be written.
        return

    udt = _embedding_udt(bind)
    if udt is None:
        return  # No table yet.
    if udt == "vector":
        #  Built by create_all with the flag on. Only the index may be missing.
        _create_index(bind)
        return

    dim = _embedding_dim()
    wrong = bind.execute(
        sa.text(
            #  json_array_length rejects a non-array, so restrict to arrays first and
            #  let anything else fall into the same refusal below.
            "SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL "
            "AND (json_typeof(embedding) <> 'array' "
            "     OR json_array_length(embedding) <> :dim)"
        ),
        {"dim": dim},
    ).scalar_one()
    if wrong:
        raise RuntimeError(
            f"{wrong} of the stored embeddings are not arrays of {dim} numbers, so the "
            f"column cannot be cast to vector({dim}). EMBEDDING_DIM is {dim}; if the "
            "corpus was embedded with a different model, re-embed it, or point "
            "EMBEDDING_DIM at the model that produced these. This migration will not "
            "discard embeddings to make the cast succeed."
        )

    op.execute(
        f"ALTER TABLE document_chunks ALTER COLUMN embedding "
        f"TYPE vector({dim}) USING embedding::text::vector"
    )
    _create_index(bind)


def _create_index(bind) -> None:
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    if _embedding_udt(bind) != "vector":
        return

    #  The index is over a vector operator class and cannot survive the column.
    op.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")
    #  ``embedding::text`` renders '[0.1,0.2]', which ``::json`` parses back to an
    #  array. ``to_json(embedding::text)`` would instead store the JSON *string*
    #  '"[0.1,0.2]"' and quietly turn every embedding into text.
    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN embedding "
        "TYPE json USING embedding::text::json"
    )
