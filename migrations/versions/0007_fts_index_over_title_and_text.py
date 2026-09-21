"""Index the lexical retriever's actual expression: document title plus chunk text.

Migration 0003 indexed ``to_tsvector('english', document_chunks.text)``. The lexical
retriever now matches against the document title *and* the chunk text, for a reason
measured on the benchmark: a table row carries none of its document's subject, so the
passage holding ``substrate_temperature 700 degC`` ranked fourth while a passage
reading "The material is LSMO, not SrTiO3" ranked third. All three of the query's
distinguishing terms were in the *title*. Scoring against title + text lifted the
right passage to second.

A functional index must match the expression the query uses or Postgres will not use
it, so this adds one over the new expression. The 0003 index is kept: it still serves
a text-only match, and dropping an index to save a few megabytes on a corpus this
size would be a poor trade.

The title is used for *scoring* only. The text returned, the quote stored, and the
citation are the chunk's own — nothing a reader checks is affected.

Because the expression spans two tables and a functional index cannot, the title is
materialised onto ``document_chunks.search_title`` and *that* is indexed. A copy can
go stale, so two triggers keep it in step: one on ``document_chunks`` for a new or
reparented chunk, and one on ``documents`` for a retitled document. The second is not
hypothetical — correcting a title from "LSMO on SrTiO3" to "LSMO on NdGaO3" left every
chunk of that document matching the *retracted* substrate and not the corrected one,
which for a platform where substrate identity is a descriptor is a wrong answer, not a
stale cache.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_document_chunks_fts_title_text"
EXPRESSION = (
    "to_tsvector('english', coalesce(documents.title, '') || ' ' || document_chunks.text)"
)


def upgrade() -> None:
    #  Postgres only: SQLite has no tsvector, and rag_backend.hybrid falls back to
    #  term overlap in Python there — which already scores title + text.
    if op.get_bind().dialect.name != "postgresql":
        return

    #  A functional index cannot span two tables, so the expression is materialised
    #  as a plain column on document_chunks and that is what gets indexed. It is a
    #  copy, so it can drift; the two triggers below are what stop it.
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS search_title text"
    )
    op.execute(
        "UPDATE document_chunks SET search_title = documents.title "
        "FROM documents WHERE documents.id = document_chunks.document_id"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON document_chunks USING gin ("
        "to_tsvector('english', coalesce(search_title, '') || ' ' || text))"
    )
    #  Keep the copy in step with the document it came from.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION document_chunks_sync_search_title()
        RETURNS trigger AS $$
        BEGIN
            SELECT title INTO NEW.search_title FROM documents WHERE id = NEW.document_id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_search_title ON document_chunks")
    op.execute(
        "CREATE TRIGGER trg_chunks_search_title BEFORE INSERT OR UPDATE OF document_id "
        "ON document_chunks FOR EACH ROW EXECUTE FUNCTION document_chunks_sync_search_title()"
    )

    #  The other direction: a retitled document has to reach the chunks that copied
    #  its old title, or the retriever keeps scoring against a title the document no
    #  longer has. Guarded on a real change so an ordinary UPDATE touching other
    #  columns does not rewrite every chunk of the document.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION documents_propagate_search_title()
        RETURNS trigger AS $$
        BEGIN
            UPDATE document_chunks SET search_title = NEW.title
            WHERE document_id = NEW.id;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_documents_search_title ON documents")
    op.execute(
        "CREATE TRIGGER trg_documents_search_title AFTER UPDATE OF title ON documents "
        "FOR EACH ROW WHEN (OLD.title IS DISTINCT FROM NEW.title) "
        "EXECUTE FUNCTION documents_propagate_search_title()"
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TRIGGER IF EXISTS trg_documents_search_title ON documents")
    op.execute("DROP FUNCTION IF EXISTS documents_propagate_search_title()")
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_search_title ON document_chunks")
    op.execute("DROP FUNCTION IF EXISTS document_chunks_sync_search_title()")
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS search_title")
