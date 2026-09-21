"""Postgres-only regressions for bug 19: schema parity and failure isolation.

These exist because bug 14 and bug 19 had the same root cause — a code path that had
never been exercised on the real backend. A test that only runs on SQLite cannot catch
either, so these run against Postgres or they are skipped **loudly**: a skip here means
"unverified", never "passing".

Point them at a database with ``CNMS_TEST_POSTGRES_URL``. The database is used
read-mostly; each test builds its own schema in a uniquely named one so a failure
cannot corrupt a real corpus.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from cnms_fom.db import models  # noqa: F401 - registers the mappers
from cnms_fom.db.base import Base
from cnms_fom.db.enums import SynthesisTechnique
from cnms_fom.db.models import Document, DocumentChunk
from cnms_fom.rag_backend.hybrid import lexical_search

POSTGRES_URL = os.environ.get("CNMS_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason=(
        "Set CNMS_TEST_POSTGRES_URL to exercise the Postgres paths. Skipped means "
        "UNVERIFIED: bug 19 existed only because these paths had never run on Postgres."
    ),
)


def _with_database(name: str) -> str:
    """The configured URL pointed at a different database.

    ``make_url`` rather than string surgery: a unix-socket URL carries ``?host=/tmp``,
    whose query string contains a slash, so splitting on the last "/" finds the wrong
    one and silently produces a host of "/postgres".

    ``render_as_string(hide_password=False)`` rather than ``str(url)``: SQLAlchemy
    masks the password in ``__str__``, so ``str(url)`` yields ``cnms:***@...`` and the
    connection fails auth. It worked locally only because that Postgres trusts a unix
    socket and ignores the password entirely — the same "passes where auth is not real"
    trap as the rest of this file guards against.
    """
    return make_url(POSTGRES_URL).set(database=name).render_as_string(
        hide_password=False
    )


@pytest.fixture
def pg_database():
    """A uniquely named throwaway Postgres database, dropped on teardown."""
    name = f"cnms_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(
        _with_database("postgres"), isolation_level="AUTOCOMMIT", future=True
    )
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = _with_database(name)
    created = create_engine(url, isolation_level="AUTOCOMMIT", future=True)
    with created.connect() as conn:
        #  A new database does not inherit extensions from the template, and with
        #  PGVECTOR_ENABLED the ORM asks for VECTOR(768). Without this, create_all
        #  fails with 'type "vector" does not exist' — which only shows up where
        #  pgvector is actually switched on, so it passed locally and failed in CI.
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    created.dispose()
    try:
        yield url
    finally:
        engine_cache_clear(url)
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


def engine_cache_clear(url: str) -> None:
    """Dispose any engine still holding a connection to the throwaway database."""
    try:
        create_engine(url, future=True).dispose()
    except Exception:  # noqa: BLE001 - teardown must not fail the test
        pass


def migrate_to_head(url: str, monkeypatch) -> None:
    """Run Alembic to head against ``url``.

    ``migrations/env.py`` overwrites ``sqlalchemy.url`` from ``get_settings()``, so
    ``config.set_main_option`` alone is silently ignored and the migration runs against
    whatever the environment points at — which in a test means the developer's real
    database. The env var is the only lever that works, and ``get_settings`` is
    ``lru_cache``d, so the cache has to be cleared for the new value to be seen.
    """
    from alembic import command
    from alembic.config import Config

    from cnms_fom.config import get_settings

    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        config = Config("alembic.ini")
        #  configparser reads "%" as interpolation and a unix-socket URL is
        #  percent-encoded (?host=%2Ftmp), so it must be escaped here too.
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()


def _seed(session) -> int:
    document = Document(
        title="Growth of HfO2 by ALD in a hot-wall reactor",
        filename="hfo2.pdf",
        content_sha256="a" * 64,
        technique=SynthesisTechnique.ALD,
    )
    session.add(document)
    session.flush()
    chunk = DocumentChunk(
        document_id=document.id,
        chunk_index=0,
        page=1,
        text="The growth per cycle was constant at 0.98 angstrom per cycle.",
    )
    session.add(chunk)
    session.commit()
    return chunk.id


def test_the_admin_url_keeps_its_password():
    """`str(url)` masks the password, which fails wherever auth is real.

    Not marked postgres-only: it is a property of the helper, and it needs to fail on
    any machine rather than only on one with password authentication. The bug reached
    CI precisely because the developer's Postgres trusts a unix socket and never
    checked the password that `str()` had already replaced with "***".
    """
    from sqlalchemy.engine.url import make_url as _make_url

    sample = "postgresql+psycopg2://someone:s3cret@localhost:5432/somedb"
    rendered = _make_url(sample).set(database="postgres").render_as_string(
        hide_password=False
    )
    assert "s3cret" in rendered
    assert "***" not in rendered
    #  And the masking this guards against is real, not imagined.
    assert "***" in str(_make_url(sample).set(database="postgres"))


# --- 1A. schema parity ----------------------------------------------------


def test_create_all_and_alembic_agree_on_document_chunk_columns(pg_database):
    """`search_title` lived only in migration 0007, so create_all schemas lacked it.

    That divergence is the whole of bug 19: the Postgres lexical query references the
    column, so a create_all schema made the query raise.
    """
    engine = create_engine(pg_database, future=True)
    Base.metadata.create_all(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("document_chunks")}
    engine.dispose()

    assert "search_title" in columns, (
        "search_title is missing from the ORM schema. The Postgres lexical query "
        "references it, so this divergence makes that query raise."
    )


def test_every_orm_column_exists_in_a_migrated_schema(pg_database, monkeypatch):
    """Table-by-table parity: ORM metadata against the Alembic head schema.

    Fails loudly when a column exists on one path and not the other, in either
    direction, which is the class of bug rather than the one instance of it.
    """
    migrate_to_head(pg_database, monkeypatch)

    engine = create_engine(pg_database, future=True)
    inspector = inspect(engine)
    live_tables = set(inspector.get_table_names())

    mismatches: list[str] = []
    for name, table in Base.metadata.tables.items():
        if name not in live_tables:
            mismatches.append(f"{name}: in ORM metadata, absent from the migrated schema")
            continue
        live = {c["name"] for c in inspector.get_columns(name)}
        declared = {c.name for c in table.columns}
        for column in sorted(declared - live):
            mismatches.append(f"{name}.{column}: in the ORM, absent after migration")
        for column in sorted(live - declared):
            mismatches.append(f"{name}.{column}: in the migrated schema, absent from the ORM")
    engine.dispose()

    assert not mismatches, "ORM and Alembic schemas diverge:\n  " + "\n  ".join(mismatches)


# --- 1B. failure isolation ------------------------------------------------


@pytest.fixture
def breaking_fulltext(monkeypatch):
    """Make only the full-text expression invalid, leaving the schema sound.

    Dropping ``search_title`` would also break the Python fallback, because that
    fallback queries through the ORM and the ORM now declares the column — which is
    itself worth knowing: after the parity fix, schema parity is load-bearing for the
    fallback too, not just for the indexed query. So the failure is injected into the
    tsvector expression instead, which is the residual real-world shape of this
    problem (a bad text-search configuration, a dropped index, an FTS extension gone).
    """
    import sqlalchemy

    real = sqlalchemy.literal_column

    def broken(expression, *args, **kwargs):
        if "to_tsvector" in str(expression):
            return real("to_tsvector('english', document_chunks.no_such_column)")
        return real(expression, *args, **kwargs)

    monkeypatch.setattr(sqlalchemy, "literal_column", broken)
    return broken


def test_a_lexical_failure_leaves_the_session_usable(pg_database, breaking_fulltext):
    """Bug 19's damaging half: the fallback ran on an already-aborted transaction.

    Postgres aborts the whole transaction on any statement error, so before the
    SAVEPOINT the fallback itself raised InFailedSqlTransaction — and so did every
    later query on that session, including the caller's.
    """
    engine = create_engine(pg_database, future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    try:
        _seed(session)

        #  Must not raise: the fallback has to be reachable.
        hits = lexical_search(session, "HfO2 growth per cycle", k=5)
        assert isinstance(hits, list)
        assert hits, "the Python term-overlap fallback should still find the passage"

        #  The session must still work. This is the assertion bug 19 failed.
        assert session.execute(text("SELECT count(*) FROM documents")).scalar() == 1
        assert session.query(Document).count() == 1
    finally:
        session.close()
        engine.dispose()


def test_a_lexical_failure_does_not_discard_the_caller_s_staged_work(
    pg_database, breaking_fulltext
):
    """The savepoint must roll back the failed query only, not the caller's work."""
    engine = create_engine(pg_database, future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    try:
        _seed(session)

        #  Work the caller owns, staged but not committed, before the failing query.
        session.add(
            Document(
                title="Staged by the caller",
                filename="staged.pdf",
                content_sha256="b" * 64,
                technique=SynthesisTechnique.PLD,
            )
        )
        session.flush()

        lexical_search(session, "HfO2 growth per cycle", k=5)

        #  Still there, and still committable.
        assert session.query(Document).filter_by(filename="staged.pdf").count() == 1
        session.commit()
        assert session.query(Document).count() == 2
    finally:
        session.close()
        engine.dispose()


def test_the_lexical_leg_actually_runs_on_a_migrated_schema(pg_database, monkeypatch):
    """The happy path, on Postgres, which no test covered before bug 14."""
    migrate_to_head(pg_database, monkeypatch)

    engine = create_engine(pg_database, future=True)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    try:
        _seed(session)
        hits = lexical_search(session, "growth per cycle", k=5)
        assert hits, "the Postgres full-text query returned nothing on a migrated schema"
        assert "0.98" in hits[0].text
        #  The trigger from migration 0007 populated the title copy.
        assert session.execute(
            text("SELECT search_title FROM document_chunks LIMIT 1")
        ).scalar() == "Growth of HfO2 by ALD in a hot-wall reactor"
    finally:
        session.close()
        engine.dispose()
