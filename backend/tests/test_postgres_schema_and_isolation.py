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

import json
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


#  ---------------------------------------------------------------------------
#  Bug 21: PGVECTOR_ENABLED describes the schema; it does not change it.
#  ---------------------------------------------------------------------------
#
#  Found by launching the server against a database migrated while pgvector was off
#  and then turning the flag on. The ORM maps ``embedding`` as VECTOR(768) because
#  ``embedding_column_type`` resolves the flag at import; Postgres still stores json.
#
#  What made it worth a named test is the size of the failure relative to how it
#  reads. The first symptom was ``operator does not exist: json <=> unknown`` from
#  the dense leg, which looks like "dense retrieval is unavailable, fall back". It is
#  not: the portable path reads the same column, so pgvector's result parser raises
#  ``'list' object has no attribute 'split'`` on *every* chunk row. Retrieval is down,
#  the traceback names a third-party file, and nothing points at the setting.
#
#  These tests are Postgres-only by nature. SQLite has no vector type, so the
#  disagreement this guards against cannot be constructed there.


def _make_embedding_json(url: str) -> None:
    """Put the schema in the state a pre-pgvector migration leaves it in."""
    engine = create_engine(url, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE document_chunks ALTER COLUMN embedding TYPE json "
                #  ::text::json, not to_json(::text): the latter stores the JSON
                #  string '"[0.1,0.2]"' and turns every embedding into text.
                "USING embedding::text::json"
            )
        )
    engine.dispose()


def test_a_json_embedding_column_under_the_pgvector_flag_is_reported(
    pg_database, monkeypatch
):
    """The mismatch names the flag, not pgvector's parser."""
    from cnms_fom.rag_backend import vectorstore

    engine = create_engine(pg_database, future=True)
    Base.metadata.create_all(engine)
    engine.dispose()
    _make_embedding_json(pg_database)

    #  Assert against the flag directly rather than the ambient environment, so the
    #  test states the same thing whether or not the suite runs with pgvector on.
    monkeypatch.setattr(
        vectorstore.get_settings(), "pgvector_enabled", True, raising=False
    )
    vectorstore._EMBEDDING_IS_VECTOR.clear()

    engine = create_engine(pg_database, future=True)
    try:
        message = vectorstore.embedding_storage_mismatch(engine)
        assert message is not None, (
            "A json column under the pgvector flag went unreported. Every chunk read "
            "fails in this state; a silent probe means the operator learns about it "
            "from a pgvector traceback."
        )
        #  The message has to name the lever. "type mismatch" sends someone into the
        #  ORM; PGVECTOR_ENABLED sends them to the thing they can actually change.
        assert "PGVECTOR_ENABLED" in message
        assert "json" in message
    finally:
        engine.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()


def test_a_matching_vector_column_is_not_reported_as_a_mismatch(
    pg_database, monkeypatch
):
    """The guard must not refuse the configuration it exists to protect."""
    from cnms_fom.rag_backend import vectorstore

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text(
                "CREATE TABLE document_chunks (id serial PRIMARY KEY, "
                "embedding vector(768))"
            )
        )
    engine.dispose()

    monkeypatch.setattr(
        vectorstore.get_settings(), "pgvector_enabled", True, raising=False
    )
    vectorstore._EMBEDDING_IS_VECTOR.clear()

    engine = create_engine(pg_database, future=True)
    try:
        assert vectorstore.embedding_storage_mismatch(engine) is None, (
            "A real vector column was reported as a mismatch, which would take "
            "retrieval down on a correctly configured deployment."
        )
    finally:
        engine.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()


def test_the_flag_off_over_a_json_column_is_the_working_configuration(
    pg_database, monkeypatch
):
    """The pairing every migrated deployment has. It must not be reported."""
    from cnms_fom.rag_backend import vectorstore

    engine = create_engine(pg_database, future=True)
    Base.metadata.create_all(engine)
    engine.dispose()
    _make_embedding_json(pg_database)

    monkeypatch.setattr(
        vectorstore.get_settings(), "pgvector_enabled", False, raising=False
    )
    vectorstore._EMBEDDING_IS_VECTOR.clear()

    engine = create_engine(pg_database, future=True)
    try:
        assert vectorstore.embedding_storage_mismatch(engine) is None
    finally:
        engine.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()


def test_a_json_column_under_the_pgvector_flag_still_returns_results(
    pg_database, monkeypatch
):
    """The mismatch costs an index, not correctness.

    This test used to assert a refusal. That was the right behaviour when the mapping
    made every chunk read raise inside pgvector's parser; it is the wrong behaviour now
    that ``db/embedding_type.py`` reads both shapes, because refusing a query that
    would have answered correctly is its own kind of wrong answer.
    """
    from cnms_fom.db.enums import SynthesisTechnique
    from cnms_fom.rag_backend import vectorstore
    from cnms_fom.rag_backend.vectorstore import search_chunks

    engine = create_engine(pg_database, future=True)
    Base.metadata.create_all(engine)
    dim = int(get_settings_dim())
    with sessionmaker(bind=engine, future=True)() as session:
        document = Document(
            title="probe", filename="probe.pdf", content_sha256="probe-sha",
            technique=SynthesisTechnique.ALD,
        )
        session.add(document)
        session.flush()
        session.add(DocumentChunk(
            document_id=document.id, chunk_index=0, text="hfo2 growth per cycle",
            embedding_model=None, embedding=[0.5] * dim,
        ))
        session.commit()
    engine.dispose()
    _make_embedding_json(pg_database)

    monkeypatch.setattr(
        vectorstore.get_settings(), "pgvector_enabled", True, raising=False
    )
    vectorstore._EMBEDDING_IS_VECTOR.clear()

    engine = create_engine(pg_database, future=True)
    session = sessionmaker(bind=engine, future=True)()
    try:
        hits = search_chunks(session, [0.5] * dim, k=4)
        assert hits, "A json column under the pgvector flag returned nothing."
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-6), (
            f"Ranked, but the similarity is {hits[0].similarity}: the embedding did "
            "not survive the read."
        )
        #  Still reported, because the missing ANN index is worth knowing about.
        assert vectorstore.embedding_storage_mismatch(session) is not None
    finally:
        session.close()
        engine.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()


def get_settings_dim() -> int:
    from cnms_fom.config import get_settings

    return int(get_settings().embedding_dim)


def test_the_vector_check_is_not_cached_across_engines(pg_database):
    """Two engines, two databases, two answers.

    The cache was keyed on ``id(engine)``. CPython reuses an id once the object is
    collected, so a disposed engine's answer could be served to a new engine pointing
    somewhere else — a cache hit that reports the wrong database's schema. Keyed on
    the engine object, the entry dies with it.
    """
    from cnms_fom.rag_backend import vectorstore

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text("CREATE TABLE document_chunks (id serial PRIMARY KEY, embedding vector(768))")
        )
    vectorstore._EMBEDDING_IS_VECTOR.clear()
    assert vectorstore._embedding_is_vector(engine) is True
    engine.dispose()
    del engine

    import gc

    gc.collect()

    #  A second engine on a database whose column is json must answer False even if
    #  the allocator handed it the first engine's address.
    other = create_engine(pg_database, future=True)
    with other.begin() as conn:
        conn.execute(text("DROP TABLE document_chunks"))
        conn.execute(text("CREATE TABLE document_chunks (id serial PRIMARY KEY, embedding json)"))
    try:
        assert vectorstore._embedding_is_vector(other) is False, (
            "A stale cache entry answered for a different engine."
        )
    finally:
        other.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()


def test_the_flag_off_over_a_vector_column_is_reported_too(pg_database, monkeypatch):
    """The quiet half of the disagreement, and the reason the probe is symmetric.

    A JSON-mapped ORM reads a vector column back as the string ``'[0.1,0.2,...]'``.
    ``_rank_in_python`` raises on ``np.asarray(..., dtype=float)``, ``hybrid_search``
    drops the dense leg as an optional retriever, and every answer afterwards comes
    from lexical search alone with nothing said. Unlike the flag-on case there is no
    crash to notice, so only this check stands between that and a silent corpus-wide
    quality regression.
    """
    from cnms_fom.rag_backend import vectorstore

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text("CREATE TABLE document_chunks (id serial PRIMARY KEY, embedding vector(768))")
        )
    engine.dispose()

    monkeypatch.setattr(
        vectorstore.get_settings(), "pgvector_enabled", False, raising=False
    )
    vectorstore._EMBEDDING_IS_VECTOR.clear()

    engine = create_engine(pg_database, future=True)
    try:
        message = vectorstore.embedding_storage_mismatch(engine)
        assert message is not None, (
            "A vector column under a JSON mapping went unreported. Dense retrieval is "
            "silently dead in this state."
        )
        assert "PGVECTOR_ENABLED" in message
        #  Reads work now, so the thing worth naming is the half that does not: an
        #  ingest into a vector column through the JSON mapping is refused outright.
        assert "ingest" in message.lower()
    finally:
        engine.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()


def test_a_json_mapped_orm_really_does_read_a_vector_column_as_a_string(pg_database):
    """The premise of the test above, checked against Postgres rather than assumed.

    This started as an assumption that JSON-over-vector was harmless ("it reads it as
    text, nothing is lost"). It is not: the value arrives as ``str`` and every numeric
    consumer downstream fails. Pinned here so the symmetric probe cannot be argued
    away later on the strength of the same wrong intuition.
    """
    from sqlalchemy import JSON, Column, Integer
    from sqlalchemy.orm import declarative_base

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE TABLE probe (id serial PRIMARY KEY, embedding vector(3))"))
        conn.execute(text("INSERT INTO probe (embedding) VALUES ('[0.1,0.2,0.3]')"))

    LocalBase = declarative_base()

    class Probe(LocalBase):
        __tablename__ = "probe"
        id = Column(Integer, primary_key=True)
        embedding = Column(JSON)  # exactly what the flag-off mapping produces

    try:
        with sessionmaker(bind=engine, future=True)() as session:
            value = session.query(Probe).one().embedding
        assert isinstance(value, str), (
            f"Expected a str from a JSON-mapped vector column, got {type(value).__name__}. "
            "If this ever returns a list, the flag-off half of the mismatch is harmless "
            "and the probe can be narrowed."
        )
    finally:
        engine.dispose()


#  ---------------------------------------------------------------------------
#  Migration 0009: the conversion the error message points at.
#  ---------------------------------------------------------------------------


def test_migration_0009_converts_the_column_and_preserves_every_value(
    pg_database, monkeypatch
):
    """Up, down, up — with the embeddings read back and compared each time.

    A conversion that lost precision or reordered components would leave retrieval
    working and every cosine similarity subtly wrong, which no schema assertion
    catches. So the values are compared, not just the column type.
    """
    from alembic import command
    from alembic.config import Config

    from cnms_fom.config import get_settings

    monkeypatch.setenv("DATABASE_URL", pg_database)
    #  0009 converts only where the ORM will map Vector, which is what this flag
    #  selects. Without it the migration correctly does nothing.
    monkeypatch.setenv("PGVECTOR_ENABLED", "true")
    get_settings.cache_clear()
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", pg_database.replace("%", "%%"))
    command.upgrade(config, "0008")

    dim = int(get_settings().embedding_dim)
    #  Values chosen to catch both precision loss and reordering: a descending ramp
    #  with enough decimal places that a float32 round trip is visible.
    original = [round(1.0 - i / dim, 6) for i in range(dim)]

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO documents (id, title, filename, content_sha256, technique) "
                "VALUES (1, 'probe', 'probe.pdf', 'probe-sha', 'ald')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO document_chunks (document_id, chunk_index, text, embedding) "
                "VALUES (1, 0, 'probe', CAST(:e AS json))"
            ),
            {"e": json.dumps(original)},
        )
    engine.dispose()

    def _embedding_as_floats() -> list[float]:
        eng = create_engine(pg_database, future=True)
        try:
            with eng.connect() as conn:
                #  Read as text and parse here, so this does not depend on which
                #  type the driver maps the column to at any point in the round trip.
                raw = conn.execute(
                    text("SELECT embedding::text FROM document_chunks WHERE chunk_index = 0")
                ).scalar_one()
        finally:
            eng.dispose()
        return [float(part) for part in raw.strip("[]").split(",")]

    def _udt() -> str:
        eng = create_engine(pg_database, future=True)
        try:
            with eng.connect() as conn:
                return conn.execute(
                    text(
                        "SELECT udt_name FROM information_schema.columns "
                        "WHERE table_name = 'document_chunks' AND column_name = 'embedding'"
                    )
                ).scalar_one()
        finally:
            eng.dispose()

    command.upgrade(config, "0009")
    assert _udt() == "vector", (
        "0009 left the column as json on a database with the vector extension, so "
        "PGVECTOR_ENABLED=true stays unusable — the bug it exists to fix."
    )
    after_up = _embedding_as_floats()

    command.downgrade(config, "0008")
    assert _udt() == "json"
    after_down = _embedding_as_floats()

    command.upgrade(config, "0009")
    assert _udt() == "vector"
    after_up_again = _embedding_as_floats()

    #  pgvector stores float32, so the round trip is not bit-exact against the
    #  float64 literals inserted. Asserting exact equality here would be asserting
    #  something untrue; 1e-6 is tighter than any similarity this feeds.
    for label, got in (
        ("after upgrade", after_up),
        ("after downgrade", after_down),
        ("after re-upgrade", after_up_again),
    ):
        assert len(got) == dim, f"{label}: {len(got)} components, expected {dim}"
        worst = max(abs(a - b) for a, b in zip(original, got, strict=True))
        assert worst < 1e-6, f"{label}: embeddings drifted by {worst}"

    engine = create_engine(pg_database, future=True)
    try:
        with engine.connect() as conn:
            assert conn.execute(
                text("SELECT 1 FROM pg_indexes WHERE indexname = 'ix_chunk_embedding_hnsw'")
            ).first(), "No HNSW index after 0009: the vector column is being scanned."
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_migration_0009_refuses_rather_than_discarding_mismatched_embeddings(
    pg_database, monkeypatch
):
    """An embedding of the wrong width is a corpus problem, not a cast problem.

    Nulling the offending rows would make the migration succeed and silently delete
    part of the corpus, which is the shape of mistake the no-imputation rule exists to
    prevent. The refusal has to name the count so the operator can size the problem.
    """
    from alembic import command
    from alembic.config import Config

    from cnms_fom.config import get_settings

    #  Everything up to 0009, then a row the cast cannot take.
    monkeypatch.setenv("DATABASE_URL", pg_database)
    monkeypatch.setenv("PGVECTOR_ENABLED", "true")
    get_settings.cache_clear()
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", pg_database.replace("%", "%%"))
    command.upgrade(config, "0008")

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO documents (id, title, filename, content_sha256, technique) "
                "VALUES (1, 'probe', 'probe.pdf', 'probe-sha', 'ald') "
                "ON CONFLICT DO NOTHING"
            )
        )
        conn.execute(
            text(
                "INSERT INTO document_chunks (document_id, chunk_index, text, embedding) "
                "VALUES (1, 0, 'probe', '[0.1, 0.2, 0.3]'::json)"
            )
        )
    engine.dispose()

    with pytest.raises(RuntimeError) as caught:
        command.upgrade(config, "0009")
    message = str(caught.value)
    assert "1 of the stored embeddings" in message, message
    #  It must say what to do, not only that it stopped.
    assert "re-embed" in message.lower()
    get_settings.cache_clear()


def test_a_vector_column_under_a_json_mapping_is_readable_and_reported(
    pg_database, monkeypatch
):
    """The quiet direction: reads work, ingestion does not, and it says so.

    A JSON-mapped ORM reads a vector column back as ``'[0.1,0.2,...]'``, which
    ``TolerantJSON`` turns into numbers — so existing rows stay searchable instead of
    the dense leg dying silently. Writing is a different matter: Postgres refuses a
    json bind into a vector column and the cast cannot be suppressed from the type, so
    this direction is reported as a fault rather than a note.
    """
    from cnms_fom.rag_backend import vectorstore

    engine = create_engine(pg_database, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(
            text("CREATE TABLE document_chunks (id serial PRIMARY KEY, embedding vector(768))")
        )
        literal = ",".join(["0.5"] * 768)
        conn.execute(text(f"INSERT INTO document_chunks (embedding) VALUES ('[{literal}]')"))
    engine.dispose()

    monkeypatch.setattr(
        vectorstore.get_settings(), "pgvector_enabled", False, raising=False
    )
    vectorstore._EMBEDDING_IS_VECTOR.clear()

    engine = create_engine(pg_database, future=True)
    try:
        message = vectorstore.embedding_storage_mismatch(engine)
        assert message is not None
        assert "PGVECTOR_ENABLED" in message
        #  It has to name the consequence that is not visible from a query.
        assert "ingest" in message.lower()
    finally:
        engine.dispose()
        vectorstore._EMBEDDING_IS_VECTOR.clear()
