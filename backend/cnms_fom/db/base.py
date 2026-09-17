"""SQLAlchemy engine, session factory, and declarative base.

The engine is built lazily.  Importing this module must not open a connection or
even require a reachable database — that keeps the test suite, the CLI, and
``--help`` working on a machine with no Postgres running.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from cnms_fom.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for every ORM model in the project."""


@lru_cache
def get_engine() -> Engine:
    """Create (once) the process-wide engine."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_pre_ping=True,  # survive a Postgres restart under a long-lived container
        future=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and background jobs."""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def embedding_column_type(dim: int | None = None):
    """Column type for an embedding vector.

    pgvector when installed *and* enabled, which buys indexed ANN search;
    otherwise JSON, so the schema still loads and the RAG store ranks in Python.
    Keeping the choice here means nothing else has to branch on it.
    """
    from sqlalchemy import JSON

    settings = get_settings()
    dim = dim or settings.embedding_dim
    if not settings.pgvector_enabled:
        return JSON
    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:  # pragma: no cover - depends on optional extra
        return JSON
    return Vector(dim)
