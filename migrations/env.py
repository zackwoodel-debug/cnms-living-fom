"""Alembic environment.

Two deliberate choices:

* The URL comes from ``cnms_fom.config`` rather than ``alembic.ini``, so
  migrations target the same database as the application and no credential is
  committed.
* ``render_as_batch`` is on. SQLite cannot ``ALTER`` a column or drop a
  constraint in place, so alembic recreates the table instead. The test suite
  runs on SQLite and production on Postgres, and batch mode is what lets one
  migration script serve both.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from cnms_fom.config import get_settings
from cnms_fom.db import models  # noqa: F401  — registers every mapper
from cnms_fom.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

#  The "%" must be doubled: set_main_option writes into configparser, which reads a
#  bare "%" as interpolation syntax and raises. Any percent-encoded URL hits this — a
#  unix-socket host (``?host=%2Ftmp``) or a password containing a special character —
#  so ``alembic upgrade head`` failed outright on a URL that SQLAlchemy accepts.
config.set_main_option(
    "sqlalchemy.url", get_settings().database_url.replace("%", "%%")
)
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Keep pgvector's ANN index out of autogenerate.

    It is created in a migration with raw DDL (the operator class has no ORM
    representation), so autogenerate would otherwise propose dropping it on
    every run.
    """
    if type_ == "index" and name and name.startswith("ix_chunk_embedding"):
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
