"""Alembic migration round-trip, exercised against a database that has rows.

An empty-database migration proves very little. The interesting failures —
adding a NOT NULL column with no backfill, installing a CHECK that existing data
violates, rewriting enum storage out from under stored values — only appear when
there is something to migrate. So this builds a database at the baseline
revision, puts data in it, and then migrates it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

alembic = pytest.importorskip("alembic", reason="alembic is needed to test migrations")
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

#  Rows written at revision 0001, using the *old* storage conventions: enum
#  member names in upper case, and no context_digest column at all.
BASELINE_ROWS = [
    (
        "INSERT INTO materials (formula, formula_reduced, polymorph, specimen_form, created_at) "
        "VALUES ('HfO2', 'HfO2', 'monoclinic', 'CRYSTALLINE_FILM', '2026-01-01')"
    ),
    (
        "INSERT INTO property_values "
        "(material_id, property_key, value, provenance_tier, temperature_k, frequency_hz, "
        " tensor_component, created_at) "
        "VALUES (1, 'k', 25.0, 'MEASURED', 300.0, 10000.0, 'zz', '2026-01-01')"
    ),
    (
        "INSERT INTO property_values "
        "(material_id, property_key, value, provenance_tier, temperature_k, frequency_hz, "
        " tensor_component, created_at) "
        "VALUES (1, 'k', 23.0, 'MEASURED', 300.0, 1000000.0, 'zz', '2026-01-01')"
    ),
]


@pytest.fixture
def alembic_config(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'migrate.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("PGVECTOR_ENABLED", "false")

    #  Settings are cached, and env.py reads them through get_settings().
    from cnms_fom.config import get_settings

    get_settings.cache_clear()

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    yield config, url
    get_settings.cache_clear()


def _constraint_names(engine, table: str) -> set[str]:
    return {c["name"] for c in inspect(engine).get_check_constraints(table)}


def test_upgrade_from_baseline_preserves_data_and_hardens_schema(alembic_config):
    config, url = alembic_config
    engine = create_engine(url, future=True)

    command.upgrade(config, "0001")
    with engine.begin() as connection:
        for statement in BASELINE_ROWS:
            connection.execute(text(statement))

    #  Baseline has none of the hardening.
    assert "context_digest" not in {c["name"] for c in inspect(engine).get_columns("property_values")}

    command.upgrade(config, "head")

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM property_values")).scalar() == 2
        assert connection.execute(text("SELECT COUNT(*) FROM materials")).scalar() == 1

        #  Enum storage rewritten from member name to member value.
        tiers = {r[0] for r in connection.execute(text("SELECT provenance_tier FROM property_values"))}
        assert tiers == {"measured"}
        forms = {r[0] for r in connection.execute(text("SELECT specimen_form FROM materials"))}
        assert forms == {"crystalline_film"}

        #  context_digest backfilled for every pre-existing row, and the two
        #  rows differ because their frequencies differ.
        digests = [r[0] for r in connection.execute(text("SELECT context_digest FROM property_values"))]
        assert all(d and len(d) == 32 for d in digests)
        assert len(set(digests)) == 2

    assert "ck_property_temperature_positive" in _constraint_names(engine, "property_values")
    engine.dispose()


def test_new_tables_appear_and_disappear_with_the_revision(alembic_config):
    config, url = alembic_config
    engine = create_engine(url, future=True)

    command.upgrade(config, "head")
    added = {
        "sensitivity_estimates",
        "integrity_checks",
        "analysis_exclusions",
        "spectral_series",
        "spectral_points",
        "external_records",
    }
    assert added <= set(inspect(engine).get_table_names())

    command.downgrade(config, "0001")
    assert not (added & set(inspect(engine).get_table_names()))
    engine.dispose()


def test_downgrade_removes_the_hardening_but_keeps_the_rows(alembic_config):
    config, url = alembic_config
    engine = create_engine(url, future=True)

    command.upgrade(config, "0001")
    with engine.begin() as connection:
        for statement in BASELINE_ROWS:
            connection.execute(text(statement))
    command.upgrade(config, "head")
    command.downgrade(config, "0001")

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM property_values")).scalar() == 2

    assert _constraint_names(engine, "property_values") == set()
    #  SQLite cannot drop a named UNIQUE by name, so the CHECK-constraint
    #  rebuild is what removes it. Asserted rather than assumed — see
    #  _drop_context_unique_constraint in revision 0002.
    with engine.connect() as connection:
        ddl = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='property_values'")
        ).scalar()
    assert "uq_property_value_context" not in ddl
    engine.dispose()


def test_upgrade_is_reapplicable_after_a_downgrade(alembic_config):
    config, url = alembic_config
    engine = create_engine(url, future=True)

    command.upgrade(config, "head")
    command.downgrade(config, "0001")
    command.upgrade(config, "head")

    assert "ck_score_not_scored_has_no_value" in _constraint_names(engine, "fom_scores")
    assert "spectral_points" in inspect(engine).get_table_names()
    engine.dispose()


def test_migrated_schema_matches_the_orm_metadata(alembic_config):
    """`alembic upgrade head` and `metadata.create_all` must agree.

    They are two paths to the same schema, and a divergence means a model change
    landed without a migration — which shows up much later as a constraint that
    exists in tests but not in production.
    """
    config, url = alembic_config
    command.upgrade(config, "head")

    engine = create_engine(url, future=True)
    migrated = set(inspect(engine).get_table_names()) - {"alembic_version"}
    engine.dispose()

    from cnms_fom.db import models  # noqa: F401  - registers the mappers
    from cnms_fom.db.base import Base

    assert migrated == set(Base.metadata.tables)
