-- Runs once, on first initialisation of the Postgres data directory.
-- The pgvector/pgvector image ships the extension; this enables it in our database.

CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram index support for substring search over formulas and document titles.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Tables are created by SQLAlchemy (`cnms-fom init-db`) or Alembic, not here:
-- the ORM is the single source of truth for the schema, and a second definition
-- in SQL would drift from it.
