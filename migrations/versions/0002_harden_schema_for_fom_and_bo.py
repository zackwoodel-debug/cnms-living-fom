"""Harden the schema for the FOM protocol and the BO loop.

Four kinds of change, in the order ``upgrade()`` applies them:

1. **Enum storage.**  Columns now hold the member *value* (``'measured'``) rather
   than the Python *name* (``'MEASURED'``), so the database speaks the same
   vocabulary as the API, the JSON payloads, and the docs.  Anyone writing
   ``WHERE provenance_tier = 'measured'`` by hand previously got zero rows.
   Existing data is rewritten before the new CHECK is installed.

2. **Measurement-context identity.**  ``property_values.context_digest``
   fingerprints the ~20 context columns so "is this the same measurement?"
   becomes a uniqueness constraint (FOM_PROOF Eq. 3).  Added nullable,
   backfilled, then tightened — autogenerate proposed it as NOT NULL with no
   backfill, which fails on any table that already has rows.

3. **New tables.**  Analysis intermediates that were computed in memory and
   never persisted (``sensitivity_estimates``, ``integrity_checks``,
   ``analysis_exclusions``), spectral storage (``spectral_series`` /
   ``spectral_points``), and the external-ingestion staging area
   (``external_records``).

4. **Constraints and indices.**  CHECK constraints encoding protocol invariants,
   plus composite indices for the query patterns the FOM and BO paths actually
   use.  Alembic's autogenerate does not detect CheckConstraints, so every one
   of them below is hand-written.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


_TIER = {
    "MEASURED": "measured",
    "CALCULATED": "calculated",
    "MODELED": "modeled",
    "UNAVAILABLE": "unavailable",
}
_TECHNIQUE = {
    "MBE": "mbe",
    "PLD": "pld",
    "ALD": "ald",
    "SPUTTERING": "sputtering",
    "CVD": "cvd",
    "SOLUTION": "solution",
    "CNMS_USER_DOC": "cnms_user_doc",
    "OTHER": "other",
}
_TRANSFORM = {"NONE": "none", "LOG10": "log10"}

#  Enum columns whose stored values change from member name to member value.
#  Spelled out as literals rather than imported from ``cnms_fom.db.enums``: a
#  migration describes one historical moment and must keep working even after
#  the application's enums gain members.
ENUM_VALUE_REWRITES: dict[tuple[str, str], dict[str, str]] = {
    ("materials", "specimen_form"): {
        "BULK_SINGLE_CRYSTAL": "bulk_single_crystal",
        "CERAMIC": "ceramic",
        "AMORPHOUS_FILM": "amorphous_film",
        "CRYSTALLINE_FILM": "crystalline_film",
        "COMPUTATIONAL": "computational",
        "OTHER": "other",
    },
    ("structures", "provenance_tier"): _TIER,
    ("descriptor_values", "provenance_tier"): _TIER,
    ("property_values", "provenance_tier"): _TIER,
    ("fom_scores", "status"): {
        "SCORED": "scored",
        "NOT_SCORED": "not_scored",
        "ILLUSTRATIVE": "illustrative",
    },
    ("correlation_results", "block"): {
        #  The one enum whose value is not simply the lowercased name.
        "SS": "R_SS",
        "SP": "R_SP",
        "PP": "R_PP",
        "FF": "R_FF",
        "SF": "R_SF",
    },
    ("correlation_results", "x_transform"): _TRANSFORM,
    ("correlation_results", "y_transform"): _TRANSFORM,
    ("correlation_results", "outcome"): {
        "SUPPORTS": "supports",
        "CONTRADICTS": "contradicts",
        "INCONCLUSIVE": "inconclusive",
    },
    ("documents", "technique"): _TECHNIQUE,
    ("instruments", "technique"): _TECHNIQUE,
}

#  CHECK constraints on pre-existing tables.  New tables carry theirs inline in
#  their ``create_table`` below.
CHECK_CONSTRAINTS: list[tuple[str, str, str]] = [
    ("materials", "ck_material_polymorph_not_blank", "length(trim(polymorph)) > 0"),
    ("structures", "ck_structure_volume_positive", "volume_ang3 IS NULL OR volume_ang3 > 0"),
    (
        "structures",
        "ck_structure_z_positive",
        "formula_units_per_cell IS NULL OR formula_units_per_cell >= 1",
    ),
    (
        "descriptor_values",
        "ck_descriptor_uncertainty_nonneg",
        "uncertainty IS NULL OR uncertainty >= 0",
    ),
    (
        "property_values",
        "ck_property_uncertainty_nonneg",
        "uncertainty IS NULL OR uncertainty >= 0",
    ),
    (
        "property_values",
        "ck_property_temperature_positive",
        "temperature_k IS NULL OR temperature_k > 0",
    ),
    (
        "property_values",
        "ck_property_frequency_nonneg",
        "frequency_hz IS NULL OR frequency_hz >= 0",
    ),
    (
        "property_values",
        "ck_property_thickness_positive",
        "thickness_nm IS NULL OR thickness_nm > 0",
    ),
    ("property_values", "ck_property_area_positive", "area_cm2 IS NULL OR area_cm2 > 0"),
    ("fom_definitions", "ck_fom_version_positive", "version >= 1"),
    ("fom_definitions", "ck_fom_floor_eps_in_unit_interval", "floor_eps > 0 AND floor_eps < 1"),
    (
        "fom_definitions",
        "ck_fom_approved_has_approver",
        "NOT approved OR approved_by IS NOT NULL",
    ),
    ("fom_scores", "ck_score_not_scored_has_no_value", "status <> 'not_scored' OR value IS NULL"),
    ("fom_scores", "ck_score_scored_has_value", "status <> 'scored' OR value IS NOT NULL"),
    ("fom_scores", "ck_score_value_nonneg", "value IS NULL OR value >= 0"),
    (
        "fom_scores",
        "ck_score_modeled_is_illustrative",
        "NOT uses_modeled_inputs OR status = 'illustrative'",
    ),
    ("analysis_runs", "ck_run_permutation_positive", "permutation_b IS NULL OR permutation_b > 0"),
    ("correlation_results", "ck_corr_n_nonneg", "n_complete >= 0"),
    (
        "correlation_results",
        "ck_corr_r_range",
        "pearson_r IS NULL OR (pearson_r >= -1 AND pearson_r <= 1)",
    ),
    (
        "correlation_results",
        "ck_corr_rho_range",
        "spearman_rho IS NULL OR (spearman_rho >= -1 AND spearman_rho <= 1)",
    ),
    (
        "correlation_results",
        "ck_corr_p_range",
        "p_permutation IS NULL OR (p_permutation > 0 AND p_permutation <= 1)",
    ),
    ("correlation_results", "ck_corr_q_range", "q_fdr IS NULL OR (q_fdr >= 0 AND q_fdr <= 1)"),
    ("documents", "ck_document_pages_positive", "n_pages IS NULL OR n_pages > 0"),
    ("document_chunks", "ck_chunk_index_nonneg", "chunk_index >= 0"),
    ("bo_runs", "ck_bo_run_objective_sense", "objective_sense IN ('max', 'min')"),
    (
        "bo_observations",
        "ck_bo_obs_noise_nonneg",
        "objective_noise IS NULL OR objective_noise >= 0",
    ),
    ("bo_suggestions", "ck_bo_sugg_batch_index_nonneg", "batch_index >= 0"),
    ("bo_suggestions", "ck_bo_sugg_std_nonneg", "predicted_std IS NULL OR predicted_std >= 0"),
]


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _checks_by_table() -> dict[str, list[tuple[str, str]]]:
    """Group ``CHECK_CONSTRAINTS`` so each table is altered exactly once."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for table, name, condition in CHECK_CONSTRAINTS:
        grouped.setdefault(table, []).append((name, condition))
    return grouped


def _relax_native_enums() -> None:
    """Postgres only: turn native ENUM columns into plain text.

    A native ENUM refuses any label outside its type, so the value rewrite below
    cannot run while one is in place.  SQLite stores these as VARCHAR already
    and needs nothing.  The orphaned types are dropped once no column references
    them.
    """
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, column in ENUM_VALUE_REWRITES:
        if _table_exists(table):
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN {column} TYPE VARCHAR(32) "
                f"USING {column}::text"
            )
    for type_name in (
        "specimen_form",
        "provenance_tier",
        "score_status",
        "correlation_block",
        "transform",
        "hypothesis_outcome",
        "synthesis_technique",
    ):
        op.execute(f"DROP TYPE IF EXISTS {type_name}")


def _rewrite_enum_values() -> None:
    """Member name -> member value, for every enum column carrying data."""
    for (table, column), mapping in ENUM_VALUE_REWRITES.items():
        if not _table_exists(table):
            continue
        for name, value in mapping.items():
            if name == value:
                continue
            op.execute(
                sa.text(
                    f"UPDATE {table} SET {column} = :new WHERE {column} = :old"
                ).bindparams(new=value, old=name)
            )


def _repair_constraint_violations() -> None:
    """Fix rows predating the constraints, so the constraints can be added.

    Both repairs restore an invariant the application already enforces; neither
    invents data.
    """
    if _table_exists("fom_scores"):
        #  Sec. 2.3: a score built on a modeled input is ILLUSTRATIVE.  Rows
        #  written before the router passed provenance through were labelled
        #  'scored' despite carrying the modeled flag.
        op.execute(
            "UPDATE fom_scores SET status = 'illustrative' "
            "WHERE uses_modeled_inputs AND status <> 'illustrative'"
        )
    if _table_exists("fom_definitions"):
        #  Sec. 6.2: weights are a policy choice and need a named approver.  An
        #  approval with nobody attached is not an approval.
        op.execute(
            "UPDATE fom_definitions SET approved = false "
            "WHERE approved AND approved_by IS NULL"
        )


def _backfill_context_digests() -> None:
    """Fill ``property_values.context_digest`` for rows predating the column.

    Imports the application's digest function on purpose.  Reimplementing the
    canonicalisation inline would let the two drift, and a digest disagreeing
    with what the ORM computes is worse than none at all — it would split
    genuine duplicates across two fingerprints.
    """
    from cnms_fom.db.context import CONTEXT_FIELDS, context_digest

    bind = op.get_bind()
    columns = ", ".join(("id", *CONTEXT_FIELDS))
    rows = bind.execute(sa.text(f"SELECT {columns} FROM property_values")).mappings().all()
    for row in rows:
        bind.execute(
            sa.text("UPDATE property_values SET context_digest = :digest WHERE id = :id"),
            {"digest": context_digest(dict(row)), "id": row["id"]},
        )


def _create_vector_index() -> None:
    """Approximate-nearest-neighbour index on the chunk embeddings.

    Postgres + pgvector only, and skipped silently otherwise: the RAG store
    falls back to ranking in Python, which is correct but linear.

    Without this, every ``/rag/query`` is a sequential scan computing a cosine
    distance per chunk. That is fine at a thousand chunks and unusable at a
    hundred thousand, which one CNMS synthesis corpus reaches easily.

    HNSW rather than IVFFlat: it needs no training pass over existing data (so
    it can be built on an empty table and stay correct as the corpus grows) and
    gives better recall at the same latency. It is pgvector >= 0.5; on an older
    build the ``CREATE INDEX`` fails and is reported rather than silently
    skipped, because a missing index here is a performance cliff worth knowing
    about.
    """
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    has_vector = bind.execute(
        sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).first()
    if not has_vector:
        return
    is_vector_column = bind.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'document_chunks' AND column_name = 'embedding' "
            "AND udt_name = 'vector'"
        )
    ).first()
    if not is_vector_column:
        return  # PGVECTOR_ENABLED was false when the table was created
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunk_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def upgrade() -> None:
    # --- 1. enum storage: names -> values, before any new CHECK is installed -
    _relax_native_enums()
    _rewrite_enum_values()
    _repair_constraint_violations()

    # --- 2. context digest, added nullable so existing rows survive ----------
    with op.batch_alter_table("property_values", schema=None) as batch_op:
        batch_op.add_column(sa.Column("context_digest", sa.String(length=32), nullable=True))

    # --- 3. new tables, indices, and the autogenerated column alterations ----
    op.create_table('analysis_exclusions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('material_key', sa.String(length=256), nullable=False),
    sa.Column('material_id', sa.Integer(), nullable=True),
    sa.Column('property_key', sa.String(length=64), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['material_id'], ['materials.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['run_id'], ['analysis_runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('analysis_exclusions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_analysis_exclusions_run_id'), ['run_id'], unique=False)
        batch_op.create_index('ix_exclusion_material', ['material_key'], unique=False)
        batch_op.create_index('ix_exclusion_run_key', ['run_id', 'property_key'], unique=False)
    op.create_table('external_records',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source_database', sa.String(length=128), nullable=False),
    sa.Column('source_table', sa.String(length=128), nullable=False),
    sa.Column('source_row_id', sa.String(length=64), nullable=False),
    sa.Column('source_sha256', sa.String(length=64), nullable=True),
    sa.Column('payload', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('parsed', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('quarantine_reason', sa.Text(), nullable=True),
    sa.Column('missing_fields', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('material_id', sa.Integer(), nullable=True),
    sa.Column('promoted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('staged', 'promoted', 'quarantined', 'superseded')", name='ck_external_status'),
    sa.ForeignKeyConstraint(['material_id'], ['materials.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_database', 'source_table', 'source_row_id', name='uq_external_record')
    )
    with op.batch_alter_table('external_records', schema=None) as batch_op:
        batch_op.create_index('ix_external_status', ['source_database', 'status'], unique=False)
    op.create_table('integrity_checks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('applications', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('n_materials', sa.Integer(), nullable=False),
    sa.Column('observed_correlation', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('null_correlation', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('excess_correlation', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('reconstruction_ok', sa.Boolean(), nullable=True),
    sa.Column('max_reconstruction_error', sa.Float(), nullable=True),
    sa.Column('leakage_passed', sa.Boolean(), nullable=True),
    sa.Column('leakage_report', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('passed', sa.Boolean(), nullable=False),
    sa.Column('failures', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['analysis_runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('integrity_checks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_integrity_checks_run_id'), ['run_id'], unique=False)
        batch_op.create_index('ix_integrity_run', ['run_id'], unique=False)
    op.create_table('sensitivity_estimates',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('descriptor_key', sa.String(length=64), nullable=False),
    sa.Column('property_key', sa.String(length=64), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.Column('elasticity', sa.Float(), nullable=True),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('std_error', sa.Float(), nullable=True),
    sa.Column('p_value', sa.Float(), nullable=True),
    sa.Column('vif', sa.Float(), nullable=True),
    sa.Column('n_complete', sa.Integer(), nullable=True),
    sa.Column('reference_point', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('confounders', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("source IN ('regression', 'theory')", name='ck_sensitivity_source'),
    sa.CheckConstraint('vif IS NULL OR vif >= 1', name='ck_sensitivity_vif_at_least_one'),
    sa.ForeignKeyConstraint(['run_id'], ['analysis_runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'descriptor_key', 'property_key', name='uq_sensitivity_cell')
    )
    with op.batch_alter_table('sensitivity_estimates', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_sensitivity_estimates_run_id'), ['run_id'], unique=False)
        batch_op.create_index('ix_sensitivity_run', ['run_id'], unique=False)
    op.create_table('spectral_series',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('material_id', sa.Integer(), nullable=False),
    sa.Column('quantity', sa.String(length=32), nullable=False),
    sa.Column('units', sa.String(length=32), nullable=True),
    sa.Column('independent_variable', sa.String(length=32), nullable=False),
    sa.Column('axis', sa.String(length=32), nullable=True),
    sa.Column('tensor_component', sa.String(length=16), nullable=True),
    sa.Column('temperature_k', sa.Float(), nullable=True),
    sa.Column('method', sa.String(length=128), nullable=True),
    sa.Column('provenance_tier', sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32), nullable=False),
    sa.Column('doi', sa.String(length=256), nullable=True),
    sa.Column('source_url', sa.String(length=512), nullable=True),
    sa.Column('database_identifier', sa.String(length=128), nullable=True),
    sa.Column('dataset_label', sa.String(length=256), nullable=True),
    sa.Column('context_digest', sa.String(length=32), nullable=False),
    sa.Column('n_points', sa.Integer(), nullable=False),
    sa.Column('x_min', sa.Float(), nullable=True),
    sa.Column('x_max', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("independent_variable IN ('wavelength_nm', 'energy_ev', 'frequency_hz', 'wavenumber_cm-1')", name='ck_spectral_independent_variable'),
    sa.CheckConstraint('n_points >= 0', name='ck_spectral_n_points_nonneg'),
    sa.ForeignKeyConstraint(['material_id'], ['materials.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('material_id', 'quantity', 'axis', 'context_digest', name='uq_spectral_series')
    )
    with op.batch_alter_table('spectral_series', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_spectral_series_context_digest'), ['context_digest'], unique=False)
        batch_op.create_index(batch_op.f('ix_spectral_series_material_id'), ['material_id'], unique=False)
        batch_op.create_index('ix_spectral_series_material_quantity', ['material_id', 'quantity'], unique=False)
    op.create_table('spectral_points',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('series_id', sa.Integer(), nullable=False),
    sa.Column('x_value', sa.Float(), nullable=False),
    sa.Column('y_value', sa.Float(), nullable=False),
    sa.Column('uncertainty', sa.Float(), nullable=True),
    sa.CheckConstraint('x_value > 0', name='ck_spectral_point_x_positive'),
    sa.ForeignKeyConstraint(['series_id'], ['spectral_series.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('series_id', 'x_value', name='uq_spectral_point_x')
    )
    with op.batch_alter_table('spectral_points', schema=None) as batch_op:
        batch_op.create_index('ix_spectral_point_series_x', ['series_id', 'x_value'], unique=False)
        batch_op.create_index(batch_op.f('ix_spectral_points_series_id'), ['series_id'], unique=False)
    with op.batch_alter_table('analysis_runs', schema=None) as batch_op:
        batch_op.create_index('ix_runs_kind_created', ['kind', 'created_at'], unique=False)
    with op.batch_alter_table('bo_observations', schema=None) as batch_op:
        batch_op.create_index('ix_bo_obs_experiment', ['experiment_id'], unique=False)
        batch_op.create_index('ix_bo_obs_run_feasible', ['bo_run_id', 'is_feasible'], unique=False)
    with op.batch_alter_table('bo_runs', schema=None) as batch_op:
        batch_op.create_index('ix_bo_runs_status', ['status'], unique=False)
    with op.batch_alter_table('bo_suggestions', schema=None) as batch_op:
        batch_op.create_index('ix_bo_sugg_run_status', ['bo_run_id', 'status'], unique=False)
    with op.batch_alter_table('correlation_results', schema=None) as batch_op:
        batch_op.alter_column('block',
               existing_type=sa.VARCHAR(length=2),
               type_=sa.Enum('R_SS', 'R_SP', 'R_PP', 'R_FF', 'R_SF', name='correlation_block', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.alter_column('x_transform',
               existing_type=sa.VARCHAR(length=5),
               type_=sa.Enum('none', 'log10', name='transform', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.alter_column('y_transform',
               existing_type=sa.VARCHAR(length=5),
               type_=sa.Enum('none', 'log10', name='transform', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.alter_column('outcome',
               existing_type=sa.VARCHAR(length=12),
               type_=sa.Enum('supports', 'contradicts', 'inconclusive', name='hypothesis_outcome', native_enum=False, length=32),
               existing_nullable=True)
        batch_op.create_index('ix_corr_pair', ['x_key', 'y_key'], unique=False)
    with op.batch_alter_table('descriptor_values', schema=None) as batch_op:
        batch_op.alter_column('provenance_tier',
               existing_type=sa.VARCHAR(length=11),
               type_=sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_descriptor_material_key', ['material_id', 'descriptor_key'], unique=False)
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.alter_column('technique',
               existing_type=sa.VARCHAR(length=13),
               type_=sa.Enum('mbe', 'pld', 'ald', 'sputtering', 'cvd', 'solution', 'cnms_user_doc', 'other', name='synthesis_technique', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_documents_technique', ['technique'], unique=False)
    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.create_index('ix_experiments_instrument_status', ['instrument_id', 'status'], unique=False)
        batch_op.create_index('ix_experiments_material', ['material_id'], unique=False)
        batch_op.create_index('ix_experiments_proposal', ['proposal_id'], unique=False)
    with op.batch_alter_table('fom_scores', schema=None) as batch_op:
        batch_op.alter_column('status',
               existing_type=sa.VARCHAR(length=12),
               type_=sa.Enum('scored', 'not_scored', 'illustrative', name='score_status', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_score_definition_status', ['fom_definition_id', 'status'], unique=False)
    with op.batch_alter_table('instruments', schema=None) as batch_op:
        batch_op.alter_column('technique',
               existing_type=sa.VARCHAR(length=13),
               type_=sa.Enum('mbe', 'pld', 'ald', 'sputtering', 'cvd', 'solution', 'cnms_user_doc', 'other', name='synthesis_technique', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_instruments_technique_available', ['technique', 'available'], unique=False)
    with op.batch_alter_table('materials', schema=None) as batch_op:
        batch_op.alter_column('specimen_form',
               existing_type=sa.VARCHAR(length=19),
               type_=sa.Enum('bulk_single_crystal', 'ceramic', 'amorphous_film', 'crystalline_film', 'computational', 'other', name='specimen_form', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_materials_form_polymorph', ['specimen_form', 'polymorph'], unique=False)
    with op.batch_alter_table('mediation_results', schema=None) as batch_op:
        batch_op.create_index('ix_mediation_application', ['application'], unique=False)
        batch_op.create_unique_constraint('uq_mediation_descriptor_application', ['run_id', 'descriptor_key', 'application'])
    with op.batch_alter_table('property_values', schema=None) as batch_op:
        batch_op.alter_column('provenance_tier',
               existing_type=sa.VARCHAR(length=11),
               type_=sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_property_key_tier', ['property_key', 'provenance_tier'], unique=False)
        batch_op.create_index('ix_property_material_key', ['material_id', 'property_key'], unique=False)
        batch_op.create_index(batch_op.f('ix_property_values_context_digest'), ['context_digest'], unique=False)
    with op.batch_alter_table('structures', schema=None) as batch_op:
        batch_op.alter_column('provenance_tier',
               existing_type=sa.VARCHAR(length=11),
               type_=sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32),
               existing_nullable=False)
        batch_op.create_index('ix_structures_material', ['material_id'], unique=False)
    # ### end Alembic commands

    # --- 4. finish the context digest now that the rows are filled -----------
    _backfill_context_digests()
    with op.batch_alter_table("property_values", schema=None) as batch_op:
        batch_op.alter_column(
            "context_digest", existing_type=sa.String(length=32), nullable=False
        )
        batch_op.create_unique_constraint(
            "uq_property_value_context", ["material_id", "property_key", "context_digest"]
        )

    # --- 5. CHECK constraints (invisible to autogenerate) --------------------
    #  One batch block per table, not per constraint.  On SQLite every batch
    #  block copies the whole table, so 28 separate blocks would mean 28 copies
    #  — and each copy is a chance for a constraint the reflector did not carry
    #  over to be quietly lost.
    for table, constraints in _checks_by_table().items():
        with op.batch_alter_table(table, schema=None) as batch_op:
            for name, condition in constraints:
                batch_op.create_check_constraint(name, condition)

    # --- 6. ANN index for retrieval (Postgres + pgvector only) --------------
    _create_vector_index()


def _drop_context_unique_constraint() -> None:
    """Remove ``uq_property_value_context``, allowing for SQLite.

    SQLite exposes an inline ``UNIQUE`` only as an anonymous auto-index
    (``sqlite_autoindex_property_values_N``), so alembic's batch mode cannot
    address it by the name we gave it and raises "No such constraint".

    On SQLite the constraint is instead removed by the CHECK-constraint batch
    below: that block rebuilds the table from reflection, and reflection does
    not carry the unique constraint across.  Relying on that would be fragile if
    it were merely assumed, so ``test_migration_round_trip`` asserts the
    constraint is genuinely gone after a downgrade.

    On Postgres — where this actually runs in production — batch mode passes
    straight through to ``ALTER TABLE ... DROP CONSTRAINT`` and the name works.
    """
    if op.get_bind().dialect.name == "sqlite":
        return
    op.drop_constraint("uq_property_value_context", "property_values", type_="unique")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunk_embedding_hnsw")
    _drop_context_unique_constraint()

    for table, constraints in _checks_by_table().items():
        with op.batch_alter_table(table, schema=None) as batch_op:
            for name, _condition in constraints:
                batch_op.drop_constraint(name, type_="check")

    with op.batch_alter_table('structures', schema=None) as batch_op:
        batch_op.drop_index('ix_structures_material')
        batch_op.alter_column('provenance_tier',
               existing_type=sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32),
               type_=sa.VARCHAR(length=11),
               existing_nullable=False)
    with op.batch_alter_table('property_values', schema=None) as batch_op:
        #  `uq_property_value_context` is dropped by _drop_context_unique_constraint()
        #  above, which handles the dialect difference. Dropping it again here failed on
        #  Postgres with "constraint ... does not exist" and was invisible on SQLite,
        #  where batch mode rebuilds the table from reflection and the second drop is a
        #  no-op. The round trip had therefore only ever been exercised on SQLite —
        #  the same shape as the other bugs in this codebase that survived because one
        #  backend never ran the code.
        batch_op.drop_index(batch_op.f('ix_property_values_context_digest'))
        batch_op.drop_index('ix_property_material_key')
        batch_op.drop_index('ix_property_key_tier')
        batch_op.alter_column('provenance_tier',
               existing_type=sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32),
               type_=sa.VARCHAR(length=11),
               existing_nullable=False)
        batch_op.drop_column('context_digest')
    with op.batch_alter_table('mediation_results', schema=None) as batch_op:
        batch_op.drop_constraint('uq_mediation_descriptor_application', type_='unique')
        batch_op.drop_index('ix_mediation_application')
    with op.batch_alter_table('materials', schema=None) as batch_op:
        batch_op.drop_index('ix_materials_form_polymorph')
        batch_op.alter_column('specimen_form',
               existing_type=sa.Enum('bulk_single_crystal', 'ceramic', 'amorphous_film', 'crystalline_film', 'computational', 'other', name='specimen_form', native_enum=False, length=32),
               type_=sa.VARCHAR(length=19),
               existing_nullable=False)
    with op.batch_alter_table('instruments', schema=None) as batch_op:
        batch_op.drop_index('ix_instruments_technique_available')
        batch_op.alter_column('technique',
               existing_type=sa.Enum('mbe', 'pld', 'ald', 'sputtering', 'cvd', 'solution', 'cnms_user_doc', 'other', name='synthesis_technique', native_enum=False, length=32),
               type_=sa.VARCHAR(length=13),
               existing_nullable=False)
    with op.batch_alter_table('fom_scores', schema=None) as batch_op:
        batch_op.drop_index('ix_score_definition_status')
        batch_op.alter_column('status',
               existing_type=sa.Enum('scored', 'not_scored', 'illustrative', name='score_status', native_enum=False, length=32),
               type_=sa.VARCHAR(length=12),
               existing_nullable=False)
    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.drop_index('ix_experiments_proposal')
        batch_op.drop_index('ix_experiments_material')
        batch_op.drop_index('ix_experiments_instrument_status')
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_index('ix_documents_technique')
        batch_op.alter_column('technique',
               existing_type=sa.Enum('mbe', 'pld', 'ald', 'sputtering', 'cvd', 'solution', 'cnms_user_doc', 'other', name='synthesis_technique', native_enum=False, length=32),
               type_=sa.VARCHAR(length=13),
               existing_nullable=False)
    with op.batch_alter_table('descriptor_values', schema=None) as batch_op:
        batch_op.drop_index('ix_descriptor_material_key')
        batch_op.alter_column('provenance_tier',
               existing_type=sa.Enum('measured', 'calculated', 'modeled', 'unavailable', name='provenance_tier', native_enum=False, length=32),
               type_=sa.VARCHAR(length=11),
               existing_nullable=False)
    with op.batch_alter_table('correlation_results', schema=None) as batch_op:
        batch_op.drop_index('ix_corr_pair')
        batch_op.alter_column('outcome',
               existing_type=sa.Enum('supports', 'contradicts', 'inconclusive', name='hypothesis_outcome', native_enum=False, length=32),
               type_=sa.VARCHAR(length=12),
               existing_nullable=True)
        batch_op.alter_column('y_transform',
               existing_type=sa.Enum('none', 'log10', name='transform', native_enum=False, length=32),
               type_=sa.VARCHAR(length=5),
               existing_nullable=False)
        batch_op.alter_column('x_transform',
               existing_type=sa.Enum('none', 'log10', name='transform', native_enum=False, length=32),
               type_=sa.VARCHAR(length=5),
               existing_nullable=False)
        batch_op.alter_column('block',
               existing_type=sa.Enum('R_SS', 'R_SP', 'R_PP', 'R_FF', 'R_SF', name='correlation_block', native_enum=False, length=32),
               type_=sa.VARCHAR(length=2),
               existing_nullable=False)
    with op.batch_alter_table('bo_suggestions', schema=None) as batch_op:
        batch_op.drop_index('ix_bo_sugg_run_status')
    with op.batch_alter_table('bo_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_bo_runs_status')
    with op.batch_alter_table('bo_observations', schema=None) as batch_op:
        batch_op.drop_index('ix_bo_obs_run_feasible')
        batch_op.drop_index('ix_bo_obs_experiment')
    with op.batch_alter_table('analysis_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_runs_kind_created')
    with op.batch_alter_table('spectral_points', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_spectral_points_series_id'))
        batch_op.drop_index('ix_spectral_point_series_x')
    op.drop_table('spectral_points')
    with op.batch_alter_table('spectral_series', schema=None) as batch_op:
        batch_op.drop_index('ix_spectral_series_material_quantity')
        batch_op.drop_index(batch_op.f('ix_spectral_series_material_id'))
        batch_op.drop_index(batch_op.f('ix_spectral_series_context_digest'))
    op.drop_table('spectral_series')
    with op.batch_alter_table('sensitivity_estimates', schema=None) as batch_op:
        batch_op.drop_index('ix_sensitivity_run')
        batch_op.drop_index(batch_op.f('ix_sensitivity_estimates_run_id'))
    op.drop_table('sensitivity_estimates')
    with op.batch_alter_table('integrity_checks', schema=None) as batch_op:
        batch_op.drop_index('ix_integrity_run')
        batch_op.drop_index(batch_op.f('ix_integrity_checks_run_id'))
    op.drop_table('integrity_checks')
    with op.batch_alter_table('external_records', schema=None) as batch_op:
        batch_op.drop_index('ix_external_status')
    op.drop_table('external_records')
    with op.batch_alter_table('analysis_exclusions', schema=None) as batch_op:
        batch_op.drop_index('ix_exclusion_run_key')
        batch_op.drop_index('ix_exclusion_material')
        batch_op.drop_index(batch_op.f('ix_analysis_exclusions_run_id'))
    op.drop_table('analysis_exclusions')

    #  Enum values are deliberately NOT rewritten back to member names.  The
    #  lowercase form is valid input for the old column type too, and reversing
    #  it would only reintroduce the API/database vocabulary split.
