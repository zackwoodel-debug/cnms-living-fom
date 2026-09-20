"""ModalFit co-refinement records, assistant conversations, and a full-text index.

Three groups of change:

1. **ModalFit records** — ``fit_records`` / ``fit_layers`` / ``fit_datasets``.  A
   co-refinement is the only place this platform measures one quantity twice by
   independent physics (an XRR thickness and an SE thickness share no forward
   model), so it is stored as a measurement record rather than as an exported
   file.  ``fit_layers.free_parameters`` carries the load: a parameter held fixed
   during refinement is an input to the fit, and ``modalfit.promote`` refuses to
   turn one into a ``PropertyValue``.

2. **Assistant conversations** — ``chat_sessions`` / ``chat_messages``.
   ``chat_messages.evidence`` and ``tool_calls`` store the chunk ids, tool
   arguments, and results behind every answer, which is what makes "where did
   that number come from?" answerable months later.

3. **A lexical index on the corpus** — a functional GIN index over
   ``to_tsvector('english', document_chunks.text)``, Postgres only.  Embedding
   search is good at paraphrase and bad at rare exact tokens, and a synthesis
   corpus is mostly rare exact tokens ("TMA", "Nevot-Croce", "HfO2").
   ``rag_backend.hybrid`` runs both retrievers and fuses them by rank; without
   this index the lexical leg is a sequential scan.

As in 0002, every CHECK constraint below is hand-written — Alembic's autogenerate
does not detect them.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

#  JSONB on Postgres, JSON elsewhere — matching db.models.JSONType so the
#  migration and the ORM agree about column types on both backends.
JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

FTS_INDEX_NAME = "ix_document_chunks_fts"

_FIT_TECHNIQUES = ("SE", "SPR", "QCM", "XRR", "NR")
_CHAT_ROLES = ("user", "assistant", "tool")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    op.create_table(
        "fit_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("stack_id", sa.String(length=128), nullable=True),
        sa.Column("sample_id", sa.String(length=128), nullable=True),
        sa.Column("source_filename", sa.String(length=512), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("datafed_record_id", sa.String(length=128), nullable=True),
        sa.Column("techniques", JSON_TYPE, nullable=False),
        sa.Column("technique_weights", JSON_TYPE, nullable=True),
        sa.Column("algorithm", sa.String(length=32), nullable=True),
        sa.Column("chi2_total", sa.Float(), nullable=True),
        sa.Column("chi2_by_technique", JSON_TYPE, nullable=True),
        sa.Column("n_free_parameters", sa.Integer(), nullable=True),
        sa.Column("resolution_smearing_applied", sa.Boolean(), nullable=True),
        sa.Column("roughness_applied_to_spr", sa.Boolean(), nullable=True),
        sa.Column("uses_placeholder_optical_constants", sa.Boolean(), nullable=False),
        sa.Column("technique_settings", JSON_TYPE, nullable=True),
        sa.Column("raw_model", JSON_TYPE, nullable=True),
        sa.Column("fitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("operator", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("material_id", sa.Integer(), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["material_id"], ["materials.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_sha256", name="uq_fit_record_content"),
        sa.CheckConstraint(
            "n_free_parameters IS NULL OR n_free_parameters >= 0",
            name="ck_fit_free_parameters_nonneg",
        ),
        sa.CheckConstraint("chi2_total IS NULL OR chi2_total >= 0", name="ck_fit_chi2_nonneg"),
    )
    with op.batch_alter_table("fit_records", schema=None) as batch_op:
        batch_op.create_index("ix_fit_records_sample", ["sample_id"], unique=False)
        batch_op.create_index("ix_fit_records_stack", ["stack_id"], unique=False)

    op.create_table(
        "fit_layers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fit_record_id", sa.Integer(), nullable=False),
        sa.Column("layer_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("material", sa.String(length=128), nullable=True),
        sa.Column("formula", sa.String(length=128), nullable=True),
        sa.Column("thickness_ang", sa.Float(), nullable=True),
        sa.Column("roughness_ang", sa.Float(), nullable=True),
        sa.Column("density_g_cm3", sa.Float(), nullable=True),
        sa.Column("parameters", JSON_TYPE, nullable=True),
        sa.Column("free_parameters", JSON_TYPE, nullable=True),
        sa.Column("bounds", JSON_TYPE, nullable=True),
        sa.Column("uncertainties", JSON_TYPE, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fit_record_id"], ["fit_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fit_record_id", "layer_index", name="uq_fit_layer_position"),
        sa.CheckConstraint("layer_index >= 0", name="ck_fit_layer_index_nonneg"),
        sa.CheckConstraint(
            _in_list("role", ("ambient", "layer", "substrate")), name="ck_fit_layer_role"
        ),
        sa.CheckConstraint(
            "thickness_ang IS NULL OR thickness_ang >= 0", name="ck_fit_layer_thickness_nonneg"
        ),
        sa.CheckConstraint(
            "roughness_ang IS NULL OR roughness_ang >= 0", name="ck_fit_layer_roughness_nonneg"
        ),
        sa.CheckConstraint(
            "density_g_cm3 IS NULL OR density_g_cm3 > 0", name="ck_fit_layer_density_positive"
        ),
    )
    with op.batch_alter_table("fit_layers", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_fit_layers_fit_record_id"), ["fit_record_id"], unique=False)
        batch_op.create_index("ix_fit_layers_label", ["label"], unique=False)

    op.create_table(
        "fit_datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fit_record_id", sa.Integer(), nullable=False),
        sa.Column(
            "technique",
            sa.Enum(*_FIT_TECHNIQUES, name="fit_technique", native_enum=False, length=8),
            nullable=False,
        ),
        sa.Column("source_filename", sa.String(length=512), nullable=True),
        sa.Column("loader", sa.String(length=64), nullable=True),
        sa.Column("datafed_record_id", sa.String(length=128), nullable=True),
        sa.Column("n_points", sa.Integer(), nullable=True),
        sa.Column("x_min", sa.Float(), nullable=True),
        sa.Column("x_max", sa.Float(), nullable=True),
        sa.Column("x_units", sa.String(length=32), nullable=True),
        sa.Column("chi2", sa.Float(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("settings", JSON_TYPE, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fit_record_id"], ["fit_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fit_record_id", "technique", name="uq_fit_dataset_technique"),
        sa.CheckConstraint("n_points IS NULL OR n_points >= 0", name="ck_fit_dataset_points_nonneg"),
        sa.CheckConstraint("chi2 IS NULL OR chi2 >= 0", name="ck_fit_dataset_chi2_nonneg"),
        sa.CheckConstraint("weight IS NULL OR weight >= 0", name="ck_fit_dataset_weight_nonneg"),
    )
    with op.batch_alter_table("fit_datasets", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_fit_datasets_fit_record_id"), ["fit_record_id"], unique=False
        )
        batch_op.create_index("ix_fit_datasets_technique", ["technique"], unique=False)

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_key", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=True),
        sa.Column("user", sa.String(length=128), nullable=True),
        sa.Column("sample_id", sa.String(length=128), nullable=True),
        sa.Column("techniques", JSON_TYPE, nullable=True),
        sa.Column("chat_model", sa.String(length=64), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_key"),
    )
    with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
        batch_op.create_index("ix_chat_sessions_created", ["created_at"], unique=False)
        batch_op.create_index("ix_chat_sessions_sample", ["sample_id"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            sa.Enum(*_CHAT_ROLES, name="chat_role", native_enum=False, length=16),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", JSON_TYPE, nullable=True),
        sa.Column("evidence", JSON_TYPE, nullable=True),
        sa.Column("insufficient_context", sa.Boolean(), nullable=False),
        sa.Column("chat_model", sa.String(length=64), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "turn_index", name="uq_chat_message_turn"),
        sa.CheckConstraint("turn_index >= 0", name="ck_chat_turn_nonneg"),
    )
    with op.batch_alter_table("chat_messages", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_chat_messages_session_id"), ["session_id"], unique=False)

    #  Functional GIN index for the lexical retriever. Postgres only: SQLite has
    #  no tsvector, and rag_backend.hybrid falls back to term overlap in Python
    #  there. Created with raw SQL because Alembic cannot express a functional
    #  index over an expression like this.
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {FTS_INDEX_NAME} ON document_chunks "
            "USING gin (to_tsvector('english', text))"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(f"DROP INDEX IF EXISTS {FTS_INDEX_NAME}")

    with op.batch_alter_table("chat_messages", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_chat_messages_session_id"))
    op.drop_table("chat_messages")

    with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
        batch_op.drop_index("ix_chat_sessions_sample")
        batch_op.drop_index("ix_chat_sessions_created")
    op.drop_table("chat_sessions")

    with op.batch_alter_table("fit_datasets", schema=None) as batch_op:
        batch_op.drop_index("ix_fit_datasets_technique")
        batch_op.drop_index(batch_op.f("ix_fit_datasets_fit_record_id"))
    op.drop_table("fit_datasets")

    with op.batch_alter_table("fit_layers", schema=None) as batch_op:
        batch_op.drop_index("ix_fit_layers_label")
        batch_op.drop_index(batch_op.f("ix_fit_layers_fit_record_id"))
    op.drop_table("fit_layers")

    with op.batch_alter_table("fit_records", schema=None) as batch_op:
        batch_op.drop_index("ix_fit_records_stack")
        batch_op.drop_index("ix_fit_records_sample")
    op.drop_table("fit_records")
