"""A content-addressed cache for per-passage model calls.

Grading and extraction call a model once per passage and are together essentially
the entire cost of a brief: measured on this scaffold, 18 of 19 calls and about 21
minutes, against 1.2 seconds for the retrieval itself. Both are pure functions of
their input — extraction's prompt does not contain the question, so a passage needs
extracting once ever — which makes both exactly cacheable.

The key is a **hash of the content that was sent**, not a chunk id. A re-ingest that
renumbers chunks therefore cannot serve a stale answer, and an edited passage misses
automatically: correctness is a property of the key rather than of an invalidation
rule somebody has to remember.

One generic table rather than a typed one per stage, which is a deliberate departure
from the rest of this schema. Everything else here is typed and constrained because
the *protocol* depends on it. This table holds a transcript of model calls — no
claim, no measurement, nothing citable — and deleting it costs time and nothing
else.

``prompt_version`` defaults to the empty string rather than NULL so that the
uniqueness constraint actually constrains: in SQL, NULL != NULL, and two rows with a
null prompt version would both be insertable.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "llm_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("payload", JSON_TYPE, nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("kind", "cache_key", "model", "prompt_version", name="uq_llm_cache_entry"),
        sa.CheckConstraint("kind IN ('extraction', 'grade')", name="ck_llm_cache_kind"),
        sa.CheckConstraint("hit_count >= 0", name="ck_llm_cache_hits_nonneg"),
        sa.CheckConstraint("length(cache_key) = 64", name="ck_llm_cache_key_is_sha256"),
    )
    with op.batch_alter_table("llm_cache", schema=None) as batch_op:
        batch_op.create_index("ix_llm_cache_lookup", ["kind", "cache_key", "model"], unique=False)
        batch_op.create_index("ix_llm_cache_last_used", ["last_used_at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("llm_cache", schema=None) as batch_op:
        batch_op.drop_index("ix_llm_cache_last_used")
        batch_op.drop_index("ix_llm_cache_lookup")
    op.drop_table("llm_cache")
