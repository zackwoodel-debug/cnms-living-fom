"""Knowledge cards: a Dynamic Knowledge Repository over the corpus.

Retrieval alone re-derives an answer on every question and keeps nothing.  These
two tables do the integration work at ingest time instead — concept pages, source
summaries, and typed edges between them — so the second time a question is asked
it starts from the first answer's conclusions rather than from raw chunks.

Typed edges rather than plain links, because a graph that only knows *that* two
pages are related cannot answer "what does this rest on?".  ``contradicts`` is the
relation this platform most needs: an unresolved disagreement between two sources
is a finding, and a flat link would bury it.

The review gate is what makes an LLM-written page safe to keep.  A card is where a
model's synthesis gets written down, and FOM_PROOF Sec. 15.2 says a synthesis is
not evidence — so ``ck_card_reviewed_has_reviewer`` makes "reviewed by nobody"
unrepresentable at the storage layer, which is precisely the state that would let
model output pass as checked.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

_CARD_TYPES = ("concept", "source", "method", "finding", "question")
_CARD_STATUSES = ("proposed", "reviewed", "superseded")
_CARD_RELATIONS = ("fed_by", "relates_to", "depends_on", "contradicts", "measured_by", "answers")


def upgrade() -> None:
    op.create_table(
        "knowledge_cards",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=256), nullable=False),
        sa.Column(
            "card_type",
            sa.Enum(*_CARD_TYPES, name="card_type", native_enum=False, length=16),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(*_CARD_STATUSES, name="card_status", native_enum=False, length=16),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("tags", JSON_TYPE, nullable=True),
        sa.Column("sources", JSON_TYPE, nullable=True),
        sa.Column("authored_by", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by", sa.String(length=128), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_body_sha256", sa.String(length=64), nullable=True),
        sa.Column("supersedes_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["supersedes_id"], ["knowledge_cards.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_card_slug"),
        #  Sec. 15.2 as a constraint: a reviewed card has a named reviewer.
        sa.CheckConstraint(
            "status <> 'reviewed' OR reviewed_by IS NOT NULL",
            name="ck_card_reviewed_has_reviewer",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_card_confidence_unit_interval",
        ),
        sa.CheckConstraint("length(trim(title)) > 0", name="ck_card_title_not_blank"),
        sa.CheckConstraint("length(trim(slug)) > 0", name="ck_card_slug_not_blank"),
    )
    with op.batch_alter_table("knowledge_cards", schema=None) as batch_op:
        batch_op.create_index("ix_cards_type_status", ["card_type", "status"], unique=False)
        batch_op.create_index("ix_cards_updated", ["updated_at"], unique=False)

    op.create_table(
        "card_links",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("from_card_id", sa.Integer(), nullable=False),
        sa.Column("to_card_id", sa.Integer(), nullable=False),
        sa.Column(
            "relation",
            sa.Enum(*_CARD_RELATIONS, name="card_relation", native_enum=False, length=16),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["from_card_id"], ["knowledge_cards.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_card_id"], ["knowledge_cards.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("from_card_id", "to_card_id", "relation", name="uq_card_link"),
        #  A self-link is a data-entry slip, and it makes every traversal a
        #  special case.
        sa.CheckConstraint("from_card_id <> to_card_id", name="ck_card_link_not_self"),
    )
    with op.batch_alter_table("card_links", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_card_links_from_card_id"), ["from_card_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_card_links_to_card_id"), ["to_card_id"], unique=False)
        batch_op.create_index("ix_card_links_relation", ["relation"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("card_links", schema=None) as batch_op:
        batch_op.drop_index("ix_card_links_relation")
        batch_op.drop_index(batch_op.f("ix_card_links_to_card_id"))
        batch_op.drop_index(batch_op.f("ix_card_links_from_card_id"))
    op.drop_table("card_links")

    with op.batch_alter_table("knowledge_cards", schema=None) as batch_op:
        batch_op.drop_index("ix_cards_updated")
        batch_op.drop_index("ix_cards_type_status")
    op.drop_table("knowledge_cards")
