"""The research loop: briefs, extracted claims, campaign context proposals.

Three tables holding the *evidence layer*, plus one column on knowledge cards.

Nothing here is a measurement, and the schema is arranged so that claiming
otherwise would need a new column rather than a set one: ``research_claims`` has no
``provenance_tier``, no ``context_digest``, and no route to ``fom_scores``. Its
tier column is ``claim_tier`` — what a *source* said about its own number — and
sharing ``provenance_tier`` would have let a literature value inherit MEASURED on a
type coercion, which is the confusion FOM_PROOF Sec. 2.2 exists to prevent.

The constraints carry most of the safety:

  ck_claim_has_a_source          every claim points at a document or a content hash
  ck_claim_has_a_quote           and at the exact text it rests on (Sec. 2.2)
  ck_claim_normalisation_explained   a converted number records its conversion
  ck_brief_reviewed_has_reviewer     reviewing needs a named person (Sec. 15.2)
  ck_context_reviewed_has_reviewer   so does reviewing a campaign change
  ck_context_applied_has_applier     and applying needs an applier as well, so a
  ck_context_applied_has_timestamp   proposal cannot walk itself into a campaign

``knowledge_cards.category`` is a closed set rather than a tag because the BO
context bridge selects cards by it, and a free-text tag would make that selection
unenforceable at the storage layer.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

_BRIEF_STATUS = ("proposed", "reviewed", "rejected", "stale")
_CLAIM_TIER = ("reported", "measured", "calculated", "modeled", "fitted", "unknown")
_CLAIM_STATUS = ("candidate", "corroborated", "disputed", "superseded", "rejected")
_CONTEXT_STATUS = ("proposed", "reviewed", "rejected", "applied", "superseded", "stale")
_CARD_CATEGORY = (
    "process_window",
    "property_prior",
    "measurement_caveat",
    "optimization_constraint",
    "hypothesis",
    "contradiction",
    "experiment_summary",
)


def upgrade() -> None:
    op.create_table(
        "research_briefs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("research_question", sa.Text(), nullable=False),
        sa.Column("bo_run_id", sa.Integer(), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("material_id", sa.Integer(), nullable=True),
        sa.Column("material", sa.String(length=128), nullable=True),
        sa.Column("specimen_form", sa.String(length=64), nullable=True),
        sa.Column("target_property", sa.String(length=64), nullable=True),
        sa.Column("fom_definition", sa.String(length=128), nullable=True),
        sa.Column(
            "status",
            sa.Enum(*_BRIEF_STATUS, name="brief_status", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("evidence", JSON_TYPE, nullable=True),
        sa.Column("contradictions", JSON_TYPE, nullable=True),
        sa.Column("data_gaps", JSON_TYPE, nullable=True),
        sa.Column("statements", JSON_TYPE, nullable=True),
        sa.Column("proposed_actions", JSON_TYPE, nullable=True),
        sa.Column("proposed_card_slugs", JSON_TYPE, nullable=True),
        sa.Column("warnings", JSON_TYPE, nullable=True),
        sa.Column("tool_calls", JSON_TYPE, nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("policy_version", sa.String(length=64), nullable=True),
        sa.Column("fingerprint", sa.String(length=64), nullable=True),
        sa.Column("reviewed_by", sa.String(length=128), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["bo_run_id"], ["bo_runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["material_id"], ["materials.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("length(trim(research_question)) > 0", name="ck_brief_has_question"),
        sa.CheckConstraint(
            "status <> 'reviewed' OR reviewed_by IS NOT NULL",
            name="ck_brief_reviewed_has_reviewer",
        ),
    )
    with op.batch_alter_table("research_briefs", schema=None) as batch_op:
        batch_op.create_index("ix_briefs_run_created", ["bo_run_id", "created_at"], unique=False)
        batch_op.create_index("ix_briefs_status", ["status"], unique=False)
        batch_op.create_index("ix_briefs_fingerprint", ["fingerprint"], unique=False)

    op.create_table(
        "research_claims",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brief_id", sa.Integer(), nullable=True),
        sa.Column("field_name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("units", sa.String(length=32), nullable=True),
        sa.Column("value_text", sa.Text(), nullable=True),
        sa.Column("normalized_value", sa.Float(), nullable=True),
        sa.Column("normalized_units", sa.String(length=32), nullable=True),
        sa.Column("normalization_note", sa.Text(), nullable=True),
        sa.Column(
            "tier",
            sa.Enum(*_CLAIM_TIER, name="claim_tier", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(*_CLAIM_STATUS, name="claim_status", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("context", JSON_TYPE, nullable=True),
        sa.Column("missing_context", JSON_TYPE, nullable=True),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
        sa.Column("document_title", sa.String(length=512), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("chunk_id", sa.Integer(), nullable=True),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("doi", sa.String(length=256), nullable=True),
        sa.Column("evidence", JSON_TYPE, nullable=True),
        sa.Column("extracted_by_model", sa.String(length=64), nullable=True),
        sa.Column("extracted_by_provider", sa.String(length=32), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("model_confidence", sa.Float(), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["brief_id"], ["research_briefs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "document_id IS NOT NULL OR content_sha256 IS NOT NULL",
            name="ck_claim_has_a_source",
        ),
        sa.CheckConstraint("length(trim(quote)) > 0", name="ck_claim_has_a_quote"),
        sa.CheckConstraint(
            "model_confidence IS NULL OR (model_confidence >= 0 AND model_confidence <= 1)",
            name="ck_claim_confidence_unit_interval",
        ),
        sa.CheckConstraint(
            "normalized_value IS NULL OR length(trim(normalization_note)) > 0",
            name="ck_claim_normalisation_explained",
        ),
    )
    with op.batch_alter_table("research_claims", schema=None) as batch_op:
        batch_op.create_index("ix_claims_field", ["field_name"], unique=False)
        batch_op.create_index("ix_claims_brief", ["brief_id"], unique=False)
        batch_op.create_index("ix_claims_document", ["document_id"], unique=False)

    op.create_table(
        "campaign_context_proposals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("bo_run_id", sa.Integer(), nullable=False),
        sa.Column("brief_id", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(*_CONTEXT_STATUS, name="context_status", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("recommended_bounds", JSON_TYPE, nullable=True),
        sa.Column("excluded_choices", JSON_TYPE, nullable=True),
        sa.Column("soft_priors", JSON_TYPE, nullable=True),
        sa.Column("process_window_hints", JSON_TYPE, nullable=True),
        sa.Column("uncertainty_notes", JSON_TYPE, nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("supporting_cards", JSON_TYPE, nullable=True),
        sa.Column("proposed_by", sa.String(length=128), nullable=False),
        sa.Column("reviewed_by", sa.String(length=128), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("applied_by", sa.String(length=128), nullable=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("campaign_fingerprint_before", sa.String(length=64), nullable=True),
        sa.Column("campaign_fingerprint_after", sa.String(length=64), nullable=True),
        sa.Column("constraints_before", JSON_TYPE, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["bo_run_id"], ["bo_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["brief_id"], ["research_briefs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status NOT IN ('reviewed', 'applied') OR reviewed_by IS NOT NULL",
            name="ck_context_reviewed_has_reviewer",
        ),
        sa.CheckConstraint(
            "status <> 'applied' OR applied_by IS NOT NULL",
            name="ck_context_applied_has_applier",
        ),
        sa.CheckConstraint(
            "status <> 'applied' OR applied_at IS NOT NULL",
            name="ck_context_applied_has_timestamp",
        ),
    )
    with op.batch_alter_table("campaign_context_proposals", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_campaign_context_proposals_bo_run_id"), ["bo_run_id"], unique=False)
        batch_op.create_index("ix_context_run_status", ["bo_run_id", "status"], unique=False)

    #  Added nullable: existing cards predate the concept, and guessing a category
    #  for them would be inventing metadata.
    with op.batch_alter_table("knowledge_cards", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "category",
                sa.Enum(*_CARD_CATEGORY, name="card_category", native_enum=False, length=32),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("knowledge_cards", schema=None) as batch_op:
        batch_op.drop_column("category")

    with op.batch_alter_table("campaign_context_proposals", schema=None) as batch_op:
        batch_op.drop_index("ix_context_run_status")
        batch_op.drop_index(batch_op.f("ix_campaign_context_proposals_bo_run_id"))
    op.drop_table("campaign_context_proposals")

    with op.batch_alter_table("research_claims", schema=None) as batch_op:
        batch_op.drop_index("ix_claims_document")
        batch_op.drop_index("ix_claims_brief")
        batch_op.drop_index("ix_claims_field")
    op.drop_table("research_claims")

    with op.batch_alter_table("research_briefs", schema=None) as batch_op:
        batch_op.drop_index("ix_briefs_fingerprint")
        batch_op.drop_index("ix_briefs_status")
        batch_op.drop_index("ix_briefs_run_created")
    op.drop_table("research_briefs")
