"""pySEA: electron microscopy records, signals and derived scalars.

pySEA (Walker, Pfeifer, Lupini, Hachtel, Pantelides, Hoglund, M&M 2026) closes the
loop from instrument configuration through scattering simulation to analysis. This
platform starts one step later, at property to FOM to optimizer. These tables are
the seam.

Three tables rather than one, for the same reason ``fit_records`` splits from
``fit_layers``: a record has many signals, a signal supports many derived numbers,
and only the derived numbers are candidates for the analysis tables. Flattening
them would make it impossible to say which signal a promoted value rests on.

**No bulk arrays.** A 4D-STEM scan is gigabytes. What lands in Postgres is shape,
axes, calibration and a locator; the array stays with pySEA and DataFed. The
question these tables answer is "which records determine this quantity, under what
microscope state", not "what did the detector read".

Two constraints carry protocol weight rather than hygiene:

``ck_pysea_simulation_metadata_present``
    A record calling itself a simulation must name what produced it. Otherwise a
    multislice spectrum with no code, no potential and no supercell sits in the
    table looking exactly like a measurement.

``ck_pysea_promoted_has_uncertainty``
    A promoted number with no uncertainty cannot be cross-checked against a second
    determination, and cross-platform comparison is what this integration is for.

The contract version is stored per row because ``pysea-canonical/0.1`` is **our**
mapping, written from the published abstracts without sight of pySEA's container
format. When the real schema arrives, every row written under a provisional mapping
has to be findable.

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

#  Matches models.JSONType: JSONB on Postgres, JSON elsewhere.
JSONType = sa.JSON().with_variant(JSONB, "postgresql")

RECORD_KIND = ("experimental", "simulation", "hybrid")
DERIVATION = ("measured", "fitted", "calculated", "simulated")
VALIDATION_STATUS = ("valid", "invalid")
PROMOTION_STATUS = ("unexamined", "eligible", "refused", "promoted")


def _enum(values: tuple[str, ...], name: str) -> sa.Enum:
    #  native_enum=False keeps these as VARCHAR + CHECK, matching the rest of the
    #  schema. A native Postgres enum cannot have a value removed without a type
    #  rewrite, and this vocabulary is provisional until pySEA's format is known.
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def upgrade() -> None:
    op.create_table(
        "pysea_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("record_id", sa.String(length=128), nullable=False),
        sa.Column("contract_version", sa.String(length=32), nullable=False),
        sa.Column("record_kind", _enum(RECORD_KIND, "pysea_record_kind"), nullable=False),
        sa.Column("sample_id", sa.String(length=128), nullable=False),
        sa.Column("material_id", sa.Integer(), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("instrument_id", sa.String(length=64), nullable=True),
        sa.Column("proposal_id", sa.String(length=64), nullable=True),
        sa.Column("operator", sa.String(length=128), nullable=True),
        sa.Column("datafed_record_id", sa.String(length=128), nullable=True),
        sa.Column("source_filename", sa.String(length=512), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("raw_metadata", JSONType, nullable=False),
        sa.Column("instrument_state", JSONType, nullable=True),
        sa.Column("calibrations", JSONType, nullable=True),
        sa.Column("simulation", JSONType, nullable=True),
        sa.Column("software", JSONType, nullable=True),
        sa.Column(
            "validation_status",
            _enum(VALIDATION_STATUS, "pysea_validation_status"),
            nullable=False,
        ),
        sa.Column("validation_issues", JSONType, nullable=True),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["material_id"], ["materials.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_sha256", name="uq_pysea_record_content"),
        sa.CheckConstraint(
            "record_kind <> 'simulation' OR simulation IS NOT NULL",
            name="ck_pysea_simulation_metadata_present",
        ),
    )
    op.create_index("ix_pysea_records_record_id", "pysea_records", ["record_id"])
    op.create_index("ix_pysea_records_sample", "pysea_records", ["sample_id"])
    op.create_index("ix_pysea_records_datafed", "pysea_records", ["datafed_record_id"])
    op.create_index("ix_pysea_records_kind", "pysea_records", ["record_kind"])

    op.create_table(
        "pysea_signals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pysea_record_id", sa.Integer(), nullable=False),
        sa.Column("signal_id", sa.String(length=128), nullable=False),
        sa.Column("signal_type", sa.String(length=64), nullable=True),
        sa.Column("shape", JSONType, nullable=False),
        sa.Column("dtype", sa.String(length=32), nullable=True),
        sa.Column("units", sa.String(length=32), nullable=True),
        sa.Column("axes", JSONType, nullable=False),
        sa.Column("calibration", JSONType, nullable=True),
        sa.Column("data_ref", JSONType, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["pysea_record_id"], ["pysea_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pysea_record_id", "signal_id", name="uq_pysea_signal_id"),
    )
    op.create_index("ix_pysea_signals_record", "pysea_signals", ["pysea_record_id"])

    op.create_table(
        "pysea_derived_scalars",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pysea_record_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("property_key", sa.String(length=64), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("uncertainty", sa.Float(), nullable=True),
        sa.Column("units", sa.String(length=32), nullable=True),
        sa.Column("derivation", _enum(DERIVATION, "pysea_derivation"), nullable=False),
        sa.Column("source_signal_ids", JSONType, nullable=False),
        sa.Column("method", sa.Text(), nullable=True),
        sa.Column("context", JSONType, nullable=True),
        sa.Column(
            "promotion_status",
            _enum(PROMOTION_STATUS, "pysea_promotion_status"),
            nullable=False,
            server_default="unexamined",
        ),
        sa.Column("refusal_reasons", JSONType, nullable=True),
        sa.Column("promoted_property_value_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["pysea_record_id"], ["pysea_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["promoted_property_value_id"], ["property_values.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "promotion_status <> 'promoted' OR uncertainty IS NOT NULL",
            name="ck_pysea_promoted_has_uncertainty",
        ),
        sa.CheckConstraint(
            "promotion_status <> 'promoted' OR promoted_property_value_id IS NOT NULL",
            name="ck_pysea_promoted_links_value",
        ),
    )
    op.create_index("ix_pysea_scalars_record", "pysea_derived_scalars", ["pysea_record_id"])
    op.create_index("ix_pysea_scalars_property", "pysea_derived_scalars", ["property_key"])


def downgrade() -> None:
    op.drop_index("ix_pysea_scalars_property", table_name="pysea_derived_scalars")
    op.drop_index("ix_pysea_scalars_record", table_name="pysea_derived_scalars")
    op.drop_table("pysea_derived_scalars")

    op.drop_index("ix_pysea_signals_record", table_name="pysea_signals")
    op.drop_table("pysea_signals")

    op.drop_index("ix_pysea_records_kind", table_name="pysea_records")
    op.drop_index("ix_pysea_records_datafed", table_name="pysea_records")
    op.drop_index("ix_pysea_records_sample", table_name="pysea_records")
    op.drop_index("ix_pysea_records_record_id", table_name="pysea_records")
    op.drop_table("pysea_records")
