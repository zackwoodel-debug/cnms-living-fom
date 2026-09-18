"""ORM schema.

Design rule, straight out of FOM_PROOF Sec. 2:

    A chemical formula is not a material identifier.

So the schema separates three things that a naive table would collapse:

  * ``Material``   — composition + polymorph + specimen form.  The identity of
    the thing.
  * ``PropertyValue`` — one measured/calculated number *with its own context*
    (temperature, frequency, direction, thickness, method, XC functional, DOI).
    Two dielectric constants for the same material at different frequencies are
    two rows, never an average.
  * ``FomScore``  — a derived number that is only meaningful alongside the
    versioned ``FomDefinition`` that produced it.

FOM_PROOF Eq. (3) defines the unit of analysis as the tuple (composition,
polymorph, specimen form, temperature, direction, method).  We store the first
three on ``Material`` and the last three on each ``PropertyValue``; an eligible
analysis row is then a ``Material`` joined to a *context-matched* set of
property values.  That filter lives in ``fom_engine.eligibility`` so the rule is
enforced in one auditable place rather than re-implemented per query.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, embedding_column_type
from .enums import (
    CorrelationBlock,
    HypothesisOutcome,
    ProvenanceTier,
    ScoreStatus,
    SpecimenForm,
    SynthesisTechnique,
    Transform,
)

# JSONB on Postgres, plain JSON elsewhere (keeps the test suite on SQLite).
JSONType = JSON().with_variant(JSONB, "postgresql")


def enum_column(enum_cls, name: str, length: int = 32) -> SAEnum:
    """A portable enum column storing the member *value*, not its Python name.

    Two deliberate choices, both of which matter for a platform whose point is
    auditability:

    ``values_callable``
        Without it SQLAlchemy stores the member *name*, so the database holds
        ``'MODELED'`` while the API, the JSON payloads, and the docs all say
        ``'modeled'``. Anyone writing ``WHERE provenance_tier = 'measured'`` by
        hand — in psql, DataGrip, or a BI tool — would silently get zero rows.
        Storing the value keeps one vocabulary everywhere.

    ``native_enum=False``
        Emits ``VARCHAR + CHECK`` on both Postgres and SQLite rather than a
        Postgres ``ENUM`` type. The constraint is just as strong, the schema is
        identical across backends, and adding a member later is an ordinary
        constraint change instead of ``ALTER TYPE`` — which matters because
        ``SpecimenForm`` and ``SynthesisTechnique`` will grow as CNMS work does.
    """
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        values_callable=lambda cls: [member.value for member in cls],
        length=length,
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )


# ---------------------------------------------------------------------------
# Materials and structures
# ---------------------------------------------------------------------------


class Material(Base, TimestampMixin):
    """A material-context record: composition + polymorph + specimen form."""

    __tablename__ = "materials"
    __table_args__ = (
        #  Eq. (3): identity is composition + polymorph + specimen form, never
        #  the formula alone.
        UniqueConstraint(
            "formula_reduced", "polymorph", "specimen_form", name="uq_material_identity"
        ),
        #  Sec. 2.1 at the storage layer: a blank polymorph would reintroduce
        #  exactly the formula-only identity the uniqueness constraint forbids.
        CheckConstraint("length(trim(polymorph)) > 0", name="ck_material_polymorph_not_blank"),
        Index("ix_materials_formula", "formula_reduced"),
        Index("ix_materials_form_polymorph", "specimen_form", "polymorph"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    formula: Mapped[str] = mapped_column(String(128), nullable=False)
    formula_reduced: Mapped[str] = mapped_column(String(128), nullable=False)
    #  "rutile", "anatase", "monoclinic-P21/c", "amorphous" ... never left blank.
    polymorph: Mapped[str] = mapped_column(String(128), nullable=False)
    space_group_symbol: Mapped[str | None] = mapped_column(String(32))
    space_group_number: Mapped[int | None] = mapped_column(Integer)
    specimen_form: Mapped[SpecimenForm] = mapped_column(
        enum_column(SpecimenForm, "specimen_form"), nullable=False
    )

    # Where this record came from (Materials Project id, ICSD collection code,
    # internal CNMS sample id, ...).
    source_database: Mapped[str | None] = mapped_column(String(64))
    source_identifier: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)

    structures: Mapped[list[StructureRecord]] = relationship(
        back_populates="material", cascade="all, delete-orphan"
    )
    descriptors: Mapped[list[DescriptorValue]] = relationship(
        back_populates="material", cascade="all, delete-orphan"
    )
    properties: Mapped[list[PropertyValue]] = relationship(
        back_populates="material", cascade="all, delete-orphan"
    )
    scores: Mapped[list[FomScore]] = relationship(
        back_populates="material", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Material {self.formula_reduced} / {self.polymorph} / {self.specimen_form.value}>"


class StructureRecord(Base, TimestampMixin):
    """A concrete atomic structure (CIF) that descriptors are computed from."""

    __tablename__ = "structures"
    __table_args__ = (
        Index("ix_structures_material", "material_id"),
        CheckConstraint("volume_ang3 IS NULL OR volume_ang3 > 0", name="ck_structure_volume_positive"),
        CheckConstraint(
            "formula_units_per_cell IS NULL OR formula_units_per_cell >= 1",
            name="ck_structure_z_positive",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    material_id: Mapped[int] = mapped_column(
        ForeignKey("materials.id", ondelete="CASCADE"), index=True
    )

    cif: Mapped[str] = mapped_column(Text, nullable=False)
    n_sites: Mapped[int | None] = mapped_column(Integer)
    formula_units_per_cell: Mapped[int | None] = mapped_column(Integer)
    volume_ang3: Mapped[float | None] = mapped_column(Float)

    # Provenance of the structure itself, not of any property measured on it.
    method: Mapped[str | None] = mapped_column(String(128))       # e.g. "DFT relaxation", "XRD refinement"
    xc_functional: Mapped[str | None] = mapped_column(String(64))  # e.g. "PBEsol"
    provenance_tier: Mapped[ProvenanceTier] = mapped_column(
        enum_column(ProvenanceTier, "provenance_tier"), nullable=False
    )
    doi: Mapped[str | None] = mapped_column(String(256))

    material: Mapped[Material] = relationship(back_populates="structures")


# ---------------------------------------------------------------------------
# Descriptors (S) and properties (P)
# ---------------------------------------------------------------------------


class DescriptorValue(Base, TimestampMixin):
    """One structural descriptor S_j for one material (FOM_PROOF Eq. 5-6)."""

    __tablename__ = "descriptor_values"
    __table_args__ = (
        UniqueConstraint("material_id", "descriptor_key", "method", name="uq_descriptor_value"),
        Index("ix_descriptor_key", "descriptor_key"),
        #  The hot path in eligibility.build_analysis_table: pull every
        #  descriptor for one material and bucket by key.
        Index("ix_descriptor_material_key", "material_id", "descriptor_key"),
        CheckConstraint(
            "uncertainty IS NULL OR uncertainty >= 0", name="ck_descriptor_uncertainty_nonneg"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    material_id: Mapped[int] = mapped_column(
        ForeignKey("materials.id", ondelete="CASCADE"), index=True
    )
    structure_id: Mapped[int | None] = mapped_column(ForeignKey("structures.id", ondelete="SET NULL"))

    # Key into descriptors.registry.STRUCTURAL_DESCRIPTORS — e.g. "V_fu", "omega_TO_min".
    descriptor_key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)  # NULL means NA (Eq. 4) — never impute
    units: Mapped[str | None] = mapped_column(String(32))
    uncertainty: Mapped[float | None] = mapped_column(Float)

    method: Mapped[str | None] = mapped_column(String(128))
    xc_functional: Mapped[str | None] = mapped_column(String(64))
    provenance_tier: Mapped[ProvenanceTier] = mapped_column(
        enum_column(ProvenanceTier, "provenance_tier"), nullable=False
    )
    # For tensor-derived scalars: which reduction rule was declared (Sec. 3.2).
    reduction_rule: Mapped[str | None] = mapped_column(String(64))
    direction_hint: Mapped[list[float] | None] = mapped_column(JSONType)

    material: Mapped[Material] = relationship(back_populates="descriptors")


class PropertyValue(Base, TimestampMixin):
    """One physical property P_q with its full measurement/calculation context.

    Every field in FOM_PROOF Table 1 has a home here.  A value whose context is
    unknown is not a usable value: leave it out rather than guessing.
    """

    __tablename__ = "property_values"
    __table_args__ = (
        #  Sec. 2.1: the same source reporting the same quantity for the same
        #  material under the same context twice is a duplicate, not two
        #  measurements.  See db/context.py for what "same context" means and
        #  why source identity is part of it.
        UniqueConstraint(
            "material_id", "property_key", "context_digest", name="uq_property_value_context"
        ),
        Index("ix_property_key", "property_key"),
        #  build_analysis_table's dominant access pattern.
        Index("ix_property_material_key", "material_id", "property_key"),
        #  Eligibility filters on tier before anything else.
        Index("ix_property_key_tier", "property_key", "provenance_tier"),
        #  Physically impossible values are a data-entry bug, not data.
        CheckConstraint(
            "uncertainty IS NULL OR uncertainty >= 0", name="ck_property_uncertainty_nonneg"
        ),
        CheckConstraint(
            "temperature_k IS NULL OR temperature_k > 0", name="ck_property_temperature_positive"
        ),
        CheckConstraint(
            "frequency_hz IS NULL OR frequency_hz >= 0", name="ck_property_frequency_nonneg"
        ),
        CheckConstraint(
            "thickness_nm IS NULL OR thickness_nm > 0", name="ck_property_thickness_positive"
        ),
        CheckConstraint("area_cm2 IS NULL OR area_cm2 > 0", name="ck_property_area_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    material_id: Mapped[int] = mapped_column(
        ForeignKey("materials.id", ondelete="CASCADE"), index=True
    )

    # Key into descriptors.registry.PHYSICAL_PROPERTIES — e.g. "k", "Eg", "tan_delta".
    property_key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)  # NULL == NA (Eq. 4)
    units: Mapped[str | None] = mapped_column(String(32))
    uncertainty: Mapped[float | None] = mapped_column(Float)

    # --- exact quantity (Table 1: "property definition") -------------------
    tensor_component: Mapped[str | None] = mapped_column(String(16))   # "xx", "zz", "iso", "perp"
    direction: Mapped[list[float] | None] = mapped_column(JSONType)    # n-hat for Eq. (8)
    reduction_rule: Mapped[str | None] = mapped_column(String(64))     # "trace/3", "n.eps.n", ...

    # --- experimental context (Table 1) ------------------------------------
    temperature_k: Mapped[float | None] = mapped_column(Float)
    frequency_hz: Mapped[float | None] = mapped_column(Float)
    field_amplitude_v_per_cm: Mapped[float | None] = mapped_column(Float)
    thickness_nm: Mapped[float | None] = mapped_column(Float)
    electrode: Mapped[str | None] = mapped_column(String(64))
    substrate: Mapped[str | None] = mapped_column(String(64))
    interface: Mapped[str | None] = mapped_column(String(128))
    area_cm2: Mapped[float | None] = mapped_column(Float)
    failure_criterion: Mapped[str | None] = mapped_column(String(128))  # required for E_bd
    processing_route: Mapped[str | None] = mapped_column(Text)

    # --- calculation context (Table 1) -------------------------------------
    method: Mapped[str | None] = mapped_column(String(128))
    software: Mapped[str | None] = mapped_column(String(64))
    xc_functional: Mapped[str | None] = mapped_column(String(64))
    pseudopotential: Mapped[str | None] = mapped_column(String(128))
    convergence: Mapped[str | None] = mapped_column(Text)

    # --- source provenance (Table 1) ---------------------------------------
    doi: Mapped[str | None] = mapped_column(String(256))
    source_url: Mapped[str | None] = mapped_column(String(512))
    database_identifier: Mapped[str | None] = mapped_column(String(128))
    source_locator: Mapped[str | None] = mapped_column(String(128))  # table/page
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    provenance_tier: Mapped[ProvenanceTier] = mapped_column(
        enum_column(ProvenanceTier, "provenance_tier"), nullable=False
    )

    #  Fingerprint of the context columns above, maintained by db/events.py.
    #  It exists so the database can enforce "same measurement" as a uniqueness
    #  constraint; ~20 nullable columns cannot be a composite key.
    context_digest: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # Link back to the experiment that produced it, when it is ours.
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL")
    )

    material: Mapped[Material] = relationship(back_populates="properties")
    experiment: Mapped[Experiment | None] = relationship(back_populates="property_values")


# ---------------------------------------------------------------------------
# Figures of merit
# ---------------------------------------------------------------------------


class FomDefinition(Base, TimestampMixin):
    """A *versioned* figure of merit — the "living" part of the platform.

    FOM_PROOF Sec. 5.3 and 6.2: the eligible material set, the min/max bounds,
    the transform, the direction correction, the numerical floor, and the
    weights are all part of the score definition.  Changing any of them creates
    a new version; it does not edit an existing one.  ``frozen`` marks a
    definition that scores have been published against.
    """

    __tablename__ = "fom_definitions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_fom_name_version"),
        CheckConstraint("version >= 1", name="ck_fom_version_positive"),
        #  Eq. (26): the floor is a small positive number, not a free parameter.
        CheckConstraint(
            "floor_eps > 0 AND floor_eps < 1", name="ck_fom_floor_eps_in_unit_interval"
        ),
        #  A definition cannot be approved by nobody (Sec. 6.2: weights are a
        #  policy choice and need a named owner).
        CheckConstraint(
            "NOT approved OR approved_by IS NOT NULL", name="ck_fom_approved_has_approver"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)      # "logic", "power", "rf"
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    application: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # {property_key: weight}; must sum to 1 (Eq. 29).  Enforced in fom_engine.
    weights: Mapped[dict] = mapped_column(JSONType, nullable=False)
    # {property_key: {transform, direction, lo, hi}} — Eqs. (23)-(25).
    normalization: Mapped[dict] = mapped_column(JSONType, nullable=False)
    floor_eps: Mapped[float] = mapped_column(Float, nullable=False, default=1e-3)  # Eq. (26)

    # How the bounds were fixed.  A definition whose bounds came from fewer than
    # a handful of materials is an artifact, not a ranking (Sec. 5.3).
    bounds_basis: Mapped[str | None] = mapped_column(Text)
    eligible_set_query: Mapped[str | None] = mapped_column(Text)
    n_materials_in_bounds: Mapped[int | None] = mapped_column(Integer)

    # Weights are policy, not measurement — so they need a named owner.
    approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String(128))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    frozen: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    scores: Mapped[list[FomScore]] = relationship(
        back_populates="definition", cascade="all, delete-orphan"
    )


class FomScore(Base, TimestampMixin):
    """F_a for one material under one FOM definition (Eq. 29)."""

    __tablename__ = "fom_scores"
    __table_args__ = (
        UniqueConstraint("material_id", "fom_definition_id", "run_id", name="uq_fom_score"),
        #  Sec. 6.2, enforced by the database rather than by convention: a
        #  not-scored material carries no score, and a scored one is not empty.
        #  This is the "no manufactured score" rule as a constraint.
        CheckConstraint(
            "status <> 'not_scored' OR value IS NULL", name="ck_score_not_scored_has_no_value"
        ),
        CheckConstraint(
            "status <> 'scored' OR value IS NOT NULL", name="ck_score_scored_has_value"
        ),
        #  Eq. (29) with z in (0, 1] and weights on a simplex: F cannot be
        #  negative, and log_value must be the log of value.
        CheckConstraint("value IS NULL OR value >= 0", name="ck_score_value_nonneg"),
        #  A modeled input forces ILLUSTRATIVE (Sec. 2.3); the converse would
        #  mean a score was labelled illustrative for no recorded reason.
        CheckConstraint(
            "NOT uses_modeled_inputs OR status = 'illustrative'",
            name="ck_score_modeled_is_illustrative",
        ),
        Index("ix_score_definition_status", "fom_definition_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    material_id: Mapped[int] = mapped_column(
        ForeignKey("materials.id", ondelete="CASCADE"), index=True
    )
    fom_definition_id: Mapped[int] = mapped_column(
        ForeignKey("fom_definitions.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_runs.id", ondelete="SET NULL"))

    value: Mapped[float | None] = mapped_column(Float)
    log_value: Mapped[float | None] = mapped_column(Float)
    status: Mapped[ScoreStatus] = mapped_column(
        enum_column(ScoreStatus, "score_status"), nullable=False
    )
    # Which required inputs were NA.  A non-empty list means status != SCORED.
    missing_inputs: Mapped[list | None] = mapped_column(JSONType)
    # Per-property {raw, transformed, normalized z, weight, ln z contribution}.
    components: Mapped[dict | None] = mapped_column(JSONType)
    # True when any input was MODELED — such a score is ILLUSTRATIVE only.
    uses_modeled_inputs: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    material: Mapped[Material] = relationship(back_populates="scores")
    definition: Mapped[FomDefinition] = relationship(back_populates="scores")


# ---------------------------------------------------------------------------
# Analysis runs and their outputs
# ---------------------------------------------------------------------------


class AnalysisRun(Base, TimestampMixin):
    """Provenance envelope for one computation (Sec. 16 reproducibility checklist)."""

    __tablename__ = "analysis_runs"
    __table_args__ = (
        Index("ix_runs_kind_created", "kind", "created_at"),
        #  Eq. (42): fewer than 10,000 permutations is allowed only as a
        #  documented exception, but zero or negative is always a bug.
        CheckConstraint(
            "permutation_b IS NULL OR permutation_b > 0", name="ck_run_permutation_positive"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # score|correlation|mediation|bo|rag
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    params: Mapped[dict | None] = mapped_column(JSONType)

    # Reproducibility: frozen eligible set, seed, permutation count, code version.
    eligible_material_ids: Mapped[list | None] = mapped_column(JSONType)
    random_seed: Mapped[int | None] = mapped_column(Integer)
    permutation_b: Mapped[int | None] = mapped_column(Integer)
    code_version: Mapped[str | None] = mapped_column(String(64))
    git_sha: Mapped[str | None] = mapped_column(String(64))

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    correlations: Mapped[list[CorrelationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    mediations: Mapped[list[MediationResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    sensitivities: Mapped[list[SensitivityEstimate]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    integrity_checks: Mapped[list[IntegrityCheck]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    exclusions: Mapped[list[AnalysisExclusion]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class CorrelationResult(Base, TimestampMixin):
    """One cell of a correlation block — every field of FOM_PROOF Table 6."""

    __tablename__ = "correlation_results"
    __table_args__ = (
        Index("ix_corr_run_block", "run_id", "block"),
        Index("ix_corr_pair", "x_key", "y_key"),
        #  Sec. 7.3: a cell always knows its own complete-case count.
        CheckConstraint("n_complete >= 0", name="ck_corr_n_nonneg"),
        CheckConstraint(
            "pearson_r IS NULL OR (pearson_r >= -1 AND pearson_r <= 1)", name="ck_corr_r_range"
        ),
        CheckConstraint(
            "spearman_rho IS NULL OR (spearman_rho >= -1 AND spearman_rho <= 1)",
            name="ck_corr_rho_range",
        ),
        #  Eq. (41) floors p at 1/(B+1), so a zero p-value means a bug.
        CheckConstraint(
            "p_permutation IS NULL OR (p_permutation > 0 AND p_permutation <= 1)",
            name="ck_corr_p_range",
        ),
        CheckConstraint(
            "q_fdr IS NULL OR (q_fdr >= 0 AND q_fdr <= 1)", name="ck_corr_q_range"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )

    block: Mapped[CorrelationBlock] = mapped_column(
        enum_column(CorrelationBlock, "correlation_block"), nullable=False
    )
    x_key: Mapped[str] = mapped_column(String(64), nullable=False)
    y_key: Mapped[str] = mapped_column(String(64), nullable=False)
    x_transform: Mapped[Transform] = mapped_column(
        enum_column(Transform, "transform"), nullable=False, default=Transform.NONE
    )
    y_transform: Mapped[Transform] = mapped_column(
        enum_column(Transform, "transform"), nullable=False, default=Transform.NONE
    )

    pearson_r: Mapped[float | None] = mapped_column(Float)
    spearman_rho: Mapped[float | None] = mapped_column(Float)
    # Sec. 7.3: per-cell complete-case count.  There is deliberately no run-level n.
    n_complete: Mapped[int] = mapped_column(Integer, nullable=False)
    p_permutation: Mapped[float | None] = mapped_column(Float)
    q_fdr: Mapped[float | None] = mapped_column(Float)

    predicted_sign: Mapped[str | None] = mapped_column(String(16))  # "+", "-", "test"
    mechanism: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[HypothesisOutcome | None] = mapped_column(
        enum_column(HypothesisOutcome, "hypothesis_outcome")
    )
    # Exactly which materials entered this cell (Table 6: "eligible material records").
    material_ids: Mapped[list | None] = mapped_column(JSONType)

    run: Mapped[AnalysisRun] = relationship(back_populates="correlations")


class MediationResult(Base, TimestampMixin):
    """M_ja = sum_q B_jq * Gamma_qa  (Eqs. 64-65) — the primary mechanistic result."""

    __tablename__ = "mediation_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "descriptor_key", "application", name="uq_mediation_descriptor_application"
        ),
        Index("ix_mediation_application", "application"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    fom_definition_id: Mapped[int | None] = mapped_column(
        ForeignKey("fom_definitions.id", ondelete="SET NULL")
    )

    descriptor_key: Mapped[str] = mapped_column(String(64), nullable=False)
    application: Mapped[str] = mapped_column(String(64), nullable=False)
    mediated_effect: Mapped[float] = mapped_column(Float, nullable=False)
    # {property_key: B_jq * Gamma_qa} — identifies the dominant channel (Eq. 65).
    contributions: Mapped[dict] = mapped_column(JSONType, nullable=False)
    dominant_property: Mapped[str | None] = mapped_column(String(64))
    # "regression" (empirical B) or "theory" (oscillator elasticities, Eqs. 50-53).
    sensitivity_source: Mapped[str | None] = mapped_column(String(32))

    run: Mapped[AnalysisRun] = relationship(back_populates="mediations")


# ---------------------------------------------------------------------------
# RAG corpus
# ---------------------------------------------------------------------------


class Document(Base, TimestampMixin):
    """A source PDF in the synthesis corpus (MBE / PLD / ALD / CNMS user docs)."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_technique", "technique"),
        CheckConstraint("n_pages IS NULL OR n_pages > 0", name="ck_document_pages_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    technique: Mapped[SynthesisTechnique] = mapped_column(
        enum_column(SynthesisTechnique, "synthesis_technique"), nullable=False
    )
    doi: Mapped[str | None] = mapped_column(String(256))
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    source_url: Mapped[str | None] = mapped_column(String(512))
    n_pages: Mapped[int | None] = mapped_column(Integer)
    doc_metadata: Mapped[dict | None] = mapped_column(JSONType)

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base, TimestampMixin):
    """An embedded passage.

    We own this table rather than delegating to a generic LangChain vector store
    because retrieval has to return a citable locator (document + page), not just
    text.  An answer the protocol can use is one you can trace to a page.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),
        CheckConstraint("chunk_index >= 0", name="ck_chunk_index_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    n_tokens: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None] = mapped_column(String(64))
    embedding = mapped_column(embedding_column_type(), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# CNMS experiments and instruments
# ---------------------------------------------------------------------------


class Instrument(Base, TimestampMixin):
    """A CNMS growth/characterization tool.

    TODO(CNMS): replace the seeded placeholders with the real instrument
    registry (tool IDs, capability envelopes, calibration state).
    """

    __tablename__ = "instruments"
    __table_args__ = (Index("ix_instruments_technique_available", "technique", "available"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    technique: Mapped[SynthesisTechnique] = mapped_column(
        enum_column(SynthesisTechnique, "synthesis_technique"), nullable=False
    )
    location: Mapped[str | None] = mapped_column(String(128))
    # {parameter: {min, max, units}} — the hard envelope the BO loop must respect.
    capabilities: Mapped[dict | None] = mapped_column(JSONType)
    available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    experiments: Mapped[list[Experiment]] = relationship(back_populates="instrument")


class Experiment(Base, TimestampMixin):
    """One growth/characterization run and the recipe that produced it."""

    __tablename__ = "experiments"
    __table_args__ = (
        Index("ix_experiments_instrument_status", "instrument_id", "status"),
        Index("ix_experiments_material", "material_id"),
        Index("ix_experiments_proposal", "proposal_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="SET NULL")
    )
    material_id: Mapped[int | None] = mapped_column(
        ForeignKey("materials.id", ondelete="SET NULL")
    )
    bo_suggestion_id: Mapped[int | None] = mapped_column(
        ForeignKey("bo_suggestions.id", ondelete="SET NULL")
    )

    # TODO(CNMS): bind to the real proposal/user-agreement system.
    proposal_id: Mapped[str | None] = mapped_column(String(64))
    operator: Mapped[str | None] = mapped_column(String(128))
    sample_id: Mapped[str | None] = mapped_column(String(64))

    # The growth recipe: {substrate_temp_c, o2_partial_pressure_torr, ...}
    recipe: Mapped[dict | None] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(16), default="planned", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument | None] = relationship(back_populates="experiments")
    property_values: Mapped[list[PropertyValue]] = relationship(back_populates="experiment")


# ---------------------------------------------------------------------------
# Bayesian optimization
# ---------------------------------------------------------------------------


class BoRun(Base, TimestampMixin):
    """A campaign: one objective, one search space, many suggestions."""

    __tablename__ = "bo_runs"
    __table_args__ = (
        Index("ix_bo_runs_status", "status"),
        CheckConstraint("objective_sense IN ('max', 'min')", name="ck_bo_run_objective_sense"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    fom_definition_id: Mapped[int | None] = mapped_column(
        ForeignKey("fom_definitions.id", ondelete="SET NULL")
    )
    # {param: {type, bounds/choices, units}} — see bo_engine.space.
    search_space: Mapped[dict] = mapped_column(JSONType, nullable=False)
    # Instrument envelope + safety limits applied on top of the space.
    constraints: Mapped[dict | None] = mapped_column(JSONType)
    acquisition: Mapped[str] = mapped_column(String(32), default="qLogEI", nullable=False)
    objective_sense: Mapped[str] = mapped_column(String(8), default="max", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    random_seed: Mapped[int | None] = mapped_column(Integer)

    observations: Mapped[list[BoObservation]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    suggestions: Mapped[list[BoSuggestion]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class BoObservation(Base, TimestampMixin):
    """An evaluated point: recipe in, objective out."""

    __tablename__ = "bo_observations"
    __table_args__ = (
        #  Every suggest() call reads exactly this slice: the feasible
        #  observations of one run.  Without the composite index it is a full
        #  scan of the run's history on every proposal.
        Index("ix_bo_obs_run_feasible", "bo_run_id", "is_feasible"),
        Index("ix_bo_obs_experiment", "experiment_id"),
        CheckConstraint(
            "objective_noise IS NULL OR objective_noise >= 0", name="ck_bo_obs_noise_nonneg"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    bo_run_id: Mapped[int] = mapped_column(ForeignKey("bo_runs.id", ondelete="CASCADE"), index=True)
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL")
    )

    parameters: Mapped[dict] = mapped_column(JSONType, nullable=False)
    objective_value: Mapped[float | None] = mapped_column(Float)
    objective_noise: Mapped[float | None] = mapped_column(Float)
    # Which FOM score this observation reports, when the objective is a FOM.
    fom_score_id: Mapped[int | None] = mapped_column(
        ForeignKey("fom_scores.id", ondelete="SET NULL")
    )
    is_feasible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    run: Mapped[BoRun] = relationship(back_populates="observations")


class BoSuggestion(Base, TimestampMixin):
    """A proposed next experiment, with the acquisition value that justified it."""

    __tablename__ = "bo_suggestions"
    __table_args__ = (
        Index("ix_bo_sugg_run_status", "bo_run_id", "status"),
        CheckConstraint("batch_index >= 0", name="ck_bo_sugg_batch_index_nonneg"),
        CheckConstraint(
            "predicted_std IS NULL OR predicted_std >= 0", name="ck_bo_sugg_std_nonneg"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    bo_run_id: Mapped[int] = mapped_column(ForeignKey("bo_runs.id", ondelete="CASCADE"), index=True)

    parameters: Mapped[dict] = mapped_column(JSONType, nullable=False)
    acquisition_value: Mapped[float | None] = mapped_column(Float)
    predicted_mean: Mapped[float | None] = mapped_column(Float)
    predicted_std: Mapped[float | None] = mapped_column(Float)
    batch_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="proposed", nullable=False)

    run: Mapped[BoRun] = relationship(back_populates="suggestions")


# ---------------------------------------------------------------------------
# Persisted analysis intermediates
#
# Sec. 16 item 11 requires transformations, bounds, and weights to be versioned,
# and Sec. 11.3 requires the integrity checks to be reproducible.  Computing
# these in memory and returning them over HTTP satisfies neither: a result you
# cannot re-read is not auditable.  The three tables below close that gap.
# ---------------------------------------------------------------------------


class SensitivityEstimate(Base, TimestampMixin):
    """One entry of the structure-to-property matrix B (FOM_PROOF Eq. 47).

    B is the empirical half of the mediated effect, and the half a reviewer will
    question.  Storing only the finished ``MediationResult`` hides which
    regression produced each coefficient, at what reference point, and with what
    collinearity — so the fit diagnostics live here alongside the number.
    """

    __tablename__ = "sensitivity_estimates"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "descriptor_key", "property_key", name="uq_sensitivity_cell"
        ),
        Index("ix_sensitivity_run", "run_id"),
        CheckConstraint(
            "source IN ('regression', 'theory')", name="ck_sensitivity_source"
        ),
        CheckConstraint("vif IS NULL OR vif >= 1", name="ck_sensitivity_vif_at_least_one"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )

    descriptor_key: Mapped[str] = mapped_column(String(64), nullable=False)
    property_key: Mapped[str] = mapped_column(String(64), nullable=False)

    #  dP_q / dS_j in natural units (Eq. 47).
    value: Mapped[float] = mapped_column(Float, nullable=False)
    #  d ln P_q / d ln S_j (Eq. 48) — dimensionless, so cells compare.
    elasticity: Mapped[float | None] = mapped_column(Float)
    #  "regression" (estimated from data) or "theory" (Eqs. 50-53 prior).
    #  They support different claims and must never be silently mixed.
    source: Mapped[str] = mapped_column(String(32), nullable=False)

    std_error: Mapped[float | None] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float)
    #  Sec. 9.2: a coefficient with a high VIF is not interpretable alone.
    vif: Mapped[float | None] = mapped_column(Float)
    n_complete: Mapped[int | None] = mapped_column(Integer)
    #  Elasticity -> natural-unit conversion is only valid at a stated point.
    reference_point: Mapped[dict | None] = mapped_column(JSONType)
    confounders: Mapped[list | None] = mapped_column(JSONType)
    notes: Mapped[str | None] = mapped_column(Text)

    run: Mapped[AnalysisRun] = relationship(back_populates="sensitivities")


class IntegrityCheck(Base, TimestampMixin):
    """Result of the Sec. 11.3 checks for one set of applications.

    Recorded whether it passes or fails.  A stored failure is the point: it is
    the evidence that a ranking was not released, and re-running until it passes
    without recording the failures would defeat the check.
    """

    __tablename__ = "integrity_checks"
    __table_args__ = (Index("ix_integrity_run", "run_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )

    applications: Mapped[list] = mapped_column(JSONType, nullable=False)
    n_materials: Mapped[int] = mapped_column(Integer, nullable=False)

    #  Row-major matrices over ``applications``.
    observed_correlation: Mapped[list | None] = mapped_column(JSONType)
    null_correlation: Mapped[list | None] = mapped_column(JSONType)   # Eq. (60)
    excess_correlation: Mapped[list | None] = mapped_column(JSONType)  # Eq. (61)

    #  Eq. (62): direct vs reconstructed R_FF.
    reconstruction_ok: Mapped[bool | None] = mapped_column(Boolean)
    max_reconstruction_error: Mapped[float | None] = mapped_column(Float)

    #  Eq. (63): leakage report, including per-application residual sd.
    leakage_passed: Mapped[bool | None] = mapped_column(Boolean)
    leakage_report: Mapped[dict | None] = mapped_column(JSONType)

    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failures: Mapped[list | None] = mapped_column(JSONType)

    run: Mapped[AnalysisRun] = relationship(back_populates="integrity_checks")


class AnalysisExclusion(Base, TimestampMixin):
    """Why one value did not enter an analysis table (FOM_PROOF Sec. 2.3).

    The protocol's headline rule is that missing data stays missing.  The
    corollary is that *why* a value is missing has to be recoverable: "excluded"
    and "never existed" are different findings, and only one of them is fixed by
    going back to the literature.  ``eligibility.build_analysis_table`` already
    produces these reasons; this table keeps them.
    """

    __tablename__ = "analysis_exclusions"
    __table_args__ = (
        Index("ix_exclusion_run_key", "run_id", "property_key"),
        Index("ix_exclusion_material", "material_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )

    #  "HfO2|monoclinic|crystalline_film" — the Eq. (3) identity as text, so the
    #  audit trail survives a material being deleted.
    material_key: Mapped[str] = mapped_column(String(256), nullable=False)
    material_id: Mapped[int | None] = mapped_column(
        ForeignKey("materials.id", ondelete="SET NULL")
    )
    property_key: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped[AnalysisRun] = relationship(back_populates="exclusions")


# ---------------------------------------------------------------------------
# Spectra
#
# omega_TO,min and S_osc (Table 2) are *derived from* an IR spectrum, and
# optical n(lambda) / k(lambda) is the most common form in which external
# databases publish dielectric information.  Without somewhere to put a curve,
# every such dataset arrives pre-reduced to a scalar by someone else, under an
# aggregation rule nobody recorded — exactly what Sec. 3.2 prohibits.
#
# Split into series + points so the measurement context is stored once per
# curve rather than once per point: the CNMS oxide import is ~300 series and
# ~125,000 points.
# ---------------------------------------------------------------------------


class SpectralSeries(Base, TimestampMixin):
    """One measured or computed curve for one material, with its context."""

    __tablename__ = "spectral_series"
    __table_args__ = (
        UniqueConstraint(
            "material_id", "quantity", "axis", "context_digest", name="uq_spectral_series"
        ),
        Index("ix_spectral_series_material_quantity", "material_id", "quantity"),
        CheckConstraint(
            "independent_variable IN ('wavelength_nm', 'energy_ev', 'frequency_hz', 'wavenumber_cm-1')",
            name="ck_spectral_independent_variable",
        ),
        CheckConstraint("n_points >= 0", name="ck_spectral_n_points_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    material_id: Mapped[int] = mapped_column(
        ForeignKey("materials.id", ondelete="CASCADE"), index=True
    )

    #  "n", "k", "eps_real", "eps_imag", "reflectance", "absorption_coefficient"
    quantity: Mapped[str] = mapped_column(String(32), nullable=False)
    units: Mapped[str | None] = mapped_column(String(32))
    independent_variable: Mapped[str] = mapped_column(String(32), nullable=False)

    #  Crystallographic/optical axis: "o-ray", "e-ray", "alpha", "beta", "gamma",
    #  "isotropic".  Sec. 3.2 — a curve without its axis cannot be reduced to a
    #  directional scalar later.
    axis: Mapped[str | None] = mapped_column(String(32))
    tensor_component: Mapped[str | None] = mapped_column(String(16))

    temperature_k: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str | None] = mapped_column(String(128))
    provenance_tier: Mapped[ProvenanceTier] = mapped_column(
        enum_column(ProvenanceTier, "provenance_tier"), nullable=False
    )
    doi: Mapped[str | None] = mapped_column(String(256))
    source_url: Mapped[str | None] = mapped_column(String(512))
    database_identifier: Mapped[str | None] = mapped_column(String(128))
    dataset_label: Mapped[str | None] = mapped_column(String(256))
    context_digest: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    n_points: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    x_min: Mapped[float | None] = mapped_column(Float)
    x_max: Mapped[float | None] = mapped_column(Float)

    points: Mapped[list[SpectralPoint]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )


class SpectralPoint(Base):
    """One (x, y) sample of a :class:`SpectralSeries`.

    No timestamp mixin and no provenance columns: both belong to the series, and
    duplicating them across ~10^5 rows would cost more than the data.
    """

    __tablename__ = "spectral_points"
    __table_args__ = (
        UniqueConstraint("series_id", "x_value", name="uq_spectral_point_x"),
        Index("ix_spectral_point_series_x", "series_id", "x_value"),
        #  A zero or negative wavelength/energy is not a data point.
        CheckConstraint("x_value > 0", name="ck_spectral_point_x_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("spectral_series.id", ondelete="CASCADE"), index=True
    )
    x_value: Mapped[float] = mapped_column(Float, nullable=False)
    y_value: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[float | None] = mapped_column(Float)

    series: Mapped[SpectralSeries] = relationship(back_populates="points")


# ---------------------------------------------------------------------------
# External ingestion staging
# ---------------------------------------------------------------------------


class ExternalRecord(Base, TimestampMixin):
    """A row from an external database, kept verbatim before promotion.

    Ingestion is two-phase on purpose.  External sources routinely lack the
    identity and context this protocol requires — the CNMS oxide database, for
    instance, has no polymorph column at all, and encodes phase and optical axis
    inside a free-text ``dataset_label``.  A single-phase importer has only two
    options at that point: invent the missing fields, or drop the row silently.
    Sec. 2.3 forbids the first and auditability forbids the second.

    So everything lands here with its raw payload, and only rows that pass
    eligibility are promoted into ``materials`` / ``property_values``.  The rest
    stay quarantined with a reason, which turns "this database is not usable
    yet" into a queryable list of exactly what is missing.
    """

    __tablename__ = "external_records"
    __table_args__ = (
        UniqueConstraint(
            "source_database", "source_table", "source_row_id", name="uq_external_record"
        ),
        Index("ix_external_status", "source_database", "status"),
        CheckConstraint(
            "status IN ('staged', 'promoted', 'quarantined', 'superseded')",
            name="ck_external_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    #  Provenance of the *import*, distinct from the provenance of the value.
    source_database: Mapped[str] = mapped_column(String(128), nullable=False)
    source_table: Mapped[str] = mapped_column(String(128), nullable=False)
    source_row_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_sha256: Mapped[str | None] = mapped_column(String(64))

    #  The original row, unmodified.  Re-parsing beats re-importing when the
    #  mapping improves, and it is the only defence against a lossy mapping.
    payload: Mapped[dict] = mapped_column(JSONType, nullable=False)
    #  What the label parser recovered: phase, axis, source tag, quantity.
    parsed: Mapped[dict | None] = mapped_column(JSONType)

    status: Mapped[str] = mapped_column(String(16), default="staged", nullable=False)
    quarantine_reason: Mapped[str | None] = mapped_column(Text)
    #  Which required fields were absent — the shopping list for making this
    #  record usable.
    missing_fields: Mapped[list | None] = mapped_column(JSONType)

    material_id: Mapped[int | None] = mapped_column(
        ForeignKey("materials.id", ondelete="SET NULL")
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


#  Registered last, once every mapped class above exists.  Importing ``models``
#  must be enough to get the invariants — a caller should not have to remember
#  to import ``events`` as well.  Kept out of ``db/__init__.py`` on purpose so
#  that ``from cnms_fom.db.enums import ...`` still works without SQLAlchemy.
from . import events  # noqa: E402,F401  (side-effecting import, must come last)
