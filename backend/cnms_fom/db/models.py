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
    BriefStatus,
    CardCategory,
    CardRelation,
    CardStatus,
    CardType,
    ChatRole,
    ClaimStatus,
    ClaimTier,
    ContextStatus,
    CorrelationBlock,
    FitTechnique,
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
    #  A copy of ``documents.title``, because the lexical retriever scores title +
    #  text and a Postgres functional index cannot span two tables (migration 0007).
    #
    #  Declared here as well as in the migration for schema parity, and that parity is
    #  not cosmetic: while this column existed only in the migration, any schema built
    #  by ``metadata.create_all`` lacked it, the Postgres lexical query raised
    #  ``UndefinedColumn``, and the documented fallback then failed too because the
    #  transaction was already aborted — poisoning every later query on that session.
    #
    #  On Postgres the value is maintained by the two triggers in migration 0007. A
    #  ``create_all`` schema has the column but no triggers, so it stays NULL; the
    #  query wraps it in ``coalesce``, so that degrades to text-only scoring rather
    #  than failing. Slower and less precise, never broken.
    search_title: Mapped[str | None] = mapped_column(Text, nullable=True)

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


# ---------------------------------------------------------------------------
# ModalFit co-refinement records
#
# ModalFit fits one shared slab model against up to five characterization
# techniques at once (SE / SPR / QCM / XRR / NR).  The reason that matters here
# is not the fitting — it is that a co-refinement is the only place the platform
# ever gets the *same* quantity from independent physics.  An XRR thickness and
# an SE thickness for one film are two measurements of one number by two
# unrelated forward models, and their disagreement is a data-quality signal
# nothing else in the schema can produce.
#
# So a fit is stored as a measurement record, not as a file: the per-layer
# fitted parameters, which techniques contributed, the per-technique
# chi-squared, and which parameters were actually free.  A parameter held fixed
# is not a measurement of anything, and the schema keeps that distinction
# because ``modalfit.promote`` refuses to turn a fixed parameter into a
# ``PropertyValue``.
# ---------------------------------------------------------------------------


class FitRecord(Base, TimestampMixin):
    """One ModalFit refinement of one slab model against one or more techniques."""

    __tablename__ = "fit_records"
    __table_args__ = (
        #  Re-importing the same exported JSON is a no-op, the same way PDF
        #  ingestion is idempotent by content hash.
        UniqueConstraint("content_sha256", name="uq_fit_record_content"),
        Index("ix_fit_records_sample", "sample_id"),
        Index("ix_fit_records_stack", "stack_id"),
        CheckConstraint(
            "n_free_parameters IS NULL OR n_free_parameters >= 0",
            name="ck_fit_free_parameters_nonneg",
        ),
        #  A reduced chi-squared is a ratio of squares; negative is a bug.
        CheckConstraint("chi2_total IS NULL OR chi2_total >= 0", name="ck_fit_chi2_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    #  ModalFit's own identifiers, carried through verbatim so a fit can be
    #  traced back to the app and to DataFed without a lookup table.
    stack_id: Mapped[str | None] = mapped_column(String(128))
    sample_id: Mapped[str | None] = mapped_column(String(128))

    #  Where the JSON came from, and its hash — a fit is evidence, and evidence
    #  needs to stay identifiable after the file is moved.
    source_filename: Mapped[str | None] = mapped_column(String(512))
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    #  DataFed record id when the model JSON was pushed or pulled through it.
    datafed_record_id: Mapped[str | None] = mapped_column(String(128))

    #  Which techniques were co-refined: ["XRR", "SE"].  A list, not a scalar —
    #  the co-refinement *is* the record.
    techniques: Mapped[list] = mapped_column(JSONType, nullable=False)
    #  Relative weight each technique carried in the combined objective.
    technique_weights: Mapped[dict | None] = mapped_column(JSONType)
    algorithm: Mapped[str | None] = mapped_column(String(32))

    chi2_total: Mapped[float | None] = mapped_column(Float)
    #  {"XRR": 1.84, "SE": 3.02} — a combined chi-squared hides which technique
    #  the model actually fails to describe.
    chi2_by_technique: Mapped[dict | None] = mapped_column(JSONType)
    n_free_parameters: Mapped[int | None] = mapped_column(Integer)

    #  Known-limitation flags, read off the fit's own settings rather than
    #  assumed.  ModalFit builds its refnx models with dq=0.0, so XRR/NR fits
    #  carry no angular-resolution smearing and show sharper fringe contrast
    #  than the instrument measured; the SPR path ignores layer roughness
    #  entirely.  Both bias the fitted values, and a downstream consumer that
    #  does not know cannot correct for it.
    resolution_smearing_applied: Mapped[bool | None] = mapped_column(Boolean)
    roughness_applied_to_spr: Mapped[bool | None] = mapped_column(Boolean)
    #  True when any layer's optical constants came from the bundled placeholder
    #  n/k tables.  ModalFit's README says these are not digitized literature
    #  values; an SE-derived number resting on them is not a citable result.
    uses_placeholder_optical_constants: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    #  Per-technique instrument settings (AOI, energy, Q-range, overtones ...).
    technique_settings: Mapped[dict | None] = mapped_column(JSONType)
    #  The exported slab-model JSON, unmodified.  Same reasoning as
    #  ``ExternalRecord.payload``: re-parsing beats re-importing.
    raw_model: Mapped[dict | None] = mapped_column(JSONType)

    fitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    operator: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)

    #  Optional links into the rest of the platform.  Both nullable: a fit is a
    #  complete record on its own, and forcing a material identity at import
    #  time would mean guessing a polymorph, which Sec. 2.1 forbids.
    material_id: Mapped[int | None] = mapped_column(
        ForeignKey("materials.id", ondelete="SET NULL")
    )
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL")
    )

    layers: Mapped[list[FitLayer]] = relationship(
        back_populates="fit", cascade="all, delete-orphan", order_by="FitLayer.layer_index"
    )
    datasets: Mapped[list[FitDataset]] = relationship(
        back_populates="fit", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        techniques = "+".join(self.techniques or [])
        return f"<FitRecord {self.sample_id or self.stack_id} [{techniques}]>"


class FitLayer(Base, TimestampMixin):
    """One slab in a fitted stack, with its refined parameters.

    Parameters live in a dict rather than in columns because the set is
    technique-dependent: a QCM-only fit has a shear modulus and no SLD, an
    XRR/NR fit has SLD and no dispersion model.  Columns for the union would be
    mostly NULL and would still need extending for the next technique.

    ``free_parameters`` is the load-bearing field.  A thickness held fixed
    during refinement is an *input* to the fit, and promoting it as a measured
    thickness would be fabrication dressed up as instrument data.
    """

    __tablename__ = "fit_layers"
    __table_args__ = (
        UniqueConstraint("fit_record_id", "layer_index", name="uq_fit_layer_position"),
        Index("ix_fit_layers_label", "label"),
        CheckConstraint("layer_index >= 0", name="ck_fit_layer_index_nonneg"),
        CheckConstraint("role IN ('ambient', 'layer', 'substrate')", name="ck_fit_layer_role"),
        CheckConstraint(
            "thickness_ang IS NULL OR thickness_ang >= 0", name="ck_fit_layer_thickness_nonneg"
        ),
        CheckConstraint(
            "roughness_ang IS NULL OR roughness_ang >= 0", name="ck_fit_layer_roughness_nonneg"
        ),
        CheckConstraint(
            "density_g_cm3 IS NULL OR density_g_cm3 > 0", name="ck_fit_layer_density_positive"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    fit_record_id: Mapped[int] = mapped_column(
        ForeignKey("fit_records.id", ondelete="CASCADE"), index=True
    )

    #  Ambient at index 0, substrate last — the slab order ModalFit uses.
    layer_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128))
    material: Mapped[str | None] = mapped_column(String(128))
    #  From the layer's ``molecular`` block; what lets refnx derive SLD from
    #  composition instead of a hand-entered number.
    formula: Mapped[str | None] = mapped_column(String(128))

    #  Promoted out of ``parameters`` because every technique shares them and
    #  they are what cross-technique comparison is actually about.  Angstroms,
    #  matching ModalFit's slab-model units.
    thickness_ang: Mapped[float | None] = mapped_column(Float)
    roughness_ang: Mapped[float | None] = mapped_column(Float)
    density_g_cm3: Mapped[float | None] = mapped_column(Float)

    #  {"structural": {...}, "optical": {...}, "xray": {...}, "neutron": {...},
    #   "viscoelastic": {...}} — the slab-model blocks, values only.
    parameters: Mapped[dict | None] = mapped_column(JSONType)
    #  Names of the parameters that were varied, e.g. ["thickness", "roughness"].
    free_parameters: Mapped[list | None] = mapped_column(JSONType)
    #  {param: {"min": ..., "max": ...}} as refined, so "hit the bound" stays
    #  detectable afterwards.  A parameter resting on its bound has not
    #  converged; it has been clamped.
    bounds: Mapped[dict | None] = mapped_column(JSONType)
    #  Per-parameter 1-sigma from the optimizer, when it reported any.  Only
    #  DREAM (emcee) produces a posterior; the four scipy minimizers do not, and
    #  a fitted value with no uncertainty must not be dressed as having one.
    uncertainties: Mapped[dict | None] = mapped_column(JSONType)

    fit: Mapped[FitRecord] = relationship(back_populates="layers")


class FitDataset(Base, TimestampMixin):
    """The experimental data one technique contributed to a fit.

    Without this a chi-squared is unfalsifiable: 1.8 over 40 points in a narrow
    Q-range and 1.8 over 400 points across two decades are not the same claim.
    """

    __tablename__ = "fit_datasets"
    __table_args__ = (
        UniqueConstraint("fit_record_id", "technique", name="uq_fit_dataset_technique"),
        Index("ix_fit_datasets_technique", "technique"),
        CheckConstraint("n_points IS NULL OR n_points >= 0", name="ck_fit_dataset_points_nonneg"),
        CheckConstraint("chi2 IS NULL OR chi2 >= 0", name="ck_fit_dataset_chi2_nonneg"),
        CheckConstraint("weight IS NULL OR weight >= 0", name="ck_fit_dataset_weight_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    fit_record_id: Mapped[int] = mapped_column(
        ForeignKey("fit_records.id", ondelete="CASCADE"), index=True
    )

    technique: Mapped[FitTechnique] = mapped_column(
        enum_column(FitTechnique, "fit_technique", length=8), nullable=False
    )
    #  Instrument file as loaded: Woollam .dat, Rigaku .ras, IMES CSV, ORSO text.
    source_filename: Mapped[str | None] = mapped_column(String(512))
    loader: Mapped[str | None] = mapped_column(String(64))
    datafed_record_id: Mapped[str | None] = mapped_column(String(128))

    n_points: Mapped[int | None] = mapped_column(Integer)
    #  The fitted window, in that technique's own units (Q in 1/A, wavelength in
    #  nm, angle in degrees, Δf in Hz).  Units are recorded, not assumed.
    x_min: Mapped[float | None] = mapped_column(Float)
    x_max: Mapped[float | None] = mapped_column(Float)
    x_units: Mapped[str | None] = mapped_column(String(32))

    chi2: Mapped[float | None] = mapped_column(Float)
    weight: Mapped[float | None] = mapped_column(Float)
    #  Instrument settings for this technique on this fit (AOI, energy, overtones).
    settings: Mapped[dict | None] = mapped_column(JSONType)

    fit: Mapped[FitRecord] = relationship(back_populates="datasets")


# ---------------------------------------------------------------------------
# Research-assistant conversations
#
# Persisted rather than held in process memory, for two reasons that have
# nothing to do with convenience:
#
#   * A retrieval answer is auditable only if the evidence behind it is
#     recoverable later.  ``ChatMessage.evidence`` stores the exact chunk ids,
#     similarities, and tool results behind each answer, so "where did that
#     number come from?" still has an answer six months on.
#   * ModalFit's own session store is explicitly in-process memory keyed by a
#     cookie, and its README names that as the thing to fix before a
#     multi-worker deployment.  Repeating the choice here would repeat the
#     defect.
# ---------------------------------------------------------------------------


class ChatSession(Base, TimestampMixin):
    """One conversation with the research assistant."""

    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_created", "created_at"),
        Index("ix_chat_sessions_sample", "sample_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    #  Opaque, client-supplied or generated; what the API exposes instead of the
    #  integer primary key.
    session_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(String(256))
    user: Mapped[str | None] = mapped_column(String(128))

    #  Optional scoping: a conversation pinned to one sample gets that sample's
    #  fits offered to the model without being asked for them by name.
    sample_id: Mapped[str | None] = mapped_column(String(128))
    #  Corpus partitions this conversation is restricted to, if any.
    techniques: Mapped[list | None] = mapped_column(JSONType)

    chat_model: Mapped[str | None] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(32))

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.turn_index",
    )


class ChatMessage(Base, TimestampMixin):
    """One turn, with the evidence behind it."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_index", name="uq_chat_message_turn"),
        CheckConstraint("turn_index >= 0", name="ck_chat_turn_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[ChatRole] = mapped_column(
        enum_column(ChatRole, "chat_role", length=16), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    #  [{"tool": "compare_fit_techniques", "arguments": {...}, "result": {...}}]
    tool_calls: Mapped[list | None] = mapped_column(JSONType)
    #  Retrieved chunk ids, similarities, and citations behind this answer.
    evidence: Mapped[list | None] = mapped_column(JSONType)
    #  True when the assistant declined for want of evidence (Sec. 2.3).  Stored
    #  because a refusal rate is a corpus-coverage metric, not a failure log.
    insufficient_context: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    chat_model: Mapped[str | None] = mapped_column(String(64))
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# Knowledge cards — a Dynamic Knowledge Repository over the corpus
#
# Ordinary retrieval re-derives an answer on every question and keeps nothing.
# Ask the same thing twice and the model reasons from scratch, having learned
# nothing in between.  The alternative, and the pattern these two tables
# implement, is to do the integration work at *ingest* time: when a source
# lands, the assistant updates the concept pages it touches, writes a summary of
# the source, and flags where it contradicts what is already recorded.  Knowledge
# then compounds instead of evaporating.
#
# What makes that safe here rather than dangerous is the review gate.  A card is
# where a language model's synthesis of the corpus gets written down, and Sec.
# 15.2 is explicit that such a synthesis is not evidence.  So an assistant-written
# card is PROPOSED: returned, clearly labelled, and not something to build on. A
# person promotes it to REVIEWED, and that promotion records who did it and
# against which body text — so an edit after review makes the review stale rather
# than silently inheriting it.
#
# The one thing a card is not, under any circumstance, is a route into the
# analysis tables. There is no code path from a card to a PropertyValue.
# ---------------------------------------------------------------------------


class KnowledgeCard(Base, TimestampMixin):
    """One page of accumulated knowledge, with its sources and its review state."""

    __tablename__ = "knowledge_cards"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_card_slug"),
        Index("ix_cards_type_status", "card_type", "status"),
        Index("ix_cards_updated", "updated_at"),
        #  Sec. 15.2 at the storage layer: a card cannot be marked reviewed
        #  without a named person having reviewed it. "Reviewed by nobody" is
        #  exactly the state that would let model output pass as checked.
        CheckConstraint(
            "status <> 'reviewed' OR reviewed_by IS NOT NULL",
            name="ck_card_reviewed_has_reviewer",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_card_confidence_unit_interval",
        ),
        CheckConstraint("length(trim(title)) > 0", name="ck_card_title_not_blank"),
        CheckConstraint("length(trim(slug)) > 0", name="ck_card_slug_not_blank"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    #  Path-like and stable: "concepts/ald-window-hfo2", "sources/kim-2024".
    #  It is the citation handle, so renaming one breaks every reference to it.
    slug: Mapped[str] = mapped_column(String(256), nullable=False)
    card_type: Mapped[CardType] = mapped_column(
        enum_column(CardType, "card_type", length=16), nullable=False
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    #  Markdown. The whole point is that a person can read and correct it.
    body: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)

    status: Mapped[CardStatus] = mapped_column(
        enum_column(CardStatus, "card_status", length=16),
        nullable=False,
        default=CardStatus.PROPOSED,
    )
    #  How much the author trusts the card, 0-1. Separate from status: a reviewed
    #  card can still record genuine uncertainty about its own claim.
    confidence: Mapped[float | None] = mapped_column(Float)
    tags: Mapped[list | None] = mapped_column(JSONType)
    #  What job this card does in the research loop — a process window, a property
    #  prior, a measurement caveat. Orthogonal to ``card_type``, which is the page's
    #  shape: a process window is a CONCEPT in shape and a PROCESS_WINDOW in
    #  purpose. A closed set rather than a tag because the BO context bridge selects
    #  cards by category, and a free-text tag would make that unenforceable.
    category: Mapped[CardCategory | None] = mapped_column(
        enum_column(CardCategory, "card_category", length=32)
    )

    #  [{"kind": "document", "document_id": 3, "page": 12, "doi": "..."},
    #   {"kind": "fit_record", "fit_record_id": 7}, ...]
    #  A card making a factual claim with an empty sources list is an assertion,
    #  not knowledge, and ``knowledge.cards.review`` refuses to sign one off.
    sources: Mapped[list | None] = mapped_column(JSONType)

    #  "assistant" or a person's name. Which it is changes how the card should be
    #  read, so it is recorded rather than inferred from the status.
    authored_by: Mapped[str] = mapped_column(String(128), nullable=False, default="assistant")
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #  Hash of the body as it stood when reviewed. An edit afterwards makes the
    #  review stale — without this, a card could be approved and then rewritten,
    #  and would still read as checked.
    reviewed_body_sha256: Mapped[str | None] = mapped_column(String(64))

    supersedes_id: Mapped[int | None] = mapped_column(
        ForeignKey("knowledge_cards.id", ondelete="SET NULL")
    )

    links_out: Mapped[list[CardLink]] = relationship(
        back_populates="from_card",
        cascade="all, delete-orphan",
        foreign_keys="CardLink.from_card_id",
    )
    links_in: Mapped[list[CardLink]] = relationship(
        back_populates="to_card",
        foreign_keys="CardLink.to_card_id",
    )

    @property
    def review_is_stale(self) -> bool:
        """True when the body changed after the card was reviewed."""
        import hashlib

        if self.status is not CardStatus.REVIEWED or not self.reviewed_body_sha256:
            return False
        current = hashlib.sha256((self.body or "").encode("utf-8")).hexdigest()
        return current != self.reviewed_body_sha256

    @property
    def citable(self) -> bool:
        """Whether anything may be built on this card.

        Reviewed, not stale, and carrying at least one source. Everything else is
        someone's notes — useful to read, not something to cite.
        """
        return (
            self.status is CardStatus.REVIEWED
            and not self.review_is_stale
            and bool(self.sources)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<KnowledgeCard {self.slug} [{self.status.value}]>"


class CardLink(Base, TimestampMixin):
    """One typed, directed edge between two cards."""

    __tablename__ = "card_links"
    __table_args__ = (
        UniqueConstraint("from_card_id", "to_card_id", "relation", name="uq_card_link"),
        Index("ix_card_links_relation", "relation"),
        #  A card related to itself is a data-entry slip, and it makes every
        #  graph traversal a special case.
        CheckConstraint("from_card_id <> to_card_id", name="ck_card_link_not_self"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    from_card_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_cards.id", ondelete="CASCADE"), index=True
    )
    to_card_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_cards.id", ondelete="CASCADE"), index=True
    )
    relation: Mapped[CardRelation] = mapped_column(
        enum_column(CardRelation, "card_relation", length=16), nullable=False
    )
    #  Why the edge exists. For CONTRADICTS especially: "which claim, and on what
    #  basis" is the whole content of the link, and an untyped pointer between two
    #  cards that disagree is worse than no link at all.
    note: Mapped[str | None] = mapped_column(Text)

    from_card: Mapped[KnowledgeCard] = relationship(
        back_populates="links_out", foreign_keys=[from_card_id]
    )
    to_card: Mapped[KnowledgeCard] = relationship(
        back_populates="links_in", foreign_keys=[to_card_id]
    )


# ---------------------------------------------------------------------------
# The research loop: briefs, extracted claims, and proposed campaign context
#
# These three tables hold the *evidence layer*.  Nothing in them is a
# measurement, and the schema is arranged so that saying otherwise would require
# adding a column rather than setting one.
#
# ``ResearchBriefRecord`` is a document about the state of the evidence for one
# question.  ``ExtractedClaimRecord`` is one number a source stated, wired to the
# page it appeared on.  ``CampaignContextProposal`` is a requested change to a BO
# campaign, which reaches ``applied`` only through a named reviewer.
#
# Two decisions worth stating.  First, claims get their own table rather than a
# JSON column on the brief: a claim is the thing people will query — "what has
# anyone reported for the permittivity of HfO2?" — and provenance kept in a blob
# cannot be constrained, indexed, or audited.  Second, the claim tier is
# ``ClaimTier`` and not ``ProvenanceTier``; see the enum docstrings for why
# sharing that vocabulary would be the single most expensive shortcut available
# here.
# ---------------------------------------------------------------------------


class ResearchBriefRecord(Base, TimestampMixin):
    """One audited research brief: what was found, what was not, what is proposed."""

    __tablename__ = "research_briefs"
    __table_args__ = (
        Index("ix_briefs_run_created", "bo_run_id", "created_at"),
        Index("ix_briefs_status", "status"),
        Index("ix_briefs_fingerprint", "fingerprint"),
        CheckConstraint("length(trim(research_question)) > 0", name="ck_brief_has_question"),
        #  Sec. 15.2 as a constraint: a brief cannot be marked reviewed without a
        #  named reviewer, the same rule knowledge cards carry.
        CheckConstraint(
            "status <> 'reviewed' OR reviewed_by IS NOT NULL",
            name="ck_brief_reviewed_has_reviewer",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    research_question: Mapped[str] = mapped_column(Text, nullable=False)

    bo_run_id: Mapped[int | None] = mapped_column(ForeignKey("bo_runs.id", ondelete="SET NULL"))
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL")
    )
    material_id: Mapped[int | None] = mapped_column(
        ForeignKey("materials.id", ondelete="SET NULL")
    )

    #  Free text, because a brief is often written before the material has an
    #  identity in this database — and Sec. 2.1 does not let us invent one.
    material: Mapped[str | None] = mapped_column(String(128))
    specimen_form: Mapped[str | None] = mapped_column(String(64))
    target_property: Mapped[str | None] = mapped_column(String(64))
    fom_definition: Mapped[str | None] = mapped_column(String(128))

    status: Mapped[BriefStatus] = mapped_column(
        enum_column(BriefStatus, "brief_status"), nullable=False, default=BriefStatus.PROPOSED
    )
    #  True when the assistant declined for want of evidence. Stored because an
    #  abstention rate is a corpus-coverage metric, not a failure log.
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #  Evidence, contradictions, gaps, labelled statements, and the tool trace.
    #  These are the brief's narrative, read as a whole; claims are the part that
    #  gets queried, and they live in their own table.
    evidence: Mapped[list | None] = mapped_column(JSONType)
    contradictions: Mapped[list | None] = mapped_column(JSONType)
    data_gaps: Mapped[list | None] = mapped_column(JSONType)
    statements: Mapped[list | None] = mapped_column(JSONType)
    proposed_actions: Mapped[list | None] = mapped_column(JSONType)
    proposed_card_slugs: Mapped[list | None] = mapped_column(JSONType)
    warnings: Mapped[list | None] = mapped_column(JSONType)
    tool_calls: Mapped[list | None] = mapped_column(JSONType)

    model: Mapped[str | None] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(32))
    #  Which retrieval/extraction policy produced this. The benchmark compares
    #  policies, so a brief that cannot name its own is not reproducible.
    policy_version: Mapped[str | None] = mapped_column(String(64))
    #  Hash of the substantive content, excluding timestamps and the trace.
    fingerprint: Mapped[str | None] = mapped_column(String(64))

    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    claims: Mapped[list[ExtractedClaimRecord]] = relationship(
        back_populates="brief", cascade="all, delete-orphan"
    )
    context_proposals: Mapped[list[CampaignContextProposal]] = relationship(
        back_populates="brief"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ResearchBrief #{self.id} [{self.status.value}] {self.research_question[:40]!r}>"


class ExtractedClaimRecord(Base, TimestampMixin):
    """One value a source stated, with the page it appeared on.

    Not a measurement, and the columns are chosen so that it cannot be mistaken
    for one: there is no ``provenance_tier``, no ``context_digest``, and no
    relationship to ``FomScore``. Promotion into ``property_values`` is a separate,
    human act that reads this row as provenance rather than converting it.
    """

    __tablename__ = "research_claims"
    __table_args__ = (
        Index("ix_claims_field", "field_name"),
        Index("ix_claims_brief", "brief_id"),
        Index("ix_claims_document", "document_id"),
        #  Sec. 2.2: every claim is traceable. A row with neither a document id nor
        #  a content hash points at nothing.
        CheckConstraint(
            "document_id IS NOT NULL OR content_sha256 IS NOT NULL",
            name="ck_claim_has_a_source",
        ),
        CheckConstraint("length(trim(quote)) > 0", name="ck_claim_has_a_quote"),
        CheckConstraint(
            "model_confidence IS NULL OR (model_confidence >= 0 AND model_confidence <= 1)",
            name="ck_claim_confidence_unit_interval",
        ),
        #  A converted number without its conversion recorded is not auditable.
        CheckConstraint(
            "normalized_value IS NULL OR length(trim(normalization_note)) > 0",
            name="ck_claim_normalisation_explained",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    brief_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_briefs.id", ondelete="CASCADE")
    )

    field_name: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    units: Mapped[str | None] = mapped_column(String(32))
    value_text: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[float | None] = mapped_column(Float)
    normalized_units: Mapped[str | None] = mapped_column(String(32))
    normalization_note: Mapped[str | None] = mapped_column(Text)

    tier: Mapped[ClaimTier] = mapped_column(
        enum_column(ClaimTier, "claim_tier"), nullable=False, default=ClaimTier.REPORTED
    )
    status: Mapped[ClaimStatus] = mapped_column(
        enum_column(ClaimStatus, "claim_status"), nullable=False, default=ClaimStatus.CANDIDATE
    )

    #  The measurement context the source recorded: temperature, frequency,
    #  precursor, chamber, substrate. A dict because the useful set differs per
    #  field, and `missing_context` on the contract computes what is absent.
    context: Mapped[dict | None] = mapped_column(JSONType)
    #  Required context this claim lacks, computed at write time so the gap is
    #  queryable rather than only derivable.
    missing_context: Mapped[list | None] = mapped_column(JSONType)

    # --- provenance, one row per claim -----------------------------------
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    document_title: Mapped[str | None] = mapped_column(String(512))
    page: Mapped[int | None] = mapped_column(Integer)
    chunk_id: Mapped[int | None] = mapped_column(Integer)
    #  The exact supporting text. Without it a wrong extraction is
    #  indistinguishable from a right one.
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    doi: Mapped[str | None] = mapped_column(String(256))
    #  The remaining evidence items, when a claim rests on more than one passage.
    evidence: Mapped[list | None] = mapped_column(JSONType)

    #  Which model, prompt, and policy produced this extraction. An extraction is
    #  only reproducible if it can name the thing that produced it.
    extracted_by_model: Mapped[str | None] = mapped_column(String(64))
    extracted_by_provider: Mapped[str | None] = mapped_column(String(32))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    model_confidence: Mapped[float | None] = mapped_column(Float)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    brief: Mapped[ResearchBriefRecord | None] = relationship(back_populates="claims")


class CampaignContextProposal(Base, TimestampMixin):
    """A requested change to a BO campaign, and the record of who allowed it.

    The only path by which evidence may influence the optimizer. It carries the
    campaign's configuration fingerprint before and after application, so "what
    changed this campaign, and on whose authority" is answerable from one row.
    """

    __tablename__ = "campaign_context_proposals"
    __table_args__ = (
        Index("ix_context_run_status", "bo_run_id", "status"),
        #  Sec. 15.2 at the storage layer, twice over: reviewing needs a reviewer,
        #  and applying needs a reviewer *and* an applier. A proposal cannot walk
        #  itself into a live campaign.
        CheckConstraint(
            "status NOT IN ('reviewed', 'applied') OR reviewed_by IS NOT NULL",
            name="ck_context_reviewed_has_reviewer",
        ),
        CheckConstraint(
            "status <> 'applied' OR applied_by IS NOT NULL",
            name="ck_context_applied_has_applier",
        ),
        CheckConstraint(
            "status <> 'applied' OR applied_at IS NOT NULL",
            name="ck_context_applied_has_timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    bo_run_id: Mapped[int] = mapped_column(
        ForeignKey("bo_runs.id", ondelete="CASCADE"), index=True
    )
    brief_id: Mapped[int | None] = mapped_column(
        ForeignKey("research_briefs.id", ondelete="SET NULL")
    )

    status: Mapped[ContextStatus] = mapped_column(
        enum_column(ContextStatus, "context_status"), nullable=False, default=ContextStatus.PROPOSED
    )

    #  {parameter: [lower, upper]} — narrowings only, enforced by the bridge.
    recommended_bounds: Mapped[dict | None] = mapped_column(JSONType)
    excluded_choices: Mapped[dict | None] = mapped_column(JSONType)
    #  Advisory content. Written to the campaign's constraint notes on apply, never
    #  into the acquisition function.
    soft_priors: Mapped[list | None] = mapped_column(JSONType)
    process_window_hints: Mapped[list | None] = mapped_column(JSONType)
    uncertainty_notes: Mapped[list | None] = mapped_column(JSONType)
    rationale: Mapped[str | None] = mapped_column(Text)

    #  Cards this rests on, with the body hash each had at proposal time. Re-checked
    #  on apply: a card edited in between makes its own review stale, and an
    #  approval granted on the old text must not silently cover the new.
    supporting_cards: Mapped[list | None] = mapped_column(JSONType)

    proposed_by: Mapped[str] = mapped_column(String(128), nullable=False, default="assistant")
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_note: Mapped[str | None] = mapped_column(Text)
    applied_by: Mapped[str | None] = mapped_column(String(128))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #  Campaign configuration hash before and after. Two proposals applied in
    #  sequence chain through these, so the campaign's history is reconstructable.
    campaign_fingerprint_before: Mapped[str | None] = mapped_column(String(64))
    campaign_fingerprint_after: Mapped[str | None] = mapped_column(String(64))
    #  The constraints blob as it stood before application, so an apply can be
    #  reversed without guessing what it replaced.
    constraints_before: Mapped[dict | None] = mapped_column(JSONType)

    brief: Mapped[ResearchBriefRecord | None] = relationship(back_populates="context_proposals")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CampaignContextProposal #{self.id} run={self.bo_run_id} [{self.status.value}]>"


# ---------------------------------------------------------------------------
# Per-passage model-call cache
#
# The one table in this schema that is infrastructure rather than science.  It
# records what was sent to a model and what came back, keyed by a hash of the
# content — so a re-ingest that renumbers chunks cannot serve a stale answer, and an
# edited passage misses automatically.  Correctness is a property of the key rather
# than of an invalidation rule somebody has to remember.
#
# It holds no claim, no measurement, and nothing a person would cite.  Deleting it
# costs time and nothing else, which is why it is one generic table instead of a
# typed one per stage: the rest of this schema is typed because the protocol depends
# on it, and here that reasoning does not apply.
#
# Why it matters: grading and extraction call a model once per passage and are
# together essentially the whole cost of a brief — 18 of 19 calls, about 21 minutes
# against 1.2 seconds for retrieval.  Extraction in particular is a pure function of
# the passage, because its prompt does not contain the question, so a passage needs
# extracting once ever.
# ---------------------------------------------------------------------------


class LlmCacheEntry(Base, TimestampMixin):
    """One cached model call, addressed by the hash of what was sent."""

    __tablename__ = "llm_cache"
    __table_args__ = (
        UniqueConstraint(
            "kind", "cache_key", "model", "prompt_version", name="uq_llm_cache_entry"
        ),
        Index("ix_llm_cache_lookup", "kind", "cache_key", "model"),
        Index("ix_llm_cache_last_used", "last_used_at"),
        #  A closed set, because the kind is part of the key and a typo would
        #  silently create a second cache that never hits.
        CheckConstraint("kind IN ('extraction', 'grade')", name="ck_llm_cache_kind"),
        CheckConstraint("hit_count >= 0", name="ck_llm_cache_hits_nonneg"),
        CheckConstraint("length(cache_key) = 64", name="ck_llm_cache_key_is_sha256"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    #  sha256 of the passage (extraction) or of question + NUL + passage (grading).
    #  Content-addressed rather than an id: see the comment above.
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    #  Empty string rather than NULL, so the uniqueness constraint actually
    #  constrains: in SQL, NULL != NULL, and two rows with a null prompt version
    #  would both be insertable.
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    payload: Mapped[dict] = mapped_column(JSONType, nullable=False)

    #  How many calls this row has avoided. The number worth quoting when asked
    #  whether the cache is earning its keep.
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LlmCacheEntry {self.kind}/{self.model} {self.cache_key[:12]} hits={self.hit_count}>"


#  Registered last, once every mapped class above exists.  Importing ``models``
#  must be enough to get the invariants — a caller should not have to remember
#  to import ``events`` as well.  Kept out of ``db/__init__.py`` on purpose so
#  that ``from cnms_fom.db.enums import ...`` still works without SQLAlchemy.
from . import events  # noqa: E402,F401  (side-effecting import, must come last)
