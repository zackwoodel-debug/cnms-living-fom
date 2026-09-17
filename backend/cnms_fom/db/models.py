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
        UniqueConstraint(
            "formula_reduced", "polymorph", "specimen_form", name="uq_material_identity"
        ),
        Index("ix_materials_formula", "formula_reduced"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    formula: Mapped[str] = mapped_column(String(128), nullable=False)
    formula_reduced: Mapped[str] = mapped_column(String(128), nullable=False)
    #  "rutile", "anatase", "monoclinic-P21/c", "amorphous" ... never left blank.
    polymorph: Mapped[str] = mapped_column(String(128), nullable=False)
    space_group_symbol: Mapped[str | None] = mapped_column(String(32))
    space_group_number: Mapped[int | None] = mapped_column(Integer)
    specimen_form: Mapped[SpecimenForm] = mapped_column(
        SAEnum(SpecimenForm, name="specimen_form"), nullable=False
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
        SAEnum(ProvenanceTier, name="provenance_tier"), nullable=False
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
        SAEnum(ProvenanceTier, name="provenance_tier"), nullable=False
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
    __table_args__ = (Index("ix_property_key", "property_key"),)

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
        SAEnum(ProvenanceTier, name="provenance_tier"), nullable=False
    )

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
    __table_args__ = (UniqueConstraint("name", "version", name="uq_fom_name_version"),)

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
        SAEnum(ScoreStatus, name="score_status"), nullable=False
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


class CorrelationResult(Base, TimestampMixin):
    """One cell of a correlation block — every field of FOM_PROOF Table 6."""

    __tablename__ = "correlation_results"
    __table_args__ = (Index("ix_corr_run_block", "run_id", "block"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )

    block: Mapped[CorrelationBlock] = mapped_column(
        SAEnum(CorrelationBlock, name="correlation_block"), nullable=False
    )
    x_key: Mapped[str] = mapped_column(String(64), nullable=False)
    y_key: Mapped[str] = mapped_column(String(64), nullable=False)
    x_transform: Mapped[Transform] = mapped_column(
        SAEnum(Transform, name="transform"), nullable=False, default=Transform.NONE
    )
    y_transform: Mapped[Transform] = mapped_column(
        SAEnum(Transform, name="transform"), nullable=False, default=Transform.NONE
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
        SAEnum(HypothesisOutcome, name="hypothesis_outcome")
    )
    # Exactly which materials entered this cell (Table 6: "eligible material records").
    material_ids: Mapped[list | None] = mapped_column(JSONType)

    run: Mapped[AnalysisRun] = relationship(back_populates="correlations")


class MediationResult(Base, TimestampMixin):
    """M_ja = sum_q B_jq * Gamma_qa  (Eqs. 64-65) — the primary mechanistic result."""

    __tablename__ = "mediation_results"

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

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    technique: Mapped[SynthesisTechnique] = mapped_column(
        SAEnum(SynthesisTechnique, name="synthesis_technique"), nullable=False
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
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),)

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

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    technique: Mapped[SynthesisTechnique] = mapped_column(
        SAEnum(SynthesisTechnique, name="synthesis_technique"), nullable=False
    )
    location: Mapped[str | None] = mapped_column(String(128))
    # {parameter: {min, max, units}} — the hard envelope the BO loop must respect.
    capabilities: Mapped[dict | None] = mapped_column(JSONType)
    available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    experiments: Mapped[list[Experiment]] = relationship(back_populates="instrument")


class Experiment(Base, TimestampMixin):
    """One growth/characterization run and the recipe that produced it."""

    __tablename__ = "experiments"

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

    id: Mapped[int] = mapped_column(primary_key=True)
    bo_run_id: Mapped[int] = mapped_column(ForeignKey("bo_runs.id", ondelete="CASCADE"), index=True)

    parameters: Mapped[dict] = mapped_column(JSONType, nullable=False)
    acquisition_value: Mapped[float | None] = mapped_column(Float)
    predicted_mean: Mapped[float | None] = mapped_column(Float)
    predicted_std: Mapped[float | None] = mapped_column(Float)
    batch_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="proposed", nullable=False)

    run: Mapped[BoRun] = relationship(back_populates="suggestions")
