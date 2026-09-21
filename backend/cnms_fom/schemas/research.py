"""Schemas for the /research router."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from cnms_fom.db.enums import SynthesisTechnique


#  Every "who is taking responsibility for this" field runs through this. A blank
#  name is the state FOM_PROOF Sec. 15.2 is written against — "reviewed by nobody"
#  is what would let model output pass as checked — so it is refused at the schema
#  as well as in the service. ``min_length=1`` alone is not enough: a single space
#  has length one, and that is exactly the case that matters.
def _require_a_real_name(value: str) -> str:
    if not (value or "").strip():
        raise ValueError(
            "This field names the person taking responsibility and cannot be blank. "
            "'Reviewed by nobody' is the state that would let model output change a "
            "campaign (FOM_PROOF Sec. 15.2)."
        )
    return value.strip()


class BriefRequest(BaseModel):
    """Ask for a campaign-aware research brief. Read-only.

    Producing a brief changes nothing: it reads the cards, the corpus, the fits and
    the campaign and returns a document. The only thing it can cause is a person
    deciding to act.
    """

    research_question: str = Field(min_length=5)
    material: str | None = Field(default=None, description="e.g. HfO2, for the record lookups.")
    sample_id: str | None = Field(
        default=None, description="A ModalFit sample, to pull its fits and their plausibility."
    )
    target_property: str | None = Field(
        default=None, description="A registry key, e.g. k or Eg. Drives the data-gap check."
    )
    techniques: list[SynthesisTechnique] | None = Field(
        default=None, description="Restrict corpus retrieval to these partitions."
    )
    policy: str = Field(
        default="baseline",
        description="Named research policy. Recorded on the brief so a result is attributable.",
    )
    provider: str | None = Field(
        default=None, description="'ollama' (local, default) or 'anthropic'."
    )
    model: str | None = None
    include_cards: bool = True
    interpret: bool = Field(
        default=True,
        description="Generate the narrative. False assembles evidence only — what the benchmark "
        "scores, since a policy changes the numbers and the abstention, not the prose.",
    )
    persist: bool = Field(
        default=True, description="Store the brief and its claims for audit."
    )


class BriefResponse(BaseModel):
    brief_id: int | None = None
    research_question: str
    bo_run_id: int | None = None
    status: str
    abstained: bool = Field(
        description="True when the evidence did not clear the policy threshold. A brief that "
        "abstains is a successful brief; the data gaps are its actionable output."
    )
    evidence: list[dict] = Field(default_factory=list)
    claims: list[dict] = Field(default_factory=list)
    contradictions: list[dict] = Field(default_factory=list)
    data_gaps: list[dict] = Field(default_factory=list)
    statements: list[dict] = Field(default_factory=list)
    proposed_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Computed from the records, not from the model's narrative — an unapproved "
        "objective, a stalled campaign, a clamped fit parameter, a proposed card.",
    )
    model: str = ""
    provider: str = ""
    policy_version: str = ""
    fingerprint: str = ""
    n_comparable_claims: int = 0
    n_incomplete_claims: int = 0
    tool_calls: list[dict] = Field(default_factory=list)
    disclaimer: str = ""


class BriefReviewRequest(BaseModel):
    reviewed_by: str = Field(min_length=1)
    accept: bool = True

    @field_validator("reviewed_by")
    @classmethod
    def _named_reviewer(cls, value: str) -> str:
        return _require_a_real_name(value)


class BoundProposalIn(BaseModel):
    parameter: str
    lower: float
    upper: float
    rationale: str = Field(min_length=1)
    card_slug: str | None = None


class ContextProposeRequest(BaseModel):
    """Propose a change to a campaign's configuration.

    A change to what the optimizer may propose — a bound or a categorical exclusion
    — must rest on at least one reviewed, sourced card in a category the bridge
    reads. Advisory content may stand on a brief alone, because a note a human reads
    is a different risk from a bound the optimizer obeys.
    """

    recommended_bounds: list[BoundProposalIn] = Field(default_factory=list)
    excluded_choices: dict[str, list] = Field(default_factory=dict)
    soft_priors: list[str] = Field(default_factory=list)
    process_window_hints: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    rationale: str = ""
    supporting_card_slugs: list[str] = Field(default_factory=list)
    brief_id: int | None = None
    proposed_by: str = "assistant"


class ContextReviewRequest(BaseModel):
    reviewed_by: str = Field(
        min_length=1,
        description="Required. 'Reviewed by nobody' is the state that would let model output "
        "change a campaign.",
    )
    accept: bool = True
    note: str | None = None

    @field_validator("reviewed_by")
    @classmethod
    def _named_reviewer(cls, value: str) -> str:
        return _require_a_real_name(value)


class ContextApplyRequest(BaseModel):
    applied_by: str = Field(min_length=1)

    @field_validator("applied_by")
    @classmethod
    def _named_applier(cls, value: str) -> str:
        return _require_a_real_name(value)


class ContextProposalOut(BaseModel):
    proposal_id: int
    bo_run_id: int
    brief_id: int | None = None
    status: str
    changes_search_behaviour: bool = False
    recommended_bounds: dict | None = None
    excluded_choices: dict | None = None
    soft_priors: list[str] | None = None
    process_window_hints: list[str] | None = None
    uncertainty_notes: list[str] | None = None
    rationale: str | None = None
    supporting_cards: list[dict] | None = None
    proposed_by: str | None = None
    reviewed_by: str | None = None
    applied_by: str | None = None
    applied_at: str | None = None
    campaign_fingerprint_before: str | None = None
    campaign_fingerprint_after: str | None = None


class ExperimentSummaryRequest(BaseModel):
    bo_run_id: int | None = None
    sample_id: str | None = None
    observation_id: int | None = None
    provider: str | None = None
    model: str | None = None
    propose_card: bool = Field(
        default=False,
        description="Record the summary as a PROPOSED experiment-summary card. Never a side "
        "effect: the card is unreviewed and non-citable.",
    )


class BenchmarkRunRequest(BaseModel):
    policy: str = "baseline"
    case_set: str = "baseline"
    #  Off by default: a run with a real provider costs one model call per candidate
    #  per case, which is minutes on a local model.
    use_provider: bool = False
    provider: str | None = None
    model: str | None = None
    description: str = ""
    persist_results: bool = True


class BenchmarkResultResponse(BaseModel):
    policy_version: str
    case_set: str
    git_commit: str = ""
    status: str = ""
    overall_score: float = 0.0
    doc_recall: float | None = None
    page_recall: float | None = None
    reciprocal_rank: float | None = None
    page_precision: float | None = None
    citation_accuracy: float | None = None
    extraction_f1: float | None = None
    context_completeness: float | None = None
    abstention_f1: float | None = None
    unsupported_claim_rate: float = 0.0
    latency_ms: int = 0
    n_cases: int = 0
    n_cases_exercised: int = 0
    n_cases_skipped: int = 0
    extraction_available: bool = False
    retrievers: list[str] = Field(default_factory=list)
    by_category: dict = Field(default_factory=dict)
    cases: list[dict] = Field(default_factory=list)
    failures: list[dict] = Field(default_factory=list)
    notes: str = ""

