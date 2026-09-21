"""/research — the evidence loop: brief, campaign context, experiment summary, benchmark.

The endpoints divide on exactly one line, and it is the line that matters:

**Read-only.** ``/brief``, ``/summary``, ``/benchmarks`` and every ``GET``. These
read the corpus, the cards, the fits and the campaign and return a document.
Producing one changes nothing scientific.

**Writes, gated.** ``/context/propose`` records a request. ``/context/{id}/review``
needs a named person. ``/context/{id}/apply`` needs a named person *and* a reviewed
proposal, and is the only route by which evidence reaches the optimizer — where it
touches a campaign's ``constraints`` and nothing else.

There is no endpoint that writes a property value, a descriptor value, a FOM score,
or a ModalFit record. That is not an omission: FOM_PROOF Sec. 15.2 forbids a
model-mediated path into the analysis tables, and the way to forbid one is not to
build it. A measured value enters through ``/materials`` or
``/modalfit/fits/{id}/promote``, both of which require a person.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from cnms_fom.db.base import get_db
from cnms_fom.research import bo_context, store
from cnms_fom.research.brief import generate_brief
from cnms_fom.research.campaign import snapshot
from cnms_fom.research.contracts import (
    BoundProposal,
    ProposedBOContext,
    ResearchContractError,
)
from cnms_fom.research.experiment import collect_outcome, propose_summary_card, summarise
from cnms_fom.research.policy import CANDIDATES, get_policy
from cnms_fom.schemas.research import (
    BenchmarkResultResponse,
    BenchmarkRunRequest,
    BriefRequest,
    BriefResponse,
    BriefReviewRequest,
    ContextApplyRequest,
    ContextProposalOut,
    ContextProposeRequest,
    ContextReviewRequest,
    ExperimentSummaryRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


def _provider(name: str | None, model: str | None):
    from cnms_fom.rag_backend.providers import get_provider

    try:
        return get_provider(name, model)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _policy(name: str):
    try:
        return get_policy(name)
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unknown policy {name!r}. Available: {sorted(CANDIDATES)}.",
        ) from exc


# ---------------------------------------------------------------------------
# Briefs
# ---------------------------------------------------------------------------


@router.post("/campaigns/{run_id}/brief", response_model=BriefResponse)
def campaign_brief(
    run_id: int, payload: BriefRequest, db: Session = Depends(get_db)
) -> BriefResponse:
    """A campaign-aware research brief. Reads everything; changes nothing.

    The campaign's own warnings — an unapproved objective, a stalled search,
    suggestions piled on a bound, a clamped fit parameter — are computed from the
    records and attached whether or not the model's narrative mentions them. A
    warning that depended on the model noticing it would not be a guardrail.
    """
    try:
        snapshot(db, run_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    return _build_brief(db, payload, bo_run_id=run_id)


@router.post("/brief", response_model=BriefResponse)
def standalone_brief(payload: BriefRequest, db: Session = Depends(get_db)) -> BriefResponse:
    """A brief with no campaign attached — a literature question on its own."""
    return _build_brief(db, payload, bo_run_id=None)


def _build_brief(db: Session, payload: BriefRequest, *, bo_run_id: int | None) -> BriefResponse:
    policy = _policy(payload.policy)
    if payload.techniques:
        policy = policy.evolve(techniques=tuple(t.value for t in payload.techniques))
    provider = _provider(payload.provider, payload.model)

    try:
        brief = generate_brief(
            db,
            payload.research_question,
            bo_run_id=bo_run_id,
            material=payload.material,
            sample_id=payload.sample_id,
            target_property=payload.target_property,
            provider=provider,
            policy=policy,
            include_cards=payload.include_cards,
            interpret=payload.interpret,
        )
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"Retrieval extra not installed: {exc}. pip install -e '.[rag]'",
        ) from exc
    except ResearchContractError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Brief generation failed: {exc}. For the local provider, check that Ollama is "
            "reachable and that the chat and embedding models are pulled.",
        ) from exc

    brief_id = None
    if payload.persist:
        try:
            record = store.save_brief(db, brief)
            db.commit()
            brief_id = record.id
        except Exception as exc:  # noqa: BLE001 - the brief is still worth returning
            db.rollback()
            logger.warning("Could not persist the brief: %s", exc)
            brief.warnings.append(
                f"This brief was not persisted ({exc}), so it has no audit record. The content "
                "below is unaffected."
            )

    payload_dict = brief.as_dict()
    return BriefResponse(
        brief_id=brief_id,
        research_question=brief.research_question,
        bo_run_id=brief.bo_run_id,
        status=brief.status.value,
        abstained=brief.abstained,
        evidence=payload_dict["evidence"],
        claims=payload_dict["claims"],
        contradictions=payload_dict["contradictions"],
        data_gaps=payload_dict["data_gaps"],
        statements=payload_dict["statements"],
        proposed_actions=brief.proposed_actions,
        warnings=brief.warnings,
        model=brief.model,
        provider=brief.provider,
        policy_version=brief.policy_version,
        fingerprint=payload_dict["fingerprint"],
        n_comparable_claims=payload_dict["n_comparable_claims"],
        n_incomplete_claims=payload_dict["n_incomplete_claims"],
        tool_calls=brief.tool_calls,
        disclaimer=payload_dict["disclaimer"],
    )


@router.get("/briefs", response_model=list[dict])
def list_briefs(
    bo_run_id: int | None = None,
    limit: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Recent briefs as summaries, newest first."""
    return store.list_briefs(db, bo_run_id=bo_run_id, limit=limit)


@router.get("/briefs/{brief_id}", response_model=dict)
def get_brief(brief_id: int, db: Session = Depends(get_db)) -> dict:
    """One stored brief, rehydrated with its claims and their provenance."""
    try:
        return store.load_brief(db, brief_id).as_dict()
    except store.BriefNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/briefs/{brief_id}/review", response_model=dict)
def review_brief(
    brief_id: int, payload: BriefReviewRequest, db: Session = Depends(get_db)
) -> dict:
    """Record that a person read a brief.

    Promotes nothing. A reviewed brief is a usable starting point for someone
    entering a ``PropertyValue`` by hand — never a substitute for doing so.
    """
    try:
        record = store.review_brief(
            db, brief_id, reviewed_by=payload.reviewed_by, accept=payload.accept
        )
        db.commit()
    except store.BriefNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except store.ReviewRefused as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return {
        "brief_id": record.id,
        "status": record.status.value,
        "reviewed_by": record.reviewed_by,
        "note": (
            "Reviewing a brief records that someone read it. No property value, descriptor, or "
            "FOM score changed, and none can change through this endpoint."
        ),
    }


@router.get("/claims/{field_name}", response_model=dict)
def claims_for_field(
    field_name: str,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict:
    """Every extracted claim for one field, across briefs, each with its page.

    Nothing is aggregated: two sources reporting different numbers under different
    conditions do not have a summary statistic (Sec. 2.1).
    """
    rows = store.claims_for_field(db, field_name, limit=limit)
    return {
        "field_name": field_name,
        "n_claims": len(rows),
        "claims": rows,
        "note": (
            "These are literature extractions, not measurements. None has entered "
            "property_values, and each is a candidate a person would have to enter there with "
            "its DOI, page, and full context."
        ),
    }


# ---------------------------------------------------------------------------
# Campaign context
# ---------------------------------------------------------------------------


@router.get("/campaigns/{run_id}/snapshot", response_model=dict)
def campaign_snapshot(run_id: int, db: Session = Depends(get_db)) -> dict:
    """The campaign's state and the warnings it earns. Deterministic, no model."""
    try:
        return snapshot(db, run_id).as_dict()
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/campaigns/{run_id}/context", response_model=list[ContextProposalOut])
def list_context_proposals(run_id: int, db: Session = Depends(get_db)) -> list[ContextProposalOut]:
    """Every proposal for one campaign — its configuration change log."""
    return [ContextProposalOut(**row) for row in bo_context.list_proposals(db, run_id)]


@router.post(
    "/campaigns/{run_id}/context/propose",
    response_model=ContextProposalOut,
    status_code=status.HTTP_201_CREATED,
)
def propose_context(
    run_id: int, payload: ContextProposeRequest, db: Session = Depends(get_db)
) -> ContextProposalOut:
    """Record a proposed change. Writes one row; changes no campaign.

    Refused when it would widen the search space, name a parameter the campaign does
    not have, or rest on a card that cannot support it — at propose time, so a
    reviewer is never shown something that could not be applied anyway.
    """
    try:
        context = ProposedBOContext(
            bo_run_id=run_id,
            recommended_bounds=[
                BoundProposal(
                    parameter=item.parameter, lower=item.lower, upper=item.upper,
                    rationale=item.rationale, card_slug=item.card_slug,
                )
                for item in payload.recommended_bounds
            ],
            excluded_choices=payload.excluded_choices,
            soft_priors=payload.soft_priors,
            process_window_hints=payload.process_window_hints,
            uncertainty_notes=payload.uncertainty_notes,
            rationale=payload.rationale,
            supporting_card_slugs=payload.supporting_card_slugs,
        )
    except ResearchContractError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    try:
        proposal = bo_context.propose_context(
            db, context, brief_id=payload.brief_id, proposed_by=payload.proposed_by
        )
        db.commit()
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except bo_context.ContextRefused as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return _proposal_out(db, proposal.id)


@router.post("/campaigns/{run_id}/context/{proposal_id}/review", response_model=ContextProposalOut)
def review_context(
    run_id: int,
    proposal_id: int,
    payload: ContextReviewRequest,
    db: Session = Depends(get_db),
) -> ContextProposalOut:
    """Accept or reject a proposal. Still changes no campaign.

    Re-validates the cards it rests on, because one may have been edited since it
    was proposed — and an approval granted on the old text must not cover the new.
    """
    try:
        bo_context.review_context(
            db, proposal_id, reviewed_by=payload.reviewed_by,
            accept=payload.accept, note=payload.note,
        )
        db.commit()
    except bo_context.ProposalNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except bo_context.ContextRefused as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    return _proposal_out(db, proposal_id)


@router.post("/campaigns/{run_id}/context/{proposal_id}/apply", response_model=dict)
def apply_context(
    run_id: int,
    proposal_id: int,
    payload: ContextApplyRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Write a reviewed proposal into the campaign's constraints.

    The only endpoint by which evidence reaches the optimizer. It requires a
    reviewed proposal and a named person, re-validates the supporting cards, and
    records the campaign's configuration fingerprint before and after. It touches
    ``constraints`` only — never the search space, the objective, the FOM
    definition, or an observation.
    """
    try:
        result = bo_context.apply_context(db, proposal_id, applied_by=payload.applied_by)
        db.commit()
    except bo_context.ProposalNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except bo_context.ContextRefused as exc:
        #  A stale proposal is marked stale by the bridge, which is a write worth
        #  keeping even though the apply failed.
        db.commit()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return result


@router.post("/campaigns/{run_id}/context/{proposal_id}/revert", response_model=dict)
def revert_context(
    run_id: int,
    proposal_id: int,
    payload: ContextApplyRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Restore the constraints an applied proposal replaced.

    Possible only because the previous constraints were stored at apply time.
    Reverting anything but the most recent apply is refused, since it would silently
    discard everything applied after it.
    """
    try:
        result = bo_context.revert_context(db, proposal_id, reverted_by=payload.applied_by)
        db.commit()
    except bo_context.ProposalNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except bo_context.ContextRefused as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return result


def _proposal_out(db: Session, proposal_id: int) -> ContextProposalOut:
    from cnms_fom.db.models import CampaignContextProposal

    row = db.get(CampaignContextProposal, proposal_id)
    if row is None:
        #  Only reachable if the row vanished between the service call and this
        #  serialisation, which would mean a concurrent delete. Reported rather than
        #  crashing on an attribute of None.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Context proposal {proposal_id} no longer exists.",
        )
    return ContextProposalOut(
        proposal_id=row.id,
        bo_run_id=row.bo_run_id,
        brief_id=row.brief_id,
        status=row.status.value,
        changes_search_behaviour=bool(row.recommended_bounds or row.excluded_choices),
        recommended_bounds=row.recommended_bounds,
        excluded_choices=row.excluded_choices,
        soft_priors=row.soft_priors,
        process_window_hints=row.process_window_hints,
        uncertainty_notes=row.uncertainty_notes,
        rationale=row.rationale,
        supporting_cards=row.supporting_cards,
        proposed_by=row.proposed_by,
        reviewed_by=row.reviewed_by,
        applied_by=row.applied_by,
        applied_at=str(row.applied_at) if row.applied_at else None,
        campaign_fingerprint_before=row.campaign_fingerprint_before,
        campaign_fingerprint_after=row.campaign_fingerprint_after,
    )


# ---------------------------------------------------------------------------
# Experiment summaries
# ---------------------------------------------------------------------------


@router.post("/experiments/{experiment_id}/summary", response_model=dict)
def experiment_summary(
    experiment_id: int, payload: ExperimentSummaryRequest, db: Session = Depends(get_db)
) -> dict:
    """Summarise one completed experiment. Updates nothing.

    The outcome is assembled from records — the recipe, the optimizer's prediction,
    the fits, their plausibility, the FOM result and its status. Every sentence in
    the narrative is labelled evidence, interpretation, or proposal, because people
    quote summaries onward and that is the boundary they lose first.
    """
    from cnms_fom.db.models import Experiment

    if db.get(Experiment, experiment_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No experiment {experiment_id}.")

    provider = _provider(payload.provider, payload.model) if payload.provider else None
    if provider is None:
        #  Default to the configured provider, but allow a records-only summary when
        #  no model is reachable: the deterministic half is the valuable half.
        try:
            provider = _provider(None, payload.model)
        except HTTPException:
            provider = None

    outcome = collect_outcome(
        db,
        experiment_id=experiment_id,
        bo_run_id=payload.bo_run_id,
        sample_id=payload.sample_id,
        observation_id=payload.observation_id,
    )
    try:
        summary = summarise(db, outcome, provider=provider)
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, f"Provider extra not installed: {exc}"
        ) from exc

    if payload.propose_card:
        try:
            propose_summary_card(db, summary)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            outcome.warnings.append(f"The summary card could not be recorded: {exc}")

    return summary.as_dict()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@router.get("/policies", response_model=dict)
def list_policies() -> dict:
    """The named research policies a benchmark run can use."""
    from cnms_fom.research.policy import BASELINE

    return {
        "baseline": BASELINE.version,
        "policies": {
            name: {
                "version": policy.version,
                "notes": policy.notes,
                "diff_from_baseline": {k: list(v) for k, v in BASELINE.diff(policy).items()},
            }
            for name, policy in sorted(CANDIDATES.items())
        },
    }


@router.post("/benchmarks/run", response_model=BenchmarkResultResponse)
def run_benchmark_endpoint(payload: BenchmarkRunRequest) -> BenchmarkResultResponse:
    """Run the benchmark in a throwaway database. Touches nothing real.

    Offline by default: without ``use_provider`` a stub grader exercises retrieval,
    ranking, citation validity and the abstention plumbing with no model server, and
    extraction metrics come back unavailable rather than zero.
    """
    from cnms_fom.research.benchmark import DEFAULT_RESULTS_PATH, run_benchmark

    policy = _policy(payload.policy)
    provider = None
    if payload.use_provider:
        provider = _provider(payload.provider, payload.model)

    try:
        outcome = run_benchmark(
            policy=policy,
            case_set=payload.case_set,
            provider=provider,
            description=payload.description,
            results_path=DEFAULT_RESULTS_PATH if payload.persist_results else None,
        )
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown case set: {exc}"
        ) from exc

    body = outcome.result.as_dict()
    body["status"] = outcome.status
    body["notes"] = outcome.description
    return BenchmarkResultResponse(**body)


@router.get("/benchmarks/{benchmark_id}/results", response_model=dict)
def benchmark_results(benchmark_id: str) -> dict:
    """Recorded runs for one case set, newest last.

    ``benchmark_id`` is the case-set name. Rows come from the tab-separated results
    file, each naming the commit and the policy fingerprint that produced it — a
    result that cannot name the code and configuration behind it is not reproducible.
    """
    from cnms_fom.research.benchmark import CASE_SETS, read_results

    if benchmark_id not in CASE_SETS:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown benchmark {benchmark_id!r}. Available: {sorted(CASE_SETS)}.",
        )

    rows = [row for row in read_results() if f"cases={benchmark_id}" in row.get("description", "")]
    return {
        "benchmark_id": benchmark_id,
        "n_runs": len(rows),
        "runs": rows,
        "n_cases": len(CASE_SETS[benchmark_id]),
    }


# ---------------------------------------------------------------------------
# The per-passage call cache
# ---------------------------------------------------------------------------


@router.get("/cache", response_model=dict)
def cache_stats(db: Session = Depends(get_db)) -> dict:
    """What the per-passage call cache holds, and how many calls it has avoided.

    Grading and extraction are 18 of the 19 model calls a brief makes, so
    ``calls_avoided`` is the number that says whether this is earning its keep.
    """
    from cnms_fom.rag_backend import cache

    return cache.stats(db)


@router.delete("/cache", response_model=dict)
def clear_cache(
    kind: str | None = Query(default=None, description="'extraction' or 'grade'."),
    model: str | None = Query(default=None, description="Clear one model's entries only."),
    db: Session = Depends(get_db),
) -> dict:
    """Clear cached model calls.

    Safe by construction: the cache holds a transcript of model calls keyed by a hash
    of the content that was sent. It stores no measurement and nothing citable, so
    clearing it costs time and nothing else.
    """
    from cnms_fom.rag_backend import cache

    removed = cache.clear(db, kind=kind, model=model)
    db.commit()
    return {
        "removed": removed,
        "kind": kind,
        "model": model,
        "note": "Nothing scientific was deleted; the next run will re-make these calls.",
    }
