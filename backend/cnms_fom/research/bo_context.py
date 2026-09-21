"""The only path by which evidence may influence the optimizer.

Three operations, and the separation between them is the whole safety argument:

``propose``  writes a row.  Changes nothing about the campaign.
``review``   a named person accepts or rejects it.  Still changes nothing.
``apply``    a named person writes it into the campaign.  Requires a reviewed
             proposal, re-validates everything, and records the campaign's
             configuration fingerprint before and after.

Nothing a language model produces can reach ``apply``.  The status enum makes
``applied`` reachable only from ``reviewed``, two CHECK constraints require a
reviewer and an applier by name, and the cards a proposal rests on are re-checked
at *every* step — because a card edited between review and apply makes its own
review stale, and an approval granted on the old text must not silently cover the
new one.

What the bridge may change, and what it may not
-----------------------------------------------
It writes to ``BoRun.constraints`` and nothing else.  Not the search space, not the
objective, not the FOM definition, not an observation.  Within constraints it may
only *narrow*: ``ConstraintSet.apply`` already refuses to widen a space, and
:meth:`ProposedBOContext.widening_violations` refuses earlier and with a better
message.  A paper may persuade a scientist to narrow a search — that is what
evidence is for — but widening one past its instrument envelope is a claim about
what a tool can physically do, and no paper is a source for that.

Soft priors, process-window hints, and uncertainty notes are written to the
constraint ``notes``, which ``bo_engine.constraints`` already documents as "rules a
human must check".  They never become numbers the acquisition function sees: a
literature prior silently steering a GP produces a result nobody can attribute
afterwards.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cnms_fom.db.enums import ContextStatus
from cnms_fom.research.campaign import snapshot
from cnms_fom.research.contracts import BoundProposal, ProposedBOContext

logger = logging.getLogger(__name__)


class ContextRefused(PermissionError):
    """The proposal may not advance.

    ``PermissionError``, matching the other Sec. 15.2 guards: a rule about what the
    platform may assert, not a malformed input.
    """


class ProposalNotFound(LookupError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Card validation
# ---------------------------------------------------------------------------


def validate_supporting_cards(db, slugs: list[str]) -> tuple[list[dict], list[str]]:
    """Check every card a proposal rests on. Returns (accepted, problems).

    Run at propose, review, *and* apply. Not redundant: a card can be edited at any
    point between those calls, and the window between an approval and its
    application is exactly where an unnoticed edit does the most damage.

    A card qualifies only if it is reviewed, not stale, carries a resolved source,
    and sits in a category the bridge is allowed to read — a measurement caveat is
    worth a scientist's attention and is not a source of search-space bounds.
    """
    from cnms_fom.db.enums import CardStatus
    from cnms_fom.db.models import KnowledgeCard
    from cnms_fom.knowledge.cards import BO_RELEVANT_CATEGORIES, body_hash

    accepted: list[dict] = []
    problems: list[str] = []

    for slug in slugs:
        card = db.query(KnowledgeCard).filter(KnowledgeCard.slug == slug).one_or_none()
        if card is None:
            problems.append(f"Card {slug!r} does not exist.")
            continue
        if card.status is not CardStatus.REVIEWED:
            problems.append(
                f"Card {slug!r} is {card.status.value}, not reviewed. An unreviewed card is the "
                "assistant's own draft and cannot change a live campaign (FOM_PROOF Sec. 15.2)."
            )
            continue
        if card.review_is_stale:
            problems.append(
                f"Card {slug!r} was reviewed by {card.reviewed_by} but its body has been edited "
                "since, so the review no longer covers what it says. Re-review it before using it."
            )
            continue
        if not card.citable:
            problems.append(
                f"Card {slug!r} is reviewed but not citable — it carries no resolved source, so "
                "nothing on it is traceable (Sec. 2.2)."
            )
            continue
        if card.category not in BO_RELEVANT_CATEGORIES:
            problems.append(
                f"Card {slug!r} is category "
                f"{card.category.value if card.category else 'unset'}, which the bridge does not "
                "read. Only "
                + ", ".join(c.value for c in BO_RELEVANT_CATEGORIES)
                + " can supply campaign context."
            )
            continue

        accepted.append({
            "slug": card.slug,
            "category": card.category.value,
            "reviewed_by": card.reviewed_by,
            "reviewed_at": str(card.reviewed_at) if card.reviewed_at else None,
            #  The body hash at the moment it was accepted. Compared on apply.
            "body_sha256": body_hash(card.body),
        })

    return accepted, problems


def _recheck_card_hashes(db, recorded: list[dict]) -> list[str]:
    """Whether any card has changed since it was recorded on the proposal."""
    from cnms_fom.db.models import KnowledgeCard
    from cnms_fom.knowledge.cards import body_hash

    problems: list[str] = []
    for entry in recorded or []:
        slug = entry.get("slug")
        card = db.query(KnowledgeCard).filter(KnowledgeCard.slug == slug).one_or_none()
        if card is None:
            problems.append(f"Card {slug!r} has been deleted since this proposal was made.")
            continue
        if body_hash(card.body) != entry.get("body_sha256"):
            problems.append(
                f"Card {slug!r} has been edited since this proposal was made. The approval was "
                "granted against the earlier text and does not cover the current one — re-propose "
                "against the card as it now stands."
            )
    return problems


# ---------------------------------------------------------------------------
# Propose
# ---------------------------------------------------------------------------


def propose_context(
    db,
    context: ProposedBOContext,
    *,
    brief_id: int | None = None,
    proposed_by: str = "assistant",
):
    """Record a proposed change. Writes one row; changes no campaign.

    Refuses a proposal that would widen the space, name a parameter the campaign
    does not have, or rest on a card that cannot support it — at propose time rather
    than at apply time, so a reviewer is never shown something that could not be
    applied anyway.
    """
    from cnms_fom.db.models import CampaignContextProposal

    snap = snapshot(db, context.bo_run_id)

    if context.is_empty:
        raise ContextRefused(
            "The proposal is empty. A context change with no bounds, no exclusions, and no notes "
            "has nothing for a reviewer to decide."
        )

    violations = context.widening_violations(snap.search_space)
    if violations:
        raise ContextRefused(
            "This proposal cannot be applied to campaign "
            f"{context.bo_run_id}:\n  - " + "\n  - ".join(violations)
        )

    accepted_cards, card_problems = validate_supporting_cards(
        db, context.supporting_card_slugs
    )
    if card_problems:
        raise ContextRefused(
            "The cards this proposal rests on cannot support it:\n  - "
            + "\n  - ".join(card_problems)
        )
    if context.changes_search_behaviour and not accepted_cards:
        #  Advisory notes may stand on a brief alone. A change to what the optimizer
        #  may propose needs a reviewed card behind it, because that is the thing a
        #  person has actually signed off on.
        raise ContextRefused(
            "This proposal changes what the optimizer may propose (it sets bounds or excludes "
            "choices), so it must rest on at least one reviewed, sourced card in a category the "
            "bridge reads. A brief alone is not sufficient: a brief is unreviewed by construction."
        )

    proposal = CampaignContextProposal(
        bo_run_id=context.bo_run_id,
        brief_id=brief_id,
        status=ContextStatus.PROPOSED,
        recommended_bounds={
            p.parameter: [p.lower, p.upper] for p in context.recommended_bounds
        } or None,
        excluded_choices=context.excluded_choices or None,
        soft_priors=context.soft_priors or None,
        process_window_hints=context.process_window_hints or None,
        uncertainty_notes=context.uncertainty_notes or None,
        rationale=context.rationale or None,
        supporting_cards=accepted_cards or None,
        proposed_by=proposed_by,
        campaign_fingerprint_before=snap.fingerprint(),
    )
    db.add(proposal)
    db.flush()
    logger.info(
        "Proposed context #%s for campaign %s by %s (%d bound(s), %d advisory note(s))",
        proposal.id, context.bo_run_id, proposed_by,
        len(context.recommended_bounds),
        len(context.soft_priors) + len(context.process_window_hints)
        + len(context.uncertainty_notes),
    )
    return proposal


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


def review_context(
    db,
    proposal_id: int,
    *,
    reviewed_by: str,
    accept: bool = True,
    note: str | None = None,
):
    """Accept or reject a proposal. Still changes no campaign."""
    from cnms_fom.db.models import CampaignContextProposal

    proposal = db.get(CampaignContextProposal, proposal_id)
    if proposal is None:
        raise ProposalNotFound(f"No context proposal {proposal_id}.")
    if not (reviewed_by or "").strip():
        raise ContextRefused(
            "A review needs a named reviewer. 'Reviewed by nobody' is the state that would let "
            "model output change a campaign (FOM_PROOF Sec. 15.2)."
        )
    if proposal.status is ContextStatus.APPLIED:
        raise ContextRefused(
            f"Proposal {proposal_id} has already been applied by {proposal.applied_by}. Propose a "
            "new change rather than re-reviewing this one, so the campaign's history stays a chain."
        )
    if proposal.status is ContextStatus.REJECTED:
        raise ContextRefused(f"Proposal {proposal_id} was already rejected.")

    if accept:
        problems = _validate_for_advance(db, proposal)
        if problems:
            raise ContextRefused(
                f"Proposal {proposal_id} can no longer be accepted:\n  - " + "\n  - ".join(problems)
            )

    proposal.status = ContextStatus.REVIEWED if accept else ContextStatus.REJECTED
    proposal.reviewed_by = reviewed_by.strip()
    proposal.reviewed_at = _utcnow()
    proposal.review_note = note
    db.flush()
    logger.info(
        "Context #%s %s by %s", proposal_id, proposal.status.value, proposal.reviewed_by
    )
    return proposal


def _validate_for_advance(db, proposal) -> list[str]:
    """Everything that must still hold before a proposal advances."""
    problems = _recheck_card_hashes(db, proposal.supporting_cards or [])

    #  Re-validate the cards from scratch as well: one may have been un-reviewed,
    #  re-categorised, or had its sources removed without its body changing.
    slugs = [entry.get("slug") for entry in (proposal.supporting_cards or []) if entry.get("slug")]
    if slugs:
        _, card_problems = validate_supporting_cards(db, slugs)
        problems.extend(card_problems)

    #  And re-check against the live space: the campaign may have been narrowed by
    #  another proposal in the meantime, which can turn a narrowing into a widening.
    try:
        snap = snapshot(db, proposal.bo_run_id)
    except LookupError as exc:
        return [*problems, str(exc)]

    context = _context_from_record(proposal)
    problems.extend(context.widening_violations(snap.search_space))
    return problems


def _context_from_record(proposal) -> ProposedBOContext:
    return ProposedBOContext(
        bo_run_id=proposal.bo_run_id,
        recommended_bounds=[
            BoundProposal(
                parameter=name, lower=float(bounds[0]), upper=float(bounds[1]),
                rationale=proposal.rationale or "recorded on the proposal",
            )
            for name, bounds in (proposal.recommended_bounds or {}).items()
        ],
        excluded_choices=dict(proposal.excluded_choices or {}),
        soft_priors=list(proposal.soft_priors or []),
        process_window_hints=list(proposal.process_window_hints or []),
        uncertainty_notes=list(proposal.uncertainty_notes or []),
        rationale=proposal.rationale or "",
        supporting_card_slugs=[
            entry.get("slug") for entry in (proposal.supporting_cards or []) if entry.get("slug")
        ],
        status=proposal.status,
        proposed_by=proposal.proposed_by,
        reviewed_by=proposal.reviewed_by,
        reviewed_at=proposal.reviewed_at,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_context(db, proposal_id: int, *, applied_by: str) -> dict:
    """Write a reviewed proposal into the campaign's constraints.

    The only function in this package that modifies a ``BoRun``. It touches
    ``constraints`` and nothing else — never the search space, the objective, the
    FOM definition, or an observation.
    """
    from cnms_fom.bo_engine.constraints import ConstraintSet
    from cnms_fom.bo_engine.constraints import apply as intersect
    from cnms_fom.bo_engine.space import SearchSpace
    from cnms_fom.db.models import BoRun, CampaignContextProposal

    proposal = db.get(CampaignContextProposal, proposal_id)
    if proposal is None:
        raise ProposalNotFound(f"No context proposal {proposal_id}.")
    if not (applied_by or "").strip():
        raise ContextRefused("Applying a context change needs a named person.")

    if proposal.status is not ContextStatus.REVIEWED:
        raise ContextRefused(
            f"Proposal {proposal_id} is {proposal.status.value}; only a REVIEWED proposal can be "
            "applied. Nothing a language model produces reaches a live campaign without a person "
            "accepting it first (FOM_PROOF Sec. 15.2)."
        )

    #  Re-validated here even though review already did it. The window between an
    #  approval and its application is exactly where an unnoticed card edit does the
    #  most damage.
    problems = _validate_for_advance(db, proposal)
    if problems:
        proposal.status = ContextStatus.STALE
        db.flush()
        raise ContextRefused(
            f"Proposal {proposal_id} is no longer applicable and has been marked stale:\n  - "
            + "\n  - ".join(problems)
        )

    run = db.get(BoRun, proposal.bo_run_id)
    if run is None:
        raise ProposalNotFound(f"No BO campaign {proposal.bo_run_id}.")

    before = snapshot(db, run.id)
    constraints_before = dict(run.constraints or {})
    context = _context_from_record(proposal)

    merged = _merge_constraints(
        constraints_before,
        context,
        search_space=before.search_space,
        proposal_id=proposal.id,
        applied_by=applied_by,
    )

    #  Prove the result is satisfiable before storing it. An unsatisfiable campaign
    #  should fail here rather than at the next suggest() call, where the error
    #  would look like a modelling problem.
    try:
        intersect(SearchSpace.from_dict(before.search_space), ConstraintSet.from_dict(merged))
    except ValueError as exc:
        raise ContextRefused(
            f"Applying proposal {proposal_id} would leave the campaign unsatisfiable: {exc}"
        ) from exc

    run.constraints = merged
    db.flush()

    after = snapshot(db, run.id)
    proposal.status = ContextStatus.APPLIED
    proposal.applied_by = applied_by.strip()
    proposal.applied_at = _utcnow()
    proposal.constraints_before = constraints_before or None
    proposal.campaign_fingerprint_before = before.fingerprint()
    proposal.campaign_fingerprint_after = after.fingerprint()
    db.flush()

    logger.warning(
        "Campaign %s constraints changed by %s via proposal #%s: %s -> %s",
        run.id, applied_by, proposal.id,
        before.fingerprint()[:12], after.fingerprint()[:12],
    )
    return {
        "proposal_id": proposal.id,
        "bo_run_id": run.id,
        "applied_by": proposal.applied_by,
        "applied_at": str(proposal.applied_at),
        "campaign_fingerprint_before": proposal.campaign_fingerprint_before,
        "campaign_fingerprint_after": proposal.campaign_fingerprint_after,
        "fingerprint_changed": (
            proposal.campaign_fingerprint_before != proposal.campaign_fingerprint_after
        ),
        "constraints_before": constraints_before,
        "constraints_after": merged,
        "supporting_cards": proposal.supporting_cards,
        "note": (
            "Only BoRun.constraints changed. The search space, the objective, the FOM definition, "
            "and every observation are untouched. Advisory content was written to constraint "
            "notes, not to the acquisition function."
        ),
    }


def _merge_constraints(
    existing: dict,
    context: ProposedBOContext,
    *,
    search_space: dict,
    proposal_id: int,
    applied_by: str,
) -> dict:
    """Fold a proposal into a campaign's constraints, tightening only."""
    patch = context.as_constraint_patch()

    bounds = {k: list(v) for k, v in (existing.get("bounds") or {}).items()}
    for parameter, (low, high) in patch["bounds"].items():
        current = bounds.get(parameter)
        if current is None:
            bounds[parameter] = [low, high]
            continue
        #  Intersect rather than replace. Two proposals applied in sequence both
        #  narrow; a replacement would let the second silently undo the first.
        bounds[parameter] = [max(float(current[0]), low), min(float(current[1]), high)]

    allowed = {k: list(v) for k, v in (existing.get("allowed_choices") or {}).items()}
    parameters = {spec.get("name"): spec for spec in (search_space or {}).get("parameters", [])}
    for parameter, excluded in (context.excluded_choices or {}).items():
        spec = parameters.get(parameter) or {}
        current = allowed.get(parameter) or list(spec.get("choices") or [])
        allowed[parameter] = [choice for choice in current if choice not in excluded]

    notes = list(existing.get("notes") or [])
    stamp = f"[proposal #{proposal_id}, applied by {applied_by}]"
    notes.extend(f"{stamp} {note}" for note in patch["notes"])

    return {"bounds": bounds, "allowed_choices": allowed, "notes": notes}


# ---------------------------------------------------------------------------
# Revert and read
# ---------------------------------------------------------------------------


def revert_context(db, proposal_id: int, *, reverted_by: str) -> dict:
    """Restore the constraints an applied proposal replaced.

    Possible only because ``constraints_before`` was stored at apply time. Reverting
    the most recent apply is safe; reverting an older one would discard everything
    applied after it, so that is refused rather than done surprisingly.
    """
    from cnms_fom.db.models import BoRun, CampaignContextProposal

    proposal = db.get(CampaignContextProposal, proposal_id)
    if proposal is None:
        raise ProposalNotFound(f"No context proposal {proposal_id}.")
    if proposal.status is not ContextStatus.APPLIED:
        raise ContextRefused(f"Proposal {proposal_id} is {proposal.status.value}, not applied.")
    if not (reverted_by or "").strip():
        raise ContextRefused("Reverting a context change needs a named person.")

    later = (
        db.query(CampaignContextProposal)
        .filter(
            CampaignContextProposal.bo_run_id == proposal.bo_run_id,
            CampaignContextProposal.status == ContextStatus.APPLIED,
            CampaignContextProposal.applied_at > proposal.applied_at,
        )
        .count()
    )
    if later:
        raise ContextRefused(
            f"{later} proposal(s) have been applied to campaign {proposal.bo_run_id} since this "
            "one. Reverting it now would discard those as well. Revert them first, newest first."
        )

    run = db.get(BoRun, proposal.bo_run_id)
    run.constraints = proposal.constraints_before or {
        "bounds": {}, "allowed_choices": {}, "notes": []
    }
    proposal.status = ContextStatus.SUPERSEDED
    proposal.review_note = (
        f"{proposal.review_note or ''} [reverted by {reverted_by} at {_utcnow().isoformat()}]"
    ).strip()
    db.flush()

    after = snapshot(db, run.id)
    logger.warning(
        "Campaign %s constraints reverted by %s (proposal #%s)", run.id, reverted_by, proposal.id
    )
    return {
        "proposal_id": proposal.id,
        "bo_run_id": run.id,
        "reverted_by": reverted_by,
        "constraints_after": run.constraints,
        "campaign_fingerprint_after": after.fingerprint(),
    }


def list_proposals(db, bo_run_id: int) -> list[dict]:
    """Every proposal for one campaign, newest first — the change log."""
    from cnms_fom.db.models import CampaignContextProposal

    rows = (
        db.query(CampaignContextProposal)
        .filter(CampaignContextProposal.bo_run_id == bo_run_id)
        .order_by(CampaignContextProposal.id.desc())
        .all()
    )
    return [
        {
            "proposal_id": row.id,
            "bo_run_id": row.bo_run_id,
            "brief_id": row.brief_id,
            "status": row.status.value,
            "changes_search_behaviour": bool(row.recommended_bounds or row.excluded_choices),
            "recommended_bounds": row.recommended_bounds,
            "excluded_choices": row.excluded_choices,
            "soft_priors": row.soft_priors,
            "process_window_hints": row.process_window_hints,
            "uncertainty_notes": row.uncertainty_notes,
            "rationale": row.rationale,
            "supporting_cards": row.supporting_cards,
            "proposed_by": row.proposed_by,
            "reviewed_by": row.reviewed_by,
            "applied_by": row.applied_by,
            "applied_at": str(row.applied_at) if row.applied_at else None,
            "campaign_fingerprint_before": row.campaign_fingerprint_before,
            "campaign_fingerprint_after": row.campaign_fingerprint_after,
        }
        for row in rows
    ]
