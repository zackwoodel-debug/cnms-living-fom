"""Persisting research briefs, claims, and context proposals.

The translation layer between the dataclasses in :mod:`contracts` — which know the
domain rules and nothing about SQLAlchemy — and the tables that keep them.

One rule shapes the whole module: **saving a brief writes only to the evidence
tables.** There is no branch here that touches ``property_values``,
``descriptor_values``, ``fom_scores``, ``fit_records``, or a ``BoRun``. Persisting
a brief is a filing operation, and the only thing it can cause is a person reading
it later.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cnms_fom.db.enums import BriefStatus, ClaimStatus, ClaimTier
from cnms_fom.research.contracts import (
    Contradiction,
    DataGap,
    EvidenceItem,
    ExtractedClaim,
    LabelledStatement,
    ResearchBrief,
)

logger = logging.getLogger(__name__)

#  Tables this module is permitted to write. Asserted in the test suite, because a
#  comment saying "we do not write measurements" is not a guarantee and a list that
#  a test checks against the module's imports is closer to one.
WRITABLE_TABLES: frozenset[str] = frozenset(
    {"research_briefs", "research_claims", "campaign_context_proposals"}
)


class BriefNotFound(LookupError):
    pass


class ReviewRefused(PermissionError):
    """A brief may not be marked reviewed.

    ``PermissionError``, matching the other Sec. 15.2 guards in this codebase: a
    rule about what the platform may assert, not a malformed input.
    """


def save_brief(db, brief: ResearchBrief, *, material_id: int | None = None):
    """Write a brief and its claims. Returns the ``ResearchBriefRecord``.

    Claims land in their own table with the provenance columns populated from
    their first evidence item, and the remaining items in ``evidence``. The first
    item is the primary locator rather than an arbitrary one: ``brief`` orders a
    claim's evidence by grade, so the best-supported passage is the one a reader
    is sent to.
    """
    from cnms_fom.db.models import ResearchBriefRecord

    record = ResearchBriefRecord(
        research_question=brief.research_question,
        bo_run_id=brief.bo_run_id,
        experiment_id=brief.experiment_id,
        material_id=material_id,
        material=brief.material,
        specimen_form=brief.specimen_form,
        target_property=brief.target_property,
        fom_definition=brief.fom_definition,
        status=brief.status,
        abstained=brief.abstained,
        evidence=[item.as_dict() for item in brief.evidence] or None,
        contradictions=[c.as_dict() for c in brief.contradictions] or None,
        data_gaps=[g.as_dict() for g in brief.data_gaps] or None,
        statements=[s.as_dict() for s in brief.statements] or None,
        proposed_actions=brief.proposed_actions or None,
        proposed_card_slugs=brief.proposed_card_slugs or None,
        warnings=brief.warnings or None,
        tool_calls=brief.tool_calls or None,
        model=brief.model or None,
        provider=brief.provider or None,
        policy_version=brief.policy_version or None,
        fingerprint=brief.fingerprint(),
    )
    db.add(record)
    db.flush()

    for claim in brief.claims:
        db.add(_claim_record(record.id, claim))
    db.flush()

    logger.info(
        "Saved brief #%s: %d claim(s), %d gap(s), %d contradiction(s), abstained=%s",
        record.id, len(brief.claims), len(brief.data_gaps), len(brief.contradictions),
        brief.abstained,
    )
    return record


def _claim_record(brief_id: int, claim: ExtractedClaim):
    from cnms_fom.db.models import ExtractedClaimRecord

    primary = claim.evidence[0]
    return ExtractedClaimRecord(
        brief_id=brief_id,
        field_name=claim.field_name,
        value=claim.value,
        units=claim.units,
        value_text=claim.value_text,
        normalized_value=claim.normalized_value,
        normalized_units=claim.normalized_units,
        normalization_note=claim.normalization_note or None,
        tier=claim.tier,
        status=claim.status,
        context=claim.context or None,
        #  Computed at write time so "which claims are missing their frequency?" is
        #  a query rather than a full scan plus a recomputation.
        missing_context=claim.missing_context or None,
        document_id=primary.document_id,
        content_sha256=primary.content_sha256,
        document_title=primary.document_title,
        page=primary.page,
        chunk_id=primary.chunk_id,
        quote=primary.quote,
        doi=primary.doi,
        evidence=[item.as_dict() for item in claim.evidence[1:]] or None,
        extracted_by_model=claim.extracted_by_model or None,
        extracted_by_provider=claim.extracted_by_provider or None,
        prompt_version=claim.prompt_version or None,
        model_confidence=claim.model_confidence,
        extracted_at=claim.extracted_at,
        notes=claim.notes or None,
    )


def load_brief(db, brief_id: int) -> ResearchBrief:
    """Rehydrate a stored brief into its contract form.

    Round-trips so the benchmark can re-score a brief without re-running the
    model, and so a reviewer sees the same object the generator produced.
    """
    from cnms_fom.db.models import ResearchBriefRecord

    record = db.get(ResearchBriefRecord, brief_id)
    if record is None:
        raise BriefNotFound(f"No research brief {brief_id}.")
    return brief_from_record(record)


def brief_from_record(record) -> ResearchBrief:
    claims = [_claim_from_record(row) for row in record.claims]

    #  A contradiction carries its own copies of both claims rather than pointing
    #  at rows, so deleting a claim later cannot orphan the disagreement — which is
    #  the part of a brief most worth keeping.
    contradictions: list[Contradiction] = []
    for payload in record.contradictions or []:
        left, right = payload.get("left"), payload.get("right")
        if not (left and right):
            continue
        try:
            contradictions.append(
                Contradiction(
                    field_name=payload["field_name"],
                    left=_claim_from_dict(left),
                    right=_claim_from_dict(right),
                    basis=payload.get("basis") or "recorded without a basis",
                    differing_context={
                        k: tuple(v) for k, v in (payload.get("differing_context") or {}).items()
                    },
                )
            )
        except (KeyError, ValueError) as exc:  # noqa: PERF203 - one bad row must not lose the brief
            logger.warning("Skipped an unreadable contradiction on brief %s: %s", record.id, exc)

    return ResearchBrief(
        research_question=record.research_question,
        bo_run_id=record.bo_run_id,
        experiment_id=record.experiment_id,
        material=record.material,
        specimen_form=record.specimen_form,
        target_property=record.target_property,
        fom_definition=record.fom_definition,
        evidence=[_evidence_from_dict(d) for d in (record.evidence or [])],
        claims=claims,
        contradictions=contradictions,
        data_gaps=[
            DataGap(
                question=d.get("question", ""),
                what_was_searched=d.get("what_was_searched", ""),
                what_would_resolve_it=d.get("what_would_resolve_it") or "not recorded",
                field_name=d.get("field_name"),
            )
            for d in (record.data_gaps or [])
        ],
        statements=[_statement_from_dict(d) for d in (record.statements or [])],
        proposed_actions=list(record.proposed_actions or []),
        proposed_card_slugs=list(record.proposed_card_slugs or []),
        warnings=list(record.warnings or []),
        status=record.status,
        model=record.model or "",
        provider=record.provider or "",
        policy_version=record.policy_version or "",
        tool_calls=list(record.tool_calls or []),
        created_at=record.created_at or datetime.now(timezone.utc),
        abstained=record.abstained,
    )


def _evidence_from_dict(payload: dict) -> EvidenceItem:
    return EvidenceItem(
        document_id=payload.get("document_id"),
        document_title=payload.get("document_title") or "unrecorded",
        page=payload.get("page"),
        quote=payload.get("quote") or "(quote not recorded)",
        content_sha256=payload.get("content_sha256"),
        chunk_id=payload.get("chunk_id"),
        doi=payload.get("doi"),
        source_url=payload.get("source_url"),
        technique=payload.get("technique"),
        retrieval_method=payload.get("retrieval_method") or "",
        retrieval_rank=payload.get("retrieval_rank"),
        grade=payload.get("grade"),
        grade_reason=payload.get("grade_reason") or "",
    )


def _statement_from_dict(payload: dict) -> LabelledStatement:
    from cnms_fom.db.enums import StatementKind

    return LabelledStatement(
        kind=StatementKind(payload.get("kind", "interpretation")),
        text=payload.get("text", ""),
        evidence=[_evidence_from_dict(d) for d in (payload.get("evidence") or [])],
    )


def _claim_from_dict(payload: dict) -> ExtractedClaim:
    return ExtractedClaim(
        field_name=payload.get("field_name", "unrecorded"),
        evidence=[_evidence_from_dict(d) for d in (payload.get("evidence") or [])]
        or [
            EvidenceItem(
                document_id=None,
                document_title="unrecorded",
                page=None,
                quote="(quote not recorded)",
            )
        ],
        value=payload.get("value"),
        units=payload.get("units"),
        value_text=payload.get("value_text"),
        normalized_value=payload.get("normalized_value"),
        normalized_units=payload.get("normalized_units"),
        normalization_note=payload.get("normalization_note") or "",
        tier=ClaimTier(payload.get("tier", "reported")),
        status=ClaimStatus(payload.get("status", "candidate")),
        context=dict(payload.get("context") or {}),
        model_confidence=payload.get("model_confidence"),
        extracted_by_model=payload.get("extracted_by_model") or "",
        extracted_by_provider=payload.get("extracted_by_provider") or "",
        prompt_version=payload.get("prompt_version") or "",
        notes=payload.get("notes") or "",
    )


def _claim_from_record(row) -> ExtractedClaim:
    primary = EvidenceItem(
        document_id=row.document_id,
        document_title=row.document_title or "unrecorded",
        page=row.page,
        quote=row.quote,
        content_sha256=row.content_sha256,
        chunk_id=row.chunk_id,
        doi=row.doi,
    )
    return ExtractedClaim(
        field_name=row.field_name,
        evidence=[primary, *(_evidence_from_dict(d) for d in (row.evidence or []))],
        value=row.value,
        units=row.units,
        value_text=row.value_text,
        normalized_value=row.normalized_value,
        normalized_units=row.normalized_units,
        normalization_note=row.normalization_note or "",
        tier=row.tier,
        status=row.status,
        context=dict(row.context or {}),
        model_confidence=row.model_confidence,
        extracted_by_model=row.extracted_by_model or "",
        extracted_by_provider=row.extracted_by_provider or "",
        prompt_version=row.prompt_version or "",
        extracted_at=row.extracted_at or datetime.now(timezone.utc),
        notes=row.notes or "",
    )


def review_brief(db, brief_id: int, *, reviewed_by: str, accept: bool = True):
    """Mark a brief reviewed or rejected by a named person.

    Reviewing a brief does not promote anything. It records that someone read it,
    which is what makes the claims in it usable as a starting point for a human
    entering a ``PropertyValue`` — never a substitute for doing so.
    """
    from cnms_fom.db.models import ResearchBriefRecord

    record = db.get(ResearchBriefRecord, brief_id)
    if record is None:
        raise BriefNotFound(f"No research brief {brief_id}.")
    if not (reviewed_by or "").strip():
        raise ReviewRefused(
            "A review needs a named reviewer. 'Reviewed by nobody' is the state that would let "
            "model output pass as checked (FOM_PROOF Sec. 15.2)."
        )

    record.status = BriefStatus.REVIEWED if accept else BriefStatus.REJECTED
    record.reviewed_by = reviewed_by.strip()
    record.reviewed_at = datetime.now(timezone.utc)
    db.flush()
    return record


def list_briefs(db, *, bo_run_id: int | None = None, limit: int = 25) -> list[dict]:
    """Recent briefs, newest first, as summaries."""
    from cnms_fom.db.models import ResearchBriefRecord

    query = db.query(ResearchBriefRecord)
    if bo_run_id is not None:
        query = query.filter(ResearchBriefRecord.bo_run_id == bo_run_id)
    records = (
        query.order_by(ResearchBriefRecord.created_at.desc(), ResearchBriefRecord.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "brief_id": record.id,
            "research_question": record.research_question,
            "bo_run_id": record.bo_run_id,
            "status": record.status.value,
            "abstained": record.abstained,
            "n_claims": len(record.claims),
            "n_data_gaps": len(record.data_gaps or []),
            "n_contradictions": len(record.contradictions or []),
            "n_warnings": len(record.warnings or []),
            "model": record.model,
            "policy_version": record.policy_version,
            "fingerprint": record.fingerprint,
            "created_at": str(record.created_at) if record.created_at else None,
            "reviewed_by": record.reviewed_by,
        }
        for record in records
    ]


def claims_for_field(db, field_name: str, *, limit: int = 50) -> list[dict]:
    """Every extracted claim for one field, across briefs.

    The query the corpus exists to answer: "what has anyone reported for this?"
    Returns each claim separately with its context and page. Nothing is aggregated,
    because two sources reporting different numbers under different conditions do
    not have a summary statistic (Sec. 2.1).
    """
    from cnms_fom.db.models import ExtractedClaimRecord

    rows = (
        db.query(ExtractedClaimRecord)
        .filter(ExtractedClaimRecord.field_name == field_name)
        .order_by(ExtractedClaimRecord.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "claim_id": row.id,
            "brief_id": row.brief_id,
            "field_name": row.field_name,
            "value": row.value,
            "units": row.units,
            "value_text": row.value_text,
            "tier": row.tier.value,
            "status": row.status.value,
            "context": row.context or {},
            "missing_context": row.missing_context or [],
            "citation": ", ".join(
                part
                for part in (
                    row.document_title,
                    f"p. {row.page}" if row.page is not None else None,
                    f"doi:{row.doi}" if row.doi else None,
                )
                if part
            ),
            "quote": row.quote,
            "is_measurement": False,
        }
        for row in rows
    ]
