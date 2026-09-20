"""/cards — the knowledge repository: concept pages, sources, findings, questions.

Retrieval re-derives an answer per query and keeps nothing.  Cards are where the
integration work is kept instead, so the second time a question is asked it starts
from the first answer's conclusions.

The endpoint set divides on one line.  Reading is open.  Writing lands a card as
``proposed``, which is readable and explicitly not citable, and ``POST
/cards/{slug}/review`` is the only way across that line — it needs a named person
and at least one resolved source, because a card is a language model's synthesis
written down and FOM_PROOF Sec. 15.2 says a synthesis is not evidence.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from cnms_fom.db.base import get_db
from cnms_fom.db.enums import CardStatus, CardType
from cnms_fom.knowledge.cards import (
    CardError,
    ReviewRefused,
    card_as_dict,
    card_graph,
    card_stats,
)
from cnms_fom.knowledge.cards import link_cards as _link
from cnms_fom.knowledge.cards import read_card as _read
from cnms_fom.knowledge.cards import review_card as _review
from cnms_fom.knowledge.cards import search_cards as _search
from cnms_fom.knowledge.cards import upsert_card as _upsert
from cnms_fom.schemas.cards import (
    CardGraphResponse,
    CardLinkRequest,
    CardListResponse,
    CardOut,
    CardReviewRequest,
    CardStatsResponse,
    CardWriteRequest,
)

router = APIRouter(prefix="/cards", tags=["cards"])


@router.get("", response_model=CardListResponse)
def list_cards(
    query: str | None = None,
    card_type: CardType | None = None,
    card_status: CardStatus | None = Query(default=None, alias="status"),
    tags: list[str] | None = Query(default=None),
    citable_only: bool = False,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> CardListResponse:
    """Search cards by title, slug, summary, or body.

    Lexical rather than semantic, deliberately: a card corpus is small and its
    titles were chosen by people, so substring matching finds what someone meant
    without an embedding round-trip. The document *chunks* need vectors; the index
    over them does not.
    """
    result = _search(
        db,
        query,
        card_type=card_type,
        status=card_status,
        tags=tags,
        citable_only=citable_only,
        limit=limit,
    )
    return CardListResponse(
        query=result.get("query"),
        n_cards=result["n_cards"],
        cards=[CardOut(**card) for card in result["cards"]],
        note=result.get("note", ""),
    )


@router.get("/stats", response_model=CardStatsResponse)
def stats(db: Session = Depends(get_db)) -> CardStatsResponse:
    """Corpus health: the review backlog, stale reviews, unresolved contradictions.

    ``awaiting_review`` is the number that decides whether this is knowledge or
    model output written down. ``unresolved_contradictions`` is the most valuable
    one: each is a disagreement between sources that somebody has to settle.
    """
    return CardStatsResponse(**card_stats(db))


@router.get("/graph", response_model=CardGraphResponse)
def graph(
    root_slug: str | None = None,
    depth: int = Query(default=2, ge=1, le=4),
    db: Session = Depends(get_db),
) -> CardGraphResponse:
    """The card graph as nodes and typed edges.

    Returned as data rather than a rendered diagram, so the caller decides: the
    assistant reads it as adjacency, a UI lays it out.
    """
    result = card_graph(db, root_slug, depth=depth)
    if "error" in result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, result["error"])
    return CardGraphResponse(**result)


@router.get("/{slug:path}", response_model=CardOut)
def get_card(slug: str, db: Session = Depends(get_db)) -> CardOut:
    """One card with its typed links in both directions.

    Backlinks are included because the incoming edges are often the useful ones —
    an incoming ``contradicts`` is a conflict the card's own author never wrote
    down.
    """
    try:
        result = _read(db, slug)
    except CardError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if not result.get("found"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, result.get("error", f"No card {slug!r}."))
    result.pop("found", None)
    return CardOut(**result)


@router.post("", response_model=CardOut, status_code=status.HTTP_201_CREATED)
def write_card(payload: CardWriteRequest, db: Session = Depends(get_db)) -> CardOut:
    """Create or update a card. It lands ``proposed`` and is not citable yet.

    Updating an already-reviewed card leaves its status alone but makes the review
    stale, so it stops being citable until someone looks again. That is not an
    error: correcting a reviewed card is normal, and forcing a new slug per
    correction would fragment the graph.
    """
    try:
        card = _upsert(
            db,
            slug=payload.slug,
            title=payload.title,
            body=payload.body,
            card_type=payload.card_type,
            summary=payload.summary,
            sources=payload.sources,
            tags=payload.tags,
            confidence=payload.confidence,
            authored_by=payload.authored_by,
            supersedes_slug=payload.supersedes_slug,
        )
        db.commit()
    except CardError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not write the card: {exc}"
        ) from exc
    return CardOut(**card_as_dict(card))


@router.post("/{slug:path}/review", response_model=CardOut)
def review(slug: str, payload: CardReviewRequest, db: Session = Depends(get_db)) -> CardOut:
    """Mark a card reviewed by a named person — the only way it becomes citable.

    Refused when the card has no resolved source. A card whose citations are all
    free-text has not been checked against anything the platform can re-read, and
    Sec. 2.2 requires provenance a reader can follow.
    """
    try:
        card = _review(db, slug, reviewed_by=payload.reviewed_by, confidence=payload.confidence)
        db.commit()
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ReviewRefused as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except CardError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return CardOut(**card_as_dict(card))


@router.post("/links", response_model=dict, status_code=status.HTTP_201_CREATED)
def link(payload: CardLinkRequest, db: Session = Depends(get_db)) -> dict:
    """Link two cards with a typed edge. Idempotent on (from, to, relation).

    A ``contradicts`` link requires a note. Recording that two cards disagree
    while discarding what they disagree about leaves nobody able to resolve it.
    """
    try:
        created = _link(
            db,
            payload.from_slug,
            payload.to_slug,
            relation=payload.relation,
            note=payload.note,
        )
        db.commit()
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except CardError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return {
        "from": payload.from_slug,
        "to": payload.to_slug,
        "relation": created.relation.value,
        "note": created.note,
    }
