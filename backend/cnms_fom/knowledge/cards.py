"""Reading and writing knowledge cards.

The pattern is a Dynamic Knowledge Repository: integrate each source into the
wiki when it arrives, rather than re-deriving an answer from raw chunks on every
query.  Ask the same question twice of a plain retrieval pipeline and it reasons
from scratch, having learned nothing in between; ask it of a corpus of curated
cards and the second answer starts from the first one's conclusions.

What keeps that safe here is the review gate, and it is worth being precise about
why it exists.  A card is where a language model's synthesis gets written down.
FOM_PROOF Sec. 15.2 says such a synthesis is not evidence — so an
assistant-written card is ``PROPOSED``, and ``citable`` is False until a person
has read it against its sources.  Proposed cards are still returned, labelled,
because a hidden draft helps nobody; what they may not do is be built on.

Three invariants, all enforced here rather than trusted:

* A card cannot be reviewed without a named reviewer (also a CHECK constraint).
* A card cannot be reviewed with no sources. A factual claim with no source is an
  assertion, and signing one off is the exact failure Sec. 2.2 guards against.
* A body edited after review makes the review stale. Without that, a card could
  be approved and then rewritten and would still read as checked.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone

from cnms_fom.db.enums import CardRelation, CardStatus, CardType

logger = logging.getLogger(__name__)

#  Slugs are the citation handle, so they are constrained: lower case, path-like,
#  no spaces. Renaming one breaks every reference to it, which is why it is
#  validated on write rather than normalised silently.
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*(?:/[a-z0-9]+(?:[-_][a-z0-9]+)*)*$")

#  Conventional prefixes per type. Not enforced — a wiki that rejects a slug for
#  being in the wrong folder is a wiki nobody writes in — but suggested, because
#  a consistent namespace is what makes `list` readable.
SLUG_PREFIX: dict[CardType, str] = {
    CardType.CONCEPT: "concepts/",
    CardType.SOURCE: "sources/",
    CardType.METHOD: "methods/",
    CardType.FINDING: "findings/",
    CardType.QUESTION: "questions/",
}


class CardError(ValueError):
    """The card cannot be written as given."""


class ReviewRefused(PermissionError):
    """The card may not be marked reviewed.

    ``PermissionError``, matching the other Sec. 15.2 guards: this is a rule about
    what the platform may assert, not a malformed input.
    """


def body_hash(body: str) -> str:
    return hashlib.sha256((body or "").encode("utf-8")).hexdigest()


def validate_slug(slug: str) -> str:
    cleaned = (slug or "").strip().lower()
    if not SLUG_PATTERN.match(cleaned):
        raise CardError(
            f"{slug!r} is not a usable slug. Expected lower-case, path-like segments such as "
            "'concepts/ald-window-hfo2'. The slug is the citation handle, so it is validated "
            "rather than normalised — a silently rewritten slug breaks every reference to it."
        )
    return cleaned


def normalise_sources(sources: list | None) -> list[dict]:
    """Coerce a source list into citation records, dropping nothing silently.

    A string is accepted and kept as a free-text locator, because a card citing
    "Kim 2024, Table 2" before the PDF is ingested is better than one citing
    nothing. It is marked ``kind='freetext'`` so it can be found and upgraded
    later, and so ``review`` can tell it apart from a resolved citation.
    """
    out: list[dict] = []
    for entry in sources or []:
        if isinstance(entry, dict):
            kind = entry.get("kind") or (
                "document" if entry.get("document_id") else
                "fit_record" if entry.get("fit_record_id") else
                "material" if entry.get("material_id") else
                "freetext"
            )
            out.append({**entry, "kind": kind})
        elif isinstance(entry, str) and entry.strip():
            out.append({"kind": "freetext", "locator": entry.strip()})
    return out


def upsert_card(
    db,
    *,
    slug: str,
    title: str,
    body: str,
    card_type: CardType | str = CardType.CONCEPT,
    summary: str | None = None,
    sources: list | None = None,
    tags: list[str] | None = None,
    confidence: float | None = None,
    authored_by: str = "assistant",
    supersedes_slug: str | None = None,
):
    """Create a card, or update the one at this slug.

    An edit to an already-reviewed card leaves ``status`` alone but makes the
    review stale via the body hash, so the card stops being citable until someone
    looks again. That is deliberately not an error: correcting a reviewed card is
    normal, and forcing a new slug for every correction would fragment the graph.
    """
    from cnms_fom.db.models import KnowledgeCard

    slug = validate_slug(slug)
    if not (title or "").strip():
        raise CardError("A card needs a title.")
    if not (body or "").strip():
        raise CardError(
            "A card needs a body. An empty card is a slug that other cards can link to and "
            "nobody can check."
        )
    kind = CardType(card_type) if not isinstance(card_type, CardType) else card_type
    if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
        raise CardError(f"confidence must be in [0, 1], got {confidence!r}.")

    existing = db.query(KnowledgeCard).filter(KnowledgeCard.slug == slug).one_or_none()
    resolved_sources = normalise_sources(sources)

    supersedes_id = None
    if supersedes_slug:
        target = (
            db.query(KnowledgeCard)
            .filter(KnowledgeCard.slug == validate_slug(supersedes_slug))
            .one_or_none()
        )
        if target is None:
            raise CardError(f"Cannot supersede {supersedes_slug!r}: no such card.")
        if existing is not None and target.id == existing.id:
            raise CardError("A card cannot supersede itself.")
        supersedes_id = target.id
        target.status = CardStatus.SUPERSEDED

    if existing is None:
        card = KnowledgeCard(
            slug=slug,
            card_type=kind,
            title=title.strip(),
            body=body,
            summary=summary,
            sources=resolved_sources or None,
            tags=tags or None,
            confidence=confidence,
            authored_by=authored_by,
            status=CardStatus.PROPOSED,
            supersedes_id=supersedes_id,
        )
        db.add(card)
        db.flush()
        logger.info("Created card %s (%s)", slug, kind.value)
        return card

    existing.title = title.strip()
    existing.body = body
    existing.card_type = kind
    if summary is not None:
        existing.summary = summary
    if resolved_sources:
        #  Union rather than replace: a second source for the same claim is
        #  additional evidence, and an update that dropped the first one would
        #  quietly weaken a card while looking like an improvement.
        merged = {repr(sorted(s.items())): s for s in (existing.sources or [])}
        for source in resolved_sources:
            merged.setdefault(repr(sorted(source.items())), source)
        existing.sources = list(merged.values())
    if tags:
        existing.tags = sorted(set(existing.tags or []) | set(tags))
    if confidence is not None:
        existing.confidence = confidence
    if supersedes_id is not None:
        existing.supersedes_id = supersedes_id
    db.flush()
    logger.info("Updated card %s", slug)
    return existing


def review_card(db, slug: str, *, reviewed_by: str, confidence: float | None = None):
    """Mark a card reviewed by a named person.

    Refuses a card with no resolved source. A card whose only citations are
    free-text locators has not been checked against anything the platform can
    re-read, and Sec. 2.2 requires provenance a reader can follow.
    """
    from cnms_fom.db.models import KnowledgeCard

    card = db.query(KnowledgeCard).filter(KnowledgeCard.slug == validate_slug(slug)).one_or_none()
    if card is None:
        raise LookupError(f"No card {slug!r}.")
    if not (reviewed_by or "").strip():
        raise ReviewRefused(
            "A review needs a named reviewer. 'Reviewed by nobody' is the state that would let "
            "model output pass as checked (FOM_PROOF Sec. 15.2)."
        )

    resolved = [s for s in (card.sources or []) if s.get("kind") != "freetext"]
    if not resolved:
        raise ReviewRefused(
            f"Card {slug!r} has no resolved source — only "
            f"{len(card.sources or [])} free-text locator(s). Sec. 2.2 requires provenance a "
            "reader can follow, so attach the ingested document (with its page) or the fit "
            "record this rests on before signing it off."
        )

    card.status = CardStatus.REVIEWED
    card.reviewed_by = reviewed_by.strip()
    card.reviewed_at = datetime.now(timezone.utc)
    card.reviewed_body_sha256 = body_hash(card.body)
    if confidence is not None:
        card.confidence = confidence
    db.flush()
    logger.info("Card %s reviewed by %s", slug, reviewed_by)
    return card


def link_cards(
    db,
    from_slug: str,
    to_slug: str,
    *,
    relation: CardRelation | str,
    note: str | None = None,
):
    """Create a typed edge. Idempotent on (from, to, relation).

    ``CONTRADICTS`` requires a note. A bare pointer between two cards that
    disagree is worse than no link: it records that a conflict exists while
    discarding what the conflict is, which is the only part anyone can act on.
    """
    from cnms_fom.db.models import CardLink, KnowledgeCard

    kind = CardRelation(relation) if not isinstance(relation, CardRelation) else relation
    source = db.query(KnowledgeCard).filter(KnowledgeCard.slug == validate_slug(from_slug)).one_or_none()
    target = db.query(KnowledgeCard).filter(KnowledgeCard.slug == validate_slug(to_slug)).one_or_none()
    if source is None:
        raise LookupError(f"No card {from_slug!r}.")
    if target is None:
        raise LookupError(f"No card {to_slug!r}.")
    if source.id == target.id:
        raise CardError("A card cannot link to itself.")
    if kind is CardRelation.CONTRADICTS and not (note or "").strip():
        raise CardError(
            "A 'contradicts' link needs a note saying which claims conflict and on what basis. "
            "Recording that two cards disagree while discarding what they disagree about leaves "
            "nobody able to resolve it."
        )

    existing = (
        db.query(CardLink)
        .filter(
            CardLink.from_card_id == source.id,
            CardLink.to_card_id == target.id,
            CardLink.relation == kind,
        )
        .one_or_none()
    )
    if existing is not None:
        if note:
            existing.note = note
        db.flush()
        return existing

    link = CardLink(from_card_id=source.id, to_card_id=target.id, relation=kind, note=note)
    db.add(link)
    db.flush()
    return link


def card_as_dict(card, *, include_body: bool = True) -> dict:
    """Serialise a card with its review state made explicit."""
    payload = {
        "slug": card.slug,
        "card_type": card.card_type.value,
        "title": card.title,
        "summary": card.summary,
        "status": card.status.value,
        "citable": card.citable,
        "review_is_stale": card.review_is_stale,
        "confidence": card.confidence,
        "tags": card.tags or [],
        "sources": card.sources or [],
        "authored_by": card.authored_by,
        "reviewed_by": card.reviewed_by,
        "reviewed_at": str(card.reviewed_at) if card.reviewed_at else None,
        "updated_at": str(card.updated_at or card.created_at) if (card.updated_at or card.created_at) else None,
    }
    if include_body:
        payload["body"] = card.body
    if not card.citable:
        payload["caution"] = _caution_for(card)
    return payload


def _caution_for(card) -> str:
    if card.status is CardStatus.PROPOSED:
        return (
            f"This card was written by {card.authored_by} and has not been reviewed. Read it as a "
            "draft and check its sources yourself; do not build a result on it."
        )
    if card.status is CardStatus.SUPERSEDED:
        return "This card has been superseded by a later one. Kept for the audit trail."
    if card.review_is_stale:
        return (
            f"Reviewed by {card.reviewed_by}, but the body has been edited since. The review no "
            "longer covers what the card now says."
        )
    if not card.sources:
        return "This card cites no source, so nothing on it is traceable."
    return "Not citable."


def read_card(db, slug: str, *, with_links: bool = True) -> dict:
    """One card, with its typed links in both directions.

    Backlinks are included because the incoming edges are often the useful ones:
    what a concept is fed by matters less than what has come to depend on it, and
    an incoming ``contradicts`` is something the card's own author never wrote
    down.
    """
    from cnms_fom.db.models import KnowledgeCard

    card = db.query(KnowledgeCard).filter(KnowledgeCard.slug == validate_slug(slug)).one_or_none()
    if card is None:
        return {"slug": slug, "found": False, "error": f"No card {slug!r}."}

    payload = {"found": True, **card_as_dict(card)}
    if with_links:
        payload["links_out"] = [
            {
                "relation": link.relation.value,
                "slug": link.to_card.slug,
                "title": link.to_card.title,
                "status": link.to_card.status.value,
                "note": link.note,
            }
            for link in card.links_out
        ]
        payload["links_in"] = [
            {
                "relation": link.relation.value,
                "slug": link.from_card.slug,
                "title": link.from_card.title,
                "status": link.from_card.status.value,
                "note": link.note,
            }
            for link in card.links_in
        ]
        contradictions = [
            entry
            for entry in payload["links_out"] + payload["links_in"]
            if entry["relation"] == CardRelation.CONTRADICTS.value
        ]
        if contradictions:
            payload["unresolved_contradictions"] = contradictions
    return payload


def search_cards(
    db,
    query: str | None = None,
    *,
    card_type: CardType | str | None = None,
    status: CardStatus | str | None = None,
    tags: list[str] | None = None,
    citable_only: bool = False,
    limit: int = 20,
) -> dict:
    """Find cards by title, slug, summary, or body text.

    Lexical, not semantic, and deliberately so: a card corpus is small and its
    titles are chosen by people, so substring matching finds what someone meant
    without an embedding round-trip. The *chunks* need vectors; the index over
    them does not.
    """
    from cnms_fom.db.models import KnowledgeCard

    statement = db.query(KnowledgeCard)
    if query and query.strip():
        needle = f"%{query.strip()}%"
        statement = statement.filter(
            KnowledgeCard.title.ilike(needle)
            | KnowledgeCard.slug.ilike(needle)
            | KnowledgeCard.summary.ilike(needle)
            | KnowledgeCard.body.ilike(needle)
        )
    if card_type:
        statement = statement.filter(
            KnowledgeCard.card_type == (CardType(card_type) if not isinstance(card_type, CardType) else card_type)
        )
    if status:
        statement = statement.filter(
            KnowledgeCard.status == (CardStatus(status) if not isinstance(status, CardStatus) else status)
        )

    cards = statement.order_by(KnowledgeCard.slug).limit(min(int(limit), 100)).all()

    if tags:
        wanted = {t.strip().lower() for t in tags}
        cards = [c for c in cards if wanted & {t.lower() for t in (c.tags or [])}]
    if citable_only:
        cards = [c for c in cards if c.citable]

    return {
        "query": query,
        "n_cards": len(cards),
        "cards": [card_as_dict(card, include_body=False) for card in cards],
        "note": (
            "A proposed card is an unreviewed draft written by the assistant. It is returned so "
            "it can be read and corrected, not so it can be cited — check `citable`."
        ),
    }


def card_graph(db, root_slug: str | None = None, *, depth: int = 2) -> dict:
    """The card graph, as nodes and typed edges.

    Breadth-first from a root, or the whole graph when no root is given. Returned
    as data rather than a rendered diagram so the caller decides what to do with
    it — the assistant reads it as adjacency, and a UI can lay it out.
    """
    from cnms_fom.db.models import CardLink, KnowledgeCard

    if root_slug:
        root = (
            db.query(KnowledgeCard)
            .filter(KnowledgeCard.slug == validate_slug(root_slug))
            .one_or_none()
        )
        if root is None:
            return {"root": root_slug, "error": f"No card {root_slug!r}.", "nodes": [], "edges": []}
        frontier = {root.id}
        seen = {root.id}
        for _ in range(max(1, int(depth))):
            if not frontier:
                break
            links = (
                db.query(CardLink)
                .filter(
                    CardLink.from_card_id.in_(frontier) | CardLink.to_card_id.in_(frontier)
                )
                .all()
            )
            neighbours = {link.from_card_id for link in links} | {link.to_card_id for link in links}
            frontier = neighbours - seen
            seen |= neighbours
        cards = db.query(KnowledgeCard).filter(KnowledgeCard.id.in_(seen)).all()
        ids = seen
    else:
        cards = db.query(KnowledgeCard).limit(500).all()
        ids = {card.id for card in cards}

    links = (
        db.query(CardLink)
        .filter(CardLink.from_card_id.in_(ids), CardLink.to_card_id.in_(ids))
        .all()
    )
    by_id = {card.id: card for card in cards}

    #  An orphan is a card nothing points at and that points at nothing — usually
    #  a note someone wrote and never connected, which is exactly the knowledge
    #  that fails to compound.
    connected = {link.from_card_id for link in links} | {link.to_card_id for link in links}
    return {
        "root": root_slug,
        "nodes": [
            {
                "slug": card.slug,
                "title": card.title,
                "card_type": card.card_type.value,
                "status": card.status.value,
                "citable": card.citable,
            }
            for card in cards
        ],
        "edges": [
            {
                "from": by_id[link.from_card_id].slug,
                "to": by_id[link.to_card_id].slug,
                "relation": link.relation.value,
                "note": link.note,
            }
            for link in links
            if link.from_card_id in by_id and link.to_card_id in by_id
        ],
        "orphans": [card.slug for card in cards if card.id not in connected],
        "n_nodes": len(cards),
        "n_edges": len(links),
    }


def card_stats(db) -> dict:
    """Health of the card corpus: counts, review backlog, contradictions, orphans."""
    from sqlalchemy import func

    from cnms_fom.db.models import CardLink, KnowledgeCard

    by_type = dict(
        db.query(KnowledgeCard.card_type, func.count(KnowledgeCard.id))
        .group_by(KnowledgeCard.card_type)
        .all()
    )
    by_status = dict(
        db.query(KnowledgeCard.status, func.count(KnowledgeCard.id))
        .group_by(KnowledgeCard.status)
        .all()
    )
    cards = db.query(KnowledgeCard).all()
    contradictions = (
        db.query(CardLink).filter(CardLink.relation == CardRelation.CONTRADICTS).count()
    )

    return {
        "by_type": {k.value: int(v) for k, v in by_type.items()},
        "by_status": {k.value: int(v) for k, v in by_status.items()},
        "total_cards": len(cards),
        "citable": sum(1 for c in cards if c.citable),
        "awaiting_review": sum(1 for c in cards if c.status is CardStatus.PROPOSED),
        "stale_reviews": sum(1 for c in cards if c.review_is_stale),
        "unresolved_contradictions": int(contradictions),
        "total_links": db.query(CardLink).count(),
        "note": (
            "`awaiting_review` is the backlog that decides whether this corpus is knowledge or "
            "just model output written down. `unresolved_contradictions` is the most valuable "
            "number here: each one is a disagreement between sources that somebody has to settle."
        ),
    }
