"""Schemas for the /cards router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cnms_fom.db.enums import CardRelation, CardStatus, CardType


class CardWriteRequest(BaseModel):
    """Create or update a card.

    A card written here by a person still lands ``proposed``; ``authored_by``
    records who wrote it, and review is a separate, deliberate call. That keeps
    one path for both authors instead of a person-shaped shortcut past the gate.
    """

    slug: str = Field(
        min_length=1,
        description="Path-like, lower case: concepts/ald-window-hfo2. It is the citation handle, "
        "so renaming one breaks every reference to it.",
    )
    title: str = Field(min_length=1)
    body: str = Field(min_length=1, description="Markdown. Cite each claim against the sources.")
    card_type: CardType = CardType.CONCEPT
    summary: str | None = None
    sources: list[dict] | None = Field(
        default=None,
        description='Citation records: [{"kind": "document", "document_id": 3, "page": 12}, '
        '{"kind": "fit_record", "fit_record_id": 7}]. A bare string becomes a free-text locator, '
        "which cannot satisfy review.",
    )
    tags: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    authored_by: str = Field(default="assistant")
    supersedes_slug: str | None = None


class CardReviewRequest(BaseModel):
    reviewed_by: str = Field(
        min_length=1,
        description="The person signing off. Required: 'reviewed by nobody' is the state that "
        "would let model output pass as checked.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class CardLinkRequest(BaseModel):
    from_slug: str
    to_slug: str
    relation: CardRelation
    note: str | None = Field(
        default=None,
        description="Required for 'contradicts': which claims conflict, and on what basis.",
    )


class CardOut(BaseModel):
    slug: str
    card_type: str
    title: str
    summary: str | None = None
    body: str | None = None
    status: str
    citable: bool = Field(
        description="Reviewed, not stale, and carrying at least one resolved source. Anything "
        "else is someone's notes — readable, not citable."
    )
    review_is_stale: bool = False
    confidence: float | None = None
    tags: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    authored_by: str
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    updated_at: str | None = None
    caution: str | None = None
    links_out: list[dict] = Field(default_factory=list)
    links_in: list[dict] = Field(default_factory=list)
    unresolved_contradictions: list[dict] = Field(default_factory=list)


class CardListResponse(BaseModel):
    query: str | None = None
    n_cards: int
    cards: list[CardOut] = Field(default_factory=list)
    note: str = ""


class CardGraphResponse(BaseModel):
    root: str | None = None
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    orphans: list[str] = Field(default_factory=list)
    n_nodes: int = 0
    n_edges: int = 0


class CardStatsResponse(BaseModel):
    by_type: dict = Field(default_factory=dict)
    by_status: dict = Field(default_factory=dict)
    total_cards: int = 0
    citable: int = 0
    awaiting_review: int = 0
    stale_reviews: int = 0
    unresolved_contradictions: int = 0
    total_links: int = 0
    note: str = ""


class CardSearchParams(BaseModel):
    query: str | None = None
    card_type: CardType | None = None
    status: CardStatus | None = None
    tags: list[str] | None = None
    citable_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)
