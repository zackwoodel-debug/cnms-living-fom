"""Knowledge cards: a Dynamic Knowledge Repository over the synthesis corpus.

Retrieval alone re-derives an answer per question and keeps nothing.  This package
does the integration work at ingest time instead — concept pages, source
summaries, and typed links between them, so the second time a question is asked
it starts from the first answer's conclusions.

The design follows ``llm-wiki``'s: path-like slugs, typed frontmatter, and typed
graph edges rather than untyped links, because a graph that only knows *that* two
pages are related cannot say what one rests on.  ``CONTRADICTS`` is the edge this
platform needs most — an unresolved disagreement between two sources is a finding,
and a flat link would bury it.

What differs is the review gate.  A card is where a language model's synthesis
gets written down, and FOM_PROOF Sec. 15.2 says a synthesis is not evidence: an
assistant-written card is ``PROPOSED`` and not citable until a named person has
checked it against resolved sources.  There is no path from a card into
``property_values``.
"""

from .cards import (  # noqa: F401
    CardError,
    ReviewRefused,
    card_as_dict,
    card_graph,
    card_stats,
    link_cards,
    read_card,
    review_card,
    search_cards,
    upsert_card,
)
