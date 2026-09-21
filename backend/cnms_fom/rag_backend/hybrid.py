"""Hybrid retrieval: dense vectors plus lexical search, fused by rank.

Why both, for this corpus specifically
--------------------------------------
Embedding search is good at paraphrase and bad at rare exact tokens, and a
synthesis corpus is mostly rare exact tokens.  "TMA" and "trimethylaluminum"
are the same precursor; ``nomic-embed-text`` knows that.  But "HfO2" and
"HfO_2" and "hafnia" are three spellings a 768-dimension vector will happily
place near "ZrO2" as well, and a question about the *Nevot-Croce* roughness
factor retrieves a passage about roughness in general while the one passage that
names the factor ranks eleventh.  Lexical search inverts both failure modes:
useless for paraphrase, exact on tokens.

Reciprocal Rank Fusion rather than score averaging
--------------------------------------------------
Cosine similarity and ``ts_rank_cd`` are not on the same scale, do not have
comparable distributions, and a weighted sum of them is a free parameter
pretending to be a method.  RRF uses only the *ranks*:

    score(chunk) = sum over retrievers of 1 / (k + rank)

which needs no calibration, cannot be broken by one retriever's scores drifting,
and rewards a chunk that both retrievers liked over one that either loved alone.
``k`` (60 by convention) damps the top of each list so a single retriever cannot
dominate on rank-1 alone.

Scoring against the title as well as the text
--------------------------------------------
A table row cannot be scored on its own.  Measured on the benchmark's PLD case, the
passage holding ``substrate_temperature 700 degC`` ranked **fourth** while a passage
reading *"The material is LSMO, not SrTiO3"* ranked third — because the table row
never mentions SrTiO3, PLD, or deposition.  All three are in its document's *title*,
which the chunk text does not contain.

So the lexical leg scores against ``title + text``, which lifted that passage to
second.  The title is used for *scoring only*: the text returned, the quote stored,
and the citation are the chunk's own, so nothing a reader checks is affected.

Backends
--------
On Postgres, lexical search is ``to_tsvector``/``plainto_tsquery`` with
``ts_rank_cd``, against the functional GIN index created in migration 0003.
Elsewhere — the test suite runs on SQLite — it falls back to a term-overlap
scorer in Python, which is enough to keep the fusion path exercised and honest
about being a fallback.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from cnms_fom.db.enums import SynthesisTechnique

from .vectorstore import ChunkHit, search_chunks

logger = logging.getLogger(__name__)

#  The RRF damping constant. 60 is the value from the original TREC work and is
#  used unchanged; tuning it per corpus is the kind of free parameter this
#  method exists to avoid.
RRF_K = 60

#  How deep each retriever goes before fusion. Wider than the final k on
#  purpose: fusion can only promote a chunk that at least one retriever
#  surfaced, so a narrow candidate pool makes the second retriever pointless.
CANDIDATE_DEPTH = 24

#  Tokens too common in a materials corpus to carry retrieval signal.  Kept
#  short deliberately — an aggressive stoplist removes exactly the rare tokens
#  lexical search is here for.
_LEXICAL_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
        "what", "when", "which", "with",
    }
)


@dataclass
class FusedHit:
    """One chunk after fusion, with the ranks that produced its score."""

    hit: ChunkHit
    rrf_score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def retriever_count(self) -> int:
        return sum(1 for rank in (self.vector_rank, self.lexical_rank) if rank is not None)

    def as_dict(self) -> dict:
        payload = self.hit.as_dict()
        payload.update(
            {
                "rrf_score": self.rrf_score,
                "vector_rank": self.vector_rank,
                "lexical_rank": self.lexical_rank,
                "found_by": [
                    name
                    for name, rank in (("vector", self.vector_rank), ("lexical", self.lexical_rank))
                    if rank is not None
                ],
            }
        )
        return payload


def _is_postgres(session) -> bool:
    bind = session.get_bind()
    return bool(bind is not None and bind.dialect.name == "postgresql")


def tokenise(query: str) -> list[str]:
    """Query terms for the fallback lexical scorer.

    Chemical formulas survive: ``HfO2`` stays one token rather than becoming
    ``hf`` and ``2``, because splitting it is what makes a formula unfindable.
    """
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_\-]*|\d+(?:\.\d+)?", query)
    return [
        token.lower()
        for token in raw
        if len(token) > 1 and token.lower() not in _LEXICAL_STOPWORDS
    ]


def lexical_search(
    session,
    query: str,
    *,
    k: int = CANDIDATE_DEPTH,
    techniques: list[SynthesisTechnique] | None = None,
) -> list[ChunkHit]:
    """Keyword search over ``document_chunks``.

    ``similarity`` on the returned hits is the lexical rank score, not a cosine
    similarity.  They are not comparable, which is exactly why fusion uses ranks
    — but the field is populated so a caller inspecting a single retriever's
    output still sees why a chunk placed where it did.
    """
    if _is_postgres(session):
        return _postgres_fulltext(session, query, k=k, techniques=techniques)
    return _python_term_overlap(session, query, k=k, techniques=techniques)


def _postgres_fulltext(
    session, query: str, *, k: int, techniques: list[SynthesisTechnique] | None
) -> list[ChunkHit]:
    from sqlalchemy import func, literal_column

    from cnms_fom.db.models import Document, DocumentChunk

    #  websearch_to_tsquery over plainto_tsquery: it tolerates quotes and OR
    #  without raising on punctuation a user typed, which plainto_tsquery does
    #  not. ts_rank_cd (cover density) rather than ts_rank: it rewards query
    #  terms appearing close together, and in a process recipe a parameter and
    #  its units being adjacent is the whole signal.
    #  Title and text together, for the reason in the module docstring. The
    #  expression must match migration 0007's functional index *exactly* or Postgres
    #  will not use it — which is why this reads ``document_chunks.search_title``, a
    #  trigger-maintained copy of the document's title on the chunk row, rather than
    #  joining ``documents.title``: a functional index cannot span two tables.
    #
    #  If 0007 has not been applied the column is absent and this query raises, which
    #  the caller below turns into a fallback to term overlap with a warning. That is
    #  the right failure: slower, not wrong.
    searchable = (
        "to_tsvector('english', coalesce(document_chunks.search_title, '') "
        "|| ' ' || document_chunks.text)"
    )

    #  Building the statement is inside the try with executing it. It was outside, and
    #  that was the whole reason a first run against real Postgres crashed instead of
    #  degrading: `.label()` on a `text()` clause raises NotImplementedError in
    #  SQLAlchemy 2.0, at construction time, so the fallback this comment promises was
    #  unreachable. A fallback that only covers execution is not a fallback.
    try:
        #  literal_column, not text: it is a ColumnElement, so `.op("@@")` and
        #  `.label()` both work. The SQL string stays byte-identical to migration
        #  0007's index expression, which is not cosmetic — one missing space drops the
        #  plan from a 5.7 ms index scan to a 99 ms sequential scan.
        tsvector = literal_column(searchable)
        tsquery = func.websearch_to_tsquery("english", query)
        rank_expr = func.ts_rank_cd(tsvector, tsquery)

        statement = (
            session.query(DocumentChunk, Document, rank_expr.label("rank"))
            .join(Document, DocumentChunk.document_id == Document.id)
            .filter(tsvector.op("@@")(tsquery))
        )
        if techniques:
            statement = statement.filter(Document.technique.in_(list(techniques)))

        rows = statement.order_by(rank_expr.desc()).limit(k).all()
    except Exception as exc:  # noqa: BLE001 - a missing index or FTS config, not a bug in the caller
        logger.warning("Postgres full-text search failed (%s); falling back to term overlap.", exc)
        return _python_term_overlap(session, query, k=k, techniques=techniques)

    return [
        ChunkHit(
            chunk_id=chunk.id,
            document_id=document.id,
            document_title=document.title,
            technique=document.technique.value,
            page=chunk.page,
            text=chunk.text,
            similarity=float(rank),
            doi=document.doi,
            source_url=document.source_url,
        )
        for chunk, document, rank in rows
    ]


def _python_term_overlap(
    session, query: str, *, k: int, techniques: list[SynthesisTechnique] | None
) -> list[ChunkHit]:
    """Term-overlap scoring in Python, for backends without full-text search.

    Scores by the fraction of distinct query terms present, with a small bonus
    for repetition — a crude stand-in for TF-IDF that is honest about being one.
    It scans the chunk table, so it is fine for a test suite and for a corpus in
    the low thousands, and is not what a real deployment should be running on:
    that is what migration 0003's GIN index is for.
    """
    from cnms_fom.db.models import Document, DocumentChunk

    terms = tokenise(query)
    if not terms:
        return []

    statement = session.query(DocumentChunk, Document).join(
        Document, DocumentChunk.document_id == Document.id
    )
    if techniques:
        statement = statement.filter(Document.technique.in_(list(techniques)))

    scored: list[tuple[float, object, object]] = []
    for chunk, document in statement.all():
        #  Title *and* text: a table row carries none of its document's subject, so
        #  scoring the text alone ranks it below prose that merely mentions the
        #  words. The title is scored, never returned.
        lowered = f"{document.title or ''} {chunk.text}".lower()
        present = [term for term in terms if term in lowered]
        if not present:
            continue
        coverage = len(set(present)) / len(set(terms))
        repetition = sum(lowered.count(term) for term in set(present))
        scored.append((coverage + 0.01 * min(repetition, 20), chunk, document))

    scored.sort(key=lambda row: row[0], reverse=True)
    return [
        ChunkHit(
            chunk_id=chunk.id,
            document_id=document.id,
            document_title=document.title,
            technique=document.technique.value,
            page=chunk.page,
            text=chunk.text,
            similarity=float(score),
            doi=document.doi,
            source_url=document.source_url,
        )
        for score, chunk, document in scored[:k]
    ]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ChunkHit]], *, k: int = RRF_K
) -> list[FusedHit]:
    """Fuse several ranked lists by reciprocal rank.

    A chunk found by both retrievers outranks one found by either alone at the
    same position, which is the property that makes fusion worth doing.
    """
    scores: dict[int, float] = defaultdict(float)
    ranks: dict[int, dict[str, int]] = defaultdict(dict)
    hits: dict[int, ChunkHit] = {}

    for retriever, results in ranked_lists.items():
        for position, hit in enumerate(results, start=1):
            scores[hit.chunk_id] += 1.0 / (k + position)
            ranks[hit.chunk_id][retriever] = position
            #  Keep the first copy seen: both retrievers return identical text
            #  for one chunk_id, and the dense hit carries the cosine similarity
            #  worth reporting.
            hits.setdefault(hit.chunk_id, hit)

    fused = [
        FusedHit(
            hit=hits[chunk_id],
            rrf_score=score,
            vector_rank=ranks[chunk_id].get("vector"),
            lexical_rank=ranks[chunk_id].get("lexical"),
        )
        for chunk_id, score in scores.items()
    ]
    #  Ties broken by agreement then by the dense score, so the ordering is
    #  deterministic across runs — an audit trail that reorders itself is not one.
    fused.sort(key=lambda f: (f.rrf_score, f.retriever_count, f.hit.similarity), reverse=True)
    return fused


def hybrid_search(
    session,
    query: str,
    *,
    k: int = 6,
    techniques: list[SynthesisTechnique] | None = None,
    min_similarity: float = 0.0,
    depth: int = CANDIDATE_DEPTH,
    use_lexical: bool = True,
    use_vector: bool = True,
) -> list[FusedHit]:
    """Retrieve with both retrievers and fuse.

    ``min_similarity`` is applied to the *dense* leg only, before fusion.  A
    lexical hit has no cosine similarity to threshold, and an exact match on a
    rare token is worth returning whatever a vector thinks of it — that being
    the reason lexical search is in the pipeline.
    """
    lists: dict[str, list[ChunkHit]] = {}
    dense_failure: Exception | None = None

    if use_vector:
        try:
            from .embeddings import embed_query

            lists["vector"] = search_chunks(
                session,
                embed_query(query),
                k=depth,
                techniques=techniques,
                min_similarity=min_similarity,
            )
        except Exception as exc:  # noqa: BLE001 - see below
            #  Ollama unreachable, or the rag extra not installed. Either way,
            #  half a retriever beats none: lexical search alone still finds
            #  exact tokens, which is most of what this corpus is. The failure is
            #  logged loudly and re-raised only if lexical came up empty too,
            #  because then there genuinely was no retrieval and the caller needs
            #  the actionable error rather than an empty list.
            logger.warning("Dense retrieval unavailable (%s); continuing lexical-only.", exc)
            dense_failure = exc
            lists["vector"] = []

    if use_lexical:
        lists["lexical"] = lexical_search(session, query, k=depth, techniques=techniques)

    if not any(lists.values()):
        if dense_failure is not None:
            raise dense_failure
        return []

    return reciprocal_rank_fusion(lists)[:k]


def retrieval_diagnostics(
    session, query: str, *, techniques: list[SynthesisTechnique] | None = None, depth: int = 10
) -> dict:
    """Side-by-side view of what each retriever found, and what fusion did.

    For the question "why did it not find the passage I know is in there?", which
    is otherwise unanswerable from the outside.  Cheap to run and worth running
    before concluding a corpus is missing a document.
    """
    dense: list[ChunkHit] = []
    dense_error: str | None = None
    try:
        from .embeddings import embed_query

        dense = search_chunks(session, embed_query(query), k=depth, techniques=techniques)
    except Exception as exc:  # noqa: BLE001
        dense_error = str(exc)

    lexical = lexical_search(session, query, k=depth, techniques=techniques)
    fused = reciprocal_rank_fusion({"vector": dense, "lexical": lexical})

    return {
        "query": query,
        "terms": tokenise(query),
        "lexical_backend": "postgres_fulltext" if _is_postgres(session) else "python_term_overlap",
        "dense_error": dense_error,
        "vector": [
            {"chunk_id": h.chunk_id, "citation": h.citation, "similarity": h.similarity}
            for h in dense
        ],
        "lexical": [
            {"chunk_id": h.chunk_id, "citation": h.citation, "rank_score": h.similarity}
            for h in lexical
        ],
        "fused": [
            {
                "chunk_id": f.hit.chunk_id,
                "citation": f.hit.citation,
                "rrf_score": f.rrf_score,
                "vector_rank": f.vector_rank,
                "lexical_rank": f.lexical_rank,
            }
            for f in fused[:depth]
        ],
        "found_by_both": sum(1 for f in fused if f.retriever_count == 2),
    }
