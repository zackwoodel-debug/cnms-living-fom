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

import contextlib
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


#  Question scaffolding that carries no retrieval signal. Postgres drops true English
#  stopwords inside plainto_tsquery on its own; these are the words that survive that
#  and still mean nothing here, so they are excluded from the coverage denominator —
#  otherwise a chunk is penalised for not containing the word "reported".
_QUESTION_NOISE: frozenset[str] = frozenset({
    "what", "which", "how", "why", "when", "where", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and", "or", "but",
    "with", "from", "by", "as", "that", "this", "these", "those", "it", "its",
    "reported", "report", "reports", "used", "use", "uses", "give", "given",
    "tell", "show", "shows", "any", "some", "there", "their", "we", "our",
    "value", "values", "typical", "about", "agree", "sources", "source",
})

#  Kept whole rather than split on case or digit boundaries. "HfO2" must not become
#  "hfo" + "2", and a sample id or a named correction ("Nevot-Croce") is exactly the
#  rare token lexical search exists to catch.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-_/]*")


def lexical_terms(query: str) -> list[str]:
    """The query's meaningful tokens, in order, de-duplicated case-insensitively.

    Used both to build the relaxed tsquery and as the denominator for term coverage,
    so the two can never disagree about what the query was asking for.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TOKEN.finditer(query):
        token = match.group(0).strip(".-_/")
        if not token or len(token) < 2:
            continue
        lowered = token.lower()
        if lowered in _QUESTION_NOISE or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(token)
    return terms


def term_coverage(text: str, terms: list[str]) -> float:
    """Fraction of the query's meaningful terms present in a passage.

    Substring rather than token matching, deliberately: "angstrom" should count for a
    passage writing "angstroms", and the alternative here is a second stemmer that
    disagrees with Postgres's. It is a ranking signal, not a guard — nothing is
    admitted or rejected on the strength of it.
    """
    if not terms:
        return 0.0
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered) / len(terms)


def lexical_search(
    session,
    query: str,
    *,
    relaxed: bool = False,
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
        return _postgres_fulltext(
            session, query, k=k, techniques=techniques, relaxed=relaxed
        )
    #  The Python fallback already scores by term overlap, which is what the relaxed
    #  stage reproduces in SQL, so there is nothing for the flag to change here.
    return _python_term_overlap(session, query, k=k, techniques=techniques)


def _postgres_fulltext(
    session,
    query: str,
    *,
    k: int,
    techniques: list[SynthesisTechnique] | None,
    relaxed: bool = False,
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
    #
    #  The SAVEPOINT is the other half of the same lesson. Postgres aborts the whole
    #  transaction on any statement error, so when this query failed the fallback below
    #  ran on a dead transaction and raised InFailedSqlTransaction — and so did every
    #  later query on the session, including ones belonging to the caller. Rolling back
    #  to a savepoint discards only this query, which is what makes the fallback
    #  reachable at all. Measured: without it, a `select count(*)` that succeeded before
    #  this call failed after it.
    try:
        with session.begin_nested():
            #  literal_column, not text: it is a ColumnElement, so `.op("@@")` and
            #  `.label()` both work. The SQL string stays byte-identical to migration
            #  0007's index expression, which is not cosmetic — one missing space drops
            #  the plan from a 5.7 ms index scan to a 99 ms sequential scan.
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

            #  Stage 2, only when stage 1 found nothing. websearch_to_tsquery ANDs its
            #  terms, so a natural-language question demands every stem in one chunk
            #  and matches nothing: measured, every real question in §10e returned
            #  0 lexical hits while dense returned 7-12. Relaxing to OR recovers the
            #  leg; term coverage is what stops OR from ranking a chunk that shares one
            #  generic word above one that shares the material and the property.
            if relaxed and not rows:
                rows = _relaxed_rows(
                    session, query, searchable, k=k, techniques=techniques
                )
    except Exception as exc:  # noqa: BLE001 - a missing index or FTS config, not a bug in the caller
        logger.warning("Postgres full-text search failed (%s); falling back to term overlap.", exc)
        #  The savepoint has already been rolled back by the context manager, so the
        #  session is usable and the fallback can actually run.
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


def _rendered_tsqueries(session, query: str, terms: list[str]) -> dict:
    """What Postgres actually parsed the query into, for both stages.

    Asking the database rather than reconstructing it in Python: the whole failure in
    bug 18 was a mismatch between what the query looked like and what it meant.
    """
    if not _is_postgres(session):
        return {"precise": None, "relaxed": None, "note": "not a Postgres backend"}
    from sqlalchemy import text as sql_text

    out: dict = {}
    try:
        with session.begin_nested():
            out["precise"] = session.execute(
                sql_text("SELECT websearch_to_tsquery('english', :q)::text"),
                {"q": query},
            ).scalar()
            if terms:
                ored = " || ".join(
                    f"plainto_tsquery('english', :t{i})" for i in range(len(terms))
                )
                out["relaxed"] = session.execute(
                    sql_text(f"SELECT ({ored})::text"),
                    {f"t{i}": term for i, term in enumerate(terms)},
                ).scalar()
            else:
                out["relaxed"] = None
    except Exception as exc:  # noqa: BLE001 - a diagnostic never breaks its caller
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _relaxed_rows(
    session,
    query: str,
    searchable: str,
    *,
    k: int,
    techniques: list[SynthesisTechnique] | None,
) -> list:
    """Stage 2: OR the query's terms, then re-rank by how many of them a chunk has.

    Each term goes through ``plainto_tsquery`` as a bound parameter and the results are
    OR-ed with the tsquery ``||`` operator. That is deliberate over building a
    ``to_tsquery`` string: a query containing ``&``, ``!`` or an unbalanced quote would
    otherwise be tsquery syntax, and a user's question is data, not an expression.
    Postgres also drops its own stopwords inside each call, so no stopword list here has
    to be exhaustive.

    Coverage is computed in Python rather than SQL. Doing it in SQL needs one CASE per
    term and so a dynamically assembled statement; the candidate pool here is ``k * 4``
    rows, and re-ranking that in Python is both cheaper and auditable.
    """
    from sqlalchemy import func, literal_column
    from sqlalchemy.sql.elements import ColumnElement

    from cnms_fom.db.models import Document, DocumentChunk

    terms = lexical_terms(query)
    if not terms:
        return []

    #  Annotated as ColumnElement because the `||` chain re-binds a Function to a
    #  BinaryExpression, and both are ColumnElements.
    tsquery: ColumnElement = func.plainto_tsquery("english", terms[0])
    for term in terms[1:]:
        tsquery = tsquery.op("||")(func.plainto_tsquery("english", term))

    tsvector: ColumnElement = literal_column(searchable)
    rank_expr = func.ts_rank_cd(tsvector, tsquery)
    statement = (
        session.query(DocumentChunk, Document, rank_expr.label("rank"))
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(tsvector.op("@@")(tsquery))
    )
    if techniques:
        statement = statement.filter(Document.technique.in_(list(techniques)))

    #  A wider pool than k, because ts_rank_cd alone is the thing being corrected.
    candidates = statement.order_by(rank_expr.desc()).limit(max(k * 4, k)).all()

    #  Coverage first, ts_rank_cd as the tiebreak. Scored over title + text to match
    #  what the tsvector indexes: a table row carries none of its document's subject.
    def score(row) -> tuple[float, float]:
        chunk, document, rank = row
        haystack = f"{document.title or ''} {chunk.text}"
        return (term_coverage(haystack, terms), float(rank))

    ranked = sorted(candidates, key=score, reverse=True)[:k]
    logger.debug(
        "Relaxed lexical stage: %d terms, %d candidates, %d returned (top coverage %.2f)",
        len(terms), len(candidates), len(ranked),
        score(ranked[0])[0] if ranked else 0.0,
    )
    return ranked


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


@contextlib.contextmanager
def _failure_isolated(session):
    """Run a query so that its failure cannot poison the caller's transaction.

    On Postgres this is a SAVEPOINT: the statement rolls back alone and the session
    stays usable. On SQLite and on an unbound session it is a no-op, because there is
    nothing to isolate and ``begin_nested`` on some of those raises in its own right.
    """
    if not _is_postgres(session):
        yield
        return
    nested = session.begin_nested()
    try:
        yield
    except Exception:
        if nested.is_active:
            nested.rollback()
        raise
    else:
        if nested.is_active:
            nested.commit()


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
    lexical_relaxed: bool = False,
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

            #  SAVEPOINT for the same reason the lexical leg has one, and found the
            #  same way: on a live server. Postgres aborts the entire transaction on
            #  any statement error, so a dense query that fails — pgvector enabled
            #  against an `embedding` column that is still `json`, say — poisoned the
            #  session, and the lexical fallback below then died too with
            #  InFailedSqlTransaction. The whole request failed on a retriever that
            #  was supposed to be optional.
            with _failure_isolated(session):
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
        lists["lexical"] = lexical_search(
            session, query, k=depth, techniques=techniques,
            relaxed=lexical_relaxed,
        )

    if not any(lists.values()):
        if dense_failure is not None:
            raise dense_failure
        return []

    return reciprocal_rank_fusion(lists)[:k]


def retrieval_diagnostics(
    session,
    query: str,
    *,
    techniques: list[SynthesisTechnique] | None = None,
    depth: int = 10,
    lexical_relaxed: bool = False,
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

    #  Both stages separately, because "lexical found nothing" and "lexical found
    #  nothing until it was relaxed" are different answers to the question this
    #  function exists for.
    precise = lexical_search(session, query, k=depth, techniques=techniques)
    lexical = precise
    relaxed_ran = False
    if lexical_relaxed and not precise:
        lexical = lexical_search(
            session, query, k=depth, techniques=techniques, relaxed=True
        )
        relaxed_ran = True

    fused = reciprocal_rank_fusion({"vector": dense, "lexical": lexical})
    terms = lexical_terms(query)

    return {
        "query": query,
        "terms": tokenise(query),
        "lexical_terms": terms,
        "lexical_backend": "postgres_fulltext" if _is_postgres(session) else "python_term_overlap",
        "lexical_stages": {
            "precise_hits": len(precise),
            "relaxed_requested": lexical_relaxed,
            "relaxed_ran": relaxed_ran,
            "relaxed_hits": len(lexical) if relaxed_ran else None,
            "tsquery": _rendered_tsqueries(session, query, terms),
            "coverage": [
                {"chunk_id": h.chunk_id, "coverage": round(
                    term_coverage(f"{h.document_title or ''} {h.text}", terms), 3
                )}
                for h in lexical
            ],
        },
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
