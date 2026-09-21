"""Relevance grading, reranking, and the corrective retrieval loop.

Retrieval returns the k best-matching passages whether or not any of them
answers the question.  That is the single most dangerous property of a RAG
pipeline in this platform: a confident, well-cited answer built on six
irrelevant excerpts is indistinguishable from a correct one to everyone except
the person who goes and reads the pages.

So there are three stages between retrieval and generation, and all three exist
to make "the corpus does not contain this" a reachable outcome:

1. **Grade** each candidate against the question (0-3).  The similarity
   threshold in ``vectorstore.search_chunks`` is a blunt instrument — it cannot
   tell an on-topic passage that answers the question from an on-topic passage
   that merely shares its vocabulary, and in a corpus of process recipes those
   look nearly identical to an embedding.
2. **Rerank** by grade, so the passages that survive lead the context.  Position
   matters; a relevant excerpt in slot six is read less carefully than one in
   slot one.
3. **Correct** — if too little survives, rewrite the query once and retry.  Most
   retrieval failures are vocabulary failures: the user asked about "growth
   temperature" and the paper says "substrate setpoint".  One rewrite recovers a
   large fraction of those, and failing after it is a real finding rather than a
   phrasing accident.

Then, if it still fails, it refuses.  FOM_PROOF Sec. 2.3 — an unknown stays
unknown, and Sec. 15.2 is explicit that a missing value may not be filled by a
plausible number from memory.  A language model is an extremely efficient source
of plausible numbers, so the refusal path has to be as well-engineered as the
answer path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from cnms_fom.config import get_settings
from cnms_fom.db.enums import SynthesisTechnique

from . import cache
from .hybrid import FusedHit, hybrid_search
from .providers import ChatProvider, get_provider

logger = logging.getLogger(__name__)

#  Grades a passage can receive. Deliberately coarse: a four-point scale is
#  about as fine as a model can apply consistently, and asking for 0-100 buys
#  precision that is not there.
GRADE_SCALE = {
    0: "irrelevant — does not concern the question's subject",
    1: "related — same subject area, does not address the question",
    2: "useful — contains part of what the question asks for",
    3: "directly answers — contains the specific fact asked for, with its context",
}

#  A passage below this does not enter the answer context.
MIN_USEFUL_GRADE = 2

#  How many graded-useful passages count as enough to answer from. One is
#  allowed: a single passage that directly states a growth window is a better
#  basis than four that circle it. Zero is not.
MIN_USEFUL_PASSAGES = 1

#  Bumped whenever GRADER_PROMPT changes. It is part of the cache key, so an
#  edited prompt must not be served answers graded under the old one.
GRADER_PROMPT_VERSION = "grade-v3"

GRADER_PROMPT = """\
You grade whether a retrieved passage answers a question about thin-film \
synthesis. You are not answering the question.

Grade on this scale:
0 = irrelevant: does not concern the question's subject.
1 = related: same subject area, but does not address what was asked.
2 = useful: contains part of what the question asks for.
3 = directly answers: contains the specific fact asked for, together with the \
context that makes it meaningful (chamber, substrate, temperature, pressure, \
precursor).

Rules:
- A passage that mentions the right material but answers a different question \
about it is a 1, not a 2. Shared vocabulary is not relevance.
- A numerical parameter stripped of its context is a 2 at most, never a 3. A \
growth temperature with no stated chamber or substrate is not an answer.
- A question may ask several things at once ("what growth per cycle and what \
density, and do the sources agree?"). Grade each passage on the part it does answer. \
A passage carrying one of the three asked-for facts is a 2 — that is what "contains \
part of what the question asks for" means — and must not be marked down to a 1 for \
being silent on the others. No single passage answers a comparison between sources; \
comparing them happens later, and it can only happen over passages you kept.
- Judge only the passage in front of you. Do not use anything you know about the \
subject to fill in what the passage does not say.

Reply with JSON only: {"grade": <0-3>, "reason": "<one short clause>"}"""

REWRITE_PROMPT = """\
You rewrite a failed search query against a corpus of thin-film synthesis \
documents (MBE, PLD, ALD, sputtering, CVD process notes and facility user guides).

The original query retrieved nothing useful. Rewrite it once to improve lexical \
and semantic recall:
- Use the vocabulary a methods section would use, not a question's phrasing.
- Spell out abbreviations and include the abbreviation too (TMA and \
trimethylaluminum; ALD and atomic layer deposition).
- Include the likely synonyms for the process parameter asked about \
(growth temperature / substrate temperature / setpoint).
- Keep every chemical formula and named quantity from the original, verbatim.
- Do not add materials, techniques, or conditions the original did not mention. \
Broadening the vocabulary is the goal; changing the question is not.

Reply with JSON only: {"query": "<rewritten query>"}"""


def grader_provider(answer_provider: ChatProvider) -> ChatProvider:
    """The model that grades and rewrites, which need not be the one that answers.

    Grading is a per-candidate classification, so it runs N times per question and
    dominates the wall clock; answering runs once. Pointing the two at one model
    means paying answer-grade inference for a 0-3 judgement — and on a reasoning
    model it is worse than linear, because it will happily think for twenty
    seconds about whether a passage mentions a growth rate.

    Returns the same provider when ``RAG_GRADER_MODEL`` is unset, so the default
    behaviour is unchanged.
    """
    model = get_settings().rag_grader_model
    if not model or model == getattr(answer_provider, "model", None):
        return answer_provider
    #  Same provider family as the answer model, different weights: a grader
    #  reachable only through a second vendor would be a second failure mode for
    #  no benefit.
    return get_provider(getattr(answer_provider, "name", None), model)


@dataclass
class GradedHit:
    """A candidate passage with its relevance grade."""

    fused: FusedHit
    grade: int
    reason: str = ""

    @property
    def useful(self) -> bool:
        return self.grade >= MIN_USEFUL_GRADE

    def as_dict(self) -> dict:
        payload = self.fused.as_dict()
        payload.update({"grade": self.grade, "grade_reason": self.reason, "useful": self.useful})
        return payload


@dataclass
class RetrievalOutcome:
    """Everything the correction loop did, and what survived it."""

    query: str
    effective_query: str
    hits: list[GradedHit]
    attempts: list[dict]
    rewritten: bool = False
    graded: bool = True

    @property
    def useful(self) -> list[GradedHit]:
        return [hit for hit in self.hits if hit.useful]

    @property
    def sufficient(self) -> bool:
        return len(self.useful) >= MIN_USEFUL_PASSAGES

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "effective_query": self.effective_query,
            "rewritten": self.rewritten,
            "graded": self.graded,
            "sufficient": self.sufficient,
            "attempts": self.attempts,
            "n_useful": len(self.useful),
            "hits": [hit.as_dict() for hit in self.hits],
        }


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Local models wrap JSON in prose and fenced blocks more often than not, and a
    grader that fails on formatting silently degrades into "everything passes".
    """
    if not text:
        return None
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (stripped, *re.findall(r"\{[^{}]*\}", stripped, flags=re.DOTALL)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _call_grader(provider: ChatProvider, question: str, hit: FusedHit) -> dict:
    """One grading call, reduced to the ``{grade, reason}`` worth caching.

    Raises on a transport failure so :func:`cache.map_cached` records a miss rather
    than remembering it: caching a failure would make one unreachable server poison
    every later run.
    """
    message = (
        f"Question: {question}\n\n"
        f"Passage ({hit.hit.citation}):\n{hit.hit.text.strip()}\n\n"
        "Grade this passage."
    )
    result = provider.send(GRADER_PROMPT, [{"role": "user", "content": message}])
    if result.refused:
        return {"grade": 1, "reason": "grader declined to assess this passage"}

    parsed = _extract_json(result.text) or {}
    try:
        grade = int(parsed.get("grade"))
    except (TypeError, ValueError):
        logger.debug("Ungradeable reply for chunk %s: %r", hit.hit.chunk_id, result.text[:200])
        return {"grade": 1, "reason": "grader reply was not parseable"}
    return {"grade": max(0, min(3, grade)), "reason": str(parsed.get("reason", ""))[:300]}


def _graded_from_payload(
    hit: FusedHit, payload: dict | None, *, error: str | None = None
) -> GradedHit:
    """Turn a grader payload into a :class:`GradedHit`, failing closed.

    A missing or unparseable reply yields grade 1 — related but not useful.  Failing
    *closed* is the only safe default: defaulting to 3 would turn every grader hiccup
    into a fabricated basis for an answer, and defaulting to 0 would silently empty
    the context and report a data gap the corpus does not actually have.  Grade 1
    keeps the passage out of the answer while leaving it in the evidence trail.
    """
    if not payload:
        return GradedHit(
            fused=hit, grade=1,
            reason=f"grader unavailable ({error})" if error else "grader unavailable",
        )
    return GradedHit(
        fused=hit, grade=int(payload.get("grade", 1)), reason=str(payload.get("reason", ""))
    )


def grade_hit(provider: ChatProvider, question: str, hit: FusedHit) -> GradedHit:
    """Grade one passage. Uncached; kept for callers that grade a single hit."""
    try:
        payload = _call_grader(provider, question, hit)
    except Exception as exc:  # noqa: BLE001 - grading must not fail the request
        logger.warning("Grader call failed for chunk %s: %s", hit.hit.chunk_id, exc)
        return GradedHit(fused=hit, grade=1, reason=f"grader unavailable ({exc})")
    return _graded_from_payload(hit, payload)


def grade_and_rerank(
    provider: ChatProvider,
    question: str,
    candidates: list[FusedHit],
    *,
    db=None,
) -> tuple[list[GradedHit], dict]:
    """Grade every candidate and order by grade, then by fusion score.

    Grade first, fusion score second: a passage the grader called a 3 belongs ahead
    of one it called a 2 regardless of how the retrievers ranked them, and within a
    grade the fused rank is the best tiebreak available.

    With a ``db``, grades are cached on a hash of (question, passage) and the calls
    for the misses run concurrently. Both are exact: the same inputs produce the same
    grades, just with less waiting. Returns the grades and a cost report, because
    "how many model calls did that take?" is the number worth surfacing.
    """
    payloads, cached, called, errors = cache.map_cached(
        db,
        candidates,
        kind=cache.KIND_GRADE,
        key_of=lambda hit: cache.grade_key(question, hit.hit.text),
        model=getattr(provider, "model", "unknown"),
        prompt_version=GRADER_PROMPT_VERSION,
        call=lambda hit: _call_grader(provider, question, hit),
    )
    graded = [
        _graded_from_payload(hit, payload, error=errors.get(index))
        for index, (hit, payload) in enumerate(zip(candidates, payloads, strict=True))
    ]
    graded.sort(key=lambda g: (g.grade, g.fused.rrf_score), reverse=True)
    return graded, {"cached": cached, "called": called, "failed": len(errors)}


def rewrite_query(provider: ChatProvider, question: str) -> str | None:
    """One vocabulary-broadening rewrite of a failed query."""
    try:
        result = provider.send(
            REWRITE_PROMPT, [{"role": "user", "content": f"Original query: {question}"}]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed: %s", exc)
        return None
    if result.refused:
        return None

    parsed = _extract_json(result.text) or {}
    rewritten = str(parsed.get("query") or "").strip()
    if not rewritten or rewritten.lower() == question.strip().lower():
        return None
    #  A "rewrite" many times the original length is the model answering the
    #  question instead of rephrasing it, which would retrieve its own
    #  invention rather than the corpus.
    if len(rewritten) > 8 * max(len(question), 40):
        logger.debug("Discarding over-long rewrite (%d chars).", len(rewritten))
        return None
    return rewritten


def graded_and_costed(provider, question, candidates, *, db, record: dict) -> list[GradedHit]:
    """``grade_and_rerank``, with the call count folded into the attempt record."""
    graded, cost = grade_and_rerank(provider, question, candidates, db=db)
    record["grader_calls"] = cost["called"]
    record["grader_cache_hits"] = cost["cached"]
    #  Recorded, not discarded. A grader that could not be reached fails each hit
    #  closed to grade 1, which drops the passage — so a transport outage looks
    #  exactly like a corpus with nothing relevant in it unless the count survives.
    record["grader_failed"] = cost["failed"]
    return graded


def retrieve_with_correction(
    session,
    question: str,
    *,
    k: int = 6,
    techniques: list[SynthesisTechnique] | None = None,
    min_similarity: float = 0.2,
    provider: ChatProvider | None = None,
    grade: bool = True,
    allow_rewrite: bool = True,
    depth: int = 12,
    use_vector: bool = True,
    use_lexical: bool = True,
    lexical_relaxed: bool = False,
    cache_db=None,
) -> RetrievalOutcome:
    """Hybrid retrieve, grade, and retry once with a rewritten query if needed.

    Returns the outcome whether or not it succeeded.  ``sufficient`` is False
    when nothing useful survived, and the caller is expected to refuse rather
    than generate — see ``chains.answer_question``.

    ``grade=False`` skips the LLM grading pass entirely, which costs one model
    call per candidate.  Worth turning off for a broad exploratory search where
    the human is reading the passages themselves; not worth turning off when a
    model is going to write an answer from them.

    ``use_vector=False`` runs lexical-only.  Useful when the embedding server is
    known to be down and an exact-token search is better than an error — but it
    has to be asked for, because a silent lexical-only fallback would report "the
    corpus does not contain this" about a search that was never fully run.
    """
    provider = provider or get_provider()
    #  Resolved once per question, not once per candidate.
    grader = grader_provider(provider)
    attempts: list[dict] = []
    effective_query = question
    rewritten = False
    #  Graded-but-insufficient hits from the last attempt. Carried out of the
    #  loop so a final refusal can show what the corpus *did* return — "searched
    #  and came up short" is a more useful answer than an empty one.
    last_hits: list[GradedHit] = []

    for attempt in range(2 if allow_rewrite else 1):
        candidates = hybrid_search(
            session,
            effective_query,
            k=depth,
            techniques=techniques,
            min_similarity=min_similarity,
            use_vector=use_vector,
            use_lexical=use_lexical,
            lexical_relaxed=lexical_relaxed,
        )
        record = {
            "attempt": attempt + 1,
            "query": effective_query,
            "n_candidates": len(candidates),
            "grader_model": grader.model if grade else None,
        }

        if not candidates:
            attempts.append({**record, "n_useful": 0, "note": "no candidates retrieved"})
        else:
            hits = (
                graded_and_costed(
                    grader, question, candidates,
                    db=cache_db if cache_db is not None else session, record=record,
                )
                if grade
                #  Ungraded: treat fusion order as the ranking and mark every
                #  candidate useful-by-default, flagged via ``graded=False`` so
                #  nothing downstream mistakes this for a passed grading.
                else [GradedHit(fused=c, grade=MIN_USEFUL_GRADE, reason="not graded") for c in candidates]
            )
            useful = [hit for hit in hits if hit.useful]
            attempts.append({**record, "n_useful": len(useful)})

            if len(useful) >= MIN_USEFUL_PASSAGES:
                return RetrievalOutcome(
                    query=question,
                    effective_query=effective_query,
                    hits=hits[:k],
                    attempts=attempts,
                    rewritten=rewritten,
                    graded=grade,
                )
            last_hits = hits[:k]

        if attempt == 0 and allow_rewrite:
            candidate_query = rewrite_query(grader, question)
            if candidate_query:
                logger.info("Retrying retrieval with rewritten query: %r", candidate_query)
                effective_query = candidate_query
                rewritten = True
                continue
        break

    return RetrievalOutcome(
        query=question,
        effective_query=effective_query,
        hits=last_hits,
        attempts=attempts,
        rewritten=rewritten,
        graded=grade,
    )
