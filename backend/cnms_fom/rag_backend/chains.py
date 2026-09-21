"""Question answering over the synthesis corpus.

The prompt does most of the safety work here, so it is worth reading rather than
skimming.  Two constraints come straight from FOM_PROOF:

  * Sec. 2.3 and Sec. 17 item 17 — an unknown stays unknown.  The model is told
    to emit ``[DATA GAP: explicitly unresolved]`` rather than a plausible number.
  * Sec. 15.2 — "a missing value may not be filled by a plausible number from
    memory".  A language model is a very good source of plausible numbers, which
    makes this the single most dangerous interface in the platform.  Hence
    ``assert_not_property_ingestion``: retrieval output is for a human to read,
    and there is no code path from an answer into ``property_values``.

This module is the *single-shot* path: one retrieval, one answer.  It is the right
shape when the question maps onto one search ("what ALD window does this paper
report for HfO2 on Si?"), and it is what ``POST /rag/query`` serves.

Retrieval underneath it is no longer a bare vector search.  ``grading`` runs
hybrid dense+lexical retrieval, grades each candidate for whether it actually
answers the question, and rewrites the query once if too little survives — so
"the corpus does not contain this" became a reachable outcome rather than a
theoretical one.  The multi-step path, where the model chooses each retrieval in
turn and can reach the ModalFit fit records, is :mod:`agent`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cnms_fom.config import get_settings
from cnms_fom.db.enums import SynthesisTechnique

from .providers import ChatProvider, get_provider
from .vectorstore import ChunkHit

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """\
You are a synthesis-methods assistant for the Center for Nanophase Materials \
Sciences. You answer questions about thin-film growth and processing (MBE, PLD, \
ALD, sputtering, CVD) using ONLY the retrieved excerpts provided below.

Rules:
1. Use only the provided context. If the context does not contain the answer, \
say exactly: [DATA GAP: explicitly unresolved] and state what document or \
measurement would resolve it.
2. Never supply a numerical process parameter, material property, or growth \
window that does not appear in the context. Do not interpolate, convert, or \
estimate one from background knowledge.
3. Cite every factual claim inline as [n], matching the numbered excerpts.
4. Growth parameters are only meaningful with their context. When you report \
one, carry its chamber, substrate, temperature, pressure, and precursor along \
with it. A number without its context is not an answer.
5. When excerpts disagree, say so and cite both. Do not average them or pick a \
favourite.
6. Distinguish what was measured from what was proposed or modelled in the \
source.

Retrieved excerpts:
{context}
"""

USER_PROMPT = """\
Question: {question}

Answer using only the excerpts above, with inline [n] citations."""


@dataclass
class RagAnswer:
    """An answer plus the evidence it was built from.

    ``retrieval`` carries how the passages were found: which retrievers hit,
    what each candidate was graded, and whether the query had to be rewritten.
    It is here because an answer and the search that produced it are one artifact
    — without it, "why did it not find X?" is unanswerable from the response
    alone, and the usual conclusion is that the corpus is missing a document it
    already has.
    """

    question: str
    answer: str
    sources: list[ChunkHit] = field(default_factory=list)
    model: str = ""
    techniques: list[str] = field(default_factory=list)
    insufficient_context: bool = False
    provider: str = ""
    query_was_rewritten: bool = False
    effective_query: str = ""
    retrieval: dict = field(default_factory=dict)
    refused_by_provider: bool = False

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "model": self.model,
            "techniques": self.techniques,
            "insufficient_context": self.insufficient_context,
            "provider": self.provider,
            "query_was_rewritten": self.query_was_rewritten,
            "effective_query": self.effective_query,
            "retrieval": self.retrieval,
            "refused_by_provider": self.refused_by_provider,
        }


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    """A ``ChatOllama`` bound to the configured server.

    Kept for callers that want the raw LangChain object.  New code should go
    through :func:`providers.get_provider`, which is provider-agnostic and is
    what everything in this package uses — the guardrails have to behave
    identically on the local and the remote path, and that is only true if there
    is one seam.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("RAG needs the 'rag' extra: pip install -e '.[rag]'") from exc

    settings = get_settings()
    return ChatOllama(
        model=model or settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def format_context(hits: list[ChunkHit]) -> str:
    """Number the excerpts so the model's [n] citations resolve to real sources."""
    return "\n\n".join(
        f"[{i}] {hit.citation}\n{hit.text.strip()}" for i, hit in enumerate(hits, start=1)
    )


def answer_question(
    session,
    question: str,
    *,
    k: int = 6,
    techniques: list[SynthesisTechnique] | None = None,
    model: str | None = None,
    min_similarity: float = 0.2,
    provider: ChatProvider | None = None,
    grade: bool = True,
) -> RagAnswer:
    """Retrieve, then answer strictly from what was retrieved.

    Retrieval is hybrid (dense + lexical, fused by reciprocal rank), graded for
    relevance, and retried once with a rewritten query when too little survives.
    Only passages that graded *useful* reach the prompt: a passage that shares the
    question's vocabulary without answering it is worse than no passage, because
    it produces a confident answer with a real citation attached to it.

    ``grade=False`` skips the grading pass — one model call per candidate — and is
    for exploratory search a human will read themselves, not for generation.
    """
    from .grading import retrieve_with_correction

    chat = provider or get_provider(model=model)
    technique_names = [t.value for t in techniques] if techniques else []

    outcome = retrieve_with_correction(
        session,
        question,
        k=k,
        techniques=techniques,
        min_similarity=min_similarity,
        provider=chat,
        grade=grade,
    )
    retrieval_trace = {
        "attempts": outcome.attempts,
        "graded": outcome.graded,
        "candidates": [
            {
                "citation": hit.fused.hit.citation,
                "grade": hit.grade,
                "grade_reason": hit.reason,
                "rrf_score": hit.fused.rrf_score,
                "found_by": [
                    name
                    for name, rank in (
                        ("vector", hit.fused.vector_rank),
                        ("lexical", hit.fused.lexical_rank),
                    )
                    if rank is not None
                ],
            }
            for hit in outcome.hits
        ],
    }

    if not outcome.sufficient:
        #  No usable retrieval, no answer. Returning the model's unaided opinion
        #  here would be precisely the failure mode Sec. 2.3 is written against.
        #  The message distinguishes the two ways this happens, because they have
        #  different fixes: nothing retrieved means ingest or widen the filter;
        #  retrieved-but-ungraded means the corpus has adjacent material and not
        #  the answer, which is a different shopping list.
        searched = any(attempt.get("n_candidates") for attempt in outcome.attempts)
        detail = (
            "Passages were retrieved but none graded useful for this question — the corpus holds "
            "adjacent material, not the answer."
            if searched
            else "No indexed passage matched this question at all."
        )
        return RagAnswer(
            question=question,
            answer=(
                f"[DATA GAP: explicitly unresolved] {detail} Ingest the relevant process "
                "documentation, widen the technique filter, or name the specific document that "
                "would resolve it."
            ),
            sources=[hit.fused.hit for hit in outcome.hits],
            model=chat.model,
            provider=chat.name,
            techniques=technique_names,
            insufficient_context=True,
            query_was_rewritten=outcome.rewritten,
            effective_query=outcome.effective_query,
            retrieval=retrieval_trace,
        )

    hits = [hit.fused.hit for hit in outcome.useful]
    result = chat.send(
        SYNTHESIS_SYSTEM_PROMPT.format(context=format_context(hits)),
        [{"role": "user", "content": USER_PROMPT.format(question=question)}],
    )

    if result.refused:
        return RagAnswer(
            question=question,
            answer=(
                "The language model declined to answer this question"
                + (f" (category: {result.refusal_category})" if result.refusal_category else "")
                + ". The retrieved passages are returned below unchanged — read them directly, or "
                "put the same question to a different provider."
            ),
            sources=hits,
            model=result.model,
            provider=result.provider,
            techniques=technique_names,
            insufficient_context=True,
            refused_by_provider=True,
            query_was_rewritten=outcome.rewritten,
            effective_query=outcome.effective_query,
            retrieval=retrieval_trace,
        )

    return RagAnswer(
        question=question,
        answer=result.text,
        sources=hits,
        model=result.model,
        provider=result.provider,
        techniques=technique_names,
        insufficient_context="[DATA GAP" in result.text,
        query_was_rewritten=outcome.rewritten,
        effective_query=outcome.effective_query,
        retrieval=retrieval_trace,
    )


def assert_not_property_ingestion(destination: str) -> None:
    """Guard: RAG output must never become a stored property value.

    Called by any code path tempted to write model output into the analysis
    tables.  FOM_PROOF Sec. 2.3 and Sec. 15.2 rule this out completely, and the
    failure is silent by nature — a fabricated permittivity looks exactly like a
    real one once it is a float in a column.

    A value read from a paper enters through ``PropertyValue`` with its own DOI,
    page, and context, entered or reviewed by a person. That path stays separate
    from this one on purpose.
    """
    raise PermissionError(
        f"Refusing to write retrieval output into {destination!r}. FOM_PROOF Sec. 2.3: a missing "
        "property may not be replaced by a plausible number. Enter the value through "
        "PropertyValue with its DOI, page, and full measurement context instead."
    )
