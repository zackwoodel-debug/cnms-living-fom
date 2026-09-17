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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cnms_fom.config import get_settings
from cnms_fom.db.enums import SynthesisTechnique

from .vectorstore import ChunkHit, search_chunks

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
    """An answer plus the evidence it was built from."""

    question: str
    answer: str
    sources: list[ChunkHit] = field(default_factory=list)
    model: str = ""
    techniques: list[str] = field(default_factory=list)
    insufficient_context: bool = False

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "model": self.model,
            "techniques": self.techniques,
            "insufficient_context": self.insufficient_context,
        }


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    """A ``ChatOllama`` bound to the configured server.

    Temperature defaults to 0: this is a retrieval task, and sampling diversity
    here buys nothing except a wider distribution of invented numbers.
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
        f"[{i}] {hit.citation()}\n{hit.text.strip()}" for i, hit in enumerate(hits, start=1)
    )


def answer_question(
    session,
    question: str,
    *,
    k: int = 6,
    techniques: list[SynthesisTechnique] | None = None,
    model: str | None = None,
    min_similarity: float = 0.2,
) -> RagAnswer:
    """Retrieve, then answer strictly from what was retrieved."""
    from .embeddings import embed_query

    query_vector = embed_query(question)
    hits = search_chunks(
        session, query_vector, k=k, techniques=techniques, min_similarity=min_similarity
    )
    technique_names = [t.value for t in techniques] if techniques else []

    if not hits:
        #  No retrieval, no answer. Returning the model's unaided opinion here
        #  would be precisely the failure mode Sec. 2.3 is written against.
        return RagAnswer(
            question=question,
            answer=(
                "[DATA GAP: explicitly unresolved] No indexed passage met the similarity "
                "threshold for this question. Ingest the relevant process documentation, or "
                "widen the technique filter, before relying on an answer."
            ),
            sources=[],
            model=model or get_settings().ollama_chat_model,
            techniques=technique_names,
            insufficient_context=True,
        )

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("RAG needs the 'rag' extra: pip install -e '.[rag]'") from exc

    chat = get_chat_model(model)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYNTHESIS_SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | chat
    response = chain.invoke({"context": format_context(hits), "question": question})
    text = getattr(response, "content", str(response))

    return RagAnswer(
        question=question,
        answer=text,
        sources=hits,
        model=model or get_settings().ollama_chat_model,
        techniques=technique_names,
        insufficient_context="[DATA GAP" in text,
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
