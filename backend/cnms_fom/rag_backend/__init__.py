"""Retrieval-augmented generation over the synthesis corpus and the lab's own records.

Local by default: with ``RAG_LLM_PROVIDER=ollama`` (the default), MBE/PLD/ALD
process notes and CNMS user documents do not leave the machine.  ``providers``
offers an opt-in Anthropic path for when retrieval fidelity matters more than
locality, and states the trade-off rather than hiding it.

The one rule that shapes every module here: retrieval supports *reading*, never
*data entry*.  FOM_PROOF Sec. 2.3 forbids filling a missing property from a
plausible number, and a language model is an extremely efficient source of
plausible numbers.  See ``chains.SYNTHESIS_SYSTEM_PROMPT`` and
``chains.assert_not_property_ingestion``.  The one path that *does* write
measured values into the analysis tables — ``modalfit.promote`` — lives in another
package, takes instrument-derived fits rather than model output, and requires a
material identity a person supplied.

Pipeline
--------
``embeddings``   vectors, via Ollama
``vectorstore``  dense similarity over ``document_chunks``, pgvector or numpy
``hybrid``       dense + lexical retrieval fused by reciprocal rank
``grading``      relevance grading, reranking, one corrective query rewrite
``chains``       single-shot question answering (``POST /rag/query``)
``tools``        read-only tool surface over corpus, fits, properties, scores
``agent``        the multi-step assistant (``POST /rag/chat``)
``memory``       conversation persistence, with the evidence behind each answer
``providers``    the chat-model seam: Ollama or Anthropic
``ingest``       PDF ingestion with page-level citation locators
"""

from .agent import AgentAnswer, AgentStep, ask, available_models  # noqa: F401
from .chains import RagAnswer, answer_question  # noqa: F401
from .grading import GradedHit, RetrievalOutcome, retrieve_with_correction  # noqa: F401
from .hybrid import FusedHit, hybrid_search, retrieval_diagnostics  # noqa: F401
from .ingest import ingest_directory, ingest_pdf  # noqa: F401
from .memory import (  # noqa: F401
    get_or_create_session,
    list_sessions,
    load_history,
    record_turn,
    session_summary,
    transcript,
)
from .providers import ChatProvider, get_provider  # noqa: F401
from .tools import TOOLS, run_tool, tool_specs  # noqa: F401
from .vectorstore import ChunkHit, corpus_stats, search_chunks  # noqa: F401
