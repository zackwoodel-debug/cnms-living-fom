"""Schemas for the /rag router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cnms_fom.db.enums import SynthesisTechnique


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=6, ge=1, le=25)
    techniques: list[SynthesisTechnique] | None = Field(
        default=None, description="Restrict retrieval to these corpus partitions."
    )
    model: str | None = None
    min_similarity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Below this, a passage is treated as not retrieved. Leaving it at 0 lets an "
        "off-topic question return the k least-bad chunks.",
    )


class SourceOut(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    technique: str
    page: int | None = None
    text: str
    similarity: float
    doi: str | None = None
    source_url: str | None = None
    citation: str


class RagQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    model: str
    techniques: list[str] = Field(default_factory=list)
    insufficient_context: bool = Field(
        description="True when no passage graded useful, or the model reported a data gap."
    )
    provider: str = Field(default="", description="ollama (local) or anthropic.")
    query_was_rewritten: bool = Field(
        default=False,
        description="True when the first retrieval found nothing useful and the query was "
        "rewritten once before retrying.",
    )
    effective_query: str = Field(
        default="", description="The query retrieval actually ran, after any rewrite."
    )
    retrieval: dict = Field(
        default_factory=dict,
        description="How the passages were found: per-attempt counts, each candidate's relevance "
        "grade, and which retriever(s) surfaced it. Read this before concluding the corpus is "
        "missing a document.",
    )
    refused_by_provider: bool = Field(
        default=False,
        description="True when the model's safety layer declined. Distinct from a data gap — the "
        "corpus is unaffected and the retrieved passages are still returned.",
    )
    disclaimer: str = (
        "Retrieval output is for reading, not for data entry. FOM_PROOF Sec. 2.3: a missing "
        "property may not be filled from a plausible number. Enter values through /materials "
        "with their DOI, page, and measurement context."
    )


class IngestRequest(BaseModel):
    path: str = Field(description="File or directory path readable by the API container.")
    technique: SynthesisTechnique | None = None
    title: str | None = None
    doi: str | None = None
    authors: str | None = None
    year: int | None = None
    source_url: str | None = None
    embed: bool = True


class IngestResponse(BaseModel):
    ingested: list[dict]
    total_documents: int
    total_chunks: int


class CorpusStatsResponse(BaseModel):
    documents_by_technique: dict[str, int]
    total_documents: int
    total_chunks: int
    embedded_chunks: int
    pgvector: bool


# ---------------------------------------------------------------------------
# Retrieval-only search, and the multi-step assistant
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Retrieve without generating.

    For reading passages yourself, and for answering "why did it not find the
    document I know is in there?" — which is otherwise unanswerable from a
    generated answer alone.
    """

    query: str = Field(min_length=2)
    k: int = Field(default=8, ge=1, le=25)
    techniques: list[SynthesisTechnique] | None = None
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    grade: bool = Field(
        default=False,
        description="Grade each candidate for relevance. Costs one model call per candidate; "
        "off by default because a human reading the passages does not need it.",
    )
    diagnostics: bool = Field(
        default=False,
        description="Also return each retriever's own ranking, side by side with the fused one.",
    )


class SearchHitOut(BaseModel):
    chunk_id: int
    citation: str
    technique: str
    page: int | None = None
    doi: str | None = None
    text: str
    similarity: float = Field(description="Cosine similarity from the dense retriever.")
    rrf_score: float = Field(description="Fused rank score. Not comparable to similarity.")
    found_by: list[str] = Field(
        default_factory=list, description="Which retrievers surfaced it: vector, lexical, or both."
    )
    vector_rank: int | None = None
    lexical_rank: int | None = None
    relevance_grade: int | None = Field(
        default=None, description="0-3 when grading ran: 3 directly answers, 0 irrelevant."
    )
    grade_reason: str | None = None


class SearchResponse(BaseModel):
    query: str
    n_hits: int
    hits: list[SearchHitOut] = Field(default_factory=list)
    lexical_backend: str = Field(
        default="",
        description="postgres_fulltext, or python_term_overlap where full-text search is "
        "unavailable.",
    )
    diagnostics: dict | None = None


class ChatRequest(BaseModel):
    """One turn with the research assistant.

    The assistant chooses its own retrievals and may chain them — corpus search,
    ModalFit fit records, property values, FOM scores. It answers only from tool
    results; an answer produced without a single tool call is replaced with a data
    gap before it is returned.
    """

    question: str = Field(min_length=2)
    session_key: str | None = Field(
        default=None,
        description="Continue an existing conversation. Omit to start one; the key comes back on "
        "the response.",
    )
    sample_id: str | None = Field(
        default=None,
        description="Scope the conversation to one ModalFit sample, so 'this film' resolves.",
    )
    techniques: list[SynthesisTechnique] | None = Field(
        default=None, description="Restrict corpus retrieval to these partitions."
    )
    user: str | None = None
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Tool-call budget for this turn. Past the budget the assistant is asked to "
        "answer from what it has, or report a gap.",
    )
    history_turns: int | None = Field(
        default=None, ge=0, le=20, description="Prior exchanges to replay. Evidence is stored "
        "for every turn regardless.",
    )
    provider: str | None = Field(
        default=None,
        description="Override the configured provider for this turn: 'ollama' (local) or "
        "'anthropic'. Choosing anthropic sends retrieved excerpts off this machine.",
    )
    model: str | None = None


class ToolCallOut(BaseModel):
    step: int
    tool: str
    arguments: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    duration_ms: int = 0


class ChatResponse(BaseModel):
    session_key: str
    question: str
    answer: str
    tools_used: list[str] = Field(default_factory=list)
    steps: list[ToolCallOut] = Field(
        default_factory=list,
        description="Every tool call and its full result — the audit trail behind the answer.",
    )
    citations: list[dict] = Field(
        default_factory=list,
        description="Deduplicated locators: document passages, fit record ids, property values.",
    )
    model: str = ""
    provider: str = ""
    insufficient_context: bool = False
    refused_by_provider: bool = False
    refusal_category: str | None = None
    hit_step_limit: bool = Field(
        default=False,
        description="True when the tool budget ran out. The answer was then produced from what "
        "had already been retrieved.",
    )
    latency_ms: int = 0
    usage: dict = Field(default_factory=dict)
    disclaimer: str = (
        "Retrieval output is for reading, not for data entry. FOM_PROOF Sec. 2.3: a missing "
        "property may not be filled from a plausible number. Fitted values enter the analysis "
        "tables through /modalfit/fits/{id}/promote, which requires an explicit material identity."
    )


class SessionSummaryOut(BaseModel):
    session_key: str
    title: str | None = None
    user: str | None = None
    sample_id: str | None = None
    techniques: list[str] | None = None
    provider: str | None = None
    chat_model: str | None = None
    created_at: str | None = None
    messages_by_role: dict = Field(default_factory=dict)
    n_data_gaps: int = Field(
        default=0,
        description="Turns that reported a data gap. A high count is a corpus-coverage finding, "
        "not a broken assistant.",
    )


class TranscriptResponse(BaseModel):
    session: SessionSummaryOut
    messages: list[dict] = Field(default_factory=list)


class AssistantConfigResponse(BaseModel):
    provider: str
    ollama: dict
    anthropic: dict
    max_steps: int
    tools: list[str] = Field(description="The read-only tool surface, always available.")
    write_tools: list[str] = Field(
        default_factory=list,
        description="Opt-in per request, and they reach knowledge cards only.",
    )
    write_tools_note: str = ""
