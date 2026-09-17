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
        description="True when no passage cleared the threshold, or the model reported a data gap."
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
