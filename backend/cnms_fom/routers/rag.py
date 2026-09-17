"""/rag — retrieval over the synthesis corpus (MBE, PLD, ALD, CNMS user docs)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from cnms_fom.db.base import get_db
from cnms_fom.rag_backend.chains import answer_question
from cnms_fom.rag_backend.ingest import ingest_directory, ingest_pdf
from cnms_fom.rag_backend.vectorstore import corpus_stats
from cnms_fom.schemas.rag import (
    CorpusStatsResponse,
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RagQueryResponse)
def query(payload: RagQueryRequest, db: Session = Depends(get_db)) -> RagQueryResponse:
    """Answer a synthesis question strictly from retrieved passages.

    Every answer carries its sources with page-level citations. When nothing
    clears the similarity threshold the response says so rather than falling back
    on the model's own recollection — FOM_PROOF Sec. 2.3 rules out filling a gap
    with a plausible number, and a language model is very good at producing one.
    """
    try:
        result = answer_question(
            db,
            payload.question,
            k=payload.k,
            techniques=payload.techniques,
            model=payload.model,
            min_similarity=payload.min_similarity,
        )
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"RAG extra not installed: {exc}. pip install -e '.[rag]'",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface Ollama connectivity clearly
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Retrieval failed: {exc}. Is Ollama reachable, and are the chat and embedding "
            "models pulled?",
        ) from exc

    return RagQueryResponse(**result.as_dict())


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    """Ingest a PDF, or every PDF under a directory.

    Idempotent by content hash, so a directory sweep can be re-run after adding
    files without re-embedding what is already indexed.
    """
    path = Path(payload.path)
    if not path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{path} is not visible to the API. Under docker-compose, mount it and use the "
            "container path (./data is mounted at /app/data).",
        )

    try:
        if path.is_dir():
            results = ingest_directory(
                db, path, technique=payload.technique, embed=payload.embed
            )
        else:
            results = [
                ingest_pdf(
                    db,
                    path,
                    technique=payload.technique,
                    title=payload.title,
                    doi=payload.doi,
                    authors=payload.authors,
                    year=payload.year,
                    source_url=payload.source_url,
                    embed=payload.embed,
                )
            ]
        db.commit()
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"RAG extra not installed: {exc}. pip install -e '.[rag]'",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Ingestion failed: {exc}") from exc

    stats = corpus_stats(db)
    return IngestResponse(
        ingested=results,
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
    )


@router.get("/corpus", response_model=CorpusStatsResponse)
def corpus(db: Session = Depends(get_db)) -> CorpusStatsResponse:
    """Corpus coverage by technique — what the assistant can actually answer from."""
    return CorpusStatsResponse(**corpus_stats(db))
