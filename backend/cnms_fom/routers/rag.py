"""/rag — retrieval over the synthesis corpus, and the research assistant.

Three ways in, for three different jobs:

``POST /query``   one retrieval, one answer.  Right when the question maps onto a
    single search.
``POST /search``  retrieval with no generation, plus per-retriever diagnostics.
    Right when you want to read the passages yourself, or find out why a document
    you know is indexed did not come back.
``POST /chat``    the multi-step assistant.  It chooses its own retrievals, can
    chain them, and reaches the ModalFit fit records as well as the corpus.
    Right for anything that needs more than one lookup — "is this thickness
    trustworthy?" needs the fits, then the literature on why they disagree.

All three share the same floor: an answer with no evidence behind it is replaced
by a data gap.  FOM_PROOF Sec. 2.3 and Sec. 15.2 — a missing value may not be
filled by a plausible number, and a language model is very good at producing one.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from cnms_fom.config import get_settings
from cnms_fom.db.base import get_db
from cnms_fom.db.enums import SynthesisTechnique
from cnms_fom.rag_backend import agent, memory
from cnms_fom.rag_backend.chains import answer_question
from cnms_fom.rag_backend.grading import retrieve_with_correction
from cnms_fom.rag_backend.hybrid import hybrid_search, retrieval_diagnostics
from cnms_fom.rag_backend.ingest import ingest_directory, ingest_pdf
from cnms_fom.rag_backend.providers import get_provider
from cnms_fom.rag_backend.vectorstore import corpus_stats
from cnms_fom.schemas.rag import (
    AssistantConfigResponse,
    ChatRequest,
    ChatResponse,
    CorpusStatsResponse,
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    SearchRequest,
    SearchResponse,
    SessionSummaryOut,
    ToolCallOut,
    TranscriptResponse,
)

logger = logging.getLogger(__name__)

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


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    """Retrieve without generating.

    Hybrid retrieval: dense embeddings for paraphrase, Postgres full-text for the
    rare exact tokens a synthesis corpus is mostly made of, fused by reciprocal
    rank. ``found_by`` on each hit shows which retriever surfaced it — a passage
    only lexical found is usually one the embedding could never have reached.

    ``diagnostics=true`` returns each retriever's own ranking alongside the fused
    one, which is how you answer "the paper is definitely indexed, why is it not
    coming back?" without guessing.
    """
    try:
        if payload.grade:
            outcome = retrieve_with_correction(
                db,
                payload.query,
                k=payload.k,
                techniques=payload.techniques,
                min_similarity=payload.min_similarity,
                grade=True,
            )
            hits = [
                {
                    **hit.fused.as_dict(),
                    "relevance_grade": hit.grade,
                    "grade_reason": hit.reason,
                }
                for hit in outcome.hits
            ]
        else:
            hits = [
                fused.as_dict()
                for fused in hybrid_search(
                    db,
                    payload.query,
                    k=payload.k,
                    techniques=payload.techniques,
                    min_similarity=payload.min_similarity,
                )
            ]

        extras = (
            retrieval_diagnostics(db, payload.query, techniques=payload.techniques)
            if payload.diagnostics
            else None
        )
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"RAG extra not installed: {exc}. pip install -e '.[rag]'",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Retrieval failed: {exc}. Is Ollama reachable, and is the embedding model pulled?",
        ) from exc

    return SearchResponse(
        query=payload.query,
        n_hits=len(hits),
        hits=[
            {
                "chunk_id": hit["chunk_id"],
                "citation": hit["citation"],
                "technique": hit["technique"],
                "page": hit.get("page"),
                "doi": hit.get("doi"),
                "text": hit["text"],
                "similarity": hit.get("similarity", 0.0),
                "rrf_score": hit.get("rrf_score", 0.0),
                "found_by": hit.get("found_by", []),
                "vector_rank": hit.get("vector_rank"),
                "lexical_rank": hit.get("lexical_rank"),
                "relevance_grade": hit.get("relevance_grade"),
                "grade_reason": hit.get("grade_reason"),
            }
            for hit in hits
        ],
        lexical_backend=(extras or {}).get("lexical_backend", ""),
        diagnostics=extras,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """One turn with the research assistant.

    The assistant runs its own retrievals and may chain them: list a sample's
    ModalFit refinements, compare a thickness across XRR and SE, then search the
    corpus for what causes that discrepancy. Every tool call and its full result
    come back in ``steps`` and are persisted, so the answer stays traceable after
    the conversation is closed.

    An answer produced without a single tool call is replaced with a data gap
    before it is returned. That is not a safety net bolted on afterwards — it is
    the only thing separating this endpoint from a chatbot with a materials
    vocabulary.
    """
    settings = get_settings()

    try:
        provider = get_provider(payload.provider, payload.model)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    try:
        session = memory.get_or_create_session(
            db,
            payload.session_key,
            user=payload.user,
            sample_id=payload.sample_id,
            techniques=payload.techniques,
            provider=provider.name,
            chat_model=provider.model,
        )
        history = memory.load_history(
            db,
            session,
            turns=payload.history_turns
            if payload.history_turns is not None
            else settings.assistant_history_turns,
        )

        answer = agent.ask(
            db,
            payload.question,
            history=history,
            #  Session scope wins over the per-turn field: a conversation pinned
            #  to a sample stays pinned, and the request may simply have omitted it.
            sample_id=session.sample_id or payload.sample_id,
            techniques=payload.techniques,
            provider=provider,
            max_steps=payload.max_steps or settings.assistant_max_steps,
        )
        memory.record_turn(db, session, payload.question, answer)
        db.commit()
    except ImportError as exc:
        db.rollback()
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            f"Assistant extra not installed: {exc}. pip install -e '.[rag]' (or '.[anthropic]' "
            "for the Anthropic provider).",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Assistant failed: {exc}. For the local provider, check that Ollama is reachable and "
            "that the chat and embedding models are pulled.",
        ) from exc

    return ChatResponse(
        session_key=session.session_key,
        question=answer.question,
        answer=answer.answer,
        tools_used=answer.tools_used,
        steps=[ToolCallOut(**step.as_dict()) for step in answer.steps],
        citations=answer.citations,
        model=answer.model,
        provider=answer.provider,
        insufficient_context=answer.insufficient_context,
        refused_by_provider=answer.refused_by_provider,
        refusal_category=answer.refusal_category,
        hit_step_limit=answer.hit_step_limit,
        latency_ms=answer.latency_ms,
        usage=answer.usage,
    )


@router.get("/sessions", response_model=list[SessionSummaryOut])
def list_conversations(
    limit: int = Query(default=25, ge=1, le=100),
    user: str | None = None,
    db: Session = Depends(get_db),
) -> list[SessionSummaryOut]:
    """Recent conversations, newest first."""
    return [SessionSummaryOut(**summary) for summary in memory.list_sessions(db, limit=limit, user=user)]


@router.get("/sessions/{session_key}", response_model=TranscriptResponse)
def get_transcript(session_key: str, db: Session = Depends(get_db)) -> TranscriptResponse:
    """The full conversation with the evidence behind every answer."""
    from cnms_fom.db.models import ChatSession

    session = (
        db.query(ChatSession).filter(ChatSession.session_key == session_key).one_or_none()
    )
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No conversation {session_key!r}.")
    return TranscriptResponse(
        session=SessionSummaryOut(**memory.session_summary(db, session)),
        messages=memory.transcript(db, session),
    )


@router.delete("/sessions/{session_key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(session_key: str, db: Session = Depends(get_db)) -> None:
    """Delete a conversation and its evidence trail.

    Exposed because a conversation can contain a user's unpublished work and they
    are entitled to remove it. It takes the audit trail with it, which is the
    trade-off and is why it is an explicit call rather than a retention policy.
    """
    from cnms_fom.db.models import ChatSession

    session = (
        db.query(ChatSession).filter(ChatSession.session_key == session_key).one_or_none()
    )
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No conversation {session_key!r}.")
    db.delete(session)
    db.commit()


@router.get("/assistant", response_model=AssistantConfigResponse)
def assistant_config() -> AssistantConfigResponse:
    """What the assistant is configured to run on, and which tools it has."""
    return AssistantConfigResponse(**agent.available_models())


@router.post("/ingest/upload", response_model=IngestResponse)
def ingest_upload(
    files: list[UploadFile] = File(description="One or more PDFs to add to the corpus."),
    technique: SynthesisTechnique | None = Form(default=None),
    doi: str | None = Form(default=None),
    authors: str | None = Form(default=None),
    year: int | None = Form(default=None),
    source_url: str | None = Form(default=None),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Upload PDFs and index them in one call.

    This is the path for "here are some papers, read them". Files are written into
    the configured corpus directory (``CORPUS_DIR``, default ``data/pdfs``) and
    then chunked, embedded, and indexed with page-level citation locators, so
    every later answer can point at a page.

    Ingestion is idempotent by content hash, so re-uploading the same paper is a
    no-op rather than a duplicate — which matters because the same PDF arrives
    twice under two filenames more often than not.

    ``technique`` is worth setting when you know it. Without it the partition is
    guessed from the filename, and a guess puts the paper in the wrong corpus
    slice, where a technique-filtered search will not find it.
    """
    settings = get_settings()
    corpus_dir = Path(settings.corpus_dir)
    try:
        corpus_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Cannot write to the corpus directory {corpus_dir}: {exc}. Set CORPUS_DIR to a "
            "writable path.",
        ) from exc

    limit_bytes = settings.max_upload_mb * 1024 * 1024
    results: list[dict] = []
    saved: list[Path] = []

    for upload in files:
        name = Path(upload.filename or "upload.pdf").name
        if not name.lower().endswith(".pdf"):
            results.append(
                {
                    "filename": name,
                    "error": "Only PDFs are ingested here. A .docx or .txt has to be converted "
                    "first — the chunker records a page number for every passage, and a format "
                    "without pages cannot produce a citation.",
                }
            )
            continue

        destination = corpus_dir / name
        #  Never silently overwrite a file already in the corpus: two different
        #  papers can share a filename, and the loser would vanish. Content-hash
        #  deduplication happens at ingest, so a real duplicate is cheap.
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            counter = 2
            while destination.exists():
                destination = corpus_dir / f"{stem}__{counter}{suffix}"
                counter += 1

        written = 0
        try:
            with destination.open("wb") as handle:
                while chunk := upload.file.read(1 << 20):
                    written += len(chunk)
                    if written > limit_bytes:
                        raise ValueError(
                            f"exceeds the {settings.max_upload_mb} MB limit "
                            f"(MAX_UPLOAD_MB)"
                        )
                    handle.write(chunk)
        except (ValueError, OSError) as exc:
            destination.unlink(missing_ok=True)
            results.append({"filename": name, "error": str(exc)})
            continue
        finally:
            upload.file.close()

        saved.append(destination)

    for path in saved:
        try:
            results.append(
                ingest_pdf(
                    db,
                    path,
                    technique=technique,
                    #  Per-file metadata only makes sense for a single upload;
                    #  applying one DOI to a batch would mislabel every paper in it.
                    title=title if len(saved) == 1 else None,
                    doi=doi if len(saved) == 1 else None,
                    authors=authors if len(saved) == 1 else None,
                    year=year if len(saved) == 1 else None,
                    source_url=source_url if len(saved) == 1 else None,
                )
            )
            db.commit()
        except ImportError as exc:
            db.rollback()
            raise HTTPException(
                status.HTTP_501_NOT_IMPLEMENTED,
                f"RAG extra not installed: {exc}. pip install -e '.[rag]'. The file was saved to "
                f"{path}, so it can be ingested later without re-uploading.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not lose the batch
            db.rollback()
            logger.warning("Ingest failed for %s: %s", path.name, exc)
            results.append(
                {
                    "filename": path.name,
                    "error": f"{exc}",
                    "saved_to": str(path),
                    "hint": "A scanned PDF with no text layer needs OCR before it can be "
                    "chunked. The file was kept, so it can be re-ingested after OCR.",
                }
            )

    stats = corpus_stats(db)
    return IngestResponse(
        ingested=results,
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
    )
