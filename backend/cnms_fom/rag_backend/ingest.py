"""PDF ingestion for the synthesis corpus.

Chunks carry their page number so that every retrieved passage can be cited as
``document, p. N``.  Sec. 2.2 requires source provenance down to the page for
any value entering an analysis; the same standard applies to anything the
assistant tells a user about a process.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from cnms_fom.db.enums import SynthesisTechnique

logger = logging.getLogger(__name__)

#  Process recipes are dense and tabular; smaller chunks with generous overlap
#  keep a parameter and its units in the same passage.
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class ChunkPayload:
    """One embeddable passage plus its citation locator."""

    chunk_index: int
    page: int | None
    text: str


def sha256_of(path: Path) -> str:
    """Content hash — the corpus is deduplicated by content, not by filename."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_technique(path: Path) -> SynthesisTechnique:
    """Guess the corpus partition from the filename.

    A guess, and treated as one: ``ingest_pdf`` takes an explicit ``technique``
    that overrides this. Keep it for bulk loading a directory that already uses
    a naming convention.

    TODO(CNMS): replace with the real document taxonomy once the user-docs
    repository structure is known.
    """
    name = path.name.lower()
    for technique, needles in {
        SynthesisTechnique.MBE: ("mbe", "molecular-beam", "molecular_beam"),
        SynthesisTechnique.PLD: ("pld", "pulsed-laser", "pulsed_laser"),
        SynthesisTechnique.ALD: ("ald", "atomic-layer", "atomic_layer"),
        SynthesisTechnique.SPUTTERING: ("sputter",),
        SynthesisTechnique.CVD: ("cvd", "mocvd"),
        SynthesisTechnique.CNMS_USER_DOC: ("cnms", "user-guide", "sop", "user_doc"),
    }.items():
        if any(needle in name for needle in needles):
            return technique
    return SynthesisTechnique.OTHER


def load_and_split(
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[ChunkPayload], int]:
    """Read a PDF and split it, preserving the page each chunk came from."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("PDF ingestion needs the 'rag' extra: pip install -e '.[rag]'") from exc

    pages = PyPDFLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = splitter.split_documents(pages)

    chunks = [
        ChunkPayload(
            chunk_index=index,
            #  PyPDFLoader pages are 0-based; humans cite from 1.
            page=(doc.metadata.get("page") + 1) if doc.metadata.get("page") is not None else None,
            text=doc.page_content,
        )
        for index, doc in enumerate(documents)
        if doc.page_content.strip()
    ]
    return chunks, len(pages)


def ingest_pdf(
    session,
    path: Path | str,
    *,
    technique: SynthesisTechnique | None = None,
    title: str | None = None,
    doi: str | None = None,
    authors: str | None = None,
    year: int | None = None,
    source_url: str | None = None,
    embed: bool = True,
) -> dict:
    """Ingest one PDF into ``documents`` + ``document_chunks``.

    Idempotent by content hash: re-ingesting the same file is a no-op, so a
    directory sweep can be re-run safely.
    """
    from cnms_fom.db.models import Document, DocumentChunk

    from .embeddings import check_embedding_dim, embed_documents

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    content_hash = sha256_of(path)
    existing = (
        session.query(Document).filter(Document.content_sha256 == content_hash).one_or_none()
    )
    if existing is not None:
        logger.info("Already ingested (sha256 match): %s", path.name)
        return {
            "document_id": existing.id,
            "title": existing.title,
            "chunks": len(existing.chunks),
            "skipped": True,
        }

    chunks, n_pages = load_and_split(path)
    if not chunks:
        raise ValueError(f"No extractable text in {path.name}. Is it a scanned PDF needing OCR?")

    document = Document(
        title=title or path.stem,
        filename=path.name,
        content_sha256=content_hash,
        technique=technique or infer_technique(path),
        doi=doi,
        authors=authors,
        year=year,
        source_url=source_url,
        n_pages=n_pages,
    )
    session.add(document)
    session.flush()  # assign document.id

    vectors: list[list[float] | None] = [None] * len(chunks)
    embedding_model: str | None = None
    if embed:
        from cnms_fom.config import get_settings

        embedding_model = get_settings().ollama_embed_model
        vectors = embed_documents([c.text for c in chunks])
        if vectors:
            check_embedding_dim(vectors[0])

    session.add_all(
        DocumentChunk(
            document_id=document.id,
            chunk_index=chunk.chunk_index,
            page=chunk.page,
            text=chunk.text,
            n_tokens=len(chunk.text.split()),
            embedding_model=embedding_model,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
    session.flush()

    logger.info("Ingested %s: %d chunks over %d pages", path.name, len(chunks), n_pages)
    return {
        "document_id": document.id,
        "title": document.title,
        "technique": document.technique.value,
        "chunks": len(chunks),
        "pages": n_pages,
        "skipped": False,
    }


def ingest_directory(
    session, directory: Path | str, *, technique: SynthesisTechnique | None = None, embed: bool = True
) -> list[dict]:
    """Ingest every PDF under ``directory`` (recursively)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    results: list[dict] = []
    for pdf in sorted(directory.rglob("*.pdf")):
        try:
            results.append(ingest_pdf(session, pdf, technique=technique, embed=embed))
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the sweep
            logger.warning("Skipped %s: %s", pdf.name, exc)
            results.append({"filename": pdf.name, "error": str(exc)})
    return results
