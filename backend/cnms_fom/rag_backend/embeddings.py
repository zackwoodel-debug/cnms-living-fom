"""Embedding generation via Ollama.

Thin on purpose: the embedding model name is stored on every chunk row, because
vectors from two different models are not comparable and mixing them silently
degrades retrieval in a way that is very hard to notice later.
"""

from __future__ import annotations

from cnms_fom.config import get_settings


def get_embedder(model: str | None = None):
    """Return a LangChain ``OllamaEmbeddings`` bound to the configured server."""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "RAG needs the 'rag' extra: pip install -e '.[rag]'"
        ) from exc

    settings = get_settings()
    return OllamaEmbeddings(
        model=model or settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


def embed_documents(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of passages."""
    return get_embedder(model).embed_documents(texts)


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query."""
    return get_embedder(model).embed_query(text)


def check_embedding_dim(vector: list[float]) -> None:
    """Fail loudly when the model's output width disagrees with the schema.

    A pgvector column has a fixed width.  Swapping the embedding model without
    updating ``EMBEDDING_DIM`` produces an insert error at best and a silently
    truncated corpus at worst.
    """
    settings = get_settings()
    if len(vector) != settings.embedding_dim:
        raise ValueError(
            f"Embedding model returned {len(vector)} dimensions but EMBEDDING_DIM is "
            f"{settings.embedding_dim}. Update the setting and re-embed the corpus — "
            "vectors of different widths cannot share a table."
        )
