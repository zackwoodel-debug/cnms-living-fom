"""Liveness, readiness, and a quick view of which optional extras are installed."""

from __future__ import annotations

import importlib.util

from fastapi import APIRouter

from cnms_fom import __version__
from cnms_fom.config import get_settings

router = APIRouter(tags=["health"])


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@router.get("/health")
def health() -> dict:
    """Liveness. Deliberately touches nothing external."""
    return {"status": "ok", "version": __version__}


@router.get("/health/ready")
def ready() -> dict:
    """Readiness: what actually works right now.

    Reports per-subsystem rather than a single boolean, because the platform is
    useful in degraded states — the FOM engine needs only Postgres, and there is
    no reason to report the whole API as down because Ollama is not running.
    """
    settings = get_settings()
    checks: dict[str, dict] = {}

    try:
        from sqlalchemy import text

        from cnms_fom.db.base import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
            has_pgvector = bool(
                connection.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ).first()
            )
        checks["database"] = {"ok": True, "pgvector_extension": has_pgvector}
    except Exception as exc:  # noqa: BLE001 - readiness must report, not raise
        checks["database"] = {"ok": False, "error": str(exc)}

    checks["extras"] = {
        "descriptors": _installed("pymatgen") and _installed("matminer"),
        "bo": _installed("botorch"),
        "rag": _installed("langchain_ollama"),
        "pgvector": _installed("pgvector"),
    }
    checks["ollama"] = {"base_url": settings.ollama_base_url, "chat_model": settings.ollama_chat_model}

    return {
        "status": "ok" if checks["database"]["ok"] else "degraded",
        "version": __version__,
        "checks": checks,
    }
