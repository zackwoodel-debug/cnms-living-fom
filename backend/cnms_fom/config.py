"""Application settings, loaded from the environment (see .env.example)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Database ---------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg2://cnms:cnms@localhost:5432/cnms_fom",
        description="SQLAlchemy URL for the Postgres instance.",
    )
    db_echo: bool = False
    #  A claim about the database, not a feature switch. It selects the ORM type for
    #  ``document_chunks.embedding`` at import (``db/base.embedding_column_type``),
    #  and nothing can make it agree with the schema it is pointed at. Migration 0009
    #  converts the column to ``vector`` exactly when the ``vector`` extension is
    #  present, so the correct value is "does this database have that extension".
    #  Either direction of disagreement breaks retrieval — see
    #  ``rag_backend.vectorstore.embedding_storage_mismatch`` — so a mismatch is
    #  reported at startup and in ``/health/ready`` rather than left to be found.
    pgvector_enabled: bool = True
    embedding_dim: int = 768

    # --- Retrieval / LLM provider -----------------------------------------
    #  "ollama" keeps every corpus excerpt on this machine, which is why it is
    #  the default: the corpus is unpublished CNMS process documentation.
    #  "anthropic" trades that locality for a model that follows the retrieval
    #  discipline (declining when evidence is thin, carrying a parameter's full
    #  context, refusing to average disagreeing sources) more reliably.
    rag_llm_provider: str = "ollama"

    # --- Ollama -----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    #  Optional small model for relevance grading and query rewriting. Those are
    #  classification tasks — "does this passage answer the question, 0-3" — and
    #  they run once per retrieved candidate, so they dominate the wall clock
    #  while needing none of the answer model's reasoning. Measured on this
    #  scaffold: grading 7 candidates took 173 s on a 14B reasoning model and 29 s
    #  on a 1B one, for the same decisions. Unset means use the chat model.
    rag_grader_model: str | None = None
    #  Optional model for claim extraction, which runs once per retrieved passage.
    #  Extraction is structured output — read a passage, emit JSON — not reasoning,
    #  and a reasoning model spends its budget deliberating about the schema.
    #  Measured on this scaffold: a 14B reasoning model took 90-250 s *per passage*,
    #  so six passages is a quarter of an hour for one brief. Unset means use the
    #  chat model.
    rag_extraction_model: str | None = None

    # --- per-passage call cache and concurrency ---------------------------
    #  Grading and extraction call a model once per passage and are together
    #  essentially the whole cost of a brief. Both are pure functions of their
    #  input, so both are cached by a hash of the content that was sent — see
    #  rag_backend/cache.py. The cache holds no measurement and clearing it costs
    #  only time.
    llm_cache_enabled: bool = True
    #  How many per-passage calls to run at once. Independent, so the output is
    #  identical either way.
    #
    #  Measured against one local Ollama instance running qwen3:14b: 7 grading calls
    #  took 196.9 s serially and 179.2 s with four workers — **9%**, not the 4x the
    #  call count suggests. A single model instance is compute-bound, so the server
    #  time-slices concurrent requests rather than overlapping them. Parallelism pays
    #  where latency dominates — a remote API, or several models — and barely moves a
    #  saturated local one. The cache is what matters locally: the same 7 calls took
    #  0.011 s warm.
    #
    #  Left at 4 because 9% is still free, and it becomes a real win the moment the
    #  provider is remote.
    llm_max_parallel: int = 4

    # --- Anthropic (only used when rag_llm_provider == "anthropic") -------
    #  Left unset by default. When it is unset the SDK resolves credentials
    #  itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
    #  profile), so an explicit empty value here must not shadow that.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5"
    #  Server-side refusal fallback: a policy decline re-runs on a fallback
    #  model within the same call instead of returning nothing.
    anthropic_refusal_fallbacks: bool = True

    # --- Corpus storage -----------------------------------------------------
    #  Where uploaded PDFs land. Under docker-compose ./data is mounted at
    #  /app/data, so the same path works inside and outside the container.
    corpus_dir: str = "data/pdfs"
    #  Cap on one uploaded file. A 200 MB scan is almost always a mis-drag, and
    #  the failure without a cap is an OOM rather than a message.
    max_upload_mb: int = 100

    # --- Research assistant ------------------------------------------------
    #  Tool-call budget for one assistant turn. Past this the model is usually
    #  re-searching a corpus that does not hold the answer, and a data gap is
    #  the honest outcome.
    assistant_max_steps: int = 6
    #  Prior exchanges replayed to the model. Evidence is stored for every turn
    #  regardless; this only bounds what goes back into the prompt.
    assistant_history_turns: int = 6

    # --- API --------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- Analysis defaults (FOM_PROOF) ------------------------------------
    permutation_b: int = Field(
        default=10_000,
        description="FOM_PROOF Eq. (42): >= 10,000 permutations for a final reported matrix.",
    )
    random_seed: int = 20260823
    score_floor_eps: float = Field(
        default=1e-3,
        description="FOM_PROOF Eq. (26): floor applied to normalised inputs of a geometric score.",
    )

    # --- CNMS integration -------------------------------------------------
    # TODO(CNMS): populate once the facility endpoints/credentials are issued.
    cnms_proposal_api_url: str | None = None
    cnms_instrument_registry_url: str | None = None
    cnms_api_token: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — safe to call from request handlers."""
    return Settings()
