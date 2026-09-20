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
