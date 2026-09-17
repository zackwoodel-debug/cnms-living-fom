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

    # --- Ollama -----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

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
