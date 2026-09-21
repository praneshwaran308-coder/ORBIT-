"""ORBIT configuration.

All provider settings come from environment variables (optionally via a local
``.env`` file next to this package). Nothing is hardcoded: the Freebuff
endpoint, API key, and model are fully operator-controlled.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / ".data"

_MODEL = r"^[A-Za-z0-9._\-/:]+$"


class Settings(BaseSettings):
    """Environment-driven configuration for ORBIT."""

    freebuff_base_url: str = Field(
        # Freebuff's supported integration is its BYOK provider endpoint
        # (see app/llm/freebuff_byok.py). There is no api.freebuff.ai host.
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible endpoint used via Freebuff's provider integration.",
    )
    freebuff_api_key: str = Field(
        default="",
        description="Provider credential (never hardcoded; normally resolved from Freebuff's BYOK store).",
    )
    freebuff_model: str = Field(
        default="openai/gpt-4o-mini",
        pattern=_MODEL,
        description="Default model id (normally taken from the Freebuff BYOK connection).",
    )
    freebuff_timeout_seconds: float = Field(
        default=60.0, gt=0, description="Per-request HTTP timeout for the LLM client."
    )
    freebuff_max_retries: int = Field(
        default=2, ge=0, le=5, description="Bounded retries for transient LLM failures."
    )
    max_upload_bytes: int = Field(
        default=20 * 1024 * 1024, gt=0, description="Maximum accepted CSV upload size."
    )
    task_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Maximum wall-clock time for one task; a task still running after "
            "this long is marked failed (its late result is discarded)."
        ),
    )
    enable_api_docs: bool = Field(
        default=True,
        description=(
            "Expose /docs, /redoc and /openapi.json. Keep True for local "
            "development; set ENABLE_API_DOCS=0 in production."
        ),
    )
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated allowed CORS origins for the frontend.",
    )

    model_config = SettingsConfigDict(
        env_file=(DATA_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def describe_config(settings: Settings | None = None) -> dict[str, str | bool | float]:
    """Human-readable, secret-free view of the *effective* configuration."""
    s = settings or get_settings()
    from app.llm.freebuff_byok import resolve_freebuff_provider

    resolved = resolve_freebuff_provider(settings=s)
    config: dict[str, str | bool | float] = {
        "provider_source": resolved.source if resolved else "unconfigured",
        "base_url": resolved.base_url if resolved else s.freebuff_base_url,
        "model": resolved.model if resolved else s.freebuff_model,
        "api_key_configured": bool(resolved and resolved.api_key) or bool(s.freebuff_api_key),
        "timeout_seconds": s.freebuff_timeout_seconds,
    }
    if resolved and resolved.source == "byok":
        config["connection"] = resolved.detail.get("connection_name", "")
    return config
