"""Application configuration models and access helpers."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    artifact_store_path: str = "./data/artifacts"
    
    # Required API keys
    nvidia_api_key: str | None = None
    nvidia_model: str = "mistralai/mistral-large-4-1t-a52b"
    
    # LLM provider: "nvidia" (NVIDIA NIM, requires NVIDIA_API_KEY) or "offline" (deterministic
    # template narrator; no network, no key).
    llm_provider: Literal["nvidia", "offline"] = "nvidia"

    # Research agent bounds
    research_max_iterations: int = Field(default=3, ge=1, le=10)
    research_max_tool_calls: int = Field(default=80, ge=10, le=500)
    research_max_llm_tool_steps: int = Field(default=4, ge=0, le=20)
    research_max_narrative_attempts: int = Field(default=2, ge=1, le=5)
    # When set, data is replayed from frozen fixtures (e.g. benchmarks/v1/fixtures) instead of live Yahoo.
    research_fixture_dir: str | None = None
    # Legacy one-shot LLM code generation; outputs are attached as unverified supplementary analysis.
    enable_codegen_analysis: bool = False

    # Optional settings
    sec_user_agent: str = "JASPER/0.1 (support@example.com)"
    notification_email: str | None = None
    enable_prometheus: bool = True

    model_config = SettingsConfigDict(case_sensitive=False, env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached instance of the application settings."""

    return AppSettings()
