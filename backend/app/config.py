"""Application configuration via pydantic-settings.

Loads all settings from environment variables / .env file with validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory containing .env (the backend/ package root)
_ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env into os.environ so all nested BaseSettings sub-models can read it.
# KeyError=False: don't fail if .env is missing (e.g. in CI, vars are set directly).
_ENV_PATH = _ROOT_DIR / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)
else:
    load_dotenv()  # fallback: look in CWD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_comma_list(v: str | list[str]) -> list[str]:
    """Parse a comma-separated string into a list of trimmed strings.

    Accepts both ``"a,b,c"`` and ``["a","b","c"]``.
    """
    if isinstance(v, list):
        return v
    return [item.strip() for item in v.split(",") if item.strip()]


def _resolve_dir(v: str) -> str:
    """Normalise a directory path to an absolute path."""
    import os

    return os.path.abspath(v)


# ---------------------------------------------------------------------------
# Sub-models (grouped by concern)
# ---------------------------------------------------------------------------


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    provider: str = Field(
        default="dashscope",
        description="Provider: dashscope | openai | deepseek | custom",
    )
    api_key: SecretStr = Field(
        default=SecretStr("sk-xxxxxxxx"),
        description="LLM API Key",
    )
    model: str = Field(
        default="qwen-plus",
        description="Model name",
    )
    base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://dashscope.aliyuncs.com/compatible-mode/v1"),
        description="LLM endpoint base URL",
    )

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    provider: str = Field(
        default="dashscope",
        description="Provider: dashscope | openai | custom",
    )
    api_key: SecretStr = Field(
        default=SecretStr("sk-xxxxxxxx"),
        description="Embedding API Key",
    )
    model: str = Field(
        default="text-embedding-v3",
        description="Embedding model name",
    )
    base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://dashscope.aliyuncs.com/compatible-mode/v1"),
        description="Embedding endpoint base URL (OpenAI-compatible)",
    )
    dimensions: int = Field(
        default=1024,
        description="Output embedding dimensions",
    )
    batch_size: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Max texts per batch API call",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max retries on transient failures",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        description="HTTP request timeout in seconds",
    )

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")


class RerankerSettings(BaseSettings):
    """Reranker model configuration."""

    model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
    )
    device: str = Field(
        default="cpu",
        description="cpu | cuda",
    )

    model_config = SettingsConfigDict(env_prefix="RERANKER_", extra="ignore")


class ChromaSettings(BaseSettings):
    """Chroma vector store configuration."""

    persist_dir: str = Field(
        default="./data/chroma_data",
    )
    collection: str = Field(
        default="knowledge_base",
    )

    model_config = SettingsConfigDict(env_prefix="CHROMA_", extra="ignore")

    @field_validator("persist_dir", mode="before")
    @classmethod
    def _abs_persist_dir(cls, v: str) -> str:
        return _resolve_dir(v)


class MySQLSettings(BaseSettings):
    """MySQL database configuration."""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=3306)
    user: str = Field(default="root")
    password: SecretStr = Field(default=SecretStr("your_password"))
    database: str = Field(default="agentic_rag")

    model_config = SettingsConfigDict(env_prefix="MYSQL_", extra="ignore")

    @property
    def async_dsn(self) -> str:
        """Async connection string (aiomysql)."""
        return (
            f"mysql+aiomysql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
            "?charset=utf8mb4"
        )

    @property
    def sync_dsn(self) -> str:
        """Sync connection string for Alembic / admin tasks."""
        return (
            f"mysql+pymysql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
            "?charset=utf8mb4"
        )


class UploadSettings(BaseSettings):
    """File upload configuration."""

    upload_dir: str = Field(default="./data/uploads")
    max_upload_size_mb: int = Field(default=50)

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    @field_validator("upload_dir", mode="before")
    @classmethod
    def _abs_upload_dir(cls, v: str) -> str:
        return _resolve_dir(v)


class AgentSettings(BaseSettings):
    """Agent behaviour configuration."""

    max_rounds: int = Field(default=3, ge=1, le=10)
    top_k_recall: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=5, ge=1, le=50)
    chunk_size: int = Field(default=800, ge=100, le=4096)
    chunk_overlap: int = Field(default=150, ge=0)

    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")


class ServerSettings(BaseSettings):
    """HTTP server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    version: str = Field(default="0.1.0")
    cors_origins: Any = Field(
        default="http://localhost:5173",
        description="Comma-separated allowed origins (stored as list after validation)",
    )
    log_level: str = Field(
        default="INFO",
        description="DEBUG | INFO | WARNING | ERROR",
    )

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated origins into a list."""
        return _parse_comma_list(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Root settings aggregator
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root settings — aggregates all sub-settings.

    Usage::

        from app.config import settings

        print(settings.llm.model)
        print(settings.mysql.async_dsn)
    """

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    # Sub-models — each reads its own prefixed env vars
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    chroma: ChromaSettings = ChromaSettings()
    mysql: MySQLSettings = MySQLSettings()
    upload: UploadSettings = UploadSettings()
    agent: AgentSettings = AgentSettings()
    server: ServerSettings = ServerSettings()

    def model_post_init(self, __context) -> None:
        """Ensure critical secrets are set."""
        super().model_post_init(__context)
        if self.llm.api_key.get_secret_value() in ("sk-xxxxxxxx", ""):
            raise ValueError(
                "LLM_API_KEY is required. Set it in .env or the environment."
            )
        if self.embedding.api_key.get_secret_value() in ("sk-xxxxxxxx", ""):
            raise ValueError(
                "EMBEDDING_API_KEY is required. Set it in .env or the environment."
            )
        if self.mysql.password.get_secret_value() in ("your_password", ""):
            raise ValueError(
                "MYSQL_PASSWORD is required. Set it in .env or the environment."
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

settings = Settings()
