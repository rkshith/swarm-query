from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rag-queue-v2"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages:"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_documents"

    redis_url: str = "redis://localhost:6379/0"
    upload_dir: Path = Path("storage/uploads")

    max_context_chunks: int = 6
    chunk_size: int = 1000
    chunk_overlap: int = 150
    search_limit: int = 12
    min_score: float = 0.25

    allowed_mime_types: set[str] = {"application/pdf"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
