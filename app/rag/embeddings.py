from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    prefix = settings.embedding_query_prefix.strip()
    query_text = f"{prefix} {query}".strip() if prefix else query
    vector = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.tolist()
