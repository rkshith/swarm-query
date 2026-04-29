from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.rag.embeddings import embed_query
from app.services.qdrant_store import QdrantStore

settings = get_settings()


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore()


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    page_number: int
    chunk_index: int
    filename: str
    score: float


def retrieve_chunks(query: str, doc_id: str | None = None) -> list[RetrievedChunk]:
    vector = embed_query(query)
    results = get_store().search(query_vector=vector, limit=settings.search_limit, doc_id=doc_id)
    chunks: list[RetrievedChunk] = []
    seen: set[tuple[str, int, int]] = set()

    for result in results:
        payload: dict[str, Any] = result.payload or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            continue

        key = (
            str(payload.get("filename", "")),
            int(payload.get("page_number", 0)),
            int(payload.get("chunk_index", 0)),
        )
        if key in seen:
            continue
        seen.add(key)

        score = float(result.score or 0.0)
        if score < settings.min_score:
            continue

        chunks.append(
            RetrievedChunk(
                text=text,
                page_number=int(payload.get("page_number", 0)),
                chunk_index=int(payload.get("chunk_index", 0)),
                filename=str(payload.get("filename", "")),
                score=score,
            )
        )

    return chunks[: settings.max_context_chunks]


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[File: {chunk.filename} | Page: {chunk.page_number} | Chunk: {chunk.chunk_index} | Score: {chunk.score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)
