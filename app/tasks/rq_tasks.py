from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from functools import lru_cache

from app.rag.embeddings import embed_texts
from app.rag.llm import generate_answer
from app.rag.retrieval import build_context, retrieve_chunks
from app.services.pdf import chunk_page_texts, extract_page_texts
from app.services.qdrant_store import QdrantStore


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore()


def index_pdf_document(*, doc_id: str, filename: str, stored_path: str) -> dict[str, Any]:
    pdf_path = Path(stored_path)
    page_texts = extract_page_texts(pdf_path)
    chunks = chunk_page_texts(page_texts)

    if not chunks:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "stored_path": stored_path,
            "chunks_indexed": 0,
            "status": "empty_pdf",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    vectors = embed_texts([chunk["text"] for chunk in chunks])
    chunk_count = get_store().upsert_chunks(
        doc_id=doc_id,
        filename=filename,
        stored_path=stored_path,
        chunks=chunks,
        vectors=vectors,
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "stored_path": stored_path,
        "chunks_indexed": chunk_count,
        "status": "indexed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def answer_query(*, query: str, doc_id: str | None = None) -> dict[str, Any]:
    chunks = retrieve_chunks(query=query, doc_id=doc_id)
    context = build_context(chunks)
    answer = generate_answer(query=query, context=context)
    sources = [
        {
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "score": chunk.score,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    return {"query": query, "doc_id": doc_id, "answer": answer, "sources": sources}
