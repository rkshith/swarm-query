from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings

settings = get_settings()
VECTOR_SIZE = 384


class QdrantStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection
        self.ensure_collection()

    def ensure_collection(self) -> None:
        collections = {collection.name for collection in self.client.get_collections().collections}
        if self.collection_name in collections:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
        )

    def upsert_chunks(self, *, doc_id: str, filename: str, stored_path: str, chunks: list[dict], vectors: list[list[float]]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        points: list[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                qmodels.PointStruct(
                    id=uuid4().hex,
                    vector=vector,
                    payload={
                        "doc_id": doc_id,
                        "filename": filename,
                        "stored_path": stored_path,
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "created_at": now,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, *, query_vector: list[float], limit: int = 6, doc_id: str | None = None):
        query_filter = None
        if doc_id:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            )
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

    def document_chunk_count(self, doc_id: str) -> int:
        result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        return len(result)

    def document_metadata(self, doc_id: str) -> dict[str, str | int]:
        result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            return {}
        payload = result[0].payload or {}
        return {
            "doc_id": str(payload.get("doc_id", doc_id)),
            "filename": str(payload.get("filename", "unknown")),
            "stored_path": str(payload.get("stored_path", "unknown")),
            "chunks_indexed": self.document_chunk_count(doc_id),
            "created_at": str(payload.get("created_at", "unknown")),
        }
