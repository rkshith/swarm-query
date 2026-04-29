from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.services.qdrant_store import QdrantStore

app = FastAPI(title="RAG Queue v2", version="2.0.0")


@app.on_event("startup")
def startup_event() -> None:
    QdrantStore().ensure_collection()


app.include_router(router)
