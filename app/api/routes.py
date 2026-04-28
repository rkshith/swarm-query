from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from rq.job import Job

from app.core.config import get_settings
from app.core.queue import chat_queue, ingestion_queue, redis_connection
from app.models.schemas import ChatJobResponse, ChatRequest, DocumentInfo, UploadResponse
from app.services.qdrant_store import QdrantStore
from app.services.storage import save_upload_file
from app.tasks.rq_tasks import answer_query, index_pdf_document

settings = get_settings()
router = APIRouter()


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore()


@router.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "RAG backend is running"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type not in settings.allowed_mime_types:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    doc_id, stored_path = save_upload_file(file.filename or "document.pdf", content)
    job = ingestion_queue.enqueue(
        index_pdf_document,
        doc_id=doc_id,
        filename=file.filename or "document.pdf",
        stored_path=str(stored_path),
        job_timeout=1800,
        result_ttl=86400,
    )
    return UploadResponse(doc_id=doc_id, filename=file.filename or "document.pdf", job_id=job.id)


@router.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    job = chat_queue.enqueue(
        answer_query,
        query=request.query,
        doc_id=request.doc_id,
        job_timeout=600,
        result_ttl=86400,
    )
    return {"status": "queued", "job_id": job.id}


@router.get("/jobs/{job_id}", response_model=ChatJobResponse)
def job_status(job_id: str) -> ChatJobResponse:
    job = Job.fetch(job_id, connection=redis_connection)
    status = job.get_status()
    result = job.result if status == "finished" else None
    error = str(job.exc_info) if status == "failed" and job.exc_info else None
    return ChatJobResponse(job_id=job_id, status=status, result=result, error=error)


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
def document_info(doc_id: str) -> DocumentInfo:
    metadata = get_store().document_metadata(doc_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentInfo(
        doc_id=str(metadata["doc_id"]),
        filename=str(metadata["filename"]),
        stored_path=str(metadata["stored_path"]),
        chunks_indexed=int(metadata["chunks_indexed"]),
        status="indexed",
        created_at=str(metadata["created_at"]),
    )
