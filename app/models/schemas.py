from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobType(str, Enum):
    ingestion = "ingestion"
    chat = "chat"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    doc_id: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    job_id: str
    status: Literal["queued"] = "queued"


class ChatJobResponse(BaseModel):
    job_id: str
    status: str
    result: Any | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    stored_path: str
    chunks_indexed: int
    status: str
    created_at: str
