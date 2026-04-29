from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings

settings = get_settings()


def save_upload_file(filename: str, content: bytes) -> tuple[str, Path]:
    doc_id = uuid4().hex
    safe_name = Path(filename).name
    stored_name = f"{doc_id}_{safe_name}"
    destination = settings.upload_dir / stored_name
    destination.write_bytes(content)
    return doc_id, destination
