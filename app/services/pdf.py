from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import get_settings

settings = get_settings()


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str


def extract_page_texts(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number=index, text=text))
    return pages


def chunk_page_texts(page_texts: Iterable[PageText]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for page in page_texts:
        for chunk_index, chunk_text in enumerate(splitter.split_text(page.text)):
            cleaned = chunk_text.strip()
            if cleaned:
                chunks.append(
                    {
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        "text": cleaned,
                    }
                )
    return chunks
