from __future__ import annotations

from functools import lru_cache

from groq import Groq

from app.core.config import get_settings
from app.rag.prompts import SYSTEM_PROMPT

settings = get_settings()


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=settings.groq_api_key)


def generate_answer(query: str, context: str) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        temperature=0.2,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + f"\n\nContext:\n{context if context else 'No context found.'}",
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or ""
