from __future__ import annotations

from redis import Redis
from rq import Queue

from app.core.config import get_settings

settings = get_settings()
redis_connection = Redis.from_url(settings.redis_url)
ingestion_queue = Queue("ingestion", connection=redis_connection)
chat_queue = Queue("chat", connection=redis_connection)
