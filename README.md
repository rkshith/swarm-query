# Swarm Query

## Run with Docker

Use the Docker Compose v2 plugin on this machine:


```bash
docker compose up --build
```

If you see a `ModuleNotFoundError: No module named 'distutils'` error, you are running the legacy `docker-compose` v1 Python launcher. This repository expects the newer `docker compose` command instead.

## Services

- `api`: FastAPI app on port `8000`
- `worker`: RQ worker for ingestion and chat queues
- `redis`: job queue backend
- `qdrant`: vector store backend
