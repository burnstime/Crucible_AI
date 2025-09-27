# crucible_ai

Production-ready, local-deployable LLM system with FastAPI, PyTorch, Hugging Face, MLflow, Celery, Redis, MinIO, and Prometheus. See `summary.txt` for implementation notes.

## Endpoints
- `/generate` — Generate LLM response
- `/metrics` — Prometheus metrics
- `/health` — Healthcheck endpoint

## Docker Compose Usage
See `INSTRUCTIONS.md` for full details. Quick start:

```sh
cp .env.example .env
make up
```

- Visit API: http://localhost:8000/health
- Prometheus: http://localhost:9090
- MLflow: http://localhost:5000
- MinIO: http://localhost:9000

All containers run as non-root users. Healthchecks and resource limits are enforced.
