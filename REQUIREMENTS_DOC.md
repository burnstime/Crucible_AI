# crucible_ai Requirements Document

## Project Overview
A production-ready, local-deployable LLM system with modular subsystems for inference, training, data cleaning, validation, canary deployment, retraining, experiment tracking, CLI, and full CI/CD.

## Subsystems
1. **Repo & Environment**: Structured repo, reproducible env, Docker, CI, helper scripts.
2. **FastAPI Inference & Logging**: POST /generate, model wrapper, JSONL logging, error handling, Prometheus metrics.
3. **Training & Fine-tuning**: PyTorch/HF, CLI args, mixed precision, checkpointing, MLflow logging.
4. **Data Cleaning**: PII removal, deduplication, normalization, CLI callable.
5. **Validation & Safety**: Profanity, refusal, hallucination tests, report, CI fail on threshold.
6. **Canary Deployment**: Deterministic routing, Prometheus, per-request logging.
7. **Retraining Scheduler**: Celery/cron, logs, exit codes, candidate promotion.
8. **MLflow Integration**: Log params, metrics, artifacts, registry.
9. **CLI**: Click-based, serve/train/clean/test/deploy commands.
10. **Documentation**: README, usage, architecture, sample data, verification.

## Acceptance Criteria
- All tests pass locally and in CI.
- Docker Compose brings up all services; API responds to /generate.
- Training run completes with sample data and logs to MLflow.
- Canary routing and retrain scheduler work as specified.
- All dependencies pinned and reproducible.
- Verification.txt documents all commands and outputs.

## Constraints
- Python 3.11+, PyTorch, HF Transformers, FastAPI, MLflow, Celery, Redis, MinIO, Prometheus, Click, pytest, black, flake8.
- Use smallest compatible HF model for reproducibility.
- All network/model ops must have timeouts and clear logs.
- Code must be type-annotated, docstringed, and tested.
