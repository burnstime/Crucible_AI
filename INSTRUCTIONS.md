# crucible_ai Build Instructions

## Goals
- Build a modular, production-ready LLM system as described in REQUIREMENTS_DOC.md.
- Follow best practices for reproducibility, testability, and maintainability.
- All code must be type-annotated, documented, and tested.
- All dependencies must be pinned and validated.
- All subsystems must be implemented and integrated as described.

## Build Steps
1. Review REQUIREMENTS_DOC.md and this file before any implementation.
2. Scaffold the repo structure exactly as specified.
3. Implement each subsystem in its designated file(s).
4. Add sample data and test fixtures for deterministic CI.
5. Write and run all unit tests (pytest).
6. Lint and format code (black, flake8).
7. Build and run Docker Compose stack; verify all services are healthy.
8. Run a sample training and inference; verify logs and outputs.
9. Document all commands and outputs in verification.txt.
10. Zip the repo and deliver all required artifacts.

## Review Checklist
- [ ] All requirements in REQUIREMENTS_DOC.md are met.
- [ ] All acceptance criteria are satisfied.
- [ ] All code is readable, tested, and documented.
- [ ] All outputs (zip, file_tree, verification, summary) are present.
- [ ] All steps are reproducible from scratch.

## Docker Compose Usage (Production Readiness)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11+ for local development

### Environment Variables
- Copy `.env` to your root directory and edit as needed:
  - `CANARY_PERCENTAGE`, `MLFLOW_TRACKING_URI`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`

### Build and Run
```sh
# Build and start all services in the background
make up

# Or manually:
docker compose up --build -d
```

### Health Verification
- Check all services are healthy:
  - `docker compose ps` (should show "healthy" for all)
  - API: http://localhost:8000/metrics
  - MLflow: http://localhost:5000
  - MinIO: http://localhost:9000
  - Prometheus: http://localhost:9090
- Logs: `docker compose logs api` (or any service)

### Stopping
```sh
docker compose down
```

### Prometheus Monitoring
- Prometheus is pre-configured to scrape the API metrics endpoint.
- Edit `infra/prometheus/prometheus.yml` to add more targets if needed.

### Security & Best Practices
- All containers run as non-root users.
- Secrets/configs are managed via `.env` (do not commit secrets to git).
- Healthchecks and resource limits are enforced in `docker-compose.yml`.
