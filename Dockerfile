FROM python:3.11-slim as base

# Security: create non-root user
RUN useradd -m appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Switch to non-root user
USER appuser
CMD ["uvicorn", "crucible.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
