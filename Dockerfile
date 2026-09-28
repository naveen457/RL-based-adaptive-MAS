FROM python:3.13-slim

# Explicit image version tag.
# GitHub Actions reads this version and automatically publishes this tag alongside :latest.
# No git commit hash tags are generated. You can change this to "1.0.0", "1.0.1", "latest", etc.
ARG VERSION=1.0.0
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser -m appuser

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/

RUN pip install --no-cache-dir --no-deps -e . && \
    mkdir -p /app/runs && chown -R appuser:appuser /app/runs

USER appuser

EXPOSE 8000

# Server mode by default for Render and cloud deployment (binds to dynamic $PORT)
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]