FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

    RUN groupadd -r appuser && useradd -r -g appuser -m appuser

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \ 
    pip install --no-cache-dir -e .

COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p /app/runs && chown -R appuser:appuser /app

USER appuser
ENTRYPOINT ["python"]

CMD ["app/main.py"]