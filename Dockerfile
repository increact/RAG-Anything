# Dockerfile for RAG-Anything API Server
# Optimized for Alma Linux 9 / RHEL 9 compatibility

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# LibreOffice for Office document support
# Additional tools for document processing
# curl: used by the healthcheck (smaller and timeout-safe vs python requests)
RUN apt-get update && apt-get install -y \
    build-essential \
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
    fonts-liberation \
    fonts-dejavu \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user. The worker image bind-mounts host volumes
# (./output, ./rag_storage, ./logs) — running as root means any container
# escape gets host-root, so we drop privileges before COPY/runtime.
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

# Copy requirements first for better caching.
# requirements.loc[k] is a glob — it copies requirements.lock if it exists and
# is a silent no-op otherwise, so the build works with or without a lockfile.
COPY --chown=appuser:appuser requirements.txt requirements.loc[k] pyproject.toml ./

# Install Python dependencies.
# Prefer a fully-pinned requirements.lock (generate it with scripts/lock-deps.sh)
# for reproducible builds; fall back to the loose requirements.txt + explicit
# extras when no lockfile is committed yet.
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.lock ]; then \
        echo "Installing from requirements.lock (pinned)" && \
        pip install --no-cache-dir -r requirements.lock; \
    else \
        echo "Installing from requirements.txt (unpinned — see scripts/lock-deps.sh)" && \
        pip install --no-cache-dir \
        -r requirements.txt \
        "raganything[all]" \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        python-dotenv \
        pydantic \
        rq; \
    fi

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories owned by appuser
RUN mkdir -p /app/output /app/rag_storage /app/logs \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

# Set environment variables. HF/torch caches are routed to the appuser home
# so the docker-compose `rag_models_cache` volume should mount at
# /home/appuser/.cache (no longer /root/.cache).
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WORKING_DIR=/app/rag_storage
ENV OUTPUT_DIR=/app/output
ENV LOG_DIR=/app/logs
ENV HOME=/home/appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# Expose API port
EXPOSE 8000

# Health check — curl with --max-time so an OOM-hung server fails fast and
# Docker's restart policy can kick in instead of waiting on a stuck python.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail --silent --max-time 5 http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "api_server.py"]

