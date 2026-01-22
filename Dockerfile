# Dockerfile for RAG-Anything API Server
# Optimized for Alma Linux 9 / RHEL 9 compatibility

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# LibreOffice for Office document support
# Additional tools for document processing
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
# Install RAG-Anything with all optional dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    -r requirements.txt \
    "raganything[all]" \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    python-dotenv \
    pydantic \
    rq

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/output /app/rag_storage /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WORKING_DIR=/app/rag_storage
ENV OUTPUT_DIR=/app/output
ENV LOG_DIR=/app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["python", "api_server.py"]

