# ---------------------------------------------------------------------------
# 3GPP RAG Assistant — API Service
# ---------------------------------------------------------------------------
# Multi-stage build:
#   Stage 1 (builder): install Python deps into a venv
#   Stage 2 (runtime): copy venv + source, run with a non-root user
#
# Build:
#   docker build -t 3gpp-rag-api .
#
# Run (standalone, expects Ollama on the host):
#   docker run -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     3gpp-rag-api
#
# Or use docker-compose (recommended — includes Ollama sidecar):
#   docker compose up
# ---------------------------------------------------------------------------

# --------------- Stage 1: dependency builder --------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools needed by some native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (layer cache optimisation)
COPY requirements.txt pyproject.toml ./

# Create a venv and install dependencies into it
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# --------------- Stage 2: runtime image ------------------------------------
FROM python:3.10-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the venv from the builder
COPY --from=builder /venv /venv

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Create data directories (will be mounted as volumes in production)
RUN mkdir -p data/raw data/processed data/vectordb && \
    chown -R appuser:appuser /app

USER appuser

# Make the venv the default Python
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose the API port
EXPOSE 8000

# Health check — calls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command — can be overridden in docker-compose
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
