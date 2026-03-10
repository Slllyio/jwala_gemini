# ============================================================
# Van Suraksha Pipeline — Multi-stage Dockerfile
# ============================================================
# Stage 1: GDAL + Python base
# Stage 2: App with all dependencies
# ============================================================

# ── Stage 1: Base with GDAL ─────────────────────────────────
FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: App ─────────────────────────────────────────────
FROM base AS app

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.docker.txt ./
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy application code
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY config.yaml ./
COPY notebooks/ ./notebooks/

# Create output directories that are expected at runtime
RUN mkdir -p outputs/pipeline_runs \
             outputs/model_registry \
             outputs/checkpoints \
             outputs/reports \
             outputs/alerts

# ── Stage 3: API server (thin layer on top of app) ───────────
FROM app AS api

# api-specific metadata — the actual entrypoint is set in Compose
LABEL org.opencontainers.image.description="Van Suraksha FastAPI service"

# Default entrypoint for the pipeline runner image
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--help"]
