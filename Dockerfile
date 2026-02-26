# =============================================================================
# Threat-to-Governance Pipeline — Multi-stage Docker Build
#
# Three stages:
#   1. base    — Python + system deps + pip packages (cached layer)
#   2. app     — Source code + configs (changes frequently)
#   3. api     — FastAPI results server (optional, for serving experiment results)
#
# Usage:
#   docker build --target app -t ttgp:latest .           # Experiment runner
#   docker build --target api -t ttgp-api:latest .       # Results API server
#
# GPU support (NVIDIA):
#   docker run --gpus all ttgp:latest python run_experiments.py --experiment 5
#
# CPU only:
#   docker run ttgp:latest python run_experiments.py --experiment 1
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base — dependencies (cached unless pyproject.toml changes)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# System deps for scientific computing + HuggingFace datasets
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency spec first (Docker cache optimization).
# This layer rebuilds ONLY when pyproject.toml changes, not on every
# source code edit. Saves 5-10 minutes on rebuilds.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# ---------------------------------------------------------------------------
# Stage 2: App — experiment runner
# ---------------------------------------------------------------------------
FROM base AS app

LABEL maintainer="Bipin Rimal <bipinrimal314@gmail.com>"
LABEL description="Cross-domain threat transfer: insider threat → AI agent governance"
LABEL org.opencontainers.image.source="https://github.com/BipinRimal314/threat-to-governance-pipeline"

# Copy source code, configs, and experiment specs
COPY src/ ./src/
COPY configs/ ./configs/
COPY run_experiments.py generate_figures.py ./
COPY tests/ ./tests/
COPY EXPERIMENTS_9_11.md ./

# Create results directory (persisted via volume mount)
RUN mkdir -p results/tables results/figures

# HuggingFace cache — mount as volume to persist across runs
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Default: show help
CMD ["python", "run_experiments.py", "--help"]

# ---------------------------------------------------------------------------
# Stage 3: API — serve experiment results via FastAPI
# ---------------------------------------------------------------------------
FROM app AS api

RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
