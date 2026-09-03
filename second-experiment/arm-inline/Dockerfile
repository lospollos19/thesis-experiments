# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1 — base: CPU-only image used for CI (GitHub Actions runners).
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir poetry

WORKDIR /app

# Dependency layer — copy only manifests first for build caching.
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --extras headless

# Application source (tests are intentionally excluded via .dockerignore).
COPY src/ ./src/
COPY main.py ./
RUN poetry install --only-root --extras headless

CMD ["python", "main.py", "--source", "simulation", "--max-frames", "100"]

# =============================================================================
# Stage 2 — gpu: CUDA toolkit for the NVIDIA AGX Orin (ARM64 / L4T).
#
# The *devel* variant is required, not *runtime*: PyCUDA compiles the CUDA-C
# kernels at run time through SourceModule, which shells out to nvcc.
# =============================================================================
FROM nvcr.io/nvidia/l4t-cuda:12.2.12-devel AS gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
# PyCUDA (built against the local CUDA 12.2) + headless OpenCV (no display).
RUN poetry install --no-root --extras "gpu headless"

COPY src/ ./src/
COPY main.py ./
RUN poetry install --only-root --extras "gpu headless"

CMD ["python3", "main.py", "--source", "simulation", "--max-frames", "100"]
