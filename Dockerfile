# Multi-stage runtime image for ingestor / worker / persistence / admin.
#
# Base images are overridable so CI can use your GitLab registry when Docker Hub
# is unreachable from the runner:
#   docker build \
#     --build-arg NODE_IMAGE=registry.host/typhon/base/node:20-bookworm-slim \
#     --build-arg PYTHON_IMAGE=registry.host/typhon/base/python:3.11-slim-bookworm \
#     .

ARG NODE_IMAGE=node:20-bookworm-slim
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# --- Frontend (discarded after copying dist) ---
FROM ${NODE_IMAGE} AS ui
WORKDIR /ui
COPY platform/ui/package.json platform/ui/package-lock.json ./
RUN npm ci
COPY platform/ui/ ./
RUN npm run build \
    && rm -rf node_modules

# --- Python deps (wheels only; keep compilers out of final image) ---
FROM ${PYTHON_IMAGE} AS python-deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch is dramatically smaller than the default CUDA wheel.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install -r requirements.txt \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -name "*.pyc" -delete

# --- Runtime image ---
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin marketmaker \
    && mkdir -p /config /data/parquet /var/log/mm

COPY --from=python-deps /opt/venv /opt/venv
COPY mm/ mm/
COPY --from=ui /ui/dist platform/ui/dist

RUN chown -R marketmaker:marketmaker /config /data/parquet /var/log/mm /app/platform

USER marketmaker

# Deployments override the command, for example:
#   python3 -m mm.ingestor --config /config/ingestor.json
#   python3 -m mm.worker --config /config/worker.json
#   python3 -m mm.persistence --config /config/persistence.json
#   python3 -m mm.api.admin
CMD ["python3", "-m", "mm.api.admin"]
