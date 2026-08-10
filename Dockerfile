# Multi-stage runtime image for ingestor / worker / persistence / admin.
#
# Prefer a prebuilt PYTHON_DEPS_IMAGE (venv with torch) seeded via:
#   ./scripts/seed_registry_base_images.sh

ARG NODE_IMAGE=node:20-bookworm-slim
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
# Defaults to PYTHON_IMAGE; CI should override with typhon/base/python-deps:...
ARG PYTHON_DEPS_IMAGE=python:3.11-slim-bookworm

# --- Frontend ---
FROM ${NODE_IMAGE} AS ui
WORKDIR /ui
COPY platform/ui/package.json platform/ui/package-lock.json ./
RUN npm ci
COPY platform/ui/ ./
RUN npm run build \
    && rm -rf node_modules

# --- Python deps (prebuilt image already has /opt/venv + torch) ---
FROM ${PYTHON_DEPS_IMAGE} AS python-deps
COPY requirements.txt /tmp/requirements.txt
RUN set -eux; \
    if [ -x /opt/venv/bin/python ] && /opt/venv/bin/python -c "import torch"; then \
      echo "Using prebuilt /opt/venv"; \
      /opt/venv/bin/pip install -r /tmp/requirements.txt; \
      /opt/venv/bin/python -c "import torch; print('torch', torch.__version__)"; \
    else \
      echo "ERROR: PYTHON_DEPS_IMAGE has no torch. Seed with scripts/seed_registry_base_images.sh"; \
      exit 1; \
    fi

# --- Runtime ---
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

RUN set -eux; \
    /opt/venv/bin/python -c "import torch; print('torch OK', torch.__version__)"; \
    chown -R marketmaker:marketmaker /config /data/parquet /var/log/mm /app/platform

USER marketmaker

CMD ["python3", "-m", "mm.api.admin"]
