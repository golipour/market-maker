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

# ``g++`` is required by the PyTorch Inductor C++ backend, and ``ninja``
# (ninja-build) drives the codegen build. Without both, ``torch.compile`` raises
# ``InvalidCxxCompiler`` on the first call and the optimizer silently runs eager
# forever -- which is exactly what happened in production before this was added.
#
# Both come from apt rather than pip on purpose: the CI runner has no DNS for
# files.pythonhosted.org, so declaring ``ninja`` in requirements.txt fails the
# build. The Debian archive resolves fine.
#
# Inductor also needs the CPython headers, which the python:3.11 image already
# ships at /usr/local/include/python3.11.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpq5 \
        g++ \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin marketmaker \
    && mkdir -p /config /data/parquet /data/attribution /var/log/mm /var/cache/torchinductor \
    && chown marketmaker:marketmaker /var/cache/torchinductor

COPY --from=python-deps /opt/venv /opt/venv
COPY mm/ mm/
COPY --from=ui /ui/dist platform/ui/dist

# Fail the build if the Inductor toolchain is unusable, instead of shipping an
# image that silently runs eager forever. The real compile is not exercised here
# (it needs app sources and would cost seconds); only the toolchain is checked,
# which is exactly what regressed.
#
# ninja comes from apt (ninja-build), so check the binary rather than importing
# the pip ``ninja`` module, which is intentionally absent (see requirements.txt).
RUN set -eux; \
    /opt/venv/bin/python -c "import torch; print('torch OK', torch.__version__)"; \
    command -v g++ >/dev/null || { echo 'ERROR: g++ missing, torch.compile cannot work'; exit 1; }; \
    command -v ninja >/dev/null || { echo 'ERROR: ninja missing, inductor cannot build'; exit 1; }; \
    /opt/venv/bin/python -c "from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability(); print('inductor toolchain OK')" || { echo 'ERROR: inductor build toolchain unusable'; exit 1; }; \
    chown -R marketmaker:marketmaker /config /data /var/log/mm /app/platform /var/cache/torchinductor

USER marketmaker

CMD ["python3", "-m", "mm.api.admin"]
