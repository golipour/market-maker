# Diagnostics UI: FastAPI + static SPA + matplotlib PNG (Redis bridge).
# Build from repository root:
#   docker build -t mm-viz -f mm_viz/Dockerfile .
# Run (needs Redis reachable at MM_DIAGNOSTICS_REDIS_URL):
#   docker run --rm -p 8501:8501 -e MM_DIAGNOSTICS_REDIS_URL=redis://host.docker.internal:6379/0 mm-viz
#
# Prefer Compose: `docker compose up -d redis mm_viz` (from repo `docker-compose.yml`), or
# `docker compose up --build` for the full stack.
#
# Dependencies are installed in-line below (no COPY before pip) so code edits reuse the
# cached install layer. Keep versions aligned with mm_viz/requirements.txt for local venvs.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "redis>=4" \
    "matplotlib>=3.8" \
    "numpy>=1.24" \
    "fastapi>=0.109" \
    "uvicorn[standard]>=0.27"

COPY mm_viz/ /app/mm_viz/

# Server imports `mm_adapters.diagnostics.delta_codec` for v2 delta envelopes.
COPY mm_adapters/__init__.py /app/mm_adapters/__init__.py
COPY mm_adapters/diagnostics/__init__.py /app/mm_adapters/diagnostics/__init__.py
COPY mm_adapters/diagnostics/delta_codec.py /app/mm_adapters/diagnostics/delta_codec.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/config', timeout=3)"

CMD ["uvicorn", "mm_viz.server:app", "--host", "0.0.0.0", "--port", "8501"]
