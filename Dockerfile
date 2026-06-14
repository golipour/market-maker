FROM node:20-bookworm-slim AS ui

WORKDIR /ui
COPY platform/ui/package.json platform/ui/package-lock.json ./
RUN npm ci

COPY platform/ui/ ./
RUN npm run build

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mm/ mm/
COPY --from=ui /ui/dist platform/ui/dist

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin marketmaker \
    && mkdir -p /config /data/parquet /var/log/mm \
    && chown -R marketmaker:marketmaker /config /data/parquet /var/log/mm /app/platform

USER marketmaker

# Each deployment sets its own command, for example:
#   python3 -m mm.ingestor --config /config/ingestor.json
#   python3 -m mm.worker --config /config/worker.json
#   python3 -m mm.persistence --config /config/persistence.json
#   python3 -m mm.api.admin
