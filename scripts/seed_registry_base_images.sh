#!/usr/bin/env bash
# Seed base images into your GitLab Container Registry for CI.
# Run on a machine that CAN reach Docker Hub + download.pytorch.org
# (not the GitLab runner).
#
# Usage:
#   export CI_REGISTRY='registry.tmhezardastan.ir'
#   export CI_REGISTRY_USER='gitlab+deploy-token-...'
#   export CI_REGISTRY_PASSWORD='...'
#   bash ./scripts/seed_registry_base_images.sh

set -euo pipefail

REGISTRY="${CI_REGISTRY:?Set CI_REGISTRY (e.g. registry.tmhezardastan.ir)}"
USER_NAME="${CI_REGISTRY_USER:?Set CI_REGISTRY_USER}"
PASSWORD="${CI_REGISTRY_PASSWORD:?Set CI_REGISTRY_PASSWORD}"
PREFIX="${BASE_IMAGE_PREFIX:-${REGISTRY}/typhon/base}"

NODE_SRC="${NODE_SRC:-node:20-bookworm-slim}"
PYTHON_SRC="${PYTHON_SRC:-python:3.11-slim-bookworm}"
NODE_DST="${PREFIX}/node:20-bookworm-slim"
PYTHON_DST="${PREFIX}/python:3.11-slim-bookworm"
PYTHON_DEPS_DST="${PREFIX}/python-deps:3.11-slim-bookworm"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "$PASSWORD" | docker login -u "$USER_NAME" --password-stdin "$REGISTRY"

echo "Pulling ${NODE_SRC}..."
docker pull "$NODE_SRC"
echo "Pulling ${PYTHON_SRC}..."
docker pull "$PYTHON_SRC"

docker tag "$NODE_SRC" "$NODE_DST"
docker tag "$PYTHON_SRC" "$PYTHON_DST"

echo "Pushing ${NODE_DST}..."
docker push "$NODE_DST"
echo "Pushing ${PYTHON_DST}..."
docker push "$PYTHON_DST"

echo "Building ${PYTHON_DEPS_DST} (torch CPU + requirements)..."
docker build \
  --build-arg "PYTHON_IMAGE=${PYTHON_DST}" \
  -f "${ROOT}/docker/Dockerfile.python-deps" \
  -t "${PYTHON_DEPS_DST}" \
  "${ROOT}"

echo "Pushing ${PYTHON_DEPS_DST}..."
docker push "$PYTHON_DEPS_DST"

echo ""
echo "Done. Set these GitLab CI/CD variables:"
echo "  CI_REGISTRY=${REGISTRY}"
echo "  BASE_NODE_IMAGE=${NODE_DST}"
echo "  BASE_PYTHON_IMAGE=${PYTHON_DST}"
echo "  BASE_PYTHON_DEPS_IMAGE=${PYTHON_DEPS_DST}"
