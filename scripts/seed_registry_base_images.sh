#!/usr/bin/env bash
# Seed linux/amd64 base images into GitLab Container Registry for CI.
# Run on a machine that can reach Docker Hub + download.pytorch.org.
#
# Always seeds linux/amd64 (GitLab runner arch), even on Apple Silicon.
#
# Usage:
#   export CI_REGISTRY='registry.tmhezardastan.ir'
#   export CI_REGISTRY_USER='...'
#   export CI_REGISTRY_PASSWORD='...'
#   bash ./scripts/seed_registry_base_images.sh
#
# Do NOT use bare `sudo` (it drops env vars). Use `sudo -E` or `sudo env ...` if needed.

set -euo pipefail

REGISTRY="${CI_REGISTRY:?Set CI_REGISTRY (e.g. registry.tmhezardastan.ir)}"
USER_NAME="${CI_REGISTRY_USER:?Set CI_REGISTRY_USER}"
PASSWORD="${CI_REGISTRY_PASSWORD:?Set CI_REGISTRY_PASSWORD}"
PREFIX="${BASE_IMAGE_PREFIX:-${REGISTRY}/typhon/base}"
PLATFORM="${SEED_PLATFORM:-linux/amd64}"

NODE_SRC="${NODE_SRC:-node:20-bookworm-slim}"
PYTHON_SRC="${PYTHON_SRC:-python:3.11-slim-bookworm}"
NODE_DST="${PREFIX}/node:20-bookworm-slim"
PYTHON_DST="${PREFIX}/python:3.11-slim-bookworm"
PYTHON_DEPS_DST="${PREFIX}/python-deps:3.11-slim-bookworm"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "$PASSWORD" | docker login -u "$USER_NAME" --password-stdin "$REGISTRY"

echo "Seeding platform: ${PLATFORM}"

# Retag via buildx so the pushed manifest is unambiguously $PLATFORM
# (plain docker tag on Apple Silicon often keeps an arm64 image).
mirror_image() {
  local src="$1"
  local dst="$2"
  echo "Mirroring ${src} -> ${dst} (${PLATFORM})..."
  docker buildx build \
    --platform "${PLATFORM}" \
    --build-arg "SRC=${src}" \
    -t "${dst}" \
    --push \
    -f - \
    "${ROOT}" <<'EOF'
ARG SRC
FROM --platform=$TARGETPLATFORM ${SRC}
EOF
  echo "Inspect ${dst}:"
  docker buildx imagetools inspect "${dst}" | sed -n '1,20p'
}

mirror_image "${NODE_SRC}" "${NODE_DST}"
mirror_image "${PYTHON_SRC}" "${PYTHON_DST}"

echo "Building ${PYTHON_DEPS_DST} (${PLATFORM}, torch CPU + requirements)..."
docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg "PYTHON_IMAGE=${PYTHON_DST}" \
  -f "${ROOT}/docker/Dockerfile.python-deps" \
  -t "${PYTHON_DEPS_DST}" \
  --push \
  "${ROOT}"

echo "Inspect ${PYTHON_DEPS_DST}:"
docker buildx imagetools inspect "${PYTHON_DEPS_DST}" | sed -n '1,30p'

echo ""
echo "Done. Confirm Platform is amd64 above, then set CI/CD variables:"
echo "  CI_REGISTRY=${REGISTRY}"
echo "  BASE_NODE_IMAGE=${NODE_DST}"
echo "  BASE_PYTHON_IMAGE=${PYTHON_DST}"
echo "  BASE_PYTHON_DEPS_IMAGE=${PYTHON_DEPS_DST}"
