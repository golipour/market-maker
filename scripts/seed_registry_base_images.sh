#!/usr/bin/env bash
# One-time: pull Docker Hub base images on a machine that CAN reach docker.io,
# then push them into your GitLab Container Registry for the CI runner to use.
#
# Usage:
#   export CI_REGISTRY=repo.tmhezardastan.ir:5050
#   export CI_REGISTRY_USER='gitlab+deploy-token-...'
#   export CI_REGISTRY_PASSWORD='...'
#   ./scripts/seed_registry_base_images.sh
#
# Then set CI/CD variables (or rely on defaults in .gitlab-ci.yml):
#   BASE_NODE_IMAGE=$CI_REGISTRY/typhon/base/node:20-bookworm-slim
#   BASE_PYTHON_IMAGE=$CI_REGISTRY/typhon/base/python:3.11-slim-bookworm

set -euo pipefail

REGISTRY="${CI_REGISTRY:?Set CI_REGISTRY (e.g. repo.tmhezardastan.ir:5050)}"
USER_NAME="${CI_REGISTRY_USER:?Set CI_REGISTRY_USER}"
PASSWORD="${CI_REGISTRY_PASSWORD:?Set CI_REGISTRY_PASSWORD}"
PREFIX="${BASE_IMAGE_PREFIX:-${REGISTRY}/typhon/base}"

NODE_SRC="${NODE_SRC:-node:20-bookworm-slim}"
PYTHON_SRC="${PYTHON_SRC:-python:3.11-slim-bookworm}"
NODE_DST="${PREFIX}/node:20-bookworm-slim"
PYTHON_DST="${PREFIX}/python:3.11-slim-bookworm"

echo "$PASSWORD" | docker login -u "$USER_NAME" --password-stdin "$REGISTRY"

echo "Pulling ${NODE_SRC} from Docker Hub..."
docker pull "$NODE_SRC"
echo "Pulling ${PYTHON_SRC} from Docker Hub..."
docker pull "$PYTHON_SRC"

docker tag "$NODE_SRC" "$NODE_DST"
docker tag "$PYTHON_SRC" "$PYTHON_DST"

echo "Pushing ${NODE_DST}..."
docker push "$NODE_DST"
echo "Pushing ${PYTHON_DST}..."
docker push "$PYTHON_DST"

echo ""
echo "Done. In GitLab → Settings → CI/CD → Variables, set (or confirm):"
echo "  BASE_NODE_IMAGE=${NODE_DST}"
echo "  BASE_PYTHON_IMAGE=${PYTHON_DST}"
