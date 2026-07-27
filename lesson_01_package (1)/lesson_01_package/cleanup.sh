#!/usr/bin/env bash
# cleanup.sh: stop containers, remove unused Docker resources, and scrub local artifacts.
set -e

CHILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Stopping and removing Day 1 containers =="
for name in day1-lesson day1-pipeline; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        docker rm -f "${name}" && echo "Removed container: ${name}"
    fi
done

echo ""
echo "== Removing day1-pipeline Docker image =="
if docker image inspect day1-pipeline &> /dev/null; then
    docker rmi day1-pipeline --force && echo "Removed image: day1-pipeline"
else
    echo "Image day1-pipeline not found - skipping."
fi

echo ""
echo "== Pruning unused Docker resources (containers, networks, volumes, build cache) =="
docker system prune -f --volumes
echo "Docker prune complete."

echo ""
echo "== Removing local Python/Node/Istio artifacts =="

# Virtual environment
if [ -d "${CHILD_DIR}/.venv" ]; then
    rm -rf "${CHILD_DIR}/.venv" && echo "Removed .venv"
fi

# Python cache files
find "${CHILD_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${CHILD_DIR}" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "${CHILD_DIR}" -name "*.pyc" -delete 2>/dev/null || true
find "${CHILD_DIR}" -name "*.pyo" -delete 2>/dev/null || true
echo "Removed Python cache/compiled files."

# Node modules
if [ -d "${CHILD_DIR}/node_modules" ]; then
    rm -rf "${CHILD_DIR}/node_modules" && echo "Removed node_modules"
fi

# Istio artifacts
find "${CHILD_DIR}" \( -name "istio*" -o -name "*.istio" \) -exec rm -rf {} + 2>/dev/null || true
echo "Istio artifacts removed (if any)."

echo ""
echo "== Cleanup complete =="
