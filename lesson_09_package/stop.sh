#!/usr/bin/env bash
# Day 9 - stop.sh: clean up everything start.sh created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="day9-formatter"
CONTAINER_NAME="day9-formatter-run"
DASHBOARD_PORT="${DASHBOARD_PORT:-8769}"

echo "== Cleaning up =="
echo "SCRIPT_DIR=${SCRIPT_DIR}"

# Stop any leftover local / docker services for this lesson
if command -v fuser &>/dev/null; then
    fuser -k "${DASHBOARD_PORT}/tcp" 2>/dev/null || true
fi
pkill -f "python.*http.server.*${DASHBOARD_PORT}" 2>/dev/null || true
pkill -f "python.*lesson_code.py" 2>/dev/null || true

if command -v docker &>/dev/null; then
    if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        echo "Stopped and removed container: ${CONTAINER_NAME}"
    fi
fi

if [[ -d "${SCRIPT_DIR}/.venv" ]]; then
    rm -rf "${SCRIPT_DIR}/.venv"
    echo "Removed .venv"
fi

rm -f "${SCRIPT_DIR}/dashboard.html"
rm -rf "${SCRIPT_DIR}/output" "${SCRIPT_DIR}/.cache"
echo "Removed generated dashboard and output folders"

find "${SCRIPT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${SCRIPT_DIR}" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "Removed cached test/build artifacts"

if command -v docker &>/dev/null; then
    if docker image inspect "${IMAGE_NAME}" &>/dev/null; then
        docker rmi "${IMAGE_NAME}" --force
        echo "Removed Docker image: ${IMAGE_NAME}"
    fi
fi

echo "Cleanup complete."
