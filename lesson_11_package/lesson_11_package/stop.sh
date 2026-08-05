#!/usr/bin/env bash
# Day 11 - stop.sh: clean up everything start.sh created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/docker_env.sh"

IMAGE_NAME="day11-e2e-matrix"
CONTAINER_NAME="day11-e2e-matrix-run"

echo "== Cleaning up =="
echo "Working directory: $SCRIPT_DIR"

DOCKER_BIN=""
if DOCKER_BIN="$(resolve_docker)"; then
    echo "Using Docker CLI: $DOCKER_BIN"
    if "$DOCKER_BIN" ps -aq --filter "name=^/${CONTAINER_NAME}$" 2>/dev/null | grep -q .; then
        "$DOCKER_BIN" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        echo "Removed container: $CONTAINER_NAME"
    fi
    ORPHANS="$("$DOCKER_BIN" ps -aq --filter "ancestor=${IMAGE_NAME}" 2>/dev/null || true)"
    if [ -n "$ORPHANS" ]; then
        # shellcheck disable=SC2086
        "$DOCKER_BIN" rm -f $ORPHANS >/dev/null 2>&1 || true
        echo "Removed orphaned containers for $IMAGE_NAME"
    fi
    if "$DOCKER_BIN" image inspect "$IMAGE_NAME" &> /dev/null; then
        "$DOCKER_BIN" rmi "$IMAGE_NAME" --force
        echo "Removed Docker image: $IMAGE_NAME"
    fi
else
    echo "No working Docker engine found - skipping container/image cleanup"
fi

pkill -f "python.*lesson_code.py" 2>/dev/null || true
pkill -f "python -m http.server 8080" 2>/dev/null || true

if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
fi

rm -f dashboard.html
rm -rf output
echo "Removed generated dashboard and output/"

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "Removed cached test/build artifacts"

echo "Cleanup complete."
