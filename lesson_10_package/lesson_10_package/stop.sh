#!/usr/bin/env bash
# Day 10 - stop.sh: clean up everything start.sh created.
set -e

resolve_docker() {
    if command -v docker.exe >/dev/null 2>&1; then
        echo "docker.exe"
        return 0
    fi
    if [ -x "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
        echo "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
        return 0
    fi
    if command -v docker >/dev/null 2>&1; then
        echo "docker"
        return 0
    fi
    return 1
}

echo "== Cleaning up =="

if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
fi

rm -f dashboard.html
echo "Removed generated dashboard"

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "Removed cached test/build artifacts"

if DOCKER_BIN="$(resolve_docker)"; then
    echo "Using Docker CLI: $DOCKER_BIN"
    if "$DOCKER_BIN" ps -a --format '{{.Names}}' | grep -qx day10-pipeline; then
        "$DOCKER_BIN" rm -f day10-pipeline
        echo "Removed Docker container: day10-pipeline"
    fi
    if "$DOCKER_BIN" image inspect day10-pipeline >/dev/null 2>&1; then
        "$DOCKER_BIN" rmi day10-pipeline --force
        echo "Removed Docker image: day10-pipeline"
    fi
fi

echo "Cleanup complete."
