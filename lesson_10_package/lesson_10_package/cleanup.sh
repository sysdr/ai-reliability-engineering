#!/usr/bin/env bash
# Day 10 - cleanup.sh: stop project services, remove local junk, prune Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

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

echo "== Stopping local lesson servers =="
pkill -f 'http.server 8000' 2>/dev/null || true
pkill -f 'lesson_code.py' 2>/dev/null || true
echo "Local servers stopped (if any)."

echo ""
echo "== Removing local artifacts =="
rm -rf node_modules .venv venv .pytest_cache __pycache__ .mypy_cache .ruff_cache .tox dist build *.egg-info
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type d -name 'node_modules' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' \) -delete 2>/dev/null || true
# Istio manifests/configs if present
find . -type d \( -iname '*istio*' -o -iname 'istio-*' \) -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -iname '*istio*' -o -iname 'istio-*.yaml' -o -iname 'istio-*.yml' \) -delete 2>/dev/null || true
rm -f dashboard.html
echo "Removed node_modules, venv, caches, .pyc, Istio files, dashboard.html (if present)."

echo ""
echo "== Stopping project containers and pruning Docker =="
cleanup_engine() {
    local DOCKER_BIN="$1"
    echo "--- Using: $DOCKER_BIN ---"
    # Project container(s)
    "$DOCKER_BIN" ps -aq --filter name=day10-pipeline 2>/dev/null | while read -r id; do
        [ -n "$id" ] && "$DOCKER_BIN" rm -f "$id" >/dev/null && echo "Removed container $id"
    done || true
    "$DOCKER_BIN" rm -f day10-pipeline >/dev/null 2>&1 || true

    # Project image
    if "$DOCKER_BIN" image inspect day10-pipeline >/dev/null 2>&1; then
        "$DOCKER_BIN" rmi -f day10-pipeline >/dev/null
        echo "Removed image: day10-pipeline"
    fi

    # Stop all running containers on this engine (lesson cleanup)
    running="$("$DOCKER_BIN" ps -q 2>/dev/null || true)"
    if [ -n "$running" ]; then
        echo "$running" | xargs -r "$DOCKER_BIN" stop
        echo "Stopped running containers."
    fi

    # Remove all stopped containers, unused networks, dangling images, build cache
    "$DOCKER_BIN" container prune -f >/dev/null
    "$DOCKER_BIN" network prune -f >/dev/null
    "$DOCKER_BIN" image prune -f >/dev/null
    "$DOCKER_BIN" builder prune -f >/dev/null 2>/dev/null || true
    "$DOCKER_BIN" system prune -f >/dev/null
    echo "Pruned unused containers, networks, dangling images, and build cache."
}

# Clean Docker Desktop (preferred) and WSL docker if both exist
CLEANED=0
if DOCKER_DESKTOP="$(resolve_docker)"; then
    cleanup_engine "$DOCKER_DESKTOP" || true
    CLEANED=1
fi
# Also clean native WSL docker when it is a different binary
if command -v docker >/dev/null 2>&1; then
    if [ "${DOCKER_DESKTOP:-}" != "docker" ]; then
        cleanup_engine "docker" || true
        CLEANED=1
    fi
fi
if [ "$CLEANED" -eq 0 ]; then
    echo "Docker CLI not found - skipped Docker cleanup."
fi

echo ""
echo "Cleanup complete."
