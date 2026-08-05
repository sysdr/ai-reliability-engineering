#!/usr/bin/env bash
# Day 12 - cleanup.sh: stop services, remove project artifacts and Docker resources.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="day12-baseline"
CONTAINER_NAME="day12-baseline-run"

echo "== Day 12 cleanup =="
echo "Working directory: $SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Resolve Docker CLI (prefer Docker Desktop on Windows/WSL)
# ---------------------------------------------------------------------------
resolve_docker() {
    local candidates=(
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
        "/mnt/c/ProgramData/DockerDesktop/version-bin/docker.exe"
        "docker.exe"
        "docker"
    )
    local c
    for c in "${candidates[@]}"; do
        if { command -v "$c" &> /dev/null || [ -x "$c" ]; } && "$c" info &> /dev/null; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Stop local lesson / dashboard processes
# ---------------------------------------------------------------------------
echo ""
echo "== Stopping local lesson services =="
pkill -f "python.*lesson_code.py" 2>/dev/null || true
pkill -f "python -m http.server 8080" 2>/dev/null || true
pkill -f "pytest.*test_lesson" 2>/dev/null || true
echo "Local lesson processes stopped (if any were running)"

# ---------------------------------------------------------------------------
# Stop / remove project containers and images on every reachable Docker engine
# ---------------------------------------------------------------------------
cleanup_docker_engine() {
    local DOCKER_BIN="$1"
    echo ""
    echo "== Docker cleanup via: $DOCKER_BIN =="

    # Project container
    if "$DOCKER_BIN" ps -aq --filter "name=^/${CONTAINER_NAME}$" 2>/dev/null | grep -q .; then
        "$DOCKER_BIN" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        echo "Removed container: $CONTAINER_NAME"
    fi

    # Any containers from this image
    local ORPHANS
    ORPHANS="$("$DOCKER_BIN" ps -aq --filter "ancestor=${IMAGE_NAME}" 2>/dev/null || true)"
    if [ -n "$ORPHANS" ]; then
        # shellcheck disable=SC2086
        "$DOCKER_BIN" rm -f $ORPHANS >/dev/null 2>&1 || true
        echo "Removed orphaned containers for $IMAGE_NAME"
    fi

    # Project image
    if "$DOCKER_BIN" image inspect "$IMAGE_NAME" &> /dev/null; then
        "$DOCKER_BIN" rmi "$IMAGE_NAME" --force >/dev/null 2>&1 || true
        echo "Removed image: $IMAGE_NAME"
    fi

    # Stop all remaining containers on this engine (project cleanup scope)
    local ALL
    ALL="$("$DOCKER_BIN" ps -aq 2>/dev/null || true)"
    if [ -n "$ALL" ]; then
        # shellcheck disable=SC2086
        "$DOCKER_BIN" stop $ALL >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        "$DOCKER_BIN" rm -f $ALL >/dev/null 2>&1 || true
        echo "Stopped and removed all containers on this engine"
    else
        echo "No containers left on this engine"
    fi

    # Remove unused images, networks, build cache
    "$DOCKER_BIN" image prune -af >/dev/null 2>&1 || true
    "$DOCKER_BIN" container prune -f >/dev/null 2>&1 || true
    "$DOCKER_BIN" network prune -f >/dev/null 2>&1 || true
    "$DOCKER_BIN" builder prune -af >/dev/null 2>&1 || true
    echo "Pruned unused Docker resources (images/containers/networks/build cache)"
}

DOCKER_SEEN=""
if DOCKER_DESKTOP="$(resolve_docker 2>/dev/null || true)"; then
    if [ -n "$DOCKER_DESKTOP" ]; then
        cleanup_docker_engine "$DOCKER_DESKTOP"
        DOCKER_SEEN="$DOCKER_DESKTOP"
    fi
fi

# Also clean the native WSL docker daemon if it is a different binary
if command -v docker &> /dev/null && docker info &> /dev/null; then
    WSL_DOCKER="$(command -v docker)"
    if [ "$WSL_DOCKER" != "$DOCKER_SEEN" ]; then
        cleanup_docker_engine "$WSL_DOCKER"
    fi
fi

# ---------------------------------------------------------------------------
# Remove project junk: node_modules, venv, caches, pyc, Istio leftovers
# ---------------------------------------------------------------------------
echo ""
echo "== Removing project junk files =="

find "$SCRIPT_DIR" -type d -name "node_modules" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type d \( -name ".venv" -o -name "venv" -o -name "env" \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
find "$SCRIPT_DIR" \( -iname "*istio*" -o -iname "istioctl*" \) -exec rm -rf {} + 2>/dev/null || true

rm -f "$SCRIPT_DIR/dashboard.html" 2>/dev/null || true
rm -rf "$SCRIPT_DIR/output" 2>/dev/null || true
rm -rf "$SCRIPT_DIR/.mypy_cache" "$SCRIPT_DIR/.ruff_cache" "$SCRIPT_DIR/.coverage" "$SCRIPT_DIR/htmlcov" 2>/dev/null || true

echo "Removed node_modules, venv/.venv, .pytest_cache, __pycache__, *.pyc, Istio files (if present)"

# ---------------------------------------------------------------------------
# Stop Docker daemon (WSL) and Docker Desktop after cleanup
# ---------------------------------------------------------------------------
echo ""
echo "== Stopping Docker service =="

# Prefer non-interactive sudo; fall back with a clear message
if command -v systemctl &> /dev/null; then
    if systemctl is-active --quiet docker 2>/dev/null || systemctl is-active --quiet docker.socket 2>/dev/null; then
        if sudo -n systemctl stop docker.socket docker 2>/dev/null; then
            echo "Stopped WSL docker.service"
        else
            echo "WARNING: cannot stop WSL docker.service without sudo password."
            echo "         Run manually:  sudo systemctl stop docker.socket docker"
        fi
    else
        echo "WSL docker.service already stopped"
    fi
fi

# Stop Docker Desktop on Windows (best-effort)
if command -v powershell.exe &> /dev/null; then
    powershell.exe -NoProfile -Command \
        "Stop-Process -Name 'Docker Desktop','com.docker.backend' -Force -ErrorAction SilentlyContinue; Get-Service com.docker.service -ErrorAction SilentlyContinue | Stop-Service -Force -ErrorAction SilentlyContinue; 'Docker Desktop stop attempted'" \
        2>/dev/null || true
    echo "Requested Docker Desktop stop"
elif [ -x "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe" ]; then
    "/mnt/c/Program Files/Docker/Docker/Docker Desktop.exe" -Quit 2>/dev/null || true
    echo "Requested Docker Desktop quit"
fi

echo ""
echo "Cleanup complete."
