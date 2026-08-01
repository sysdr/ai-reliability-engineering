#!/usr/bin/env bash
# Day 6 - cleanup.sh: stop services/containers and remove unused Docker resources.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="day6-query-understanding"
CONTAINER_NAME="day6-query-understanding-run"
DASHBOARD_PORT="${DASHBOARD_PORT:-8766}"

echo "== Stopping local services =="
if command -v fuser &>/dev/null; then
    fuser -k "${DASHBOARD_PORT}/tcp" 2>/dev/null || true
fi
pkill -f "python.*http.server.*${DASHBOARD_PORT}" 2>/dev/null || true
pkill -f "python.*lesson_code.py" 2>/dev/null || true
pkill -f "pytest.*test_lesson" 2>/dev/null || true
echo "Local lesson/dashboard processes cleared."

echo ""
echo "== Removing project artifacts =="
# node_modules, venv, pytest cache, pyc, Istio, generated dashboards
rm -rf .venv venv node_modules .pytest_cache __pycache__ .mypy_cache .tox dist build output .cache
rm -f dashboard.html
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "venv" -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
find . -iname "*istio*" -exec rm -rf {} + 2>/dev/null || true
echo "Removed venv, caches, pyc, node_modules, dashboard.html, output/, and Istio artifacts (if present)."

if ! command -v docker &>/dev/null; then
    echo ""
    echo "Docker CLI not found - skipping container/image cleanup."
    echo "Cleanup complete."
    exit 0
fi

echo ""
echo "== Stopping and removing Day 6 containers =="
for name in "${CONTAINER_NAME}" "${IMAGE_NAME}"; do
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$name"; then
        docker stop "$name" >/dev/null 2>&1 || true
        docker rm -f "$name" >/dev/null 2>&1 || true
        echo "Removed container: $name"
    fi
done

while read -r cid; do
    [ -z "$cid" ] && continue
    docker stop "$cid" >/dev/null 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    echo "Removed container id: $cid"
done < <(docker ps -aq --filter "ancestor=${IMAGE_NAME}" 2>/dev/null || true)

echo ""
echo "== Removing Day 6 images =="
docker rmi -f "${IMAGE_NAME}" 2>/dev/null || true
docker image prune -f >/dev/null 2>&1 || true

echo ""
echo "== Pruning unused Docker resources =="
docker container prune -f || true
docker image prune -af || true
docker network prune -f || true
docker volume prune -f || true
docker builder prune -af >/dev/null 2>&1 || true

echo ""
echo "== Remaining Docker state =="
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null || true
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}' 2>/dev/null || true

echo ""
echo "== Stopping Docker engine service =="
if command -v systemctl &>/dev/null; then
    sudo systemctl stop docker.socket 2>/dev/null || true
    sudo systemctl stop docker 2>/dev/null || true
    echo "Attempted: systemctl stop docker / docker.socket"
elif command -v service &>/dev/null; then
    sudo service docker stop 2>/dev/null || true
    echo "Attempted: service docker stop"
else
    echo "No systemctl/service helper found — stop Docker Desktop manually if needed."
fi

echo ""
echo "Cleanup complete."
echo "If Docker Desktop is still running on Windows/macOS, quit it from the system tray."
