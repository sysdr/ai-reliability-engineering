#!/usr/bin/env bash
# Day 3 - cleanup.sh: stop containers and remove unused Docker resources.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== Stopping local services =="
pkill -f "python.*lesson_code.py" 2>/dev/null || true
pkill -f "pytest.*test_lesson" 2>/dev/null || true
echo "Local lesson processes cleared."

echo ""
echo "== Removing project artifacts =="
rm -rf .venv venv node_modules .pytest_cache __pycache__ .mypy_cache .tox dist build
rm -f embeddings.json
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "venv" -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
find . -iname "*istio*" -exec rm -rf {} + 2>/dev/null || true
echo "Removed venv, caches, pyc, node_modules, and Istio artifacts (if present)."

if ! command -v docker &>/dev/null; then
    echo ""
    echo "Docker CLI not found - skipping container/image cleanup."
    echo "Cleanup complete."
    exit 0
fi

echo ""
echo "== Stopping and removing Day 3 containers =="
for name in day3-embeddings-run day3-embeddings; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
        docker stop "$name" >/dev/null 2>&1 || true
        docker rm -f "$name" >/dev/null 2>&1 || true
        echo "Removed container: $name"
    fi
done

# Also stop/remove any containers started from day3-embeddings images
while read -r cid; do
    [ -z "$cid" ] && continue
    docker stop "$cid" >/dev/null 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    echo "Removed container id: $cid"
done < <(docker ps -aq --filter ancestor=day3-embeddings 2>/dev/null || true)

echo ""
echo "== Removing Day 3 images =="
docker rmi -f day3-embeddings 2>/dev/null || true
# Remove dangling images left by untagged rebuilds of this lesson
docker image prune -f >/dev/null

echo ""
echo "== Pruning unused Docker resources =="
docker container prune -f
docker image prune -af
docker network prune -f
docker volume prune -f
docker builder prune -af >/dev/null 2>&1 || true

echo ""
echo "== Remaining Docker state =="
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' || true
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}' || true

echo ""
echo "Cleanup complete."
echo "To stop the Docker engine itself afterwards, run:"
echo "  sudo service docker stop    # or quit Docker Desktop"
