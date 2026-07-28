#!/usr/bin/env bash
# Day 3 - stop.sh: clean up everything start.sh created.
set -e

echo "== Cleaning up =="

if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
fi

rm -f embeddings.json
echo "Removed generated embeddings file"

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "Removed cached test/build artifacts"

if command -v docker &> /dev/null; then
    if docker image inspect day3-embeddings &> /dev/null; then
        docker rmi day3-embeddings --force
        echo "Removed Docker image: day3-embeddings"
    fi
fi

echo "Cleanup complete."
