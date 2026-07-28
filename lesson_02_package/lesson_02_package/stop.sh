#!/usr/bin/env bash
# Day 2 - stop.sh: clean up everything start.sh created.
set -e

echo "== Cleaning up =="

if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
fi

rm -f passages.json
echo "Removed generated passage store"

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "Removed cached test/build artifacts"

if command -v docker &> /dev/null; then
    if docker image inspect day2-ingestion &> /dev/null; then
        docker rmi day2-ingestion --force
        echo "Removed Docker image: day2-ingestion"
    fi
fi

echo "Cleanup complete."
