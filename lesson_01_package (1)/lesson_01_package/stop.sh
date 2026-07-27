#!/usr/bin/env bash
# Day 1 - stop.sh: clean up everything start.sh created.
set -e

echo "== Cleaning up =="

if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
fi

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "Removed cached test/build artifacts"

if command -v docker &> /dev/null; then
    if docker image inspect day1-pipeline &> /dev/null; then
        docker rmi day1-pipeline --force
        echo "Removed Docker image: day1-pipeline"
    fi
fi

echo "Cleanup complete."
