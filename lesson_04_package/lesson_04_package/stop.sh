#!/usr/bin/env bash
# Day 4 - stop.sh: clean up everything start.sh created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

if command -v docker &> /dev/null; then
    if docker image inspect day4-hybrid-search &> /dev/null; then
        docker rmi day4-hybrid-search --force
        echo "Removed Docker image: day4-hybrid-search"
    fi
fi

echo "Cleanup complete."
